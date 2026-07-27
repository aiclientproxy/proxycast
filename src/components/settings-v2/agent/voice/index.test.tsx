import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { Config } from "@/lib/api/appConfig";
import {
  persistVoiceModelSettingsFocusRequest,
  VOICE_MODEL_SETTINGS_FOCUS_STORAGE_KEY,
  VOICE_MODEL_SETTINGS_SECTION_ID,
} from "@/lib/voiceModelSettingsNavigation";
import {
  cleanupMountedVoiceSettings,
  createDeferred,
  createVoiceInputConfig,
  flushEffects,
  renderComponent,
} from "./voiceSettingsTestFixtures";

const {
  mockGetConfig,
  mockSaveConfig,
  mockUpdateConfig,
  mockGetVoiceInputConfig,
  mockSaveVoiceInputConfig,
  mockListVoiceModelCatalog,
  mockGetVoiceModelInstallState,
  mockDownloadVoiceModel,
  mockListenVoiceModelDownloadProgress,
  mockDeleteVoiceModel,
  mockSetDefaultVoiceModel,
  mockTestTranscribeVoiceModelFile,
  mockOpenDialog,
} = vi.hoisted(() => {
  return {
    mockGetConfig: vi.fn(),
    mockSaveConfig: vi.fn(),
    mockUpdateConfig: vi.fn(),
    mockGetVoiceInputConfig: vi.fn(),
    mockSaveVoiceInputConfig: vi.fn(),
    mockListVoiceModelCatalog: vi.fn(),
    mockGetVoiceModelInstallState: vi.fn(),
    mockDownloadVoiceModel: vi.fn(),
    mockListenVoiceModelDownloadProgress: vi.fn(),
    mockDeleteVoiceModel: vi.fn(),
    mockSetDefaultVoiceModel: vi.fn(),
    mockTestTranscribeVoiceModelFile: vi.fn(),
    mockOpenDialog: vi.fn(),
  };
});

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
  saveConfig: mockSaveConfig,
  updateConfig: mockUpdateConfig,
}));

vi.mock("@/lib/api/asrProvider", () => ({
  getVoiceInputConfig: mockGetVoiceInputConfig,
  saveVoiceInputConfig: mockSaveVoiceInputConfig,
}));

vi.mock("@/lib/api/voiceModels", () => ({
  listVoiceModelCatalog: mockListVoiceModelCatalog,
  getVoiceModelInstallState: mockGetVoiceModelInstallState,
  downloadVoiceModel: mockDownloadVoiceModel,
  listenVoiceModelDownloadProgress: mockListenVoiceModelDownloadProgress,
  deleteVoiceModel: mockDeleteVoiceModel,
  setDefaultVoiceModel: mockSetDefaultVoiceModel,
  testTranscribeVoiceModelFile: mockTestTranscribeVoiceModelFile,
}));

vi.mock("@/lib/desktop-host/plugin-dialog", () => ({
  open: mockOpenDialog,
}));

vi.mock("@/hooks/useConfiguredProviders", () => ({
  useConfiguredProviders: () => ({
    providers: [
      {
        key: "openai",
        label: "OpenAI",
        registryId: "openai",
        type: "openai",
        providerId: "openai",
        models: ["gpt-4.1-mini", "gpt-4o-mini-tts"],
      },
    ],
    loading: false,
  }),
  findConfiguredProviderBySelection: (
    providers: Array<{
      key: string;
      providerId?: string;
    }>,
    selection?: string,
  ) =>
    providers.find(
      (provider) =>
        provider.key === selection || provider.providerId === selection,
    ) ?? null,
}));

vi.mock("@/components/input-kit", () => ({
  ModelSelector: ({
    providerType,
    model,
  }: {
    providerType: string;
    model: string;
  }) => (
    <div data-testid="voice-model-selector">
      {providerType || "自动选择"} / {model || "自动选择"}
    </div>
  ),
}));

vi.mock("@/components/voice/InstructionEditor", () => ({
  InstructionEditor: ({
    defaultInstructionId,
    onDefaultChange,
    onInstructionsChange,
  }: {
    defaultInstructionId?: string;
    onDefaultChange?: (id: string) => void;
    onInstructionsChange?: (
      instructions: Array<{
        id: string;
        name: string;
        prompt: string;
        is_preset: boolean;
      }>,
    ) => void;
  }) => (
    <div data-testid="instruction-editor">
      <span>{defaultInstructionId}</span>
      <button type="button" onClick={() => onDefaultChange?.("email")}>
        设置默认指令
      </button>
      <button
        type="button"
        onClick={() =>
          onInstructionsChange?.([
            {
              id: "default",
              name: "默认润色",
              prompt: "{{text}}",
              is_preset: true,
            },
            {
              id: "email",
              name: "邮件格式",
              prompt: "{{text}}",
              is_preset: false,
            },
          ])
        }
      >
        同步指令
      </button>
    </div>
  ),
}));

let emitVoiceModelProgress:
  | ((event: {
      model_id: string;
      phase: string;
      downloaded_bytes: number;
      total_bytes?: number | null;
      overall_progress: number;
      message: string;
    }) => void)
  | null = null;
let persistedConfig: Config;
const scrollIntoViewMock = vi.fn();
const originalScrollIntoViewDescriptor = Object.getOwnPropertyDescriptor(
  Element.prototype,
  "scrollIntoView",
);
const originalHtmlScrollIntoViewDescriptor = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  "scrollIntoView",
);

async function waitForScrollIntoView() {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await flushEffects(2);
    if (scrollIntoViewMock.mock.calls.length > 0) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("等待语音模型区块滚动聚焦超时");
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  await changeLimeLocale("en-US");
  scrollIntoViewMock.mockClear();
  emitVoiceModelProgress = null;
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    writable: true,
    value: scrollIntoViewMock,
  });
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    writable: true,
    value: scrollIntoViewMock,
  });
  vi.stubGlobal(
    "requestAnimationFrame",
    (callback: (timestamp: number) => void) => {
      callback(0);
      return 1;
    },
  );

  persistedConfig = {
    default_provider: "openai",
    workspace_preferences: {
      media_defaults: {
        voice: {
          preferredProviderId: "openai",
          preferredModelId: "gpt-4o-mini-tts",
          allowFallback: false,
        },
      },
    },
  } as Config;
  mockGetConfig.mockImplementation(async () => persistedConfig);

  mockGetVoiceInputConfig.mockResolvedValue(createVoiceInputConfig());

  mockListVoiceModelCatalog.mockResolvedValue([
    {
      id: "sensevoice-small-int8-2024-07-17",
      name: "SenseVoice Small INT8",
      provider: "FunAudioLLM / sherpa-onnx",
      description: "",
      version: "2024-07-17",
      languages: ["zh", "en", "ja", "ko", "yue"],
      size_bytes: 262144000,
      download_url: "https://example.test/sensevoice.tar.bz2",
      vad_model_id: "silero-vad-onnx",
      vad_download_url: "https://example.test/silero_vad.onnx",
      runtime: "sherpa-onnx",
      bundled: false,
      checksum_sha256: null,
    },
  ]);
  mockGetVoiceModelInstallState.mockResolvedValue({
    model_id: "sensevoice-small-int8-2024-07-17",
    installed: false,
    installing: false,
    install_dir: "/mock/lime/models/voice/sensevoice-small-int8-2024-07-17",
    model_file: null,
    tokens_file: null,
    vad_file: null,
    installed_bytes: 0,
    last_verified_at: 1,
    missing_files: ["model.int8.onnx", "tokens.txt", "silero_vad.onnx"],
    default_credential_id: null,
  });
  mockDownloadVoiceModel.mockResolvedValue({
    state: {
      model_id: "sensevoice-small-int8-2024-07-17",
      installed: true,
      installing: false,
      install_dir: "/mock/lime/models/voice/sensevoice-small-int8-2024-07-17",
      model_file: "/mock/model.int8.onnx",
      tokens_file: "/mock/tokens.txt",
      vad_file: "/mock/silero_vad.onnx",
      installed_bytes: 262144000,
      last_verified_at: 2,
      missing_files: [],
      default_credential_id: null,
    },
  });
  mockListenVoiceModelDownloadProgress.mockImplementation(async (callback) => {
    emitVoiceModelProgress = callback;
    return vi.fn();
  });
  mockDeleteVoiceModel.mockResolvedValue({
    model_id: "sensevoice-small-int8-2024-07-17",
    installed: false,
    installing: false,
    install_dir: "/mock/lime/models/voice/sensevoice-small-int8-2024-07-17",
    model_file: null,
    tokens_file: null,
    vad_file: null,
    installed_bytes: 0,
    last_verified_at: 3,
    missing_files: ["model.int8.onnx", "tokens.txt", "silero_vad.onnx"],
    default_credential_id: null,
  });
  mockSetDefaultVoiceModel.mockResolvedValue({
    id: "sensevoice-local-sensevoice-small-int8-2024-07-17",
    provider: "sensevoice_local",
    name: "SenseVoice Small 本地",
    is_default: true,
    disabled: false,
    language: "auto",
  });
  mockTestTranscribeVoiceModelFile.mockResolvedValue({
    text: "这是测试音频的本地转写结果。",
    duration_secs: 2.5,
    sample_rate: 16000,
    language: "auto",
  });
  mockOpenDialog.mockResolvedValue("/tmp/interview.wav");
  mockSaveConfig.mockResolvedValue(undefined);
  mockUpdateConfig.mockImplementation(
    async (updater: (current: Config) => Config) => {
      persistedConfig = updater(persistedConfig);
      await mockSaveConfig(persistedConfig);
      return persistedConfig;
    },
  );
  mockSaveVoiceInputConfig.mockResolvedValue(undefined);
});

afterEach(async () => {
  cleanupMountedVoiceSettings();
  window.sessionStorage.removeItem(VOICE_MODEL_SETTINGS_FOCUS_STORAGE_KEY);
  if (originalScrollIntoViewDescriptor) {
    Object.defineProperty(
      Element.prototype,
      "scrollIntoView",
      originalScrollIntoViewDescriptor,
    );
  } else {
    delete (Element.prototype as Partial<Element>).scrollIntoView;
  }
  if (originalHtmlScrollIntoViewDescriptor) {
    Object.defineProperty(
      HTMLElement.prototype,
      "scrollIntoView",
      originalHtmlScrollIntoViewDescriptor,
    );
  } else {
    delete (HTMLElement.prototype as Partial<HTMLElement>).scrollIntoView;
  }
  vi.unstubAllGlobals();
  await changeLimeLocale("zh-CN");
});

describe("VoiceSettings", () => {
  it("应渲染语音模型、语音处理和语音服务模型设置，并下线实时语音入口", async () => {
    const container = renderComponent();
    await flushEffects(6);

    const text = container.textContent ?? "";
    expect(text).toContain("Voice Models");
    expect(text).toContain("SenseVoice Small");
    expect(text).toContain("Local");
    expect(text).toContain("Not installed (ONNX int8 quantized");
    expect(text).toContain(
      "Local offline ASR; the model is downloaded to user data on demand.",
    );
    expect(text).toContain("Download model");
    expect(text).toContain("Voice Processing");
    expect(text).toContain("Voice Service Model");
    expect(text).toContain("openai / gpt-4.1-mini");
    expect(text).toContain("openai / gpt-4o-mini-tts");
    expect(text).toContain("Translation mode instruction");
    expect(text).not.toContain("Voice Input");
    expect(text).not.toContain("Voice input shortcut");
    expect(text).not.toContain("Hold to record, release to transcribe");
    expect(text).not.toContain("🌐 Fn");
    expect(
      container.querySelector("button[aria-label='Toggle voice input']"),
    ).toBeNull();
    expect(
      container.querySelector("[data-testid='microphone-test']"),
    ).toBeNull();
    expect(text).not.toContain("Translation mode shortcut registered");
    expect(text).not.toContain("settings.voice");
  });

  it("点击下载模型时应调用本地模型下载命令", async () => {
    const container = renderComponent();
    await flushEffects(6);

    const downloadButton = Array.from(
      container.querySelectorAll("button"),
    ).find((element) => element.textContent?.includes("Download model"));
    expect(downloadButton).toBeInstanceOf(HTMLButtonElement);

    await act(async () => {
      downloadButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flushEffects(4);
    });

    expect(mockDownloadVoiceModel).toHaveBeenCalledWith(
      "sensevoice-small-int8-2024-07-17",
    );
    expect(container.textContent ?? "").toContain("Installed");
  });

  it("缺模型跳转进入设置页时应聚焦语音模型区块", async () => {
    persistVoiceModelSettingsFocusRequest({
      source: "inputbar",
      reason: "missing-model",
      modelId: "sensevoice-small-int8-2024-07-17",
    });

    const container = renderComponent();
    await flushEffects(6);

    const voiceModelSection = container.querySelector(
      `#${VOICE_MODEL_SETTINGS_SECTION_ID}`,
    );
    expect(voiceModelSection).toBeInstanceOf(HTMLElement);
    await waitForScrollIntoView();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
    expect(
      window.sessionStorage.getItem(VOICE_MODEL_SETTINGS_FOCUS_STORAGE_KEY),
    ).toBeNull();
  });

  it("模型下载中应从 0 开始并跟随真实进度事件", async () => {
    const pendingDownload = createDeferred<{
      state: {
        model_id: string;
        installed: boolean;
        installing: boolean;
        install_dir: string;
        model_file: string;
        tokens_file: string;
        vad_file: string;
        installed_bytes: number;
        last_verified_at: number;
        missing_files: string[];
        default_credential_id: null;
      };
    }>();
    mockDownloadVoiceModel.mockReturnValueOnce(pendingDownload.promise);

    const container = renderComponent();
    await flushEffects(6);

    const downloadButton = Array.from(
      container.querySelectorAll("button"),
    ).find((element) => element.textContent?.includes("Download model"));
    expect(downloadButton).toBeInstanceOf(HTMLButtonElement);

    await act(async () => {
      downloadButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    const progressbar = container.querySelector(
      "[role='progressbar'][aria-label='Voice model download progress']",
    ) as HTMLDivElement | null;
    expect(progressbar).toBeInstanceOf(HTMLDivElement);
    expect(progressbar?.getAttribute("aria-valuenow")).toBe("0");
    expect(progressbar?.style.width).toBe("0%");
    expect(container.textContent ?? "").toContain("Preparing model download");
    expect(container.textContent ?? "").toContain("0%");

    await act(async () => {
      emitVoiceModelProgress?.({
        model_id: "sensevoice-small-int8-2024-07-17",
        phase: "archive",
        downloaded_bytes: 42,
        total_bytes: 100,
        overall_progress: 0.42,
        message: "正在下载模型包",
      });
      await Promise.resolve();
    });

    expect(progressbar?.getAttribute("aria-valuenow")).toBe("42");
    expect(progressbar?.style.width).toBe("42%");
    expect(container.textContent ?? "").toContain("正在下载模型包");

    pendingDownload.resolve({
      state: {
        model_id: "sensevoice-small-int8-2024-07-17",
        installed: true,
        installing: false,
        install_dir: "/mock/lime/models/voice/sensevoice-small-int8-2024-07-17",
        model_file: "/mock/model.int8.onnx",
        tokens_file: "/mock/tokens.txt",
        vad_file: "/mock/silero_vad.onnx",
        installed_bytes: 262144000,
        last_verified_at: 2,
        missing_files: [],
        default_credential_id: null,
      },
    });
    await flushEffects(6);
  });

  it("模型已安装时应支持设为默认", async () => {
    mockGetVoiceModelInstallState.mockResolvedValue({
      model_id: "sensevoice-small-int8-2024-07-17",
      installed: true,
      installing: false,
      install_dir: "/mock/lime/models/voice/sensevoice-small-int8-2024-07-17",
      model_file: "/mock/model.int8.onnx",
      tokens_file: "/mock/tokens.txt",
      vad_file: "/mock/silero_vad.onnx",
      installed_bytes: 262144000,
      last_verified_at: 4,
      missing_files: [],
      default_credential_id: null,
    });

    const container = renderComponent();
    await flushEffects(6);

    const defaultButton = Array.from(container.querySelectorAll("button")).find(
      (element) => element.textContent?.includes("Set as default"),
    );
    expect(defaultButton).toBeInstanceOf(HTMLButtonElement);

    await act(async () => {
      defaultButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flushEffects(6);
    });

    expect(mockSetDefaultVoiceModel).toHaveBeenCalledWith(
      "sensevoice-small-int8-2024-07-17",
    );
  });

  it("模型已安装时应支持选择 WAV 文件并测试转写", async () => {
    mockGetVoiceModelInstallState.mockResolvedValue({
      model_id: "sensevoice-small-int8-2024-07-17",
      installed: true,
      installing: false,
      install_dir: "/mock/lime/models/voice/sensevoice-small-int8-2024-07-17",
      model_file: "/mock/model.int8.onnx",
      tokens_file: "/mock/tokens.txt",
      vad_file: "/mock/silero_vad.onnx",
      installed_bytes: 262144000,
      last_verified_at: 4,
      missing_files: [],
      default_credential_id: null,
    });

    const container = renderComponent();
    await flushEffects(6);

    const input = container.querySelector(
      "input[aria-label='WAV file path']",
    ) as HTMLInputElement | null;
    expect(input).toBeInstanceOf(HTMLInputElement);

    const selectButton = Array.from(container.querySelectorAll("button")).find(
      (element) => element.textContent?.includes("Choose WAV"),
    );
    expect(selectButton).toBeInstanceOf(HTMLButtonElement);

    await act(async () => {
      selectButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flushEffects(4);
    });

    expect(mockOpenDialog).toHaveBeenCalledWith({
      title: "Choose WAV test audio",
      multiple: false,
      directory: false,
      filters: [{ name: "WAV audio", extensions: ["wav"] }],
    });
    expect(input?.value).toBe("/tmp/interview.wav");

    const testButton = Array.from(container.querySelectorAll("button")).find(
      (element) => element.textContent?.includes("Test transcription"),
    );
    expect(testButton).toBeInstanceOf(HTMLButtonElement);
    expect((testButton as HTMLButtonElement).disabled).toBe(false);

    await act(async () => {
      testButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flushEffects(6);
    });

    expect(mockTestTranscribeVoiceModelFile).toHaveBeenCalledWith(
      "sensevoice-small-int8-2024-07-17",
      "/tmp/interview.wav",
    );
    expect(container.textContent ?? "").toContain(
      "这是测试音频的本地转写结果。",
    );
  });

  it("切换默认润色指令时应保存 processor.default_instruction_id", async () => {
    const container = renderComponent();
    await flushEffects(6);

    const select = container.querySelector(
      "select[aria-label='Default polish instruction']",
    ) as HTMLSelectElement | null;
    expect(select).toBeInstanceOf(HTMLSelectElement);

    await act(async () => {
      if (select) {
        select.value = "email";
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
      await flushEffects(2);
    });

    expect(mockSaveVoiceInputConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveVoiceInputConfig.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        processor: expect.objectContaining({
          default_instruction_id: "email",
        }),
      }),
    );
  });

  it("恢复默认后应清空语音生成任务覆盖", async () => {
    const container = renderComponent();
    await flushEffects(6);

    const resetButton = Array.from(container.querySelectorAll("button")).find(
      (element) => element.textContent?.includes("Restore defaults"),
    );
    expect(resetButton).toBeInstanceOf(HTMLButtonElement);

    await act(async () => {
      resetButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flushEffects(2);
    });

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    const savedConfig = mockSaveConfig.mock.calls[0][0];
    expect(
      savedConfig.workspace_preferences.media_defaults.voice,
    ).toBeUndefined();
  });
});
