import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import { invalidateAppConfigCache } from "./appConfig";
import {
  addAsrCredential,
  cancelRecording,
  deleteAsrCredential,
  deleteVoiceInstruction,
  getAsrCredentials,
  getRecordingSegment,
  getRecordingSnapshot,
  getRecordingStatus,
  getVoiceInputConfig,
  getVoiceInstructions,
  outputVoiceText,
  polishVoiceInputText,
  polishVoiceText,
  saveVoiceInputConfig,
  saveVoiceInstruction,
  setDefaultAsrCredential,
  startRecording,
  stopRecording,
  testAsrCredential,
  transcribeAudio,
  transcribeVoiceInputAudio,
  updateAsrCredential,
} from "./asrProvider";

const appServerMocks = vi.hoisted(() => ({
  request: vi.fn(),
  listVoiceAsrCredentials: vi.fn(),
  createVoiceAsrCredential: vi.fn(),
  updateVoiceAsrCredential: vi.fn(),
  deleteVoiceAsrCredential: vi.fn(),
  setDefaultVoiceAsrCredential: vi.fn(),
  testVoiceAsrCredential: vi.fn(),
  listVoiceInstructions: vi.fn(),
  saveVoiceInstruction: vi.fn(),
  deleteVoiceInstruction: vi.fn(),
  polishVoiceText: vi.fn(),
  transcribeVoiceAudio: vi.fn(),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

vi.mock("./appServer", () => ({
  AppServerClient: class {
    request = appServerMocks.request;
  },
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_LIST: "voiceAsrCredential/list",
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_CREATE: "voiceAsrCredential/create",
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_UPDATE: "voiceAsrCredential/update",
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DELETE: "voiceAsrCredential/delete",
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET:
    "voiceAsrCredential/default/set",
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_TEST: "voiceAsrCredential/test",
  APP_SERVER_METHOD_VOICE_INSTRUCTION_LIST: "voiceInstruction/list",
  APP_SERVER_METHOD_VOICE_INSTRUCTION_SAVE: "voiceInstruction/save",
  APP_SERVER_METHOD_VOICE_INSTRUCTION_DELETE: "voiceInstruction/delete",
  APP_SERVER_METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO:
    "voiceTranscription/transcribeAudio",
  APP_SERVER_METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT:
    "voiceTranscription/polishText",
  createAppServerClient: () => appServerMocks,
}));

function configReadResult(config: unknown, version = "version-1") {
  return {
    result: {
      config,
      layers: [
        {
          name: { type: "user", file: "/tmp/lime/config.yaml", profile: null },
          version,
          config,
        },
      ],
    },
  };
}

describe("asrProvider API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(safeInvoke).mockReset();
    Object.values(appServerMocks).forEach((mock) => {
      mock.mockReset();
    });
    invalidateAppConfigCache();
  });

  it("ASR 凭证读写应走 App Server current 并投影 provider 名", async () => {
    appServerMocks.listVoiceAsrCredentials.mockResolvedValueOnce({
      result: {
        credentials: [
          {
            id: "cred-1",
            provider: "sense_voice_local",
            is_default: true,
            disabled: false,
            language: "auto",
          },
        ],
      },
    });
    appServerMocks.createVoiceAsrCredential.mockResolvedValueOnce({
      result: {
        credential: {
          id: "cred-2",
          provider: "sense_voice_local",
          is_default: true,
          disabled: false,
          language: "zh-CN",
        },
      },
    });
    appServerMocks.updateVoiceAsrCredential.mockResolvedValueOnce({
      result: {},
    });
    appServerMocks.deleteVoiceAsrCredential.mockResolvedValueOnce({
      result: {},
    });
    appServerMocks.setDefaultVoiceAsrCredential.mockResolvedValueOnce({
      result: {},
    });
    appServerMocks.testVoiceAsrCredential.mockResolvedValueOnce({
      result: { success: true, message: "ok" },
    });

    await expect(getAsrCredentials()).resolves.toEqual([
      expect.objectContaining({
        id: "cred-1",
        provider: "sensevoice_local",
      }),
    ]);
    await expect(
      addAsrCredential({
        provider: "sensevoice_local",
        is_default: true,
        disabled: false,
        language: "zh-CN",
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        id: "cred-2",
        provider: "sensevoice_local",
      }),
    );
    await expect(
      updateAsrCredential({
        id: "cred-2",
        provider: "sensevoice_local",
        is_default: true,
        disabled: false,
        language: "zh-CN",
      }),
    ).resolves.toBeUndefined();
    await expect(deleteAsrCredential("cred-2")).resolves.toBeUndefined();
    await expect(setDefaultAsrCredential("cred-2")).resolves.toBeUndefined();
    await expect(testAsrCredential("cred-2")).resolves.toEqual(
      expect.objectContaining({ success: true }),
    );

    expect(appServerMocks.listVoiceAsrCredentials).toHaveBeenCalledWith();
    expect(appServerMocks.createVoiceAsrCredential).toHaveBeenCalledWith({
      provider: "sense_voice_local",
      is_default: true,
      disabled: false,
      language: "zh-CN",
    });
    expect(appServerMocks.updateVoiceAsrCredential).toHaveBeenCalledWith({
      credential: {
        id: "cred-2",
        provider: "sense_voice_local",
        is_default: true,
        disabled: false,
        language: "zh-CN",
      },
    });
    expect(appServerMocks.deleteVoiceAsrCredential).toHaveBeenCalledWith({
      id: "cred-2",
    });
    expect(appServerMocks.setDefaultVoiceAsrCredential).toHaveBeenCalledWith({
      id: "cred-2",
    });
    expect(appServerMocks.testVoiceAsrCredential).toHaveBeenCalledWith({
      id: "cred-2",
    });
    expect(safeInvoke).not.toHaveBeenCalledWith("get_asr_credentials");
  });

  it("旧转写、输出与录音控制入口应本地 fail closed，不再调用旧命令", async () => {
    const currentBlockedMessage =
      "旧实时语音转写、润色、输出与录音控制入口已退役，请使用 App Server current 语音转写通道，旧 Tauri in-process command 已退役。";

    await expect(
      transcribeAudio(new Uint8Array([1, 2, 3]), 16000, "cred-1"),
    ).rejects.toThrow(currentBlockedMessage);
    await expect(polishVoiceText("你好")).rejects.toThrow(
      currentBlockedMessage,
    );
    await expect(outputVoiceText("hello", "type")).rejects.toThrow(
      currentBlockedMessage,
    );
    await expect(startRecording("default")).rejects.toThrow(
      currentBlockedMessage,
    );
    await expect(stopRecording()).rejects.toThrow(currentBlockedMessage);
    await expect(getRecordingSnapshot()).rejects.toThrow(currentBlockedMessage);
    await expect(getRecordingSegment(16000, 0.8)).rejects.toThrow(
      currentBlockedMessage,
    );
    await expect(cancelRecording()).rejects.toThrow(currentBlockedMessage);
    await expect(getRecordingStatus()).rejects.toThrow(currentBlockedMessage);
    expect(safeInvoke).not.toHaveBeenCalledWith("transcribe_audio", {
      audioData: [1, 2, 3],
      sampleRate: 16000,
      credentialId: "cred-1",
    });
    for (const command of [
      "polish_voice_text",
      "output_voice_text",
      "start_recording",
      "stop_recording",
      "get_recording_snapshot",
      "get_recording_segment",
      "cancel_recording",
      "get_recording_status",
    ]) {
      expect(safeInvoke).not.toHaveBeenCalledWith(command, expect.anything());
      expect(safeInvoke).not.toHaveBeenCalledWith(command);
    }
  });

  it("输入框语音转写应走 App Server current transcribeAudio 方法", async () => {
    appServerMocks.transcribeVoiceAudio.mockResolvedValueOnce({
      result: {
        text: "你好，继续整理。",
        provider: "sense_voice_local",
        duration_secs: 1.25,
        sample_rate: 16000,
        language: "auto",
      },
    });

    await expect(
      transcribeVoiceInputAudio({
        audioBase64: "UklGRg==",
        mimeType: "audio/wav",
        credentialId: "cred-1",
      }),
    ).resolves.toEqual({
      text: "你好，继续整理。",
      provider: "sensevoice_local",
      durationSecs: 1.25,
      sampleRate: 16000,
      language: "auto",
    });

    expect(appServerMocks.transcribeVoiceAudio).toHaveBeenCalledWith({
      audio_base64: "UklGRg==",
      mime_type: "audio/wav",
      credential_id: "cred-1",
    });
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "transcribe_audio",
      expect.anything(),
    );
  });

  it("ASR 凭证 App Server 返回错误形态时不应吞成成功", async () => {
    appServerMocks.listVoiceAsrCredentials.mockResolvedValueOnce({
      result: { credentials: {} },
    });
    appServerMocks.createVoiceAsrCredential.mockResolvedValueOnce({
      result: { success: true },
    });
    appServerMocks.updateVoiceAsrCredential.mockResolvedValueOnce({
      result: { success: true },
    });
    appServerMocks.deleteVoiceAsrCredential.mockResolvedValueOnce({
      result: { success: true },
    });
    appServerMocks.setDefaultVoiceAsrCredential.mockResolvedValueOnce({
      result: { success: true },
    });
    appServerMocks.testVoiceAsrCredential.mockResolvedValueOnce({
      result: { success: true },
    });

    await expect(getAsrCredentials()).rejects.toThrow(
      "voiceAsrCredential/list did not return ASR credentials",
    );
    await expect(
      addAsrCredential({
        provider: "openai",
        is_default: true,
        disabled: false,
        language: "zh-CN",
      }),
    ).rejects.toThrow(
      "voiceAsrCredential/create did not return an ASR credential",
    );
    await expect(
      updateAsrCredential({
        id: "cred-2",
        provider: "openai",
        is_default: true,
        disabled: false,
        language: "zh-CN",
      }),
    ).rejects.toThrow(
      "voiceAsrCredential/update did not return an empty result",
    );
    await expect(deleteAsrCredential("cred-2")).rejects.toThrow(
      "voiceAsrCredential/delete did not return an empty result",
    );
    await expect(setDefaultAsrCredential("cred-2")).rejects.toThrow(
      "voiceAsrCredential/default/set did not return an empty result",
    );
    await expect(testAsrCredential("cred-2")).rejects.toThrow(
      "voiceAsrCredential/test did not return a test result",
    );
  });

  it("Voice instructions App Server 返回错误形态时不应吞成成功", async () => {
    appServerMocks.listVoiceInstructions.mockResolvedValueOnce({
      result: { instructions: {} },
    });
    appServerMocks.saveVoiceInstruction.mockResolvedValueOnce({
      result: { success: true },
    });
    appServerMocks.deleteVoiceInstruction.mockResolvedValueOnce({
      result: { success: true },
    });

    await expect(getVoiceInstructions()).rejects.toThrow(
      "voiceInstruction/list did not return voice instructions",
    );
    await expect(
      saveVoiceInstruction({
        id: "inst-2",
        name: "润色",
        prompt: "请优化",
        is_preset: false,
      }),
    ).rejects.toThrow("voiceInstruction/save did not return an empty result");
    await expect(deleteVoiceInstruction("inst-2")).rejects.toThrow(
      "voiceInstruction/delete did not return an empty result",
    );
  });

  it("应通过 app config 读写语音输入配置", async () => {
    const config = {
      enabled: true,
      shortcut: "CommandOrControl+Shift+V",
      processor: {
        polish_enabled: true,
        default_instruction_id: "default",
      },
      output: {
        mode: "type" as const,
        type_delay_ms: 10,
      },
      instructions: [],
      sound_enabled: true,
      translate_instruction_id: "default",
    };

    const currentConfig = {
      default_provider: "openai",
      experimental: {
        webmcp: { enabled: false },
        voice_input: {
          ...config,
          asr_credentials: [{ id: "cred-1" }],
        },
      },
    };
    appServerMocks.request
      .mockResolvedValueOnce(configReadResult(currentConfig))
      .mockResolvedValueOnce(
        configReadResult(
          {
            ...currentConfig,
            experimental: {
              ...currentConfig.experimental,
              voice_input: {
                ...currentConfig.experimental.voice_input,
                enabled: false,
              },
            },
          },
          "version-2",
        ),
      )
      .mockResolvedValueOnce({
        result: {
          status: "ok",
          version: "version-3",
          filePath: "/tmp/lime/config.yaml",
          overriddenMetadata: null,
        },
      });
    appServerMocks.listVoiceInstructions.mockResolvedValueOnce({
      result: {
        instructions: [
          { id: "inst-1", name: "默认", prompt: "优化", is_preset: true },
        ],
      },
    });
    appServerMocks.saveVoiceInstruction.mockResolvedValueOnce({ result: {} });
    appServerMocks.deleteVoiceInstruction.mockResolvedValueOnce({ result: {} });

    await expect(getVoiceInputConfig()).resolves.toEqual(
      expect.objectContaining({ enabled: true }),
    );
    await expect(saveVoiceInputConfig(config)).resolves.toBeUndefined();
    await expect(getVoiceInstructions()).resolves.toEqual([
      expect.objectContaining({ id: "inst-1" }),
    ]);
    await expect(
      saveVoiceInstruction({
        id: "inst-2",
        name: "润色",
        prompt: "请优化",
        is_preset: false,
      }),
    ).resolves.toBeUndefined();
    await expect(deleteVoiceInstruction("inst-2")).resolves.toBeUndefined();

    expect(appServerMocks.request).toHaveBeenNthCalledWith(
      1,
      "config/read",
      { includeLayers: true },
    );
    expect(appServerMocks.request).toHaveBeenNthCalledWith(
      2,
      "config/read",
      { includeLayers: true },
    );
    expect(appServerMocks.request).toHaveBeenNthCalledWith(
      3,
      "config/batchWrite",
      expect.objectContaining({
        expectedVersion: "version-2",
        reloadUserConfig: true,
        edits: expect.any(Array),
      }),
    );
    expect(safeInvoke).not.toHaveBeenCalledWith("get_config");
    expect(safeInvoke).not.toHaveBeenCalledWith("save_config", expect.anything());
    expect(safeInvoke).not.toHaveBeenCalledWith("get_voice_input_config");
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "save_voice_input_config",
      expect.anything(),
    );
    expect(appServerMocks.listVoiceInstructions).toHaveBeenCalledWith();
    expect(appServerMocks.saveVoiceInstruction).toHaveBeenCalledWith({
      instruction: {
        id: "inst-2",
        name: "润色",
        prompt: "请优化",
        is_preset: false,
      },
    });
    expect(appServerMocks.deleteVoiceInstruction).toHaveBeenCalledWith({
      id: "inst-2",
    });
    expect(safeInvoke).not.toHaveBeenCalledWith("get_voice_instructions");
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "save_voice_instruction",
      expect.anything(),
    );
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "delete_voice_instruction",
      expect.anything(),
    );
  });

  it("输入框语音润色应走 App Server current polishText 方法", async () => {
    appServerMocks.polishVoiceText.mockResolvedValueOnce({
      result: {
        text: "请继续整理这段内容。",
        instruction_name: "默认润色",
        polished: true,
      },
    });

    await expect(
      polishVoiceInputText({
        text: "请继续整理这个这个内容",
        instructionId: "default",
      }),
    ).resolves.toEqual({
      text: "请继续整理这段内容。",
      instructionName: "默认润色",
      polished: true,
    });

    expect(appServerMocks.polishVoiceText).toHaveBeenCalledWith({
      text: "请继续整理这个这个内容",
      instruction_id: "default",
    });
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "polish_voice_text",
      expect.anything(),
    );
  });
});
