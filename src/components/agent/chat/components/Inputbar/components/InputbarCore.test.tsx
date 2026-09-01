import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { agentZhCNResource } from "@/i18n/agentResources";
import { InputbarCore } from "./InputbarCore";
import {
  buildInputbarCoreCopy,
  type InputbarCoreCopyKey,
} from "./inputbarCoreCopy";

vi.mock("./InputbarTools", () => ({
  InputbarTools: () => <div data-testid="inputbar-tools">tools</div>,
}));

const asrProviderMocks = vi.hoisted(() => ({
  polishVoiceInputText: vi.fn(),
  transcribeVoiceInputAudio: vi.fn(),
}));

vi.mock("@/lib/api/asrProvider", () => asrProviderMocks);

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

type TestMediaRecorderState = "inactive" | "recording" | "paused";
type TestEventListener = (event: Event) => void;

function translateResource(
  resource: Partial<Record<InputbarCoreCopyKey, string>>,
  key: InputbarCoreCopyKey,
  values?: Record<string, number | string>,
) {
  return Object.entries(values ?? {}).reduce(
    (text, [name, value]) => text.split(`{{${name}}}`).join(String(value)),
    resource[key] ?? key,
  );
}

const TEST_INPUTBAR_CORE_COPY = buildInputbarCoreCopy((key, values) =>
  translateResource(agentZhCNResource, key, values),
);

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
  Object.values(asrProviderMocks).forEach((mock) => {
    mock.mockReset();
  });
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const renderInputbarCore = async (
  props?: Partial<Omit<React.ComponentProps<typeof InputbarCore>, "uiCopy">> & {
    uiCopy?: React.ComponentProps<typeof InputbarCore>["uiCopy"];
  },
) => {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <InputbarCore
        uiCopy={TEST_INPUTBAR_CORE_COPY}
        text=""
        setText={vi.fn()}
        onSend={vi.fn()}
        activeTools={{}}
        onToolClick={vi.fn()}
        toolMode="attach-only"
        visualVariant="floating"
        {...props}
      />,
    );
  });
  await act(async () => {
    await Promise.resolve();
  });

  mountedRoots.push({ root, container });
  return container;
};

describe("InputbarCore", () => {
  it("图片草稿没有 base64 data 时应使用 sourceUri 作为预览", async () => {
    const container = await renderInputbarCore({
      pendingImages: [
        {
          data: "",
          mediaType: "image/png",
          sourceUri: "file://queued.png",
          sourcePath: "/project/queued.png",
          previewUrl: "file://queued.png",
        },
      ],
    });

    const image = container.querySelector("img") as HTMLImageElement | null;
    expect(image?.getAttribute("src")).toBe("file://queued.png");
  });

  it("应在输入框右侧主操作区渲染 current 录音按钮", async () => {
    const container = await renderInputbarCore({
      visualVariant: "default",
      toolMode: "default",
    });

    const primaryActions = container.querySelector(
      '[data-testid="inputbar-primary-actions"]',
    );
    const dictationButton = container.querySelector(
      '[data-testid="inputbar-dictation-toggle"]',
    ) as HTMLButtonElement | null;
    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;

    expect(primaryActions).toBeTruthy();
    expect(dictationButton).toBeTruthy();
    expect(dictationButton?.getAttribute("aria-label")).toBe("开始语音输入");
    expect(
      dictationButton?.closest('[data-testid="inputbar-primary-actions"]'),
    ).toBe(primaryActions);
    expect(
      sendButton?.closest('[data-testid="inputbar-primary-actions"]'),
    ).toBe(primaryActions);
    expect(document.querySelector('[aria-live="polite"]')).toBeNull();
  });

  it("录音中应隐藏发送按钮并阻止 Enter 发送", async () => {
    const onSend = vi.fn();
    const originalMediaDevices = navigator.mediaDevices;
    const stopTrack = vi.fn();
    const mediaDevices = {
      getUserMedia: vi.fn(async () => ({
        getTracks: () => [{ stop: stopTrack }],
      })),
    };

    class FakeMediaRecorder {
      static isTypeSupported() {
        return true;
      }

      mimeType = "audio/webm";
      state: TestMediaRecorderState = "inactive";
      private readonly listeners = new Map<string, Set<TestEventListener>>();

      addEventListener(type: string, listener: TestEventListener) {
        const listeners =
          this.listeners.get(type) ?? new Set<TestEventListener>();
        listeners.add(listener);
        this.listeners.set(type, listeners);
      }

      removeEventListener(type: string, listener: TestEventListener) {
        this.listeners.get(type)?.delete(listener);
      }

      start() {
        this.state = "recording";
      }

      stop() {
        this.state = "inactive";
        this.listeners
          .get("stop")
          ?.forEach((listener) => listener(new Event("stop")));
      }

      requestData() {}
    }

    class FakeAudioContext {
      close() {
        return Promise.resolve();
      }
    }

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: mediaDevices,
    });
    vi.stubGlobal(
      "MediaRecorder",
      FakeMediaRecorder as unknown as typeof MediaRecorder,
    );
    vi.stubGlobal(
      "AudioContext",
      FakeAudioContext as unknown as typeof AudioContext,
    );

    const container = await renderInputbarCore({
      text: "可以发送",
      onSend,
    });
    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;
    const dictationButton = container.querySelector(
      '[data-testid="inputbar-dictation-toggle"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      dictationButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });

    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
    expect(container.querySelector('[data-testid="send-btn"]')).toBeNull();
    expect(textarea?.disabled).toBe(true);

    act(() => {
      textarea?.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          bubbles: true,
        }),
      );
    });

    expect(onSend).not.toHaveBeenCalled();

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: originalMediaDevices,
    });
  });

  it("录音中应将本地 ASR 的实时识别结果流式写入输入框草稿", async () => {
    vi.useFakeTimers();
    const setText = vi.fn();
    asrProviderMocks.transcribeVoiceInputAudio.mockResolvedValue({
      text: "实时识别文本",
      provider: "sensevoice_local",
      durationSecs: 1.2,
      sampleRate: 16000,
      language: "auto",
    });
    const mediaDevices = {
      getUserMedia: vi.fn(async () => ({
        getTracks: () => [{ stop: vi.fn() }],
      })),
    };
    let processor: {
      onaudioprocess: ((event: AudioProcessingEvent) => void) | null;
    } | null = null;

    class FakeMediaRecorder {
      static isTypeSupported() {
        return true;
      }

      mimeType = "audio/webm";
      state: TestMediaRecorderState = "inactive";

      addEventListener() {}

      removeEventListener() {}

      start() {
        this.state = "recording";
      }

      requestData() {}
    }

    class FakeAudioContext {
      sampleRate = 16000;
      destination = {};

      createMediaStreamSource() {
        return {
          connect: vi.fn(),
          disconnect: vi.fn(),
        };
      }

      createScriptProcessor() {
        processor = {
          onaudioprocess: null,
        };
        return {
          connect: vi.fn(),
          disconnect: vi.fn(),
          get onaudioprocess() {
            return processor?.onaudioprocess ?? null;
          },
          set onaudioprocess(
            listener: ((event: AudioProcessingEvent) => void) | null,
          ) {
            if (processor) {
              processor.onaudioprocess = listener;
            }
          },
        };
      }

      resume() {
        return Promise.resolve();
      }

      close() {
        return Promise.resolve();
      }
    }

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: mediaDevices,
    });
    vi.stubGlobal(
      "MediaRecorder",
      FakeMediaRecorder as unknown as typeof MediaRecorder,
    );
    vi.stubGlobal(
      "AudioContext",
      FakeAudioContext as unknown as typeof AudioContext,
    );

    const container = await renderInputbarCore({ setText });
    const dictationButton = container.querySelector(
      '[data-testid="inputbar-dictation-toggle"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      dictationButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });
    await act(async () => {
      processor?.onaudioprocess?.({
        inputBuffer: {
          length: 12000,
          numberOfChannels: 1,
          getChannelData: () => new Float32Array(12000).fill(0.2),
        },
      } as unknown as AudioProcessingEvent);
      vi.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();
    });

    const preview = container.querySelector(
      '[data-testid="inputbar-dictation-live-transcript"]',
    );
    expect(preview?.textContent).toContain("实时识别文本");
    expect(setText).toHaveBeenCalledWith("实时识别文本");
    expect(asrProviderMocks.transcribeVoiceInputAudio).toHaveBeenCalledTimes(1);
  });

  it("主题工作台空输入时应保持单行紧凑态，聚焦后也不应放大", async () => {
    const container = await renderInputbarCore();
    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;
    const inputBar = container.querySelector(
      '[data-testid="inputbar-core-container"]',
    ) as HTMLDivElement | null;
    expect(textarea).toBeTruthy();
    expect(inputBar).toBeTruthy();
    expect(textarea?.className).toContain("floating-collapsed");
    expect(
      container.querySelector('[data-testid="inputbar-tools"]'),
    ).toBeNull();

    act(() => {
      inputBar?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
      textarea?.focus();
    });

    expect(textarea?.className).toContain("floating-collapsed");
    expect(
      container.querySelector('[data-testid="inputbar-tools"]'),
    ).toBeNull();
  });

  it("主题工作台有输入内容时应展开为常规编辑态", async () => {
    const container = await renderInputbarCore({
      text: "继续补充当前分析",
    });
    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;

    expect(textarea).toBeTruthy();
    expect(textarea?.className).not.toContain("floating-collapsed");
    expect(
      container.querySelector('button[aria-label="添加图片"]'),
    ).toBeTruthy();
  });

  it("应在 textarea 上暴露当前会话标识，便于发送链路回归定位", async () => {
    const container = await renderInputbarCore({
      sessionId: "session-inputbar-1",
    });
    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;

    expect(textarea?.getAttribute("data-session-id")).toBe(
      "session-inputbar-1",
    );
  });

  it("左侧加号应打开输入设置浮层并触发工具动作", async () => {
    const onAddFiles = vi.fn();
    const onToggleTask = vi.fn();
    const onToggleObjective = vi.fn();
    const onToggleSubagent = vi.fn();
    const container = await renderInputbarCore({
      text: "继续处理",
      plusMenu: {
        labels: {
          open: "打开输入设置",
          addFiles: "添加照片和文件",
          attachKnowledge: "附加资料",
          planMode: "计划模式",
          subagent: "子代理",
          objective: "追求目标",
          skills: "技能",
          plugins: "插件",
          unavailable: "当前会话暂不可用",
        },
        taskEnabled: true,
        subagentEnabled: false,
        knowledgeActive: true,
        objectiveActive: true,
        skillsActive: true,
        onAddFiles,
        onToggleTask,
        onToggleObjective,
        onToggleSubagent,
        knowledgePanel: <div>资料面板</div>,
        skillsPanel: <div>技能面板</div>,
      },
    });

    const plusButton = container.querySelector(
      'button[aria-label="打开输入设置"]',
    ) as HTMLButtonElement | null;
    expect(plusButton).toBeTruthy();

    await act(async () => {
      plusButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    const menu = document.body.querySelector(
      '[data-testid="inputbar-plus-menu"]',
    );
    expect(menu).toBeTruthy();
    expect(menu?.textContent).toContain("添加照片和文件");
    expect(menu?.textContent).toContain("计划模式");
    expect(menu?.textContent).toContain("子代理");
    expect(menu?.textContent).toContain("追求目标");
    expect(menu?.textContent).toContain("技能");

    const planModeRow = document.body.querySelector(
      '[data-testid="inputbar-plus-plan-mode"]',
    );
    const subagentModeRow = document.body.querySelector(
      '[data-testid="inputbar-plus-subagent-mode"]',
    );
    const objectiveRow = document.body.querySelector(
      '[data-testid="inputbar-plus-objective"]',
    );
    expect(planModeRow?.getAttribute("role")).toBe("menuitemcheckbox");
    expect(planModeRow?.getAttribute("aria-checked")).toBe("true");
    expect(planModeRow?.querySelector('[data-state="checked"]')).toBeTruthy();
    expect(subagentModeRow?.getAttribute("role")).toBe("menuitemcheckbox");
    expect(subagentModeRow?.getAttribute("aria-checked")).toBe("false");
    expect(
      subagentModeRow?.querySelector('[data-state="unchecked"]'),
    ).toBeTruthy();
    expect(objectiveRow?.getAttribute("role")).toBe("menuitemcheckbox");
    expect(objectiveRow?.getAttribute("aria-checked")).toBe("true");

    await act(async () => {
      planModeRow
        ?.querySelector('[data-state="checked"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onToggleTask).toHaveBeenCalledTimes(1);

    await act(async () => {
      subagentModeRow
        ?.querySelector('[data-state="unchecked"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onToggleSubagent).toHaveBeenCalledTimes(1);

    await act(async () => {
      objectiveRow?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onToggleObjective).toHaveBeenCalledTimes(1);
    expect(
      document.body.querySelector(
        '[data-testid="inputbar-plus-panel-objective"]',
      ),
    ).toBeNull();

    await act(async () => {
      document.body
        .querySelector('[data-testid="inputbar-plus-knowledge"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(
      document.body.querySelector(
        '[data-testid="inputbar-plus-panel-knowledge"]',
      )?.textContent,
    ).toContain("资料面板");

    await act(async () => {
      document.body
        .querySelector('[data-testid="inputbar-plus-add-files"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onAddFiles).toHaveBeenCalledTimes(1);
  });

  it("底部 meta 应支持左侧状态和右侧模型分区", async () => {
    const container = await renderInputbarCore({
      text: "继续处理",
      plusMenu: {
        labels: {
          open: "打开输入设置",
          addFiles: "添加照片和文件",
          attachKnowledge: "附加资料",
          planMode: "计划模式",
          subagent: "子代理",
          objective: "追求目标",
          skills: "技能",
          plugins: "插件",
          unavailable: "当前会话暂不可用",
        },
        taskEnabled: false,
        onAddFiles: vi.fn(),
        onToggleTask: vi.fn(),
        onToggleObjective: vi.fn(),
      },
      leftExtra: <span data-testid="left-meta-item">完全访问</span>,
      trailingMeta: <span data-testid="trailing-meta-item">模型切换</span>,
    });

    const leftMeta = container.querySelector(
      '[data-testid="inputbar-meta-left"]',
    );
    const rightMeta = container.querySelector(
      '[data-testid="inputbar-meta-trailing"]',
    );

    expect(
      leftMeta?.querySelector('[data-testid="left-meta-item"]'),
    ).toBeTruthy();
    expect(
      rightMeta?.querySelector('[data-testid="trailing-meta-item"]'),
    ).toBeTruthy();
    expect(
      leftMeta?.querySelector('[data-testid="trailing-meta-item"]'),
    ).toBeNull();
  });

  it("添加路径引用时应显示 chip 并允许移除", async () => {
    const onRemovePathReference = vi.fn();
    const container = await renderInputbarCore({
      pathReferences: [
        {
          id: "dir:/Users/demo/Downloads",
          path: "/Users/demo/Downloads",
          name: "Downloads",
          isDir: true,
          source: "file_manager",
        },
      ],
      onRemovePathReference,
    });

    expect(container.textContent).toContain("Downloads");
    expect(container.textContent).toContain("本地文件夹");
    expect(container.textContent).not.toContain("/Users/demo/Downloads");
    expect(
      container.querySelector('[data-testid="inputbar-path-reference-chip"]'),
    ).toBeTruthy();

    const removeButton = container.querySelector(
      'button[aria-label="移除路径 Downloads"]',
    ) as HTMLButtonElement | null;
    expect(removeButton).toBeTruthy();

    await act(async () => {
      removeButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onRemovePathReference).toHaveBeenCalledWith(
      "dir:/Users/demo/Downloads",
    );
  });

  it("文本路径引用应提供设为项目资料动作", async () => {
    const onImportPathReferenceAsKnowledge = vi.fn();
    const reference = {
      id: "file:/Users/demo/brief.txt",
      path: "/Users/demo/brief.txt",
      name: "brief.txt",
      isDir: false,
      mimeType: "text/plain",
      source: "file_manager" as const,
    };
    const container = await renderInputbarCore({
      pathReferences: [reference],
      onImportPathReferenceAsKnowledge,
    });

    const importButton = container.querySelector(
      'button[aria-label="设为项目资料 brief.txt"]',
    ) as HTMLButtonElement | null;
    expect(importButton).toBeTruthy();

    await act(async () => {
      importButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onImportPathReferenceAsKnowledge).toHaveBeenCalledWith(reference);
  });

  it("非文本路径引用不应展示设为项目资料动作", async () => {
    const onImportPathReferenceAsKnowledge = vi.fn();
    const reference = {
      id: "file:/Users/demo/contract.pdf",
      path: "/Users/demo/contract.pdf",
      name: "contract.pdf",
      isDir: false,
      mimeType: "application/pdf",
      source: "file_manager" as const,
    };
    const container = await renderInputbarCore({
      pathReferences: [reference],
      onImportPathReferenceAsKnowledge,
    });

    expect(
      container.querySelector('button[aria-label="设为项目资料 contract.pdf"]'),
    ).toBeNull();
    expect(container.textContent).toContain("contract.pdf");
  });

  it("从输入框正文区域拖放时应由容器优先接收 drop", async () => {
    const onDrop = vi.fn((event: React.DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
    });
    const onDragOver = vi.fn((event: React.DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
    });
    const container = await renderInputbarCore({
      onDrop,
      onDragOver,
    });
    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;
    expect(textarea).toBeTruthy();

    await act(async () => {
      textarea?.dispatchEvent(new Event("dragover", { bubbles: true }));
      textarea?.dispatchEvent(new Event("drop", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onDragOver).toHaveBeenCalledTimes(1);
    expect(onDrop).toHaveBeenCalledTimes(1);
  });

  it("点击展开按钮应切换输入框展开态", async () => {
    const container = await renderInputbarCore({
      visualVariant: "default",
      toolMode: "default",
    });
    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;
    const expandButton = container.querySelector(
      'button[aria-label="展开输入框"]',
    ) as HTMLButtonElement | null;

    expect(textarea?.className).not.toContain("composer-expanded");
    expect(expandButton).toBeTruthy();

    await act(async () => {
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(textarea?.className).toContain("composer-expanded");
    expect(
      container.querySelector('button[aria-label="收起输入框"]'),
    ).toBeTruthy();
  });

  it("生成中且没有下一条草稿时应显示正在输出与停止按钮", async () => {
    const onSend = vi.fn();
    const onStop = vi.fn();
    const container = await renderInputbarCore({
      text: "",
      onSend,
      onStop,
      isLoading: true,
    });

    const runningButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("正在输出"),
    ) as HTMLButtonElement | undefined;
    const stopButton = container.querySelector(
      'button[aria-label="停止"]',
    ) as HTMLButtonElement | null;

    expect(runningButton).toBeTruthy();
    expect(runningButton?.disabled).toBe(true);
    expect(stopButton).toBeTruthy();
    expect(container.textContent).not.toContain("稍后处理");

    act(() => {
      stopButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSend).not.toHaveBeenCalled();
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("生成中有草稿时应允许显式加入 canonical queue", async () => {
    const onSend = vi.fn();
    const container = await renderInputbarCore({
      text: "下一步检查测试",
      onSend,
      onStop: vi.fn(),
      isLoading: true,
    });
    const queueButton = container.querySelector(
      'button[aria-label="加入队列"]',
    ) as HTMLButtonElement | null;

    expect(queueButton).toBeTruthy();
    await act(async () => queueButton?.click());
    expect(onSend).toHaveBeenCalledWith(
      expect.objectContaining({
        triggerSource: "button",
        submitTarget: "queue",
      }),
    );
  });

  it("点击图片删除按钮应触发 onRemoveImage", async () => {
    const onRemoveImage = vi.fn();
    const container = await renderInputbarCore({
      pendingImages: [
        {
          data: "aGVsbG8=",
          mediaType: "image/png",
        },
      ],
      onRemoveImage,
    });

    const removeButton = container.querySelector(
      'button[aria-label="移除图片 1"]',
    ) as HTMLButtonElement | null;

    expect(removeButton).toBeTruthy();

    act(() => {
      removeButton?.dispatchEvent(
        new MouseEvent("mousedown", { bubbles: true }),
      );
      removeButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onRemoveImage).toHaveBeenCalledWith(0);
  });
});
