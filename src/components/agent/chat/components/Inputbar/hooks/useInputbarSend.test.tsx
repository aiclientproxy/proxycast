import React, { useEffect } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Skill } from "@/lib/api/skills";
import type { MessageImage, MessagePathReference } from "../../../types";
import type { InputCapabilitySelection } from "../../../skill-selection/inputCapabilitySelection";
import type { InputbarSendHandler } from "../inputbarSendPayload";
import type { InputbarPluginSelection } from "../pluginInputCapability";
import { useInputbarSend } from "./useInputbarSend";

const { recordAgentUiPerformanceMetricMock } = vi.hoisted(() => ({
  recordAgentUiPerformanceMetricMock: vi.fn(),
}));

vi.mock("@/lib/agentUiPerformanceMetrics", () => ({
  recordAgentUiPerformanceMetric: recordAgentUiPerformanceMetricMock,
}));

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

interface HarnessProps {
  activeCapability?: InputCapabilitySelection | null;
  activePluginSelection?: InputbarPluginSelection | null;
  activeTools?: Record<string, boolean>;
  getInputRestoreEpoch?: () => number;
  input?: string;
  onSend: InputbarSendHandler;
  pathReferences?: MessagePathReference[];
  pendingImages?: MessageImage[];
  projectId?: string | null;
  sessionId?: string | null;
}

const mountedRoots: MountedHarness[] = [];
let latestSend:
  | ((metadata?: {
      triggeredAt?: number;
      triggerSource?: "button" | "enter" | "ime" | "adapter";
    }) => Promise<void>)
  | null = null;
let clearPathReferencesMock: ReturnType<typeof vi.fn>;
let clearActiveCapabilityMock: ReturnType<typeof vi.fn>;
let clearPendingImagesMock: ReturnType<typeof vi.fn>;

function Harness({
  activeCapability = null,
  activePluginSelection = null,
  activeTools = {},
  getInputRestoreEpoch,
  input = "",
  onSend,
  pathReferences = [],
  pendingImages = [],
  projectId = null,
  sessionId = null,
}: HarnessProps) {
  const handleSend = useInputbarSend({
    input,
    pendingImages,
    pathReferences,
    activeCapability,
    activePluginSelection,
    knowledgePackSelection: null,
    activeTools,
    projectId,
    sessionId,
    onSend,
    clearPendingImages: clearPendingImagesMock,
    clearPathReferences: clearPathReferencesMock,
    clearActiveCapability: clearActiveCapabilityMock,
    getInputRestoreEpoch,
  });

  useEffect(() => {
    latestSend = handleSend;
    return () => {
      if (latestSend === handleSend) {
        latestSend = null;
      }
    };
  }, [handleSend]);

  return null;
}

function renderHarness(props: HarnessProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoots.push({ container, root });

  act(() => {
    root.render(<Harness {...props} />);
  });
}

async function send() {
  expect(latestSend).toBeTypeOf("function");
  await act(async () => {
    await latestSend?.();
  });
}

async function sendWithMetadata(metadata: {
  triggeredAt?: number;
  triggerSource?: "button" | "enter" | "ime" | "adapter";
}) {
  expect(latestSend).toBeTypeOf("function");
  await act(async () => {
    await latestSend?.(metadata);
  });
}

function createPathReference(
  overrides: Partial<MessagePathReference> = {},
): MessagePathReference {
  return {
    id: "file:/tmp/report.md",
    path: "/tmp/report.md",
    name: "report.md",
    isDir: false,
    size: 128,
    mimeType: "text/markdown",
    source: "file_manager",
    ...overrides,
  };
}

function createInstalledSkillSelection(): InputCapabilitySelection {
  const skill: Skill = {
    key: "local:capability-report",
    name: "Capability Report",
    description: "生成能力报告",
    directory: "capability-report",
    installed: true,
    sourceKind: "other",
  };

  return {
    kind: "installed_skill",
    skill,
  };
}

describe("useInputbarSend", () => {
  afterEach(() => {
    for (const { root, container } of mountedRoots.splice(0)) {
      act(() => {
        root.unmount();
      });
      container.remove();
    }
    latestSend = null;
    vi.clearAllMocks();
  });

  beforeEach(() => {
    clearPathReferencesMock = vi.fn();
    clearActiveCapabilityMock = vi.fn();
    clearPendingImagesMock = vi.fn();
  });

  it("普通文本发送应显式下传当前输入文本", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    renderHarness({
      input: "继续优化 Skill runtime",
      onSend,
    });

    await send();

    expect(onSend).toHaveBeenCalledWith({
      images: undefined,
      textOverride: "继续优化 Skill runtime",
      sendOptions: undefined,
    });
    expect(clearPendingImagesMock).toHaveBeenCalledTimes(1);
  });

  it("纯文本无附件和模式时应走最小发送快路径", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const triggeredAt = Date.now() - 12;
    renderHarness({
      activeTools: {
        objective_mode: false,
        subagent_mode: false,
        task_mode: false,
      },
      input: "你好",
      onSend,
    });

    await sendWithMetadata({ triggeredAt, triggerSource: "button" });

    expect(onSend).toHaveBeenCalledWith({
      images: undefined,
      textOverride: "你好",
      sendOptions: undefined,
      triggeredAt,
      triggerSource: "button",
    });
    expect(recordAgentUiPerformanceMetricMock).toHaveBeenCalledWith(
      "inputbar.send.plainTextFastPath",
      expect.objectContaining({
        inputLength: 2,
        sessionId: null,
        triggerSource: "button",
      }),
    );
  });

  it("插件 chip 激活但输入未含触发词时应补齐插件触发词发送", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    renderHarness({
      activePluginSelection: {
        plugin: {
          pluginId: "content-factory-app@personal",
          displayName: "写文章",
          trigger: "@写文章",
        },
        trigger: "@写文章",
        text: "帮我整理项目资料",
        preserveInput: true,
      },
      input: "帮我整理项目资料",
      onSend,
    });

    await send();

    const payload = onSend.mock.calls[0]?.[0];
    expect(payload?.textOverride).toBe("@写文章 帮我整理项目资料");
    expect(payload?.sendOptions?.inputMentions).toEqual([
      {
        type: "mention",
        name: "写文章",
        path: "plugin://content-factory-app@personal",
      },
    ]);
    expect(payload?.sendOptions?.inputRestoreDraft?.textElements).toEqual([
      { type: "text", text: "@写文章 帮我整理项目资料" },
    ]);
  });

  it("只有路径引用时仍应发送路径占位文本并保留 metadata", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const pathReferences = [createPathReference()];
    renderHarness({
      input: "   ",
      onSend,
      pathReferences,
    });

    await send();

    const payload = onSend.mock.calls[0]?.[0];
    expect(payload?.sendOptions?.inputRestoreDraft?.pathReferences).not.toBe(
      pathReferences,
    );
    expect(onSend).toHaveBeenCalledWith({
      images: undefined,
      textOverride: "请查看这些文件或文件夹。",
      sendOptions: {
        inputRestoreDraft: {
          text: "",
          images: [],
          pathReferences: [createPathReference()],
          textElements: [],
          inputCapabilityRoute: undefined,
        },
        requestMetadata: {
          path_references: [
            {
              path: "/tmp/report.md",
              name: "report.md",
              is_dir: false,
              isDir: false,
              size: 128,
              mime_type: "text/markdown",
              mimeType: "text/markdown",
              source: "file_manager",
            },
          ],
          harness: {
            file_references: [
              {
                path: "/tmp/report.md",
                name: "report.md",
                is_dir: false,
                isDir: false,
                size: 128,
                mime_type: "text/markdown",
                mimeType: "text/markdown",
                source: "file_manager",
              },
            ],
            fileReferences: [
              {
                path: "/tmp/report.md",
                name: "report.md",
                is_dir: false,
                isDir: false,
                size: 128,
                mime_type: "text/markdown",
                mimeType: "text/markdown",
                source: "file_manager",
              },
            ],
          },
        },
      },
    });
    expect(clearPathReferencesMock).toHaveBeenCalledTimes(1);
  });

  it("富输入恢复草稿应使用发送瞬间的独立快照", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const image: MessageImage = {
      data: "image-data",
      mediaType: "image/png",
      sourcePath: "/tmp/screenshot.png",
    };
    const pendingImages = [image];
    const pathReferences = [createPathReference()];
    renderHarness({
      activeCapability: createInstalledSkillSelection(),
      input: "请结合截图和文件生成能力报告",
      onSend,
      pathReferences,
      pendingImages,
    });

    await send();

    const payload = onSend.mock.calls[0]?.[0];
    expect(payload?.sendOptions?.inputRestoreDraft).toMatchObject({
      text: "请结合截图和文件生成能力报告",
      images: [image],
      pathReferences: [createPathReference()],
      inputCapabilityRoute: {
        kind: "installed_skill",
        skillKey: "capability-report",
        skillName: "Capability Report",
      },
    });
    expect(payload?.sendOptions?.inputRestoreDraft?.images).not.toBe(
      pendingImages,
    );
    expect(payload?.sendOptions?.inputRestoreDraft?.pathReferences).not.toBe(
      pathReferences,
    );
  });

  it("发送期间发生输入恢复时不应再清理刚恢复的富输入状态", async () => {
    let inputRestoreEpoch = 1;
    const image: MessageImage = {
      data: "image-data",
      mediaType: "image/png",
      sourcePath: "/tmp/screenshot.png",
    };
    const pathReferences = [createPathReference()];
    const onSend = vi.fn(async () => {
      inputRestoreEpoch = 2;
      return true;
    });

    renderHarness({
      activeCapability: createInstalledSkillSelection(),
      getInputRestoreEpoch: () => inputRestoreEpoch,
      input: "请结合截图和文件生成能力报告",
      onSend,
      pathReferences,
      pendingImages: [image],
    });

    await send();

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(clearPendingImagesMock).not.toHaveBeenCalled();
    expect(clearPathReferencesMock).not.toHaveBeenCalled();
    expect(clearActiveCapabilityMock).not.toHaveBeenCalled();
  });

  it("技能路由和 typed 计划模式不应因显式文本发送丢失", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    renderHarness({
      activeCapability: createInstalledSkillSelection(),
      activeTools: {
        task_mode: true,
        subagent_mode: true,
      },
      input: "生成一份能力报告",
      onSend,
      sessionId: "thread-skill-runtime",
    });

    await send();

    expect(onSend).toHaveBeenCalledWith({
      images: undefined,
      textOverride: "生成一份能力报告",
      sendOptions: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "capability-report",
          skillName: "Capability Report",
        },
        inputRestoreDraft: {
          text: "生成一份能力报告",
          images: [],
          pathReferences: [],
          textElements: [{ type: "text", text: "生成一份能力报告" }],
          inputCapabilityRoute: {
            kind: "installed_skill",
            skillKey: "capability-report",
            skillName: "Capability Report",
          },
        },
        displayContent: "生成一份能力报告",
        collaborationMode: "plan",
        toolPreferencesOverride: {
          task: true,
          subagent: true,
        },
      },
    });
    expect(clearActiveCapabilityMock).toHaveBeenCalledTimes(1);
  });

  it("目标模式应通过 typed threadGoal 发送同一文本", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    renderHarness({
      activeTools: {
        objective_mode: true,
      },
      input: "保持当前修复目标",
      onSend,
      projectId: "workspace-1",
      sessionId: "thread-goal-1",
    });

    await send();

    expect(onSend).toHaveBeenCalledWith(
      expect.objectContaining({
        textOverride: "保持当前修复目标",
        sendOptions: expect.objectContaining({
          threadGoal: { objective: "保持当前修复目标" },
        }),
      }),
    );
  });
});
