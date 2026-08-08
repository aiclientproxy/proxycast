import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { Artifact } from "@/lib/artifact/types";
import { createInitialDocumentState } from "@/components/workspace/canvas/canvasUtils";
import type { ArtifactBlockRewriteCompletion } from "./artifactWorkbenchRewrite";
import { useWorkspaceCanvasWorkflowActions } from "./useWorkspaceCanvasWorkflowActions";

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: vi.fn(),
}));

vi.mock("@/lib/api/session-files", () => ({
  importDocument: vi.fn(),
}));

const toastError = vi.fn();

vi.mock("sonner", () => ({
  toast: {
    info: vi.fn(),
    success: vi.fn(),
    error: (...args: unknown[]) => toastError(...args),
  },
}));

type HookProps = Parameters<typeof useWorkspaceCanvasWorkflowActions>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createArtifact(): Artifact {
  return {
    id: "artifact-1",
    type: "document",
    title: "board-review.artifact.json",
    content: "",
    status: "complete",
    meta: {
      filePath: ".lime/artifacts/thread-1/board-review.artifact.json",
      filename: "board-review.artifact.json",
      artifactRequestId: "artifact:analysis:board-review",
    },
    position: { start: 0, end: 0 },
    createdAt: 1,
    updatedAt: 1,
  };
}

function createDocument(): ArtifactDocumentV1 {
  return {
    schemaVersion: "artifact_document.v1",
    artifactId: "artifact-document:artifact:analysis:board-review",
    kind: "analysis",
    title: "董事会季度复盘",
    status: "ready",
    language: "zh-CN",
    summary: "聚焦经营效率与风险收口。",
    blocks: [
      {
        id: "body-1",
        type: "rich_text",
        contentFormat: "markdown",
        content: "当前正文",
        markdown: "当前正文",
        sourceIds: ["source-1"],
      },
    ],
    sources: [
      {
        id: "source-1",
        type: "web",
        label: "季度经营看板",
        locator: {
          url: "https://example.com/board-review",
        },
      },
    ],
    metadata: {},
  };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<typeof useWorkspaceCanvasWorkflowActions> | null =
    null;

  const defaultProps: HookProps = {
    sendRef: {
      current: vi.fn().mockResolvedValue(true),
    },
    setCanvasState: vi.fn(),
    setTopicStatus: vi.fn(),
    projectId: "project-1",
    projectName: "董事会项目",
    canvasState: null,
    contentId: "content-1",
    selectedText: "",
    onRunImageWorkbenchCommand: vi.fn(),
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceCanvasWorkflowActions(currentProps);
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    await act(async () => {
      root.render(<Probe {...defaultProps} {...props} {...nextProps} />);
      await Promise.resolve();
    });
  };

  mountedRoots.push({ root, container });

  return {
    render,
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
    defaultProps: { ...defaultProps, ...props },
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  toastError.mockReset();
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.restoreAllMocks();
});

describe("useWorkspaceCanvasWorkflowActions", () => {
  it("文稿配图入口应根据当前选中文本推断目标小节", async () => {
    const onRunImageWorkbenchCommand = vi.fn().mockResolvedValue(undefined);
    const { render, getValue } = renderHook({
      canvasState: createInitialDocumentState(`# 标题

## 开场
这是开场内容。

## 核心观点
这里是被选中的关键段落，用于说明主结论。

## 收尾
这里是结尾。`),
      selectedText: "这里是被选中的关键段落，用于说明主结论。",
      onRunImageWorkbenchCommand,
    });

    await render();

    await act(async () => {
      await getValue().handleAddImage();
    });

    expect(onRunImageWorkbenchCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        applyTarget: expect.objectContaining({
          kind: "canvas-insert",
          canvasType: "document",
          anchorHint: "section_end",
          sectionTitle: "核心观点",
          anchorText: "这里是被选中的关键段落，用于说明主结论。",
        }),
      }),
    );
  });

  it("应通过统一发送主线发起当前块 AI 改写，并附带 rewrite metadata", async () => {
    const sendCurrent = vi
      .fn()
      .mockImplementation(async (...args: unknown[]) => {
        const sendOptions = args[6] as
          | {
              observer?: {
                onComplete?: (content: string) => void;
              };
            }
          | undefined;
        sendOptions?.observer?.onComplete?.(`{
        "type": "artifact_rewrite_patch",
        "artifactId": "artifact-document:artifact:analysis:board-review",
        "targetBlockId": "body-1",
        "summary": "压缩冗余表达",
        "block": {
          "id": "body-1",
          "type": "rich_text",
          "contentFormat": "markdown",
          "content": "AI 改写后的正文"
        }
      }`);
        return true;
      });
    const { render, getValue } = renderHook({
      sendRef: {
        current: sendCurrent,
      },
    });

    await render();

    let result: ArtifactBlockRewriteCompletion | null = null;

    await act(async () => {
      result = await getValue().handleArtifactBlockRewriteRun({
        artifact: createArtifact(),
        document: createDocument(),
        entry: {
          blockId: "body-1",
          label: "正文块 1",
          detail: "执行摘要",
          editorKind: "rich_text",
          draft: {
            editorKind: "rich_text",
            markdown: "当前正文",
          },
        },
        draft: {
          editorKind: "rich_text",
          markdown: "当前正文",
        },
        instruction: "请压缩冗余表达，适合董事会快速阅读。",
      });
    });

    expect(result).toMatchObject({
      suggestion: {
        draft: {
          editorKind: "rich_text",
          markdown: "AI 改写后的正文",
        },
      },
    });
    expect(sendCurrent).toHaveBeenCalledTimes(1);
    const sendArgs = sendCurrent.mock.calls[0];
    expect(sendArgs?.[3]).toContain("请压缩冗余表达，适合董事会快速阅读。");
    expect(sendCurrent).toHaveBeenCalledWith(
      [],
      undefined,
      undefined,
      expect.stringContaining("Lime Artifact Workbench 的局部改写任务"),
      undefined,
      undefined,
      expect.objectContaining({
        skipThemeSkillPrefix: true,
        purpose: "style_rewrite",
        requestMetadata: {
          artifact: {
            artifact_mode: "rewrite",
            artifact_stage: "rewrite",
            artifact_kind: "analysis",
            artifact_request_id: "artifact:analysis:board-review",
            artifact_target_block_id: "body-1",
            artifact_rewrite_instruction:
              "请压缩冗余表达，适合董事会快速阅读。",
            source_policy: "required",
            workbench_surface: "right_panel",
          },
        },
      }),
    );
  });

  it("缺少 request id 时应抛错并提示", async () => {
    const sendCurrent = vi.fn().mockResolvedValue(true);
    const artifact = createArtifact();
    artifact.meta = {
      filePath: ".lime/artifacts/thread-1/board-review.artifact.json",
    };
    const document = createDocument();
    document.artifactId = "";

    const { render, getValue } = renderHook({
      sendRef: {
        current: sendCurrent,
      },
    });
    await render();

    await expect(
      getValue().handleArtifactBlockRewriteRun({
        artifact,
        document,
        entry: {
          blockId: "body-1",
          label: "正文块 1",
          editorKind: "rich_text",
          draft: {
            editorKind: "rich_text",
            markdown: "当前正文",
          },
        },
        draft: {
          editorKind: "rich_text",
          markdown: "当前正文",
        },
      }),
    ).rejects.toThrow("当前 Artifact 缺少 request id");

    expect(sendCurrent).not.toHaveBeenCalled();
    expect(toastError).toHaveBeenCalledWith(
      "当前 Artifact 缺少 request id，暂时无法发起局部改写",
    );
  });
});
