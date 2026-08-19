import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  useWorkspaceWorkbenchRequests,
  type WorkspaceWorkbenchRequestsController,
} from "./useWorkspaceWorkbenchRequests";

let latest: WorkspaceWorkbenchRequestsController | null = null;

function Harness() {
  latest = useWorkspaceWorkbenchRequests();
  return null;
}

describe("useWorkspaceWorkbenchRequests", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    latest = null;
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    latest = null;
  });

  function mount() {
    act(() => {
      root.render(<Harness />);
    });
    if (!latest) {
      throw new Error("hook 尚未初始化");
    }
    return latest;
  }

  it("生成 canvas preview 打开请求并递增 requestKey", () => {
    const controller = mount();

    act(() =>
      controller.requestCanvasWorkbenchPreviewOpen({
        filePath: "notes.md",
        selectionKey: "artifact:a",
      }),
    );
    expect(latest?.canvasWorkbenchPreviewOpenRequest).toEqual({
      requestKey: 1,
      filePath: "notes.md",
      selectionKey: "artifact:a",
    });

    act(() =>
      latest?.requestCanvasWorkbenchPreviewOpen({
        filePath: "",
      }),
    );
    expect(latest?.canvasWorkbenchPreviewOpenRequest).toEqual({
      requestKey: 2,
      filePath: null,
      selectionKey: null,
    });
  });

  it("只清除匹配的 canvas preview 请求", () => {
    const controller = mount();

    act(() =>
      controller.requestCanvasWorkbenchPreviewOpen({
        filePath: "output.md",
      }),
    );
    act(() => latest?.handleCanvasWorkbenchPreviewOpenRequestHandled(2));
    expect(latest?.canvasWorkbenchPreviewOpenRequest?.requestKey).toBe(1);

    act(() => latest?.handleCanvasWorkbenchPreviewOpenRequestHandled(1));
    expect(latest?.canvasWorkbenchPreviewOpenRequest).toBeNull();
  });

  it("聚焦 artifact block 会裁剪空白并递增请求 key，空值不生效", () => {
    const controller = mount();

    act(() => controller.focusArtifactBlock("  block-1  "));
    expect(latest?.focusedArtifactBlockId).toBe("block-1");
    expect(latest?.artifactBlockFocusRequestKey).toBe(1);

    act(() => latest?.focusArtifactBlock("   "));
    expect(latest?.focusedArtifactBlockId).toBe("block-1");
    expect(latest?.artifactBlockFocusRequestKey).toBe(1);

    act(() => latest?.clearFocusedArtifactBlock());
    expect(latest?.focusedArtifactBlockId).toBeNull();
    expect(latest?.artifactBlockFocusRequestKey).toBe(1);
  });

  it("跳转 timeline item 时返回是否生效并递增请求 key", () => {
    const controller = mount();

    act(() => {
      expect(controller.jumpToTimelineItem("  item-1  ")).toBe(true);
    });
    expect(latest?.focusedTimelineItemId).toBe("item-1");
    expect(latest?.timelineFocusRequestKey).toBe(1);

    act(() => {
      expect(latest?.jumpToTimelineItem(" ")).toBe(false);
    });
    expect(latest?.focusedTimelineItemId).toBe("item-1");
    expect(latest?.timelineFocusRequestKey).toBe(1);
  });
});
