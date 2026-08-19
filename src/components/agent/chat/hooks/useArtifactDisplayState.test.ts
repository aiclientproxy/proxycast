import { describe, expect, it } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import {
  resolveArtifactDisplayState,
  settleLiveArtifactAfterStreamStops,
} from "./useArtifactDisplayState";

function createArtifact(overrides: Partial<Artifact> = {}): Artifact {
  const content = overrides.content ?? "# 内容";
  return {
    id: overrides.id ?? "artifact-1",
    type: overrides.type ?? "document",
    title: overrides.title ?? "demo.md",
    content,
    status: overrides.status ?? "complete",
    meta: {
      filePath: overrides.meta?.filePath ?? "workspace/demo.md",
      filename: overrides.meta?.filename ?? "demo.md",
      ...overrides.meta,
    },
    position: overrides.position ?? { start: 0, end: content.length },
    createdAt: overrides.createdAt ?? 1,
    updatedAt: overrides.updatedAt ?? 1,
    error: overrides.error,
  };
}

describe("resolveArtifactDisplayState", () => {
  it("有内容时应直接展示 live artifact", () => {
    const liveArtifact = createArtifact({
      id: "artifact-live",
      content: "# 最新版本",
      status: "streaming",
      meta: {
        filePath: "workspace/live.md",
        writePhase: "streaming",
      },
    });

    const state = resolveArtifactDisplayState({
      liveArtifact,
      artifacts: [liveArtifact],
    });

    expect(state.mode).toBe("content");
    expect(state.displayArtifact?.id).toBe("artifact-live");
    expect(state.overlay).toBeNull();
    expect(state.showPreviousVersionBadge).toBe(false);
  });

  it("新 artifact 为空且存在上一版本时应保留旧内容并显示 overlay", () => {
    const previousArtifact = createArtifact({
      id: "artifact-prev",
      title: "old.md",
      content: "# 旧版本",
      status: "complete",
      meta: {
        filePath: "workspace/old.md",
      },
    });
    const liveArtifact = createArtifact({
      id: "artifact-live",
      title: "new.md",
      content: "",
      status: "streaming",
      meta: {
        filePath: "workspace/new.md",
        writePhase: "streaming",
      },
    });

    const state = resolveArtifactDisplayState({
      liveArtifact,
      artifacts: [previousArtifact, liveArtifact],
      previousRenderableArtifact: previousArtifact,
    });

    expect(state.mode).toBe("overlay-on-previous");
    expect(state.displayArtifact?.id).toBe("artifact-prev");
    expect(state.overlay?.phase).toBe("streaming_content");
    expect(state.overlay?.displayName).toBe("new.md");
    expect(state.showPreviousVersionBadge).toBe(true);
  });

  it("没有上一版本时应回退到类型化 skeleton", () => {
    const liveArtifact = createArtifact({
      id: "artifact-live",
      title: "index.ts",
      type: "code",
      content: "",
      status: "pending",
      meta: {
        filePath: "workspace/index.ts",
        writePhase: "preparing",
        language: "typescript",
      },
    });

    const state = resolveArtifactDisplayState({
      liveArtifact,
      artifacts: [liveArtifact],
    });

    expect(state.mode).toBe("typed-skeleton");
    expect(state.displayArtifact?.id).toBe("artifact-live");
    expect(state.overlay).toBeNull();
    expect(state.showPreviousVersionBadge).toBe(false);
  });

  it("完成但仍无内容时应保留上一版本并提示完成态", () => {
    const previousArtifact = createArtifact({
      id: "artifact-prev",
      title: "report.md",
      content: "# 上一版报告",
      status: "complete",
      meta: {
        filePath: "workspace/report.md",
      },
    });
    const liveArtifact = createArtifact({
      id: "artifact-live",
      title: "summary.md",
      content: "",
      status: "complete",
      meta: {
        filePath: "workspace/summary.md",
        writePhase: "completed",
      },
    });

    const state = resolveArtifactDisplayState({
      liveArtifact,
      artifacts: [previousArtifact, liveArtifact],
      previousRenderableArtifact: previousArtifact,
    });

    expect(state.mode).toBe("overlay-on-previous");
    expect(state.displayArtifact?.id).toBe("artifact-prev");
    expect(state.overlay?.phase).toBe("finalized_empty");
    expect(state.overlay?.title).toContain("写入已结束");
  });

  it("写入失败且没有上一版本时应展示错误态空画布", () => {
    const liveArtifact = createArtifact({
      id: "artifact-live",
      title: "broken.md",
      content: "",
      status: "error",
      error: "磁盘写入失败",
      meta: {
        filePath: "workspace/broken.md",
        writePhase: "failed",
      },
    });

    const state = resolveArtifactDisplayState({
      liveArtifact,
      artifacts: [liveArtifact],
    });

    expect(state.mode).toBe("error");
    expect(state.displayArtifact?.id).toBe("artifact-live");
    expect(state.overlay).toBeNull();
  });

});

describe("settleLiveArtifactAfterStreamStops", () => {
  it("流结束后应把仍停留在 streaming 的 artifact 收尾为 complete", () => {
    const artifact = createArtifact({
      id: "artifact-live",
      content: "# 已完成正文",
      status: "streaming",
      meta: {
        filePath: "workspace/live.md",
        writePhase: "streaming",
        isPartial: true,
      },
    });

    const settled = settleLiveArtifactAfterStreamStops(artifact, {
      streamActive: false,
    });

    expect(settled).not.toBeNull();
    expect(settled).not.toBe(artifact);
    expect(settled?.status).toBe("complete");
    expect(settled?.meta.writePhase).toBe("completed");
    expect(settled?.meta.isPartial).toBe(false);
  });

  it("流仍在进行时不应提前收尾 artifact 状态", () => {
    const artifact = createArtifact({
      id: "artifact-live",
      content: "# 增量正文",
      status: "streaming",
      meta: {
        filePath: "workspace/live.md",
        writePhase: "streaming",
      },
    });

    const settled = settleLiveArtifactAfterStreamStops(artifact, {
      streamActive: true,
    });

    expect(settled).toBe(artifact);
  });
});
