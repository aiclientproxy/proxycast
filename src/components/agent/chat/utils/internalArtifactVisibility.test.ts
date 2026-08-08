import { describe, expect, it } from "vitest";
import {
  isHiddenConversationArtifact,
  isHiddenConversationArtifactPath,
  isHiddenInternalArtifactPath,
} from "./internalArtifactVisibility";

describe("isHiddenInternalArtifactPath", () => {
  it("应隐藏 .lime/tasks 下的内部任务快照 JSON", () => {
    expect(
      isHiddenInternalArtifactPath(
        ".lime/tasks/image_generate/task-image-1.json",
      ),
    ).toBe(true);
    expect(
      isHiddenInternalArtifactPath(
        "/workspace/demo/.lime/tasks/image_generate/task-image-1.json",
      ),
    ).toBe(true);
  });

  it("不应隐藏用户可消费的正式产物", () => {
    expect(
      isHiddenInternalArtifactPath("content-posts/demo.publish-pack.json"),
    ).toBe(false);
    expect(
      isHiddenInternalArtifactPath(
        ".lime/artifacts/thread-1/report.artifact.json",
      ),
    ).toBe(false);
    expect(isHiddenInternalArtifactPath("content-posts/demo-cover.png")).toBe(
      false,
    );
  });

  it("聊天区应隐藏 .lime/artifacts 下的内部 artifact 文稿 JSON", () => {
    expect(
      isHiddenConversationArtifactPath(
        ".lime/artifacts/thread-1/report.artifact.json",
      ),
    ).toBe(true);
    expect(
      isHiddenConversationArtifactPath(
        "/workspace/demo/.lime/artifacts/thread-1/report.artifact.json",
      ),
    ).toBe(true);
    expect(
      isHiddenConversationArtifactPath("content-posts/demo.publish-pack.json"),
    ).toBe(false);
    expect(
      isHiddenConversationArtifactPath("exports/x-article/google/index.md"),
    ).toBe(false);
  });

  it("聊天区应隐藏辅助运行时投影工件", () => {
    expect(
      isHiddenConversationArtifactPath(
        ".lime/harness/sessions/session-1/auxiliary-runtime/title-generation-aux-1.json",
      ),
    ).toBe(true);
    expect(
      isHiddenConversationArtifactPath(
        "/workspace/demo/.lime/harness/sessions/session-1/auxiliary-runtime/title-generation-aux-1.json",
      ),
    ).toBe(true);
    expect(
      isHiddenConversationArtifactPath(
        ".lime/harness/sessions/session-1/evidence/runtime.json",
      ),
    ).toBe(false);
  });

  it("聊天区应隐藏 workspace patch 原始工件", () => {
    expect(
      isHiddenConversationArtifactPath(
        ".lime/artifacts/article-workspace/workspace-patch.json",
      ),
    ).toBe(true);
    expect(
      isHiddenConversationArtifactPath(
        ".lime/artifacts/workspace-patch.json",
      ),
    ).toBe(true);
    expect(
      isHiddenConversationArtifact(
        {
          title: "内容工厂工作区补丁",
          content: "{}",
          meta: {
            kind: "workspace_patch",
          },
        },
        "workspace-patch.json",
      ),
    ).toBe(true);
  });

  it("聊天区不应隐藏可见的文章预览产物", () => {
    expect(
      isHiddenConversationArtifact(
        {
          title: "公众号文章草稿",
          content: "# 公众号文章草稿\n\n正文",
          meta: {
            openedFrom: "right_surface_article_workspace",
            artifactKind: "report",
            workspacePatch: {
              appId: "content-factory-app",
              objects: [],
            },
          },
        },
        "公众号文章草稿.md",
      ),
    ).toBe(false);
  });
});
