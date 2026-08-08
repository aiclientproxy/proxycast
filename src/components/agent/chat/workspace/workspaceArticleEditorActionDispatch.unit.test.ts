import { describe, expect, it, vi } from "vitest";
import {
  buildWorkspaceArticleEditorActionSystemPrompt,
  submitWorkspaceArticleEditorActionIntent,
  type SubmitWorkspaceArticleEditorAction,
} from "./workspaceArticleEditorActionDispatch";
import type { WorkspaceArticleWorkspaceActionIntent } from "./workspaceArticleWorkspaceModel";

const intent: WorkspaceArticleWorkspaceActionIntent = {
  action: {
    key: "regenerate",
    intent: "regenerate",
    risk: "write",
    taskKind: "content.image.generate",
    labelKey: "workspace.articleWorkspace.action.regenerate",
    promptKey: "workspace.articleWorkspace.actionPrompt.regenerate",
  },
  object: {
    ref: {
      appId: "content-factory-app",
      kind: "imageGenerationSet",
      id: "image-set-1",
      sessionId: "session-main",
      artifactIds: ["artifact-image-1"],
    },
    title: "配图组",
    status: "needs_review",
    summary: "等待选择主图",
    source: {
      outputArtifactKind: "workspace_patch",
    },
  },
  articleWorkspace: {
    schemaVersion: "article-workspace.v1",
    appId: "content-factory-app",
    sessionId: "session-main",
    workspaceId: "workspace-main",
    source: "threadRead",
    objectCount: 1,
    actionHistory: [],
    objects: [],
  },
  prompt: "请重新生成「配图组」",
};

describe("workspaceArticleEditorActionDispatch", () => {
  it("应通过 Claw turn submit 发送 Article Editor action metadata", async () => {
    const submit = vi.fn<SubmitWorkspaceArticleEditorAction>(async () => true);
    const restoreInput = vi.fn();

    await expect(
      submitWorkspaceArticleEditorActionIntent({
        intent: {
          ...intent,
          editedMarkdown: "# 本地编辑正文\n\n这是用户在 Article Editor 里改过的内容。",
        },
        restoreInput,
        submit,
      }),
    ).resolves.toBe(true);

    expect(submit).toHaveBeenCalledWith(
      "请重新生成「配图组」",
      expect.objectContaining({
        displayContent: "请重新生成「配图组」",
        skipSceneCommandRouting: true,
        searchMode: "disabled",
        explicitToolPreferences: true,
        systemPromptOverride: expect.stringContaining(
          "right_surface_article_workspace",
        ),
        requestMetadata: expect.objectContaining({
          plugin: expect.objectContaining({
            source: "right_surface_article_workspace",
            app_id: "content-factory-app",
            article_workspace_action: expect.objectContaining({
              key: "regenerate",
              task_kind: "content.image.generate",
              edited_markdown:
                "# 本地编辑正文\n\n这是用户在 Article Editor 里改过的内容。",
            }),
          }),
        }),
      }),
    );
    const [, sendOptions] = submit.mock.calls[0] ?? [];
    expect(sendOptions?.systemPromptOverride).toContain(
      "content.image.generate",
    );
    expect(sendOptions?.systemPromptOverride).toContain(
      "workspace_patch",
    );
    expect(sendOptions?.systemPromptOverride).toContain(
      "imageGenerationSet/image-set-1",
    );
    expect(sendOptions?.systemPromptOverride).toContain("artifact-image-1");
    expect(sendOptions?.systemPromptOverride).toContain("artifact.snapshot");
    expect(restoreInput).not.toHaveBeenCalled();
  });

  it("应从 Article Editor action intent 生成 runtime 指令", () => {
    const systemPrompt = buildWorkspaceArticleEditorActionSystemPrompt({
      ...intent,
      editedMarkdown: "# 本地编辑正文\n\n这是用户在 Article Editor 里改过的内容。",
    });

    expect(systemPrompt).toContain("Article Editor action");
    expect(systemPrompt).toContain("right_surface_article_workspace");
    expect(systemPrompt).toContain("content-factory-app");
    expect(systemPrompt).toContain("workspace-main");
    expect(systemPrompt).toContain("imageGenerationSet/image-set-1");
    expect(systemPrompt).toContain("regenerate");
    expect(systemPrompt).toContain("content.image.generate");
    expect(systemPrompt).toContain("workspace_patch");
    expect(systemPrompt).toContain("Current edited article markdown");
    expect(systemPrompt).toContain("本地编辑正文");
    expect(systemPrompt).toContain("workspace patch");
    expect(systemPrompt).toContain("skill_search");
  });

  it("发送失败时应把 prompt 恢复到 Claw 输入框", async () => {
    const submit = vi.fn<SubmitWorkspaceArticleEditorAction>(
      async () => false,
    );
    const restoreInput = vi.fn();

    await expect(
      submitWorkspaceArticleEditorActionIntent({
        intent,
        restoreInput,
        submit,
      }),
    ).resolves.toBe(false);

    expect(restoreInput).toHaveBeenCalledWith("请重新生成「配图组」");
  });

  it("空 prompt 不应发送 action", async () => {
    const submit = vi.fn<SubmitWorkspaceArticleEditorAction>(async () => true);
    const restoreInput = vi.fn();

    await expect(
      submitWorkspaceArticleEditorActionIntent({
        intent: {
          ...intent,
          prompt: "   ",
        },
        restoreInput,
        submit,
      }),
    ).resolves.toBe(false);

    expect(submit).not.toHaveBeenCalled();
    expect(restoreInput).not.toHaveBeenCalled();
  });
});
