import { describe, expect, it } from "vitest";
import {
  buildWorkspaceArticleWorkspaceActionRequestMetadata,
  buildWorkspaceArticleWorkspaceFromPendingRequests,
  buildWorkspaceArticleWorkspaceFromThreadRead,
  buildWorkspaceArticleWorkspaceViewModel,
  hasWorkspaceArticleWorkspaceThreadReadMetadata,
  selectWorkspaceArticleDraftObject,
} from "./workspaceArticleWorkspaceModel";

const workspacePatch = {
  schemaVersion: "article-workspace.v1",
  appId: "content-factory-app",
  sessionId: "session-main",
  workspaceId: "workspace-main",
  primaryObjectRef: {
    appId: "content-factory-app",
    kind: "articleDraft",
    id: "article-1",
    sessionId: "session-main",
  },
  selectedObjectRef: {
    appId: "content-factory-app",
    kind: "imageGenerationSet",
    id: "image-set-1",
    sessionId: "session-main",
  },
  objects: [
    {
      ref: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "article-1",
        sessionId: "session-main",
        artifactIds: ["artifact-article-1"],
        sourceTurnId: "turn-1",
      },
      title: "公众号文章草稿",
      status: "ready",
      summary: "已生成首版文章",
      previewArtifactId: "artifact-article-1",
      source: {
        taskKind: "content.article.generate",
        artifactIds: ["artifact-article-1"],
        documentText: "# 标题\n\n这是首版文章正文。",
        finalMarkdown: "# 标题\n\n这是首版文章正文。",
        researchRounds: [{ id: "research-1", title: "资料检索" }],
      },
    },
    {
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
        taskKind: "content.image.generate",
        images: [
          {
            id: "artifact-image-1",
            title: "主图",
            url: "https://lime.local/image-1.png",
            prompt: "明亮的中文内容工厂主图",
          },
          {
            id: "artifact-image-local",
            title: "本地缓存图",
            filePath: "/tmp/lime-content-factory/local-image.png",
            prompt: "已缓存到本地的内容工厂配图",
          },
        ],
      },
    },
  ],
  layoutState: {
    activeTabKind: "articleWorkspace",
    activePaneKind: "imageGrid",
    openTabKinds: ["articleWorkspace", "files"],
    splitMode: "chat-right-dock",
  },
  sourceArtifacts: [{ artifactRef: "artifact-workspace-patch-1" }],
  workerEvidence: [
    {
      id: "evt-worker-success:workerEvidence",
      status: "completed",
      source: "workspace_patch",
      appId: "content-factory-app",
      taskId: "task-article-1",
      taskKind: "content.article.generate",
      turnId: "turn-action-1",
      artifactRef: "artifact-workspace-patch-1",
      artifactKind: "workspace_patch",
      updatedAt: "2026-06-24T00:00:00.000Z",
    },
    {
      id: "evt-worker-failed:workerEvidence",
      status: "failed",
      source: "workspace_patch",
      appId: "content-factory-app",
      taskId: "task-image-1",
      taskKind: "content.image.generate",
      turnId: "turn-action-2",
      errorCode: "workspace_patch_invalid",
      errorMessage: "Workspace patch is invalid",
      failureCategory: "workspace_patch",
      retryable: false,
      retryAdvice: "inspect_workspace_patch",
      retryAttempt: 0,
      retryMaxAttempts: 0,
      updatedAt: "2026-06-24T00:00:02.000Z",
    },
  ],
  actionHistory: [
    {
      id: "turn-action-1:articleWorkspaceAction:regenerate",
      key: "regenerate",
      intent: "regenerate",
      risk: "write",
      status: "completed",
      turnStatus: "completed",
      turnId: "turn-action-1",
      sessionId: "session-main",
      threadId: "thread-main",
      appId: "content-factory-app",
      objectRef: {
        appId: "content-factory-app",
        kind: "imageGenerationSet",
        id: "image-set-1",
        sessionId: "session-main",
      },
      objectTitle: "配图组",
      objectStatus: "needs_review",
      taskKind: "content.image.generate",
      prompt: "请重新生成「配图组」",
      submittedAt: "2026-06-24T00:00:00.000Z",
      completedAt: "2026-06-24T00:00:01.000Z",
    },
  ],
  updatedAt: "2026-06-24T00:00:00.000Z",
};

describe("workspaceArticleWorkspaceModel", () => {
  it("普通 thread_read 不应触发 Article Editor 投影", () => {
    const threadRead = {
      thread_id: "thread-news",
      outputs: [],
      tool_calls: [],
    };

    expect(hasWorkspaceArticleWorkspaceThreadReadMetadata(threadRead)).toBe(
      false,
    );
    expect(buildWorkspaceArticleWorkspaceFromThreadRead(threadRead)).toBeNull();
  });

  it("应从 thread_read.article_workspace 投影内容工厂文章编辑器", () => {
    expect(
      hasWorkspaceArticleWorkspaceThreadReadMetadata({
        thread_id: "thread-main",
        article_workspace: workspacePatch,
      }),
    ).toBe(true);
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      article_workspace: workspacePatch,
    });

    expect(profile).toMatchObject({
      appId: "content-factory-app",
      sessionId: "session-main",
      workspaceId: "workspace-main",
      source: "threadRead",
      objectCount: 2,
      layoutState: {
        activePaneKind: "imageGrid",
        openTabKinds: ["articleWorkspace", "files"],
      },
    });
    expect(profile?.objects.map((object) => object.ref.kind)).toEqual([
      "articleDraft",
      "imageGenerationSet",
    ]);
  });

  it("应从 thread_read.article_workspace.editedDraft 恢复 Article Editor 编辑正文", () => {
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      article_workspace: {
        ...workspacePatch,
        edited_draft: {
          object_ref: {
            app_id: "content-factory-app",
            kind: "articleDraft",
            id: "article-1",
            session_id: "session-main",
          },
          markdown: "# 用户编辑稿\n\n这是从历史恢复的正文。",
          updated_at: "2026-06-29T10:00:00.000Z",
        },
      },
    });

    expect(profile?.editedDraft).toEqual({
      objectKey: "content-factory-app:session-main:articleDraft:article-1",
      markdown: "# 用户编辑稿\n\n这是从历史恢复的正文。",
      updatedAt: "2026-06-29T10:00:00.000Z",
    });
    expect(profile?.objects[0]?.source).toMatchObject({
      documentText: "# 用户编辑稿\n\n这是从历史恢复的正文。",
      finalMarkdown: "# 用户编辑稿\n\n这是从历史恢复的正文。",
      updatedAt: "2026-06-29T10:00:00.000Z",
      researchRounds: [{ id: "research-1", title: "资料检索" }],
    });
  });

  it("view model 应优先展示 editedDraft 对应的文章对象", () => {
    const oldArticle = workspacePatch.objects[0]!;
    const finalArticle = {
      ...oldArticle,
      ref: {
        ...oldArticle.ref,
        id: "article-final",
        version: "v3",
        sourceTaskId: "content-factory-worker-task",
      },
      title: "多轮检索后的公众号文章草稿",
      source: {
        documentText: "# 多轮检索后的公众号文章草稿\n\n这是完整正文。",
        finalMarkdown: "# 多轮检索后的公众号文章草稿\n\n这是完整正文。",
        researchRounds: [
          { id: "research-1", title: "确认用户目标" },
          { id: "research-2", title: "整理场景痛点" },
          { id: "research-3", title: "收敛结构" },
        ],
      },
    };
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: {
        ...workspacePatch,
        selectedObjectRef: finalArticle.ref,
        primaryObjectRef: finalArticle.ref,
        editedDraft: {
          objectKey: "content-factory-app:session-main:articleDraft:article-1",
          markdown: "# Inline 配图恢复验证\n\n核心观点段落",
          updatedAt: "2026-07-04T00:00:00.000Z",
        },
        objects: [oldArticle, workspacePatch.objects[1], finalArticle],
      },
    });
    expect(profile).not.toBeNull();

    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);

    expect(viewModel.selectedObject.ref.id).toBe("article-1");
    expect(viewModel.selectedPreview.documentText).toContain(
      "Inline 配图恢复验证",
    );
  });

  it("应从 Article Editor pending metadata 投影并保留请求来源", () => {
    const profile = buildWorkspaceArticleWorkspaceFromPendingRequests([
      {
        requestId: "right_surface_article_workspace_1",
        workspaceId: "workspace-main",
        sessionId: "session-main",
        surfaceKind: "articleWorkspace",
        origin: "runtime",
        priority: "foreground",
        status: "pending",
        reason: "article_workspace_ready",
        requestedAt: "2026-06-24T00:00:00.000Z",
        metadata: {
          workspacePatch,
        },
      },
    ]);

    expect(profile).toMatchObject({
      source: "rightSurfacePending",
      appId: "content-factory-app",
      sessionId: "session-main",
      workspaceId: "workspace-main",
      updatedAt: "2026-06-24T00:00:00.000Z",
    });
  });

  it("Article Editor pending metadata 应拒绝旧 contentFactoryWorkspacePatch", () => {
    const profile = buildWorkspaceArticleWorkspaceFromPendingRequests([
      {
        requestId: "right_surface_article_workspace_legacy",
        workspaceId: "workspace-main",
        sessionId: "session-main",
        surfaceKind: "articleWorkspace",
        origin: "runtime",
        priority: "foreground",
        status: "pending",
        reason: "article_workspace_ready",
        requestedAt: "2026-06-24T00:00:00.000Z",
        metadata: {
          contentFactoryWorkspacePatch: workspacePatch,
        },
      },
    ]);

    expect(profile).toBeNull();
  });

  it("Article Editor pending metadata 只使用 workspacePatch", () => {
    const profile = buildWorkspaceArticleWorkspaceFromPendingRequests([
      {
        requestId: "right_surface_article_workspace_current_first",
        workspaceId: "workspace-main",
        sessionId: "session-main",
        surfaceKind: "articleWorkspace",
        origin: "runtime",
        priority: "foreground",
        status: "pending",
        reason: "article_workspace_ready",
        requestedAt: "2026-06-24T00:00:00.000Z",
        metadata: {
          workspacePatch,
        },
      },
    ]);

    expect(profile?.objects[0]?.title).toBe("公众号文章草稿");
  });

  it("view model 应优先选择 selectedObjectRef 并统计状态", () => {
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: workspacePatch,
    });
    expect(profile).not.toBeNull();

    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);

    expect(viewModel.selectedObject.ref.kind).toBe("imageGenerationSet");
    expect(viewModel.selectedObject.title).toBe("配图组");
    expect(viewModel.selectedSurface).toMatchObject({
      layout: "imageGrid",
      titleKey: "workspace.articleWorkspace.surface.imageGrid",
    });
    expect(viewModel.selectedActions.map((action) => action.key)).toEqual([
      "regenerate",
      "create_variant",
      "apply_to_article",
    ]);
    expect(viewModel.selectedArtifactIds).toEqual(["artifact-image-1"]);
    expect(viewModel.selectedPreview.images).toEqual([
      expect.objectContaining({
        id: "artifact-image-1",
        title: "主图",
        url: "https://lime.local/image-1.png",
      }),
      expect.objectContaining({
        id: "artifact-image-local",
        localPath: "/tmp/lime-content-factory/local-image.png",
        filePath: "/tmp/lime-content-factory/local-image.png",
        url: null,
      }),
    ]);
    expect(viewModel.selectedActionHistory).toEqual([
      expect.objectContaining({
        key: "regenerate",
        status: "completed",
        turnId: "turn-action-1",
      }),
    ]);
    expect(viewModel.latestSelectedAction?.taskKind).toBe(
      "content.image.generate",
    );
    expect(viewModel.statusCounts.ready).toBe(1);
    expect(viewModel.statusCounts.needs_review).toBe(1);
    expect(viewModel.sourceArtifacts).toEqual([
      { artifactRef: "artifact-workspace-patch-1" },
    ]);
    expect(viewModel.latestWorkerEvidence).toEqual(
      expect.objectContaining({
        status: "failed",
        taskId: "task-image-1",
        errorCode: "workspace_patch_invalid",
        failureCategory: "workspace_patch",
        retryable: false,
        retryAdvice: "inspect_workspace_patch",
      }),
    );
    expect(viewModel.workerEvidence).toHaveLength(2);
  });

  it("有多个文章草稿时应默认选择多轮检索后的最终草稿", () => {
    const oldArticle = workspacePatch.objects[0];
    const finalArticle = {
      ...oldArticle,
      ref: {
        ...oldArticle.ref,
        id: "article-final",
        artifactIds: ["artifact-article-final"],
        sourceTaskId: "content-factory-worker-task",
        version: "v3",
      },
      title: "多轮检索后的公众号文章草稿",
      summary: "已完成 3 轮资料检索、5 段大纲、3 个配图占位和首版正文。",
      previewArtifactId: "artifact-article-final",
      source: {
        taskKind: "content.article.generate",
        taskId: "content-factory-worker-task",
        documentText: [
          "# 多轮检索后的公众号文章草稿",
          "",
          "## 三轮资料检索",
          "",
          "- 第一轮：确认用户目标。",
          "- 第二轮：整理场景痛点。",
          "- 第三轮：收敛结构和发布检查。",
          "",
          "## 正文草稿",
          "",
          "这是经过多轮检索后写出的完整正文。",
        ].join("\n"),
        finalMarkdown: [
          "# 多轮检索后的公众号文章草稿",
          "",
          "## 三轮资料检索",
          "",
          "- 第一轮：确认用户目标。",
          "- 第二轮：整理场景痛点。",
          "- 第三轮：收敛结构和发布检查。",
          "",
          "## 正文草稿",
          "",
          "这是经过多轮检索后写出的完整正文。",
        ].join("\n"),
        researchRounds: [
          { id: "research-1", title: "确认目标" },
          { id: "research-2", title: "整理痛点" },
          { id: "research-3", title: "收敛结构" },
        ],
        titleCandidates: [
          { id: "title-1", title: "内容工厂不是聊天框" },
          { id: "title-2", title: "写作应该先搜索再成文" },
        ],
        outline: [
          { id: "intro", title: "开场", points: [], evidenceIds: [] },
          { id: "research", title: "检索", points: [], evidenceIds: [] },
          { id: "draft", title: "正文", points: [], evidenceIds: [] },
        ],
        keyTakeaways: ["先搜索再写作", "产物框承载完整正文"],
        citations: [
          { id: "citation-1", title: "规划文档" },
          { id: "citation-2", title: "用户反馈" },
        ],
        imageSlots: [
          { id: "hero", title: "首图" },
          { id: "workflow", title: "流程图" },
          { id: "canvas", title: "画布图" },
        ],
        writingPlan: [
          { id: "plan-1", title: "资料检索" },
          { id: "plan-2", title: "正文写作" },
        ],
        updatedAt: "2026-06-24T00:00:03.000Z",
      },
    };
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: {
        ...workspacePatch,
        selectedObjectRef: oldArticle.ref,
        primaryObjectRef: oldArticle.ref,
        objects: [oldArticle, workspacePatch.objects[1], finalArticle],
      },
    });
    expect(profile).not.toBeNull();

    const selected = selectWorkspaceArticleDraftObject(profile!.objects);
    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);

    expect(selected?.ref.id).toBe("article-final");
    expect(viewModel.selectedObject.ref.id).toBe("article-final");
    expect(viewModel.selectedPreview.researchRounds).toHaveLength(3);
    expect(viewModel.selectedPreview.documentText).toContain("三轮资料检索");
  });

  it("应从对象 source 投影结构化预览内容", () => {
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: {
        ...workspacePatch,
        selectedObjectRef: {
          appId: "content-factory-app",
          kind: "articleDraft",
          id: "article-1",
          sessionId: "session-main",
        },
      },
    });
    expect(profile).not.toBeNull();

    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);

    expect(viewModel.selectedPreview.documentText).toContain("首版文章正文");
    expect(viewModel.selectedPreview.images).toEqual([]);
  });

  it("应将过程稿与正式稿分开投影，正式稿优先进入 documentText", () => {
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: {
        ...workspacePatch,
        objects: [
          {
            ...workspacePatch.objects[0],
            source: {
              ...workspacePatch.objects[0].source,
              processMarkdown: "## 过程稿\n\n只用于编排与检索。",
              documentText: "# 最终稿\n\n这是正式可编辑文章。",
              finalMarkdown: "# 最终稿\n\n这是正式可编辑文章。",
            },
          },
        ],
        selectedObjectRef: workspacePatch.objects[0].ref,
        primaryObjectRef: workspacePatch.objects[0].ref,
      },
    });
    expect(profile).not.toBeNull();

    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);

    expect(viewModel.selectedPreview.processMarkdown).toContain("过程稿");
    expect(viewModel.selectedPreview.documentText).toContain("正式可编辑文章");
  });

  it("旧 read model 没有 workerEvidence 时应从 diagnostics 投影失败记录", () => {
    const legacyWorkspacePatch = {
      ...workspacePatch,
    } as Record<string, unknown>;
    delete legacyWorkspacePatch.workerEvidence;
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      active_turn_id: "turn-worker-failed",
      articleWorkspace: legacyWorkspacePatch,
      diagnostics: {
        latest_turn_error_message: "Plugin task worker failed: invalid JSON",
        latest_turn_completed_at: "2026-06-24T00:00:03.000Z",
        warning_count: 0,
        context_compaction_count: 0,
        failed_tool_call_count: 0,
        failed_command_count: 0,
        pending_request_count: 0,
      },
    });
    expect(profile).not.toBeNull();

    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);

    expect(viewModel.latestWorkerEvidence).toEqual(
      expect.objectContaining({
        status: "failed",
        turnId: "turn-worker-failed",
        errorMessage: "Plugin task worker failed: invalid JSON",
      }),
    );
  });

  it("应构造可随 Claw turn 透传的 Article Editor action metadata", () => {
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: workspacePatch,
    });
    expect(profile).not.toBeNull();
    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);
    const action = viewModel.selectedActions.find(
      (candidate) => candidate.key === "regenerate",
    );
    expect(action).toBeDefined();

    const metadata = buildWorkspaceArticleWorkspaceActionRequestMetadata({
      action: action!,
      editedMarkdown:
        "# 本地编辑正文\n\n这是用户在 Article Editor 里改过的内容。",
      object: viewModel.selectedObject,
      articleWorkspace: profile!,
      prompt: "请重新生成「配图组」",
    });

    expect(metadata).toMatchObject({
      plugin: {
        source: "right_surface_article_workspace",
        app_id: "content-factory-app",
        session_id: "session-main",
        workspace_id: "workspace-main",
        article_workspace_action: {
          key: "regenerate",
          intent: "regenerate",
          risk: "write",
          task_kind: "content.image.generate",
          output_artifact_kind: "workspace_patch",
          prompt: "请重新生成「配图组」",
          edited_markdown:
            "# 本地编辑正文\n\n这是用户在 Article Editor 里改过的内容。",
          object: {
            app_id: "content-factory-app",
            kind: "imageGenerationSet",
            id: "image-set-1",
            session_id: "session-main",
            title: "配图组",
            status: "needs_review",
            artifact_ids: ["artifact-image-1"],
          },
        },
        pane_action: {
          key: "regenerate",
          intent: "regenerate",
          risk: "write",
          task_kind: "content.image.generate",
          output_artifact_kind: "workspace_patch",
          prompt: "请重新生成「配图组」",
          pane_kind: "imageGenerationSet",
          surface_kind: "articleWorkspace",
          source_artifact_ids: ["artifact-image-1"],
          object: {
            app_id: "content-factory-app",
            kind: "imageGenerationSet",
            id: "image-set-1",
            session_id: "session-main",
            title: "配图组",
            status: "needs_review",
            artifact_ids: ["artifact-image-1"],
          },
        },
      },
      right_surface: {
        surface_kind: "articleWorkspace",
        pane_kind: "imageGenerationSet",
        source: "threadRead",
        action_key: "regenerate",
      },
    });
  });

  it("Article Editor action 应保留 canonical workspace patch kind", () => {
    const profile = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      articleWorkspace: {
        ...workspacePatch,
        objects: workspacePatch.objects.map((object) =>
          object.ref.kind === "imageGenerationSet"
            ? {
                ...object,
                source: {
                  ...object.source,
                  outputArtifactKind: "workspace_patch",
                },
              }
            : object,
        ),
      },
    });
    expect(profile).not.toBeNull();
    const viewModel = buildWorkspaceArticleWorkspaceViewModel(profile!);
    const action = viewModel.selectedActions.find(
      (candidate) => candidate.key === "regenerate",
    );
    expect(action).toBeDefined();

    const metadata = buildWorkspaceArticleWorkspaceActionRequestMetadata({
      action: action!,
      object: viewModel.selectedObject,
      articleWorkspace: profile!,
      prompt: "请重新生成「配图组」",
    });

    expect(metadata).toMatchObject({
      plugin: {
        article_workspace_action: {
          output_artifact_kind: "workspace_patch",
        },
        pane_action: {
          output_artifact_kind: "workspace_patch",
        },
      },
    });
  });
});
