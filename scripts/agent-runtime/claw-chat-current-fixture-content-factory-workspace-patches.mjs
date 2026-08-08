import {
  CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID,
} from "./claw-chat-current-fixture-constants.mjs";

const CONTENT_FACTORY_APP_ID = "content-factory-app";
const ARTICLE_OBJECT_ID = "article-1";
const IMAGE_SET_OBJECT_ID = "image-set-1";
const STORYBOARD_OBJECT_ID = "storyboard-1";
const CHECKLIST_OBJECT_ID = "delivery-checklist-1";
const CONTENT_FACTORY_ARTICLE_WORKSPACE_STORYBOARD_ARTIFACT_ID =
  "artifact-video-storyboard";
const CONTENT_FACTORY_ARTICLE_WORKSPACE_CHECKLIST_ARTIFACT_ID =
  "artifact-delivery-checklist";

export function buildWorkspacePatch(workspace, identity) {
  const { sessionId, sourceTaskId } = identity;
  return {
    schemaVersion: "article-workspace.v1",
    appId: CONTENT_FACTORY_APP_ID,
    sessionId,
    workspaceId: workspace.workspaceId,
    primaryObjectRef: {
      appId: CONTENT_FACTORY_APP_ID,
      kind: "articleDraft",
      id: ARTICLE_OBJECT_ID,
      sessionId,
      artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID],
      sourceTaskId,
    },
    selectedObjectRef: {
      appId: CONTENT_FACTORY_APP_ID,
      kind: "articleDraft",
      id: ARTICLE_OBJECT_ID,
      sessionId,
      artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID],
      sourceTaskId,
    },
    objects: [
      {
        ref: {
          appId: CONTENT_FACTORY_APP_ID,
          kind: "articleDraft",
          id: ARTICLE_OBJECT_ID,
          sessionId,
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID],
          sourceTaskId,
        },
        title: "公众号文章草稿",
        status: "ready",
        summary: "已生成首版文章",
        previewArtifactId: CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID,
        source: {
          taskKind: "content.article.generate",
          taskId: sourceTaskId,
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID],
          markdown: [
            "# 内容工厂首版文章",
            "",
            "## 三轮资料检索",
            "",
            "本次 fixture 使用结构化 workspace patch 验证文章工作区投影，不调用独立 Plugin worker。资料卡保留主题、引用与编辑恢复所需的最小事实。",
            "",
            "## 正文草稿",
            "",
            "文章正文通过 artifact/write 进入 App Server，再由 Thread/Turn/Item read model 投影到右侧 Article Editor。这个路径只验证 current artifact owner，不声称执行了真实 Provider 写作任务。",
          ].join("\n"),
          researchRounds: [
            {
              id: "research-1",
              title: "检索行业背景",
              query: "AI Agent 写作工作流",
              status: "completed",
              summary: "整理内容工厂和子流程编排的行业背景。",
              citations: ["citation-1", "citation-2"],
            },
          ],
          titleCandidates: [
            {
              id: "title-1",
              title: "内容工厂不是聊天框",
              angle: "产品设计复盘",
              score: 0.92,
            },
          ],
          outline: [
            {
              id: "intro",
              title: "开场：为什么要把写作变成工作流",
              purpose: "解释用户目标",
              points: ["从搜索开始", "通过内容框沉淀产物"],
              evidenceIds: ["citation-1"],
            },
          ],
          keyTakeaways: ["写作应该经过检索、提纲、正文、配图和复核"],
          citations: [
            {
              id: "citation-1",
              title: "产品规划文档",
              sourceType: "internal",
              summary: "Writing 路线图要求内容框输出和右侧展开。",
              status: "selected",
            },
          ],
          imageSlots: [
            {
              id: "hero",
              title: "首图",
              sectionId: "intro",
              purpose: "解释内容工厂工作流",
              prompt: "桌面端内容工厂写作流程图，中文标签",
              status: "planned",
            },
          ],
          writingPlan: [
            {
              id: "plan-1",
              title: "先做资料检索",
              owner: "research-writer",
              skillRef: "article-research",
              output: "结构化资料卡",
              done: true,
            },
          ],
          reviewNotes: ["正文需要保留真实引用来源。"],
          evidenceIds: ["evidence-article-1"],
        },
      },
      {
        ref: {
          appId: CONTENT_FACTORY_APP_ID,
          kind: "imageGenerationSet",
          id: IMAGE_SET_OBJECT_ID,
          sessionId,
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID],
          sourceTaskId: "image_job_1",
        },
        title: "配图组",
        status: "needs_review",
        summary: "等待选择主图",
        previewArtifactId: CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID,
        source: {
          taskKind: "content.image.generate",
          taskId: "image_job_1",
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID],
          images: [
            {
              id: CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID,
              title: "主图",
              url: "https://example.com/content-factory-image-1.png",
              prompt: "明亮的中文内容工厂主图",
            },
          ],
          evidenceIds: ["evidence-image-1"],
        },
      },
      {
        ref: {
          appId: CONTENT_FACTORY_APP_ID,
          kind: "videoStoryboard",
          id: STORYBOARD_OBJECT_ID,
          sessionId,
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_STORYBOARD_ARTIFACT_ID],
          sourceTaskId: "storyboard_job_1",
        },
        title: "视频分镜",
        status: "ready",
        summary: "3 镜头短视频分镜",
        previewArtifactId:
          CONTENT_FACTORY_ARTICLE_WORKSPACE_STORYBOARD_ARTIFACT_ID,
        source: {
          taskKind: "content.video.storyboard.generate",
          taskId: "storyboard_job_1",
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_STORYBOARD_ARTIFACT_ID],
          rendererContract: {
            pluginId: CONTENT_FACTORY_APP_ID,
            rendererKind: "app_declared",
            artifactType: "videoStoryboard",
            outputArtifactKind: "workspace_patch",
            surfaceKind: "articleWorkspace",
            paneKind: "storyboard",
            entry: "./renderer/storyboard.tsx",
            actionKeys: ["open_storyboard"],
            runtimeAuthorization: {
              status: "placeholder_only",
              executionMode: "host_placeholder",
              reasonCode: "app_declared_renderer_placeholder_only",
              requestedOutputArtifactKind: "workspace_patch",
              allowedOutputArtifactKinds: ["workspace_patch"],
            },
          },
          scenes: [
            {
              id: "shot-1",
              title: "厨房开场",
              description: "镜头推近产品",
              visualPrompt: "明亮厨房，自然光",
              duration: "3s",
            },
          ],
        },
      },
      {
        ref: {
          appId: CONTENT_FACTORY_APP_ID,
          kind: "deliveryChecklist",
          id: CHECKLIST_OBJECT_ID,
          sessionId,
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_CHECKLIST_ARTIFACT_ID],
          sourceTaskId: "delivery_checklist_job_1",
        },
        title: "交付检查清单",
        status: "ready",
        summary: "发布前检查项",
        previewArtifactId:
          CONTENT_FACTORY_ARTICLE_WORKSPACE_CHECKLIST_ARTIFACT_ID,
        source: {
          taskKind: "content.delivery.review",
          taskId: "delivery_checklist_job_1",
          artifactIds: [CONTENT_FACTORY_ARTICLE_WORKSPACE_CHECKLIST_ARTIFACT_ID],
          items: [
            {
              id: "article",
              title: "文章已生成",
              status: "done",
            },
            {
              id: "image-license",
              title: "确认图片授权",
              notes: "发布前需复核",
              status: "todo",
            },
          ],
        },
      },
    ],
    layoutState: {
      activeTabKind: "articleWorkspace",
      activePaneKind: "documentCanvas",
      openTabKinds: ["articleWorkspace", "files"],
      splitMode: "chat-right-dock",
    },
    sourceArtifacts: [{ artifactRef: "artifact-workspace-patch-1" }],
    workerEvidence: [
      {
        id: "artifact-workspace-patch-1:workerEvidence",
        status: "completed",
        source: "workspace_patch",
        eventType: "artifact.snapshot",
        appId: CONTENT_FACTORY_APP_ID,
        taskId: sourceTaskId,
        taskKind: "content.article.generate",
        artifactRef: "artifact-workspace-patch-1",
        artifactKind: "workspace_patch",
        outputObjectCount: 4,
        updatedAt: "2026-06-24T00:00:00.000Z",
      },
    ],
    updatedAt: "2026-06-24T00:00:00.000Z",
  };
}

export function buildContentFactoryActionResultWorkspacePatch(
  workspace,
  identity,
) {
  const { sessionId } = identity;
  return {
    schemaVersion: "article-workspace.v1",
    appId: CONTENT_FACTORY_APP_ID,
    sessionId,
    workspaceId: workspace.workspaceId,
    selectedObjectRef: {
      appId: CONTENT_FACTORY_APP_ID,
      kind: "imageGenerationSet",
      id: IMAGE_SET_OBJECT_ID,
      sessionId,
    },
    objects: [
      {
        ref: {
          appId: CONTENT_FACTORY_APP_ID,
          kind: "imageGenerationSet",
          id: IMAGE_SET_OBJECT_ID,
          sessionId,
          artifactIds: ["artifact-image-regenerated"],
          sourceTaskId: "image_regenerate_job_1",
          version: "2",
        },
        title: "配图组",
        status: "ready",
        summary: "已重新生成 2 张候选图",
        previewArtifactId: "artifact-image-regenerated",
        source: {
          taskKind: "content.image.generate",
          taskId: "image_regenerate_job_1",
          artifactIds: ["artifact-image-regenerated"],
          images: [
            {
              id: "image-regenerated-1",
              title: "厨房台面主图",
              url: "https://example.com/content-factory-image-regenerated-1.png",
              prompt: "厨房台面主图，明亮自然光",
            },
            {
              id: "image-regenerated-2",
              title: "产品细节图",
              url: "https://example.com/content-factory-image-regenerated-2.png",
              prompt: "产品细节图，突出质感",
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
    sourceArtifacts: [
      { artifactRef: "artifact-image-regenerate-workspace-patch" },
    ],
    workerEvidence: [
      {
        id: "artifact-image-regenerate-workspace-patch:workerEvidence",
        status: "completed",
        source: "workspace_patch",
        eventType: "artifact.snapshot",
        appId: CONTENT_FACTORY_APP_ID,
        taskId: "image_regenerate_job_1",
        taskKind: "content.image.generate",
        artifactRef: "artifact-image-regenerate-workspace-patch",
        artifactKind: "workspace_patch",
        outputObjectCount: 1,
        updatedAt: "2026-06-24T00:01:00.000Z",
      },
    ],
    updatedAt: "2026-06-24T00:01:00.000Z",
  };
}
