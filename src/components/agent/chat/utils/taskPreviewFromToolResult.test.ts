import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { loadNamespaceResource } from "@/i18n/loadNamespace";
import { SUPPORTED_LOCALES } from "@/i18n/locales";
import {
  buildImageTaskPreviewFromToolResult,
  buildTaskPreviewFromToolResult,
  buildToolResultArtifactFromToolResult,
} from "./taskPreviewFromToolResult";

beforeEach(async () => {
  await changeLimeLocale("zh-CN");
});

afterEach(async () => {
  await changeLimeLocale("zh-CN");
});

describe("buildImageTaskPreviewFromToolResult", () => {
  it("应在图片任务完成后输出更友好的完成态摘要", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-1",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "未来感青柠实验室",
        size: "1024x1024",
        count: 2,
      }),
      toolResult: {
        metadata: {
          task_id: "task-1",
          task_type: "image_generate",
          status: "succeeded",
          prompt: "未来感青柠实验室",
          size: "1024x1024",
          project_id: "project-1",
          content_id: "content-1",
          path: "/tmp/task-1.json",
          artifact_path: ".lime/tasks/image_generate/task-1.json",
          requested_count: 2,
          received_count: 2,
        },
      },
      fallbackPrompt: "@配图 未来感青柠实验室",
    });

    expect(preview).toMatchObject({
      taskId: "task-1",
      prompt: "未来感青柠实验室",
      status: "complete",
      imageCount: 2,
      size: "1024x1024",
      projectId: "project-1",
      contentId: "content-1",
      taskFilePath: "/tmp/task-1.json",
      artifactPath: ".lime/tasks/image_generate/task-1.json",
      phase: "succeeded",
      statusMessage: "图片生成完成。",
    });
  });

  it("应从 typed 图片工具恢复任务失败态摘要", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-2",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "未来感青柠实验室",
      }),
      toolResult: {
        metadata: {
          task_id: "task-2",
          task_type: "image_generate",
          status: "failed",
        },
      },
      fallbackPrompt: "@配图 未来感青柠实验室",
    });

    expect(preview).toMatchObject({
      taskId: "task-2",
      status: "failed",
      phase: "failed",
      statusMessage: "图片生成失败。",
    });
  });

  it("图片任务完成但尚未带回数量时，应输出面向用户的完成态文案", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-3",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "清晨广州塔",
        size: "1024x1024",
      }),
      toolResult: {
        metadata: {
          task_id: "task-3",
          task_type: "image_generate",
          status: "succeeded",
          prompt: "清晨广州塔",
          size: "1024x1024",
        },
      },
      fallbackPrompt: "@配图 清晨广州塔",
    });

    expect(preview).toMatchObject({
      taskId: "task-3",
      status: "complete",
      phase: "succeeded",
      statusMessage: "图片生成完成。",
    });
  });

  it("图片任务刚提交时，应输出同会话生成态文案", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-4",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "广州塔夜景",
      }),
      toolResult: {
        metadata: {
          task_id: "task-4",
          task_type: "image_generate",
          status: "queued",
          prompt: "广州塔夜景",
        },
      },
      fallbackPrompt: "@配图 广州塔夜景",
    });

    expect(preview).toMatchObject({
      taskId: "task-4",
      status: "running",
      phase: "queued",
      statusMessage: "正在生成图片。",
    });
  });

  it("v2 图片任务只有 image_generation task_family 时，也应恢复图片轻卡", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-v2-family",
      toolName: "mediaTaskArtifact/image/create",
      toolArguments: JSON.stringify({
        prompt: "画一张广州夏天的图",
      }),
      toolResult: {
        metadata: {
          task_id: "task-v2-family",
          task_family: "image_generation",
          status: "pending_submit",
          normalized_status: "pending",
          artifact_path: ".lime/tasks/image_generate/task-v2-family.json",
          record: {
            payload: {
              prompt: "画一张广州夏天的图",
              count: 1,
              session_id: "session-v2",
            },
          },
        },
      },
      fallbackPrompt: "@配图 画一张广州夏天的图",
    });

    expect(preview).toMatchObject({
      taskId: "task-v2-family",
      prompt: "画一张广州夏天的图",
      status: "running",
      phase: "queued",
      artifactPath: ".lime/tasks/image_generate/task-v2-family.json",
      expectedImageCount: 1,
      imageCount: 1,
    });
  });

  it("应从纯 JSON 工具输出恢复图片轻卡，而不是依赖额外 metadata", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-json-1",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "青柠插画",
        count: 1,
        size: "1024x1024",
      }),
      toolResult: {
        success: true,
        output: JSON.stringify({
          success: true,
          task_id: "task-json-image-1",
          task_type: "image_generate",
          task_family: "image",
          status: "pending_submit",
          normalized_status: "pending",
          path: ".lime/tasks/image_generate/task-json-image-1.json",
          absolute_path:
            "/workspace/.lime/tasks/image_generate/task-json-image-1.json",
          artifact_path: ".lime/tasks/image_generate/task-json-image-1.json",
          progress: {
            phase: "pending_submit",
            message: "任务已创建，等待进入队列",
          },
          record: {
            payload: {
              prompt: "青柠插画",
              count: 1,
              size: "1024x1024",
              session_id: "session-json-image-1",
            },
          },
        }),
      },
      fallbackPrompt: "@配图 青柠插画",
    });

    expect(preview).toMatchObject({
      taskId: "task-json-image-1",
      prompt: "青柠插画",
      status: "running",
      phase: "queued",
      taskFilePath:
        "/workspace/.lime/tasks/image_generate/task-json-image-1.json",
      artifactPath: ".lime/tasks/image_generate/task-json-image-1.json",
      expectedImageCount: 1,
      imageCount: 1,
      size: "1024x1024",
      statusMessage: "任务已创建，等待进入队列",
    });
  });

  it("应从顶层 task JSON 工具结果恢复图片轻卡和模型标签", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-top-level-image-1",
      toolName: "mediaTaskArtifact/image/create",
      toolArguments: undefined,
      toolResult: {
        success: true,
        task_id: "task-top-level-image-1",
        task_type: "image_generate",
        task_family: "image",
        status: "pending_submit",
        artifact_path: ".lime/tasks/image_generate/task-top-level-image-1.json",
        record: {
          payload: {
            prompt: "画一张深圳夏天的图",
            provider_id: "fal",
            model: "fal-ai/nano-banana-pro",
            count: 1,
          },
        },
      },
      fallbackPrompt: "@Nanobanana Pro 画一张深圳夏天的图",
    });

    expect(preview).toMatchObject({
      taskId: "task-top-level-image-1",
      prompt: "画一张深圳夏天的图",
      status: "running",
      phase: "queued",
      artifactPath: ".lime/tasks/image_generate/task-top-level-image-1.json",
      expectedImageCount: 1,
      imageCount: 1,
      providerName: "fal",
      modelName: "fal-ai/nano-banana-pro",
      caption: null,
    });
  });

  it("图片任务只有相对 path 时，应结合工具参数里的项目根目录恢复 task file", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-relative-image-1",
      toolName: "mediaTaskArtifact/image/create",
      toolArguments: JSON.stringify({
        prompt: "画一张深圳夏天的图",
        projectRootPath:
          "/Users/coso/Library/Application Support/lime/projects/demo",
      }),
      toolResult: {
        success: true,
        task_id: "task-relative-image-1",
        task_type: "image_generate",
        task_family: "image",
        status: "pending_submit",
        path: ".lime/tasks/image_generate/task-relative-image-1.json",
        record: {
          payload: {
            prompt: "画一张深圳夏天的图",
            count: 1,
          },
        },
      },
      fallbackPrompt: "@配图 画一张深圳夏天的图",
    });

    expect(preview).toMatchObject({
      taskId: "task-relative-image-1",
      taskFilePath:
        "/Users/coso/Library/Application Support/lime/projects/demo/.lime/tasks/image_generate/task-relative-image-1.json",
      artifactPath: ".lime/tasks/image_generate/task-relative-image-1.json",
      status: "running",
    });
  });

  it("图片任务完成态应把结果描述投影为图片下方 caption", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-caption-image-1",
      toolName: "mediaTaskArtifact/image/create",
      toolArguments: undefined,
      toolResult: {
        success: true,
        task_id: "task-caption-image-1",
        task_type: "image_generate",
        task_family: "image",
        status: "succeeded",
        received_count: 1,
        record: {
          payload: {
            prompt: "从花城汇看广州塔的春天照片",
            model: "fal-ai/nano-banana-pro",
            presentation: {
              result_caption:
                "搞定，从花城汇看广州塔的春日景象已经好了，要调整的话直接说。",
            },
          },
        },
      },
      fallbackPrompt: "@Nanobanana Pro 从花城汇看广州塔",
    });

    expect(preview).toMatchObject({
      taskId: "task-caption-image-1",
      prompt: "从花城汇看广州塔的春天照片",
      status: "complete",
      imageCount: 1,
      modelName: "fal-ai/nano-banana-pro",
      caption: "搞定，从花城汇看广州塔的春日景象已经好了，要调整的话直接说。",
    });
  });

  it("图片任务排队态应保留后端 completion caption 供完成态展示", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-pending-caption-image-1",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "从花城汇看广州塔的春天照片",
        model: "fal-ai/nano-banana-pro",
      }),
      toolResult: {
        success: true,
        task_id: "task-pending-caption-image-1",
        task_type: "image_generate",
        task_family: "image_generation",
        status: "pending_submit",
        normalized_status: "pending",
        record: {
          payload: {
            prompt: "从花城汇看广州塔的春天照片",
            model: "fal-ai/nano-banana-pro",
            presentation: {
              result_captions: {
                complete: "完成了，花城汇望向广州塔的春日画面已经生成。",
              },
            },
          },
        },
      },
      fallbackPrompt: "@Nanobanana Pro 从花城汇看广州塔",
    });

    expect(preview).toMatchObject({
      taskId: "task-pending-caption-image-1",
      prompt: "从花城汇看广州塔的春天照片",
      status: "running",
      modelName: "fal-ai/nano-banana-pro",
      caption: "完成了，花城汇望向广州塔的春日画面已经生成。",
    });
  });

  it("图片任务完成态应从 result.images 读取真实预览图", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-result-image-url",
      toolName: "mediaTaskArtifact/image/create",
      toolArguments: undefined,
      toolResult: {
        success: true,
        task_id: "task-result-image-url",
        task_type: "image_generate",
        task_family: "image",
        status: "succeeded",
        normalized_status: "succeeded",
        record: {
          task_id: "task-result-image-url",
          task_type: "image_generate",
          status: "succeeded",
          normalized_status: "succeeded",
          payload: {
            prompt: "深圳夏天午后的城市照片",
            model: "agnes-image-2.1-flash",
          },
          result: {
            images: [
              {
                url: "https://platform-outputs.agnes-ai.space/images/t2i/example.png",
                slot_id: "image-slot-1",
                slot_index: 1,
              },
            ],
          },
        },
      },
      fallbackPrompt: "@配图 深圳夏天午后的城市照片",
    });

    expect(preview).toMatchObject({
      taskId: "task-result-image-url",
      status: "complete",
      imageUrl:
        "https://platform-outputs.agnes-ai.space/images/t2i/example.png",
      previewImages: [
        "https://platform-outputs.agnes-ai.space/images/t2i/example.png",
      ],
      modelName: "agnes-image-2.1-flash",
    });
  });

  it("图片任务完成态会清理 caption 里的混合命令标签", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-caption-image-polluted",
      toolName: "mediaTaskArtifact/image/create",
      toolArguments: undefined,
      toolResult: {
        success: true,
        task_id: "task-caption-image-polluted",
        task_type: "image_generate",
        task_family: "image",
        status: "succeeded",
        received_count: 1,
        record: {
          payload: {
            prompt: "Generate 深圳夏day午后的城市照片，阳光明亮，真实摄影Style",
            presentation: {
              result_caption:
                "搞定，深圳夏day午后的城市照片，真实摄影Style 已经做好了。",
            },
          },
        },
      },
      fallbackPrompt:
        "@配图 用 Agnes Generate一张深圳夏day午后的城市照片，真实摄影Style",
    });

    expect(preview?.caption).toContain("真实摄影Style");
    expect(preview?.caption).toContain("深圳夏day");
  });

  it("应从 structuredContent 工具结果恢复图片轻卡，避免 Skill 子会话漏卡", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-structured-image-1",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "广州夏天的骑楼街景",
        count: 1,
      }),
      toolResult: {
        success: true,
        output: "任务正在排队生成",
        structuredContent: {
          success: true,
          task_id: "task-structured-image-1",
          task_type: "image_generate",
          task_family: "image",
          status: "pending_submit",
          normalized_status: "pending",
          path: ".lime/tasks/image_generate/task-structured-image-1.json",
          absolute_path:
            "/workspace/.lime/tasks/image_generate/task-structured-image-1.json",
          artifact_path:
            ".lime/tasks/image_generate/task-structured-image-1.json",
          record: {
            task_id: "task-structured-image-1",
            task_type: "image_generate",
            payload: {
              prompt: "广州夏天的骑楼街景",
              size: "1024x1024",
              provider_id: "openai",
              model: "gpt-images-2",
            },
          },
        },
      },
      fallbackPrompt: "@配图 广州夏天的骑楼街景",
    });

    expect(preview).toMatchObject({
      taskId: "task-structured-image-1",
      prompt: "广州夏天的骑楼街景",
      status: "running",
      phase: "queued",
      taskFilePath:
        "/workspace/.lime/tasks/image_generate/task-structured-image-1.json",
      artifactPath: ".lime/tasks/image_generate/task-structured-image-1.json",
      size: "1024x1024",
      statusMessage: "正在生成图片。",
    });
  });

  it("3x3 分镜完成后应输出更贴近布局语义的摘要", () => {
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-5",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "三国主要人物",
        count: 9,
        layout_hint: "storyboard_3x3",
      }),
      toolResult: {
        metadata: {
          task_id: "task-5",
          task_type: "image_generate",
          status: "succeeded",
          prompt: "三国主要人物",
          requested_count: 9,
          received_count: 9,
          layout_hint: "storyboard_3x3",
        },
      },
      fallbackPrompt: "@分镜 生成 三国主要人物，3x3 分镜",
    });

    expect(preview).toMatchObject({
      taskId: "task-5",
      status: "complete",
      imageCount: 9,
      layoutHint: "storyboard_3x3",
      statusMessage: "3x3 分镜生成完成。",
    });
  });

  it("英文界面不应在图片任务默认状态里混入中文 fallback", async () => {
    await changeLimeLocale("en-US");
    const preview = buildImageTaskPreviewFromToolResult({
      toolId: "tool-image-en",
      toolName: "lime_create_image_generation_task",
      toolArguments: JSON.stringify({
        prompt: "workspace concept",
        layout_hint: "storyboard_3x3",
      }),
      toolResult: {
        metadata: {
          task_id: "task-image-en",
          task_type: "image_generate",
          status: "succeeded",
          requested_count: 9,
          received_count: 9,
          layout_hint: "storyboard_3x3",
        },
      },
      fallbackPrompt: "",
    });

    expect(preview).toMatchObject({
      taskId: "task-image-en",
      prompt: "workspace concept",
      statusMessage: "3x3 storyboard generation completed.",
    });
    expect(JSON.stringify(preview)).not.toMatch(
      /[图片圖片分镜分鏡生成进行進行]/,
    );
  });

  it("图片任务预览文案资源应覆盖所有支持语言", () => {
    const requiredKeys = [
      "agentChat.taskPreview.image.fallbackPrompt",
      "agentChat.taskPreview.image.status.cancelled",
      "agentChat.taskPreview.image.status.complete.default",
      "agentChat.taskPreview.image.status.complete.storyboard3x3",
      "agentChat.taskPreview.image.status.failed",
      "agentChat.taskPreview.image.status.partial.default",
      "agentChat.taskPreview.image.status.partial.storyboard3x3",
      "agentChat.taskPreview.image.status.running",
    ];

    for (const locale of SUPPORTED_LOCALES) {
      const resource = loadNamespaceResource(locale, "agent");
      for (const key of requiredKeys) {
        expect(resource[key], `${locale} missing ${key}`).toBeTruthy();
      }
    }
  });
});

describe("buildTaskPreviewFromToolResult web image search", () => {
  it("联网搜图工具应输出图片候选预览与带来源的 artifact document", () => {
    const toolResult = {
      metadata: {
        provider: "pexels",
        result: {
          provider: "pexels",
          query: "cozy coffee table",
          returnedCount: 2,
          aspect: "landscape",
          hits: [
            {
              id: "hit-1",
              thumbnail_url: "https://pexels.example/1-thumb.jpg",
              content_url: "https://pexels.example/1.jpg",
              host_page_url: "https://www.pexels.com/photo/1",
              width: 1600,
              height: 900,
              name: "cozy coffee table 1",
            },
            {
              id: "hit-2",
              thumbnail_url: "https://pexels.example/2-thumb.jpg",
              content_url: "https://pexels.example/2.jpg",
              host_page_url: "https://www.pexels.com/photo/2",
              width: 1600,
              height: 900,
              name: "cozy coffee table 2",
            },
          ],
        },
      },
    };

    const preview = buildTaskPreviewFromToolResult({
      toolId: "tool-web-image-1",
      toolName: "lime_search_web_images",
      toolArguments: JSON.stringify({
        query: "cozy coffee table",
        count: 2,
        aspect: "landscape",
      }),
      toolResult,
      fallbackPrompt: "帮我找咖啡馆木桌背景图",
    });

    expect(preview).toMatchObject({
      kind: "modal_resource_search",
      taskId: "resource-search:tool-web-image-1",
      taskType: "modal_resource_search",
      prompt: "cozy coffee table",
      title: "Pexels 图片候选",
      status: "complete",
      artifactPath: ".lime/runtime/resource-search/tool-web-image-1.md",
      providerId: "pexels",
      phase: "completed",
      statusMessage:
        "已找到 2 张 Pexels 图片候选，打开查看可继续挑选与查看来源。",
      metaItems: ["Pexels", "2 个候选", "landscape"],
      imageCandidates: [
        expect.objectContaining({
          id: "hit-1",
          thumbnailUrl: "https://pexels.example/1-thumb.jpg",
          contentUrl: "https://pexels.example/1.jpg",
          hostPageUrl: "https://www.pexels.com/photo/1",
        }),
        expect.objectContaining({
          id: "hit-2",
          thumbnailUrl: "https://pexels.example/2-thumb.jpg",
          contentUrl: "https://pexels.example/2.jpg",
          hostPageUrl: "https://www.pexels.com/photo/2",
        }),
      ],
    });

    const artifact = buildToolResultArtifactFromToolResult({
      toolId: "tool-web-image-1",
      toolName: "lime_search_web_images",
      toolArguments: JSON.stringify({
        query: "cozy coffee table",
        count: 2,
        aspect: "landscape",
      }),
      toolResult,
      fallbackPrompt: "帮我找咖啡馆木桌背景图",
    });

    expect(artifact).toMatchObject({
      filePath: ".lime/runtime/resource-search/tool-web-image-1.md",
      metadata: {
        artifact_type: "document",
        previewText: "已找到 2 张 Pexels 图片候选",
        provider: "pexels",
        query: "cozy coffee table",
        returnedCount: 2,
        aspect: "landscape",
      },
    });
    expect(artifact?.metadata.artifactDocument).toMatchObject({
      artifactId: "resource-search:tool-web-image-1",
      title: "Pexels 图片候选",
      sources: [
        expect.objectContaining({
          id: "source-1",
          label: "cozy coffee table 1",
          locator: { url: "https://www.pexels.com/photo/1" },
        }),
        expect.objectContaining({
          id: "source-2",
          label: "cozy coffee table 2",
          locator: { url: "https://www.pexels.com/photo/2" },
        }),
      ],
    });
  });

  it("英文界面不应在联网搜图预览和 artifact 文档里混入中文拼接", async () => {
    await changeLimeLocale("en-US");
    const toolResult = {
      metadata: {
        result: {
          query: "workspace moodboard",
          returnedCount: 1,
          hits: [
            {
              thumbnail_url: "https://images.example/thumb.jpg",
              content_url: "https://images.example/full.jpg",
            },
          ],
        },
      },
    };

    const preview = buildTaskPreviewFromToolResult({
      toolId: "tool-web-image-en",
      toolName: "lime_search_web_images",
      toolArguments: JSON.stringify({ query: "workspace moodboard" }),
      toolResult,
      fallbackPrompt: "find a workspace moodboard",
    });

    expect(preview).toMatchObject({
      kind: "modal_resource_search",
      title: "web image library image candidates",
      statusMessage:
        "Found 1 web image library image candidate(s). Open them to keep reviewing and checking sources.",
      metaItems: ["web image library", "1 candidate(s)"],
      imageCandidates: [
        expect.objectContaining({
          name: "Image candidate 1",
        }),
      ],
    });

    const artifact = buildToolResultArtifactFromToolResult({
      toolId: "tool-web-image-en",
      toolName: "lime_search_web_images",
      toolArguments: JSON.stringify({ query: "workspace moodboard" }),
      toolResult,
      fallbackPrompt: "find a workspace moodboard",
    });

    expect(artifact?.metadata.previewText).toBe(
      "Found 1 web image library image candidate(s)",
    );
    expect(artifact?.metadata.artifactDocument).toMatchObject({
      language: "en-US",
      title: "web image library image candidates",
      summary:
        '1 image asset candidate(s) were returned for "workspace moodboard".',
      blocks: [
        expect.objectContaining({
          eyebrow: "Asset search",
          summary:
            "1 high-relevance image candidate(s) were returned. Open the right panel to keep reviewing and checking sources.",
          highlights: ["Source: web image library", "Candidates: 1"],
        }),
        expect.objectContaining({
          alt: "Image candidate 1",
        }),
      ],
      sources: [
        expect.objectContaining({
          label: "Image candidate 1",
        }),
      ],
    });

    const visibleCopy = JSON.stringify({
      preview,
      artifactDocument: artifact?.metadata.artifactDocument,
    });
    expect(visibleCopy).not.toMatch(/[候選候选素材畫幅画幅来源來源]/);
  });

  it("联网搜图预览文案资源应覆盖所有支持语言", () => {
    const requiredKeys = [
      "agentChat.taskPreview.webImageSearch.artifact.eyebrow",
      "agentChat.taskPreview.webImageSearch.artifact.heroSummary",
      "agentChat.taskPreview.webImageSearch.artifact.summary",
      "agentChat.taskPreview.webImageSearch.candidateLabel",
      "agentChat.taskPreview.webImageSearch.countMeta",
      "agentChat.taskPreview.webImageSearch.highlight.aspect",
      "agentChat.taskPreview.webImageSearch.highlight.candidates",
      "agentChat.taskPreview.webImageSearch.highlight.source",
      "agentChat.taskPreview.webImageSearch.previewText",
      "agentChat.taskPreview.webImageSearch.provider.generic",
      "agentChat.taskPreview.webImageSearch.queryFallback",
      "agentChat.taskPreview.webImageSearch.status",
      "agentChat.taskPreview.webImageSearch.title",
    ];

    for (const locale of SUPPORTED_LOCALES) {
      const resource = loadNamespaceResource(locale, "agent");
      for (const key of requiredKeys) {
        expect(resource[key], `${locale} missing ${key}`).toBeTruthy();
      }
    }
  });
});

describe("buildTaskPreviewFromToolResult video", () => {
  it("视频任务完成但尚未带回结果地址时，应输出完成态同步文案", () => {
    const preview = buildTaskPreviewFromToolResult({
      toolId: "tool-video-1",
      toolName: "video_generate",
      toolArguments: JSON.stringify({
        prompt: "广州塔城市宣传片",
        duration: 15,
        aspect_ratio: "16:9",
        resolution: "720p",
      }),
      toolResult: {
        metadata: {
          task_id: "task-video-1",
          task_type: "video_generate",
          status: "succeeded",
          prompt: "广州塔城市宣传片",
        },
      },
      fallbackPrompt: "@视频 15秒 广州塔城市宣传片，16:9，720p",
    });

    expect(preview).toMatchObject({
      kind: "video_generate",
      taskId: "task-video-1",
      status: "complete",
      phase: "succeeded",
      statusMessage: "视频已经生成完成，正在同步最终结果。",
    });
  });

  it("视频任务排队中时，应输出用户可理解的排队态文案", () => {
    const preview = buildTaskPreviewFromToolResult({
      toolId: "tool-video-2",
      toolName: "video_generate",
      toolArguments: JSON.stringify({
        prompt: "新品发布短视频",
        duration: 15,
        aspect_ratio: "16:9",
        resolution: "720p",
      }),
      toolResult: {
        metadata: {
          task_id: "task-video-2",
          task_type: "video_generate",
          status: "queued",
          prompt: "新品发布短视频",
        },
      },
      fallbackPrompt: "@视频 15秒 新品发布短视频，16:9，720p",
    });

    expect(preview).toMatchObject({
      kind: "video_generate",
      taskId: "task-video-2",
      status: "running",
      phase: "queued",
      statusMessage: "视频任务已进入排队队列，稍后会自动开始生成。",
    });
  });

  it("英文界面不应在视频任务默认状态里混入中文 fallback", async () => {
    await changeLimeLocale("en-US");
    const preview = buildTaskPreviewFromToolResult({
      toolId: "tool-video-en",
      toolName: "video_generate",
      toolArguments: JSON.stringify({
        prompt: "launch film",
        duration: 15,
        aspect_ratio: "16:9",
        resolution: "720p",
      }),
      toolResult: {
        metadata: {
          task_id: "task-video-en",
          task_type: "video_generate",
          status: "queued",
        },
      },
      fallbackPrompt: "",
    });

    expect(preview).toMatchObject({
      kind: "video_generate",
      taskId: "task-video-en",
      prompt: "launch film",
      statusMessage: "The video task is queued and will start automatically.",
    });
    expect(JSON.stringify(preview)).not.toMatch(
      /[视频影片任務任务生成排队排隊]/,
    );
  });

  it("视频任务预览文案资源应覆盖所有支持语言", () => {
    const requiredKeys = [
      "agentChat.taskPreview.video.fallbackPrompt",
      "agentChat.taskPreview.video.status.cancelled",
      "agentChat.taskPreview.video.status.complete.synced",
      "agentChat.taskPreview.video.status.complete.waitingResult",
      "agentChat.taskPreview.video.status.failed",
      "agentChat.taskPreview.video.status.partial.synced",
      "agentChat.taskPreview.video.status.partial.waitingResult",
      "agentChat.taskPreview.video.status.queued",
      "agentChat.taskPreview.video.status.running",
    ];

    for (const locale of SUPPORTED_LOCALES) {
      const resource = loadNamespaceResource(locale, "agent");
      for (const key of requiredKeys) {
        expect(resource[key], `${locale} missing ${key}`).toBeTruthy();
      }
    }
  });
});

describe("buildTaskPreviewFromToolResult audio", () => {
  it("配音任务应输出 audio_generate 预览并指向可打开的运行时文档", () => {
    const preview = buildTaskPreviewFromToolResult({
      toolId: "tool-audio-1",
      toolName: "lime_create_audio_generation_task",
      toolArguments: JSON.stringify({
        sourceText: "欢迎来到 Lime 多模态工作台。",
        voice: "warm_female",
        voiceStyle: "温暖克制",
        targetLanguage: "zh-CN",
        durationMs: 8200,
      }),
      toolResult: {
        metadata: {
          task_id: "task-audio-1",
          task_type: "audio_generate",
          status: "pending_submit",
          prompt: "欢迎来到 Lime 多模态工作台。",
          artifact_path: ".lime/tasks/audio_generate/task-audio-1.json",
          provider_id: "voice-runtime",
          model: "voice-pro",
        },
      },
      fallbackPrompt: "@配音 欢迎来到 Lime 多模态工作台。",
    });

    expect(preview).toMatchObject({
      kind: "audio_generate",
      taskId: "task-audio-1",
      taskType: "audio_generate",
      prompt: "欢迎来到 Lime 多模态工作台。",
      status: "running",
      phase: "queued",
      artifactPath: ".lime/runtime/audio-generate/task-audio-1.md",
      taskFilePath: ".lime/tasks/audio_generate/task-audio-1.json",
      providerId: "voice-runtime",
      model: "voice-pro",
      voice: "warm_female",
      durationMs: 8200,
      metaItems: ["warm_female", "温暖克制", "zh-CN", "8 秒"],
      statusMessage:
        "配音任务已写入 audio_task/audio_output，工作区会继续同步音频结果。",
    });
  });

  describe("buildTaskPreviewFromToolResult transcription", () => {
    it("转写任务应输出 transcription_generate 预览并指向运行时文档", () => {
      const preview = buildTaskPreviewFromToolResult({
        toolId: "tool-transcription-1",
        toolName: "lime_create_transcription_task",
        toolArguments: JSON.stringify({
          prompt: "请转写访谈音频",
          sourcePath: "materials/interview.wav",
          language: "zh-CN",
          outputFormat: "txt",
        }),
        toolResult: {
          metadata: {
            task_id: "task-transcription-1",
            task_type: "transcription_generate",
            status: "pending_submit",
            prompt: "请转写访谈音频",
            artifact_path:
              ".lime/tasks/transcription_generate/task-transcription-1.json",
            provider_id: "openai-asr",
            model: "gpt-4o-transcribe",
          },
        },
        fallbackPrompt: "@转写 materials/interview.wav",
      });

      expect(preview).toMatchObject({
        kind: "transcription_generate",
        taskId: "task-transcription-1",
        taskType: "transcription_generate",
        prompt: "请转写访谈音频",
        status: "running",
        phase: "queued",
        artifactPath:
          ".lime/runtime/transcription-generate/task-transcription-1.md",
        taskFilePath:
          ".lime/tasks/transcription_generate/task-transcription-1.json",
        sourcePath: "materials/interview.wav",
        language: "zh-CN",
        outputFormat: "txt",
        providerId: "openai-asr",
        model: "gpt-4o-transcribe",
        metaItems: ["materials/interview.wav", "zh-CN", "txt"],
        statusMessage: "转写任务已提交，工作区会继续同步最新进度。",
      });
    });

    it("转写任务工具结果应生成 transcript viewer 文档，避免打开隐藏 task json", () => {
      const artifact = buildToolResultArtifactFromToolResult({
        toolId: "tool-transcription-2",
        toolName: "lime_create_transcription_task",
        toolArguments: JSON.stringify({
          prompt: "请转写访谈音频",
          sourcePath: "materials/interview.wav",
          language: "zh-CN",
          outputFormat: "txt",
        }),
        toolResult: {
          metadata: {
            task_id: "task-transcription-2",
            task_type: "transcription_generate",
            status: "succeeded",
            artifact_path:
              ".lime/tasks/transcription_generate/task-transcription-2.json",
            transcript_path:
              ".lime/runtime/transcripts/task-transcription-2.txt",
            transcript_text: "欢迎来到 Lime 访谈节目。",
            transcript_segments: [
              {
                start: 1,
                end: 3.2,
                speaker: "主持人",
                text: "欢迎来到 Lime 访谈节目。",
              },
            ],
          },
        },
        fallbackPrompt: "@转写 materials/interview.wav",
      });

      expect(artifact).toMatchObject({
        filePath:
          ".lime/runtime/transcription-generate/task-transcription-2.md",
        metadata: {
          artifact_type: "document",
          taskId: "task-transcription-2",
          taskType: "transcription_generate",
          taskFilePath:
            ".lime/tasks/transcription_generate/task-transcription-2.json",
          transcriptPath: ".lime/runtime/transcripts/task-transcription-2.txt",
          transcriptText: "欢迎来到 Lime 访谈节目。",
          modalityContractKey: "audio_transcription",
        },
      });
      expect(artifact?.metadata.artifactDocument).toMatchObject({
        artifactId: "transcription-generate:task-transcription-2",
        title: "内容转写任务",
        metadata: {
          taskFilePath:
            ".lime/tasks/transcription_generate/task-transcription-2.json",
          transcriptPath: ".lime/runtime/transcripts/task-transcription-2.txt",
          transcriptText: "欢迎来到 Lime 访谈节目。",
          transcriptCorrectionEnabled: true,
          transcriptCorrectionStatus: "available",
          transcriptCorrectionSource: "artifact_document_version",
          transcriptCorrectionPatchKind: "artifact_document_version",
          transcriptCorrectionOriginalImmutable: true,
          transcriptSegments: [
            {
              id: "segment-1",
              index: 1,
              startMs: 1000,
              endMs: 3200,
              speaker: "主持人",
              text: "欢迎来到 Lime 访谈节目。",
            },
          ],
        },
      });
      expect(artifact?.metadata.artifactDocument).toMatchObject({
        blocks: expect.arrayContaining([
          expect.objectContaining({
            id: "transcript-segments",
            type: "table",
            title: "转写时间轴（可逐段编辑校对）",
            rows: [["00:01 - 00:03", "主持人", "欢迎来到 Lime 访谈节目。"]],
          }),
          expect.objectContaining({
            id: "transcript-text",
            type: "code_block",
            title: "转写文本（可编辑校对）",
            code: "欢迎来到 Lime 访谈节目。",
          }),
          expect.objectContaining({
            id: "transcript-output",
            type: "callout",
            title: "Transcript 已同步，可校对保存",
            body: expect.stringContaining("不改写原始 ASR 输出"),
          }),
        ]),
      });
    });
  });

  it("配音任务工具结果应生成轻量 artifact document，避免打开隐藏 task json", () => {
    const artifact = buildToolResultArtifactFromToolResult({
      toolId: "tool-audio-2",
      toolName: "lime_create_audio_generation_task",
      toolArguments: JSON.stringify({
        sourceText: "请用轻快语气播报新品发布。",
        voice: "brand_voice",
        audioPath: "https://cdn.example/audio/task-audio-2.mp3",
      }),
      toolResult: {
        metadata: {
          task_id: "task-audio-2",
          task_type: "audio_generate",
          status: "succeeded",
          artifact_path: ".lime/tasks/audio_generate/task-audio-2.json",
          mime_type: "audio/mpeg",
        },
      },
      fallbackPrompt: "@配音 请用轻快语气播报新品发布。",
    });

    expect(artifact).toMatchObject({
      filePath: ".lime/runtime/audio-generate/task-audio-2.md",
      metadata: {
        artifact_type: "document",
        taskId: "task-audio-2",
        taskType: "audio_generate",
        taskFilePath: ".lime/tasks/audio_generate/task-audio-2.json",
        audioUrl: "https://cdn.example/audio/task-audio-2.mp3",
        modalityContractKey: "voice_generation",
      },
    });
    expect(artifact?.metadata.artifactDocument).toMatchObject({
      artifactId: "audio-generate:task-audio-2",
      title: "配音生成任务",
      metadata: {
        taskFilePath: ".lime/tasks/audio_generate/task-audio-2.json",
        audioUrl: "https://cdn.example/audio/task-audio-2.mp3",
      },
    });
  });

  it("英文界面的配音与转写 artifact 文档不应混入中文 fallback", async () => {
    await changeLimeLocale("en-US");

    const audioArtifact = buildToolResultArtifactFromToolResult({
      toolId: "tool-audio-en",
      toolName: "lime_create_audio_generation_task",
      toolArguments: JSON.stringify({
        sourceText: "Welcome to the workspace.",
        voice: "warm_female",
        audioPath: "https://cdn.example/audio/task-audio-en.mp3",
      }),
      toolResult: {
        metadata: {
          task_id: "task-audio-en",
          task_type: "audio_generate",
          status: "succeeded",
          artifact_path: ".lime/tasks/audio_generate/task-audio-en.json",
          model: "voice-pro",
        },
      },
      fallbackPrompt: "@voice Welcome to the workspace.",
    });

    expect(audioArtifact?.metadata.previewText).toBe(
      "Audio results are synced. Open them to continue previewing and managing the task.",
    );
    expect(audioArtifact?.metadata.artifactDocument).toMatchObject({
      language: "en-US",
      title: "Voice generation task",
      blocks: expect.arrayContaining([
        expect.objectContaining({
          eyebrow: "Voice generation",
          highlights: expect.arrayContaining([
            "Status: complete",
            "Voice: warm_female",
            "Model: voice-pro",
          ]),
        }),
        expect.objectContaining({
          id: "source-text",
          markdown: expect.stringContaining("### Text to voice"),
        }),
        expect.objectContaining({
          id: "audio-output",
          title: "Audio result synced",
          body: "Audio path: https://cdn.example/audio/task-audio-en.mp3",
        }),
      ]),
    });

    const transcriptionArtifact = buildToolResultArtifactFromToolResult({
      toolId: "tool-transcription-en",
      toolName: "lime_create_transcription_task",
      toolArguments: JSON.stringify({
        prompt: "Transcribe the interview",
        sourcePath: "materials/interview.wav",
        language: "en-US",
        outputFormat: "txt",
      }),
      toolResult: {
        metadata: {
          task_id: "task-transcription-en",
          task_type: "transcription_generate",
          status: "succeeded",
          artifact_path:
            ".lime/tasks/transcription_generate/task-transcription-en.json",
          transcript_path:
            ".lime/runtime/transcripts/task-transcription-en.txt",
          transcript_text: "Welcome to the interview.",
          transcript_segments: [
            {
              start: 1,
              end: 3.2,
              speaker: "",
              text: "Welcome to the interview.",
            },
          ],
          model: "gpt-4o-transcribe",
        },
      },
      fallbackPrompt: "@transcribe materials/interview.wav",
    });

    expect(transcriptionArtifact?.metadata.previewText).toBe(
      "Transcription results are synced. Open them to continue reviewing the text.",
    );
    expect(transcriptionArtifact?.metadata.artifactDocument).toMatchObject({
      language: "en-US",
      title: "Transcription task",
      blocks: expect.arrayContaining([
        expect.objectContaining({
          eyebrow: "Transcription",
          highlights: expect.arrayContaining([
            "Status: complete",
            "Language: en-US",
            "Format: txt",
            "Model: gpt-4o-transcribe",
            "Segments: 1",
            "Characters: 25",
          ]),
        }),
        expect.objectContaining({
          id: "source",
          markdown: expect.stringContaining("### Transcription source"),
        }),
        expect.objectContaining({
          id: "transcript-segments",
          title: "Transcript timeline (editable by segment)",
          columns: ["Time", "Speaker", "Content"],
          rows: [["00:01 - 00:03", "Unlabeled", "Welcome to the interview."]],
        }),
        expect.objectContaining({
          id: "transcript-output",
          title: "Transcript synced for review",
          body: expect.stringContaining(
            "without rewriting the original ASR output",
          ),
        }),
      ]),
    });

    const visibleCopy = JSON.stringify({
      audio: audioArtifact?.metadata.artifactDocument,
      transcription: transcriptionArtifact?.metadata.artifactDocument,
      audioPreviewText: audioArtifact?.metadata.previewText,
      transcriptionPreviewText: transcriptionArtifact?.metadata.previewText,
    });
    expect(visibleCopy).not.toMatch(
      /[配音转寫轉写音频音訊路径路徑来源來源等待校对校對候选候選]/,
    );
  });
});
