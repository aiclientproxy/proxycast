import { describe, expect, it } from "vitest";
import { compileServiceSkillCatalogProjection } from "./serviceSkillCatalogProjection";
import { compileCommandCatalogProjection } from "./commandCatalogProjection";
import type { BaseSetupPackage } from "../types";

function createBaseSetupPackage(): BaseSetupPackage {
  return {
    id: "command-starter-pack",
    version: "0.1.0",
    title: "Command Starter Pack",
    summary: "把 command_catalog projection 编译到统一目录。",
    bundleRefs: [
      {
        id: "scene-bundle",
        source: "remote",
        pathOrUri: "lime://bundles/scene",
        kind: "skill_bundle",
      },
    ],
    catalogProjections: [
      {
        id: "voice-runtime-service",
        targetCatalog: "service_skill_catalog",
        entryKey: "voice-runtime-service",
        skillKey: "story-video-suite",
        title: "短视频编排服务",
        summary: "把脚本、镜头和配音串成一条内容工作流。",
        category: "Workflows",
        outputHint: "结果包",
        bundleRefId: "scene-bundle",
        slotProfileRef: "voice-runtime-slots",
        bindingProfileRef: "agent-turn-binding",
        artifactProfileRef: "voice-runtime-artifact",
        scorecardProfileRef: "voice-runtime-scorecard",
        policyProfileRef: "command-policy",
      },
      {
        id: "voice-runtime-command",
        targetCatalog: "command_catalog",
        entryKey: "voice-runtime-service",
        skillKey: "voice_runtime",
        title: "配音",
        summary: "从输入栏直接进入配音工作流。",
        category: "Workflows",
        outputHint: "结果包",
        bundleRefId: "scene-bundle",
        slotProfileRef: "voice-runtime-slots",
        bindingProfileRef: "agent-turn-binding",
        artifactProfileRef: "voice-runtime-artifact",
        scorecardProfileRef: "voice-runtime-scorecard",
        policyProfileRef: "command-policy",
        aliases: ["配音", "旁白"],
        triggerHints: ["@配音", "/voice-runtime", "@配音"],
      },
    ],
    slotProfiles: [
      {
        id: "voice-runtime-slots",
        slots: [
          {
            key: "topic",
            label: "主题",
            type: "text",
            required: true,
            placeholder: "输入主题",
          },
        ],
      },
    ],
    bindingProfiles: [
      {
        id: "agent-turn-binding",
        bindingFamily: "agent_turn",
      },
    ],
    artifactProfiles: [
      {
        id: "voice-runtime-artifact",
        deliveryContract: "artifact_bundle",
        requiredParts: ["index.md"],
        viewerKind: "artifact_bundle",
      },
    ],
    scorecardProfiles: [
      {
        id: "voice-runtime-scorecard",
        metrics: ["success_rate"],
      },
    ],
    policyProfiles: [
      {
        id: "command-policy",
        enabled: true,
        surfaceScopes: ["mention", "workspace"],
      },
    ],
    compatibility: {
      minAppVersion: "1.11.0",
      requiredKernelCapabilities: ["agent_turn"],
      seededFallback: true,
      compatCatalogProjection: true,
    },
  };
}

describe("compileCommandCatalogProjection", () => {
  it("应把 command_catalog projection 编译成统一 command entry", () => {
    const pkg = createBaseSetupPackage();
    const compiled = compileServiceSkillCatalogProjection(pkg, {
      tenantId: "tenant-a",
      syncedAt: "2026-04-15T00:00:00.000Z",
    });
    const entries = compileCommandCatalogProjection(
      pkg,
      compiled.catalog.items,
    );

    expect(entries).toEqual([
      expect.objectContaining({
        id: "command:voice_runtime",
        title: "配音",
        summary: "从输入栏直接进入配音工作流。",
        commandKey: "voice_runtime",
        aliases: ["配音", "旁白"],
        surfaceScopes: ["mention", "workspace"],
        triggers: [
          { mode: "mention", prefix: "@配音" },
          { mode: "slash", prefix: "/voice-runtime" },
        ],
        binding: expect.objectContaining({
          skillId: "voice-runtime-service",
          skillLocator: {
            source: "catalog",
            name: "voice_runtime",
          },
          executionKind: "agent_turn",
        }),
        renderContract: expect.objectContaining({
          resultKind: "tool_timeline",
          detailKind: "json",
        }),
      }),
    ]);
  });

  it("缺少显式命令 trigger 时应直接失败", () => {
    const pkg = createBaseSetupPackage();
    pkg.catalogProjections[1] = {
      ...pkg.catalogProjections[1]!,
      triggerHints: ["这是说明文案，不是命令"],
    };
    const compiled = compileServiceSkillCatalogProjection(pkg, {
      tenantId: "tenant-a",
      syncedAt: "2026-04-15T00:00:00.000Z",
    });

    expect(() =>
      compileCommandCatalogProjection(pkg, compiled.catalog.items),
    ).toThrow(/缺少可识别的 triggerHints/);
  });

  it("应优先使用 commandBinding 和 commandRenderContract 显式覆盖命令侧契约", () => {
    const pkg = createBaseSetupPackage();
    pkg.catalogProjections[1] = {
      ...pkg.catalogProjections[1]!,
      entryKey: "poster-generate-runtime",
      skillKey: "poster_generate",
      title: "海报",
      summary: "显式声明 task_queue 海报命令契约。",
      bindingProfileRef: "agent-turn-binding",
      triggerHints: ["@海报"],
      commandBinding: {
        skillId: "image_generate",
        executionKind: "task_queue",
        requestDefaults: {
          imageWorkbench: "true",
        },
        intentConfirmation: {
          id: "plain_image_generation",
          ruleKey: "agentChat.inputIntent.imageGeneration.rules",
          confirmationKey: "agentChat.inputIntent.imageGeneration.confirm",
          systemPromptKey: "agentChat.inputIntent.imageGeneration.systemPrompt",
        },
      },
      commandRenderContract: {
        resultKind: "image_gallery",
        detailKind: "media_detail",
        supportsStreaming: true,
        supportsTimeline: true,
      },
    };
    const compiled = compileServiceSkillCatalogProjection(pkg, {
      tenantId: "tenant-a",
      syncedAt: "2026-04-15T00:00:00.000Z",
    });

    expect(
      compileCommandCatalogProjection(pkg, compiled.catalog.items),
    ).toEqual([
      expect.objectContaining({
        id: "command:poster_generate",
        commandKey: "poster_generate",
        binding: expect.objectContaining({
          skillId: "image_generate",
          skillLocator: {
            source: "catalog",
            name: "poster_generate",
          },
          executionKind: "task_queue",
          requestDefaults: {
            imageWorkbench: "true",
          },
          intentConfirmation: {
            id: "plain_image_generation",
            ruleKey: "agentChat.inputIntent.imageGeneration.rules",
            confirmationKey: "agentChat.inputIntent.imageGeneration.confirm",
            systemPromptKey:
              "agentChat.inputIntent.imageGeneration.systemPrompt",
          },
        }),
        renderContract: {
          resultKind: "image_gallery",
          detailKind: "media_detail",
          supportsStreaming: true,
          supportsTimeline: true,
        },
      }),
    ]);
  });
});
