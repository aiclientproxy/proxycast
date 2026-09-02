import { getSeededSkillCatalog, type SkillCatalog } from "./skillCatalog";

export function buildLegacyCatalogWithSiteEntries(): SkillCatalog {
  const seeded = getSeededSkillCatalog();
  const generalSkill = seeded.items[0]!;

  return {
    version: "tenant-2026-03-30",
    tenantId: "tenant-demo",
    syncedAt: "2026-03-30T12:00:00.000Z",
    groups: [
      {
        key: "github",
        title: "GitHub",
        summary: "围绕仓库与 Issue 的只读研究技能。",
        sort: 10,
        itemCount: 1,
      },
      {
        key: "general",
        title: "通用技能",
        summary: "不依赖站点登录态的业务技能。",
        sort: 90,
        itemCount: 1,
      },
    ],
    entries: [],
    items: [
      {
        ...generalSkill,
        id: "legacy-site-skill",
        title: "旧版 GitHub 站点技能",
        skillType: "site",
        defaultExecutorBinding: "browser_assist",
        siteCapabilityBinding: {
          adapterName: "github/search",
          autoRun: true,
          requireAttachedSession: true,
          saveMode: "current_content",
          slotArgMap: {
            reference_topic: "query",
          },
        },
        groupKey: "github",
        execution: {
          kind: "site_adapter",
          siteAdapterBinding: {
            adapterName: "github/search",
            autoRun: true,
            requireAttachedSession: true,
            saveMode: "current_content",
            slotArgMap: {
              reference_topic: "query",
            },
          },
        },
      },
      {
        ...generalSkill,
        id: "tenant-daily-briefing",
        title: "租户日报摘要",
        summary: "远端同步后的目录项",
        groupKey: "general",
        execution: {
          kind: "agent_turn",
        },
      },
    ],
  };
}

export function buildBaseSetupPackage() {
  return {
    id: "content-workflow-base-setup",
    version: "tenant-2026-04-15",
    title: "Content Workflow Base Setup",
    summary: "通过基础设置包定义多模态场景入口",
    bundle_refs: [
      {
        id: "content-workflow-bundle",
        source: "remote",
        path_or_uri: "lime://bundles/content-workflow",
        kind: "skill_bundle",
      },
    ],
    catalog_projections: [
      {
        id: "content-workflow-service",
        target_catalog: "service_skill_catalog",
        entry_key: "content-workflow-service",
        skill_key: "story-video-suite",
        title: "短视频编排",
        summary: "把文本、线框图、配乐和短视频串起来。",
        category: "Workflows",
        output_hint: "结果包",
        bundle_ref_id: "content-workflow-bundle",
        slot_profile_ref: "content-workflow-slot-profile",
        binding_profile_ref: "content-workflow-binding-profile",
        artifact_profile_ref: "content-workflow-artifact-profile",
        scorecard_profile_ref: "content-workflow-scorecard-profile",
        policy_profile_ref: "content-workflow-policy-profile",
        scene_binding: {
          scene_key: "story-video-suite",
          command_prefix: "/legacy-story-video",
          title: "旧版自动场景标题",
          summary: "旧版自动场景摘要",
          aliases: ["story-video-auto"],
        },
      },
      {
        id: "content-workflow-scene",
        target_catalog: "scene_catalog",
        entry_key: "content-workflow-service",
        skill_key: "story-video-suite",
        title: "短视频编排显式场景",
        summary: "用显式 projection 覆盖 auto scene。",
        category: "Workflows",
        output_hint: "结果包",
        bundle_ref_id: "content-workflow-bundle",
        slot_profile_ref: "content-workflow-slot-profile",
        binding_profile_ref: "content-workflow-binding-profile",
        artifact_profile_ref: "content-workflow-artifact-profile",
        scorecard_profile_ref: "content-workflow-scorecard-profile",
        policy_profile_ref: "content-workflow-policy-profile",
        scene_binding: {
          scene_key: "story-video-suite",
          command_prefix: "/story-video-suite",
          title: "短视频编排",
          summary: "把文本生成线框图、配乐、剧本和短视频串成一条场景链。",
          aliases: ["story-video", "mv-pipeline"],
        },
      },
      {
        id: "content-workflow-command",
        target_catalog: "command_catalog",
        entry_key: "content-workflow-service",
        skill_key: "voice_runtime",
        title: "短视频配音入口",
        summary: "用显式 command projection 覆盖 seeded voice_runtime。",
        category: "Workflows",
        output_hint: "结果包",
        bundle_ref_id: "content-workflow-bundle",
        slot_profile_ref: "content-workflow-slot-profile",
        binding_profile_ref: "content-workflow-binding-profile",
        artifact_profile_ref: "content-workflow-artifact-profile",
        scorecard_profile_ref: "content-workflow-scorecard-profile",
        policy_profile_ref: "content-workflow-policy-profile",
        aliases: ["短视频配音", "story-voice"],
        trigger_hints: ["@配音", "/voice-runtime"],
        command_binding: {
          request_defaults: {
            launch_hint: "voice_scene",
          },
          intent_confirmation: {
            id: "plain_voice_request",
            rule_key: "agentChat.voice.intentRules",
            confirmation_key: "agentChat.voice.confirmPlainRequest",
            system_prompt_key: "agentChat.voice.confirmPlainRequestPrompt",
          },
        },
      },
    ],
    slot_profiles: [
      {
        id: "content-workflow-slot-profile",
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
    binding_profiles: [
      {
        id: "content-workflow-binding-profile",
        binding_family: "agent_turn",
      },
    ],
    artifact_profiles: [
      {
        id: "content-workflow-artifact-profile",
        delivery_contract: "artifact_bundle",
        required_parts: ["index.md"],
        viewer_kind: "artifact_bundle",
      },
    ],
    scorecard_profiles: [
      {
        id: "content-workflow-scorecard-profile",
        metrics: ["success_rate"],
      },
    ],
    policy_profiles: [
      {
        id: "content-workflow-policy-profile",
        surface_scopes: ["mention", "workspace"],
      },
    ],
    compatibility: {
      min_app_version: "1.11.0",
      required_kernel_capabilities: ["agent_turn"],
      seeded_fallback: true,
    },
  };
}
