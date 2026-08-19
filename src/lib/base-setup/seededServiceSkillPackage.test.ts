import { describe, expect, it } from "vitest";
import {
  createSeededCloudServiceSkillCatalog,
  createSeededServiceSkillBaseSetupPackage,
  SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID,
  SEEDED_SERVICE_SKILL_CATALOG_VERSION,
} from "./seededServiceSkillPackage";

describe("seededServiceSkillPackage", () => {
  it("应提供 seeded 基础设置包事实源", () => {
    const pkg = createSeededServiceSkillBaseSetupPackage();

    expect(pkg.id).toBe("lime-seeded-service-skills");
    expect(pkg.version).toBe(SEEDED_SERVICE_SKILL_CATALOG_VERSION);
    expect(pkg.catalogProjections).toHaveLength(17);
    expect(pkg.catalogProjections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "personal-ip-knowledge-builder",
          targetCatalog: "service_skill_catalog",
          bindingProfileRef: "native-skill-instant",
          skillBundleMetadata: expect.objectContaining({
            Lime_knowledge_pack_type: "personal-profile",
            Lime_agent_knowledge_runtime_mode: "persona",
            Lime_skill_bundle_path:
              "lime-rs/resources/default-skills/personal-ip-knowledge-builder",
          }),
        }),
        expect.objectContaining({
          id: "brand-persona-knowledge-builder",
          targetCatalog: "service_skill_catalog",
          bindingProfileRef: "native-skill-instant",
          skillBundleMetadata: expect.objectContaining({
            Lime_knowledge_pack_type: "brand-persona",
            Lime_agent_knowledge_runtime_mode: "persona",
            Lime_skill_bundle_path:
              "lime-rs/resources/default-skills/brand-persona-knowledge-builder",
          }),
        }),
        expect.objectContaining({
          id: "content-operations-knowledge-builder",
          targetCatalog: "service_skill_catalog",
          bindingProfileRef: "native-skill-instant",
          skillBundleMetadata: expect.objectContaining({
            Lime_knowledge_pack_type: "content-operations",
            Lime_agent_knowledge_runtime_mode: "data",
            Lime_skill_bundle_path:
              "lime-rs/resources/default-skills/content-operations-knowledge-builder",
          }),
        }),
        expect.objectContaining({
          id: "brand-product-knowledge-builder",
          targetCatalog: "service_skill_catalog",
          bindingProfileRef: "native-skill-instant",
          skillBundleMetadata: expect.objectContaining({
            Lime_knowledge_pack_type: "brand-product",
            Lime_agent_knowledge_runtime_mode: "data",
            Lime_skill_bundle_path:
              "lime-rs/resources/default-skills/brand-product-knowledge-builder",
          }),
        }),
      ]),
    );
    expect(pkg.automationProfiles).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "daily-trend-briefing-automation",
        }),
        expect.objectContaining({
          id: "account-performance-tracking-automation",
        }),
      ]),
    );
    expect(pkg.bindingProfiles.map((profile) => profile.id)).toEqual(
      expect.arrayContaining(["agent-turn-instant", "automation-job-managed"]),
    );
    expect(pkg.compatibility.requiredKernelCapabilities).not.toContain(
      "cloud_scene",
    );
  });

  it("应把 seeded 基础设置包编译成默认云端目录", () => {
    const catalog = createSeededCloudServiceSkillCatalog();

    expect(catalog.tenantId).toBe(SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID);
    expect(catalog.version).toBe(SEEDED_SERVICE_SKILL_CATALOG_VERSION);
    expect(catalog.items).toHaveLength(17);
    expect(catalog.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "carousel-post-replication",
          title: "复制轮播帖",
          skillBundle: expect.objectContaining({
            name: "carousel-post-replication",
            metadata: expect.objectContaining({
              Lime_base_setup_package_id: "lime-seeded-service-skills",
              Lime_prompt_template_key: "replication",
            }),
          }),
        }),
        expect.objectContaining({
          id: "account-performance-tracking",
          runnerType: "managed",
          defaultExecutorBinding: "automation_job",
          skillBundle: expect.objectContaining({
            metadata: expect.objectContaining({
              Lime_automation_profile_ref:
                "account-performance-tracking-automation",
            }),
          }),
        }),
        expect.objectContaining({
          id: "cloud-video-dubbing",
          title: "视频配音",
          defaultExecutorBinding: "agent_turn",
          executionLocation: "client_default",
        }),
        expect.objectContaining({
          id: "personal-ip-knowledge-builder",
          source: "cloud_catalog",
          defaultExecutorBinding: "native_skill",
          skillBundle: expect.objectContaining({
            name: "personal-ip-knowledge-builder",
            resourceSummary: {
              hasScripts: true,
              hasReferences: true,
              hasAssets: true,
            },
            metadata: expect.objectContaining({
              Lime_knowledge_builder: "true",
              Lime_knowledge_pack_type: "personal-profile",
              Lime_knowledge_template: "personal-ip",
              Lime_knowledge_family: "persona",
            }),
          }),
        }),
        expect.objectContaining({
          id: "brand-persona-knowledge-builder",
          source: "cloud_catalog",
          defaultExecutorBinding: "native_skill",
          skillBundle: expect.objectContaining({
            name: "brand-persona-knowledge-builder",
            resourceSummary: {
              hasScripts: false,
              hasReferences: true,
              hasAssets: false,
            },
            metadata: expect.objectContaining({
              Lime_knowledge_builder: "true",
              Lime_knowledge_pack_type: "brand-persona",
              Lime_knowledge_template: "brand-persona",
              Lime_knowledge_family: "persona",
            }),
          }),
        }),
        expect.objectContaining({
          id: "content-operations-knowledge-builder",
          source: "cloud_catalog",
          defaultExecutorBinding: "native_skill",
          skillBundle: expect.objectContaining({
            name: "content-operations-knowledge-builder",
            resourceSummary: {
              hasScripts: false,
              hasReferences: true,
              hasAssets: false,
            },
            metadata: expect.objectContaining({
              Lime_knowledge_builder: "true",
              Lime_knowledge_pack_type: "content-operations",
              Lime_knowledge_family: "data",
            }),
          }),
        }),
        expect.objectContaining({
          id: "private-domain-operations-knowledge-builder",
          defaultExecutorBinding: "native_skill",
        }),
        expect.objectContaining({
          id: "live-commerce-operations-knowledge-builder",
          defaultExecutorBinding: "native_skill",
        }),
        expect.objectContaining({
          id: "campaign-operations-knowledge-builder",
          defaultExecutorBinding: "native_skill",
        }),
        expect.objectContaining({
          id: "brand-product-knowledge-builder",
          defaultExecutorBinding: "native_skill",
          skillBundle: expect.objectContaining({
            name: "brand-product-knowledge-builder",
            metadata: expect.objectContaining({
              Lime_knowledge_pack_type: "brand-product",
              Lime_knowledge_family: "data",
            }),
          }),
        }),
        expect.objectContaining({
          id: "organization-knowhow-knowledge-builder",
          defaultExecutorBinding: "native_skill",
        }),
        expect.objectContaining({
          id: "growth-strategy-knowledge-builder",
          defaultExecutorBinding: "native_skill",
        }),
      ]),
    );
  });

});
