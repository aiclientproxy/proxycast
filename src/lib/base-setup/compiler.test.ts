import { describe, expect, it } from "vitest";
import type { BaseSetupPackage } from "./types";
import { compileBaseSetupPackage } from "./compiler";

function createBaseSetupPackage(): BaseSetupPackage {
  return {
    id: "service-skill-starter-pack",
    version: "0.2.0",
    title: "Service Skill Starter Pack",
    summary: "给现有工作区提供 compat ServiceSkillCatalog projection。",
    bundleRefs: [
      {
        id: "article-export-bundle",
        source: "builtin",
        pathOrUri: "bundles/article-export",
        kind: "skill_bundle",
      },
    ],
    catalogProjections: [
      {
        id: "article-export-skill",
        targetCatalog: "service_skill_catalog",
        entryKey: "x-article-export",
        skillKey: "x-article-export",
        title: "文章转存",
        summary: "把页面转成项目资料包。",
        category: "采集",
        outputHint: "Markdown bundle",
        entryHint: "输入 URL 开始转存",
        bundleRefId: "article-export-bundle",
        slotProfileRef: "article-export-input",
        bindingProfileRef: "browser-export",
        artifactProfileRef: "article-export-bundle-profile",
        scorecardProfileRef: "article-export-scorecard",
        policyProfileRef: "default-exposure",
        aliases: ["x转存"],
        triggerHints: ["转存页面", "文章导出"],
      },
    ],
    slotProfiles: [
      {
        id: "article-export-input",
        slots: [
          {
            key: "url",
            label: "页面链接",
            type: "url",
            required: true,
            placeholder: "https://example.com",
          },
        ],
      },
    ],
    bindingProfiles: [
      {
        id: "browser-export",
        bindingFamily: "agent_turn",
        runnerType: "instant",
      },
    ],
    artifactProfiles: [
      {
        id: "article-export-bundle-profile",
        deliveryContract: "artifact_bundle",
        requiredParts: ["index.md", "meta.json"],
        viewerKind: "artifact_bundle",
        defaultArtifactKind: "report",
        outputDestination: "project_resource",
      },
    ],
    scorecardProfiles: [
      {
        id: "article-export-scorecard",
        metrics: ["success_rate"],
      },
    ],
    policyProfiles: [
      {
        id: "default-exposure",
        enabled: true,
        surfaceScopes: ["home", "workspace"],
      },
    ],
    compatibility: {
      minAppVersion: "1.11.0",
      requiredKernelCapabilities: ["agent_turn", "artifact_viewer"],
      seededFallback: true,
      compatCatalogProjection: true,
    },
  };
}

describe("compileBaseSetupPackage", () => {
  it("把基础设置包编译成当前 ServiceSkillCatalog projection", () => {
    const compiled = compileBaseSetupPackage(createBaseSetupPackage(), {
      tenantId: "tenant-a",
      syncedAt: "2026-04-15T00:00:00.000Z",
    });

    expect(compiled.packageId).toBe("service-skill-starter-pack");
    expect(compiled.serviceSkillCatalogProjection.tenantId).toBe("tenant-a");
    expect(compiled.serviceSkillCatalogProjection.items).toHaveLength(1);
    expect(compiled.serviceSkillCatalogProjection.items[0]).toEqual(
      expect.objectContaining({
        id: "article-export-skill",
        skillKey: "x-article-export",
        title: "文章转存",
        defaultExecutorBinding: "agent_turn",
        source: "cloud_catalog",
        outputDestination: "project_resource",
      }),
    );
    expect(
      compiled.projectionIndex.artifactProfileRefsByProjectionId[
        "article-export-skill"
      ],
    ).toBe("article-export-bundle-profile");
  });

  it("在包校验失败时抛出错误", () => {
    const pkg = createBaseSetupPackage();
    pkg.catalogProjections[0].bundleRefId = "missing-bundle";

    expect(() => compileBaseSetupPackage(pkg)).toThrow(
      /Base Setup Package 校验失败/,
    );
  });
});
