import { describe, expect, it } from "vitest";
import type { AppServerPluginCatalogSummary } from "@/lib/api/appServerTypes";
import {
  detailCapabilityCount,
  filterPluginCatalog,
  listPluginCatalogSources,
  mergePluginCatalogSummary,
} from "./PluginCatalogPageViewModel";

function summary(
  id: string,
  overrides: Partial<AppServerPluginCatalogSummary> = {},
): AppServerPluginCatalogSummary {
  return {
    authPolicy: "ON_USE",
    availability: "installed",
    description: `${id} description`,
    enabled: true,
    hooksCount: 1,
    id,
    marketplaceId: "personal",
    contentDigest: "sha256:test",
    installPolicy: "AVAILABLE",
    installed: true,
    localVersion: "1.0.0",
    mcpServersCount: 1,
    name: id,
    skillsCount: 1,
    source: "local",
    sourceUri: `/tmp/${id}`,
    version: "1.0.0",
    ...overrides,
  };
}

describe("PluginCatalogPageViewModel", () => {
  it("按视图、来源和查询过滤同一 catalog projection", () => {
    const plugins = [
      summary("writer", { name: "Writing Tools" }),
      summary("browser", {
        installed: false,
        source: "bundled",
        name: "Browser Tools",
      }),
    ];

    expect(
      filterPluginCatalog(plugins, {
        query: "writing",
        source: "local",
        view: "installed",
      }).map((plugin) => plugin.id),
    ).toEqual(["writer"]);
    expect(listPluginCatalogSources(plugins)).toEqual(["bundled", "local"]);
  });

  it("以 plugin identity 合并状态并统计详情能力", () => {
    const updated = summary("writer", { enabled: false });
    expect(mergePluginCatalogSummary([summary("writer")], updated)).toEqual([
      updated,
    ]);
    expect(
      detailCapabilityCount({
        summary: updated,
        skills: [
          { id: "skill", name: "Skill", description: "", requiresAuth: false },
        ],
        mcpServers: [],
        apps: [
          { id: "app", name: "App", description: "", requiresAuth: false },
        ],
        hooks: [{ id: "turn", event: "turn" }],
        uiResources: [],
      }),
    ).toBe(3);
  });
});
