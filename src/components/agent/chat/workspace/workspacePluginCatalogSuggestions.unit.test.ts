import { describe, expect, it } from "vitest";
import type { AppServerPluginCatalogSummary } from "@/lib/api/appServerTypes";
import { buildWorkspacePluginCatalogSuggestions } from "./workspacePluginCatalogSuggestions";

function plugin(
  overrides: Partial<AppServerPluginCatalogSummary> = {},
): AppServerPluginCatalogSummary {
  return {
    id: "browser",
    name: "Browser",
    version: "1.0.0",
    marketplaceId: "openai-bundled",
    contentDigest: "sha256:browser",
    description: "Control a browser",
    source: "bundled",
    sourceUri: "/plugins/browser",
    installed: true,
    enabled: true,
    installPolicy: "AVAILABLE",
    authPolicy: "ON_USE",
    availability: "installed",
    skillsCount: 1,
    mcpServersCount: 0,
    appsCount: 0,
    hooksCount: 0,
    ...overrides,
  };
}

describe("workspacePluginCatalogSuggestions", () => {
  it("只从 v2 installed summary 投影稳定 Plugin 候选", () => {
    expect(
      buildWorkspacePluginCatalogSuggestions([
        plugin(),
        plugin({ id: "available", name: "Available", installed: false }),
        plugin({ id: "disabled", name: "Disabled", enabled: false }),
      ]),
    ).toEqual([
      {
        pluginId: "browser",
        displayName: "Browser",
        description: "Control a browser",
        disabled: false,
        blockerCodes: [],
      },
      {
        pluginId: "disabled",
        displayName: "Disabled",
        description: "Control a browser",
        disabled: true,
        blockerCodes: ["PLUGIN_DISABLED"],
      },
    ]);
  });
});
