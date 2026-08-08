import { act, useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppServerPluginCatalogSummary } from "@/lib/api/appServerTypes";
import { PLUGIN_CATALOG_CHANGED_EVENT } from "@/lib/api/pluginCatalog";
import {
  cleanupMountedRoots,
  flushEffects,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "@/components/workspace/hooks/testUtils";
import {
  useWorkspacePluginCatalogSuggestions,
  type UseWorkspacePluginCatalogSuggestionsOptions,
  type UseWorkspacePluginCatalogSuggestionsResult,
} from "./useWorkspacePluginCatalogSuggestions";

const mountedRoots: MountedRoot[] = [];

function plugin(enabled = true): AppServerPluginCatalogSummary {
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
    enabled,
    installPolicy: "AVAILABLE",
    authPolicy: "ON_USE",
    availability: "installed",
    skillsCount: 1,
    mcpServersCount: 0,
    appsCount: 0,
    hooksCount: 0,
  };
}

function Harness({
  options,
  onReady,
}: {
  options: UseWorkspacePluginCatalogSuggestionsOptions;
  onReady: (value: UseWorkspacePluginCatalogSuggestionsResult) => void;
}) {
  const value = useWorkspacePluginCatalogSuggestions(options);
  useEffect(() => onReady(value), [onReady, value]);
  return null;
}

describe("useWorkspacePluginCatalogSuggestions", () => {
  let latest: UseWorkspacePluginCatalogSuggestionsResult | null = null;

  beforeEach(() => {
    setupReactActEnvironment();
    latest = null;
  });

  afterEach(() => {
    cleanupMountedRoots(mountedRoots);
    vi.clearAllMocks();
  });

  it("按需读取 v2 installed catalog，并在 v2 变更事件后刷新", async () => {
    const listInstalled = vi
      .fn()
      .mockResolvedValueOnce({ plugins: [plugin()], generatedAt: "first" })
      .mockResolvedValueOnce({
        plugins: [plugin(false)],
        generatedAt: "second",
      });

    mountHarness(
      Harness,
      {
        options: { enabled: true, listInstalled },
        onReady: (value) => {
          latest = value;
        },
      },
      mountedRoots,
    );
    await flushEffects(6);

    expect(latest?.suggestions[0]).toMatchObject({
      pluginId: "browser@openai-bundled",
      disabled: false,
    });

    act(() => {
      window.dispatchEvent(new CustomEvent(PLUGIN_CATALOG_CHANGED_EVENT));
    });
    await flushEffects(6);

    expect(listInstalled).toHaveBeenCalledTimes(2);
    expect(latest?.suggestions[0]).toMatchObject({
      pluginId: "browser@openai-bundled",
      disabled: true,
      blockerCodes: ["PLUGIN_DISABLED"],
    });
  });
});
