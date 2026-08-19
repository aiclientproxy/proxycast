import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const SITE_ADAPTER_SYMBOLS = [
  "siteApplyAdapterCatalogBootstrap",
  "siteClearAdapterCatalogCache",
  "siteDebugRunAdapter",
  "siteGetAdapterCatalogStatus",
  "siteGetAdapterInfo",
  "siteGetAdapterLaunchReadiness",
  "siteImportAdapterYamlBundle",
  "siteListAdapters",
  "siteRecommendAdapters",
  "siteRunAdapter",
  "siteSaveAdapterResult",
  "siteSearchAdapters",
];

function readRepoFile(path: string): string {
  return readFileSync(resolve(cwd(), path), "utf8");
}

describe("agentRuntime siteClient current boundary", () => {
  it("agentRuntime 公共聚合入口已删除且不得恢复", () => {
    for (const path of [
      "src/lib/api/agentRuntime.ts",
      "src/lib/api/agentRuntime.d.ts",
      "src/lib/api/agentRuntime/index.ts",
      "src/lib/api/agentRuntime/index.d.ts",
    ]) {
      expect(existsSync(resolve(cwd(), path)), path).toBe(false);
    }
  });

  it("retired siteClient 与 webview-api 已物理删除且不得恢复", () => {
    for (const path of [
      "src/lib/api/agentRuntime/siteClient.ts",
      "src/lib/api/agentRuntime/siteClient.d.ts",
      "src/lib/webview-api.ts",
      "src/lib/webview-api.d.ts",
    ]) {
      expect(existsSync(resolve(cwd(), path)), path).toBe(false);
    }
  });

  it("createAgentRuntimeClient 不再混入 retired Site Adapter 方法", () => {
    const source = readRepoFile("src/lib/api/agentRuntime/clientFactory.ts");
    const declarations = readRepoFile(
      "src/lib/api/agentRuntime/clientFactory.d.ts",
    );

    expect(source).not.toContain("createSiteClient");
    expect(source).not.toContain("./siteClient");
    for (const symbol of SITE_ADAPTER_SYMBOLS) {
      expect(source).not.toContain(symbol);
      expect(declarations).not.toContain(symbol);
    }
  });
});
