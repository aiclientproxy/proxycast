import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

describe("agentRuntime siteClient retired boundary", () => {
  it("Site Adapter 客户端与旧聚合网关不得恢复", () => {
    for (const path of [
      "src/lib/api/agentRuntime/siteClient.ts",
      "src/lib/api/agentRuntime/siteClient.d.ts",
      "src/lib/webview-api.ts",
      "src/lib/webview-api.d.ts",
    ]) {
      expect(existsSync(resolve(cwd(), path)), path).toBe(false);
    }
  });

  it("current client factory 不得重新暴露 Site Adapter 方法", () => {
    const source = readFileSync(
      resolve(cwd(), "src/lib/api/agentRuntime/clientFactory.ts"),
      "utf8",
    );

    expect(source).not.toContain("createSiteClient");
    expect(source).not.toContain("siteListAdapters");
    expect(source).not.toContain("siteRunAdapter");
  });
});
