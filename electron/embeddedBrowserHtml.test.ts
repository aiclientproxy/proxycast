import { describe, expect, it } from "vitest";
import {
  buildEmbeddedBrowserHtmlDataUrl,
  EMBEDDED_BROWSER_HTML_MAX_BYTES,
  readEmbeddedBrowserHtmlPayload,
} from "./embeddedBrowserHtml";

describe("embeddedBrowserHtml", () => {
  it("只接受受控 MCP App HTML 载荷", () => {
    expect(
      readEmbeddedBrowserHtmlPayload({
        html: "<!doctype html><title>Plugin</title>",
        source: "mcpApp",
        sourceUri: "ui://plugin/report.html",
      }),
    ).toEqual({
      csp: {
        baseUriDomains: [],
        connectDomains: [],
        frameDomains: [],
        resourceDomains: [],
      },
      html: "<!doctype html><title>Plugin</title>",
      source: "mcpApp",
      sourceUri: "ui://plugin/report.html",
    });
  });

  it("拒绝非 MCP App 来源、非 ui URI 与超限 HTML", () => {
    expect(() =>
      readEmbeddedBrowserHtmlPayload({
        html: "<p>bad source</p>",
        source: "browser",
        sourceUri: "ui://plugin/report.html",
      }),
    ).toThrow("source 必须是 mcpApp");
    expect(() =>
      readEmbeddedBrowserHtmlPayload({
        html: "<p>bad uri</p>",
        source: "mcpApp",
        sourceUri: "https://example.com/report.html",
      }),
    ).toThrow("sourceUri 必须使用 ui:// 协议");
    expect(() =>
      readEmbeddedBrowserHtmlPayload({
        html: "a".repeat(EMBEDDED_BROWSER_HTML_MAX_BYTES + 1),
        source: "mcpApp",
        sourceUri: "ui://plugin/report.html",
      }),
    ).toThrow("HTML 超过");
  });

  it("将 HTML 与受控 CSP 编码为不暴露原文的数据 URL", () => {
    const dataUrl = buildEmbeddedBrowserHtmlDataUrl({
      csp: {
        baseUriDomains: [],
        connectDomains: ["https://api.example.com"],
        frameDomains: [],
        resourceDomains: ["https://cdn.example.com"],
      },
      html: "<html><head></head><body><h1>报告</h1></body></html>",
    });
    expect(dataUrl).toMatch(/^data:text\/html;charset=utf-8;base64,/);
    expect(dataUrl).not.toContain("报告");
    const encoded = dataUrl.slice(dataUrl.indexOf(",") + 1);
    const decoded = Buffer.from(encoded, "base64").toString("utf8");
    expect(decoded).toContain('http-equiv="Content-Security-Policy"');
    expect(decoded).toContain("connect-src https://api.example.com");
    expect(decoded).toContain("script-src 'unsafe-inline' https://cdn.example.com");
    expect(decoded).toContain("frame-src 'none'");
  });

  it("拒绝 CSP 指令注入、路径与不受支持协议", () => {
    for (const connectDomain of [
      "https://api.example.com; script-src *",
      "https://api.example.com/path",
      "file:///tmp/app",
    ]) {
      expect(() =>
        readEmbeddedBrowserHtmlPayload({
          csp: { connectDomains: [connectDomain] },
          html: "<main>app</main>",
          source: "mcpApp",
          sourceUri: "ui://plugin/report.html",
        }),
      ).toThrow("包含无效 origin");
    }
  });
});
