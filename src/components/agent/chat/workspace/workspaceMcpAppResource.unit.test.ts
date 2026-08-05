import { describe, expect, it } from "vitest";
import {
  buildWorkspaceMcpAppHtmlParams,
  WorkspaceMcpAppResourceError,
} from "./workspaceMcpAppResource";

describe("workspaceMcpAppResource", () => {
  it("将标准 MCP App HTML 与 CSP 投影为受控宿主参数", () => {
    expect(
      buildWorkspaceMcpAppHtmlParams({
        content: {
          uri: "ui://plugin/report.html",
          mime_type: "text/html;profile=mcp-app",
          text: "<!doctype html><main>report</main>",
          meta: {
            ui: {
              csp: {
                connectDomains: ["https://api.example.com"],
                resourceDomains: ["https://cdn.example.com"],
              },
            },
          },
        },
        expectedUri: "ui://plugin/report.html",
        viewId: "plugin-item-1",
      }),
    ).toEqual({
      viewId: "plugin-item-1",
      csp: {
        baseUriDomains: undefined,
        connectDomains: ["https://api.example.com"],
        frameDomains: undefined,
        resourceDomains: ["https://cdn.example.com"],
      },
      html: "<!doctype html><main>report</main>",
      source: "mcpApp",
      sourceUri: "ui://plugin/report.html",
    });
  });

  it.each([
    [
      "uriMismatch",
      { uri: "ui://other/report.html", mime_type: "text/html;profile=mcp-app", text: "<p>x</p>" },
    ],
    [
      "invalidMimeType",
      { uri: "ui://plugin/report.html", mime_type: "text/html", text: "<p>x</p>" },
    ],
    [
      "missingHtml",
      { uri: "ui://plugin/report.html", mime_type: "text/html;profile=mcp-app" },
    ],
  ] as const)("拒绝不符合 MCP Apps 合同的资源: %s", (code, content) => {
    expect(() =>
      buildWorkspaceMcpAppHtmlParams({
        content,
        expectedUri: "ui://plugin/report.html",
        viewId: "plugin-item-1",
      }),
    ).toThrow(expect.objectContaining<Partial<WorkspaceMcpAppResourceError>>({ code }));
  });
});
