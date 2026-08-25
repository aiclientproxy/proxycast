import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "@/i18n/config";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { UseMcpReturn } from "@/hooks/useMcp";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import type {
  McpPromptDefinition,
  McpResourceDefinition,
  McpServerInfo,
  McpToolDefinition,
} from "@/lib/api/mcp";
import { McpPanel } from "./McpPanel";
import { MCP_RESOURCE_TEXT_PREVIEW_CHAR_LIMIT } from "./mcpResourcePreview";

const useMcpMock = vi.hoisted(() => vi.fn<() => UseMcpReturn>());
const toastMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/hooks/useMcp", () => ({
  useMcp: useMcpMock,
}));

vi.mock("sonner", () => ({
  toast: toastMock,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: vi.fn(),
}));

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: RenderResult[] = [];

function createServer(overrides: Partial<McpServerInfo> = {}): McpServerInfo {
  return {
    id: "server-demo",
    name: "demo",
    description: "Demo MCP server",
    config: { command: "npx", args: ["demo"] },
    is_running: true,
    server_info: {
      name: "demo",
      version: "1.0.0",
      supports_tools: true,
      supports_prompts: true,
      supports_resources: true,
    },
    enabled_lime: true,
    enabled_claude: true,
    enabled_codex: true,
    enabled_gemini: false,
    ...overrides,
  };
}

function createTool(
  overrides: Partial<McpToolDefinition> = {},
): McpToolDefinition {
  return {
    name: "mcp__demo__search_docs",
    description: "搜索文档",
    input_schema: { type: "object" },
    server_name: "demo",
    ...overrides,
  };
}

function createMcpState(overrides: Partial<UseMcpReturn> = {}): UseMcpReturn {
  return {
    servers: [createServer()],
    tools: [createTool()],
    prompts: [
      {
        name: "write_summary",
        description: "生成摘要",
        arguments: [],
        server_name: "demo",
      } satisfies McpPromptDefinition,
    ],
    resources: [
      {
        uri: "file://demo/readme.md",
        name: "README",
        description: "项目说明",
        server_name: "demo",
      } satisfies McpResourceDefinition,
    ],
    loading: false,
    error: null,
    serverConnectionStates: {},
    oauthCompletion: null,
    eventStreams: {},
    startServer: vi.fn(async () => undefined),
    stopServer: vi.fn(async () => undefined),
    reconnectServer: vi.fn(async () => undefined),
    loginOAuthServer: vi.fn(async () => ({
      authorizationUrl: "https://auth.example/authorize",
      state: "state-1",
    })),
    refreshServers: vi.fn(async () => undefined),
    refreshTools: vi.fn(async () => undefined),
    callTool: vi.fn(async () => ({ content: [], is_error: false })),
    refreshPrompts: vi.fn(async () => undefined),
    getPrompt: vi.fn(async () => ({ messages: [] })),
    refreshResources: vi.fn(async () => undefined),
    readResource: vi.fn(async () => ({ uri: "file://demo/readme.md" })),
    subscribeResource: vi.fn(async () => undefined),
    unsubscribeResource: vi.fn(async () => undefined),
    ...overrides,
  };
}

async function renderPanel(
  props: Partial<React.ComponentProps<typeof McpPanel>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  await act(async () => {
    root.render(<McpPanel {...props} />);
    await Promise.resolve();
    await Promise.resolve();
  });

  mountedRoots.push({ container, root });
  return container;
}

function findButton(container: HTMLElement, text: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find(
    (candidate) => candidate.textContent?.includes(text),
  );

  if (!button) {
    throw new Error(`未找到按钮：${text}`);
  }

  return button as HTMLButtonElement;
}

async function openResourcesTab(container: HTMLElement): Promise<void> {
  await act(async () => {
    findButton(container, "资源").dispatchEvent(
      new MouseEvent("click", { bubbles: true }),
    );
    await Promise.resolve();
  });
  await act(async () => {
    findButton(container, "demo").dispatchEvent(
      new MouseEvent("click", { bubbles: true }),
    );
    await Promise.resolve();
  });
}

function findResourcePreviewButtons(
  container: HTMLElement,
): HTMLButtonElement[] {
  return Array.from(
    container.querySelectorAll<HTMLButtonElement>('button[title="读取资源"]'),
  );
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  return changeLimeLocale("zh-CN").then(() => {
    vi.mocked(openExternalUrlWithSystemBrowser).mockResolvedValue(undefined);
    useMcpMock.mockReturnValue(createMcpState());
  });
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe("McpPanel", () => {
  it("设置页内嵌时仍渲染统一页头和摘要指标", async () => {
    const container = await renderPanel({ hideHeader: true });

    expect(container.textContent).toContain("MCP 服务器");
    expect(container.textContent).toContain("Model Context Protocol");
    expect(container.textContent).toContain("1 个运行中");
    expect(container.textContent).toContain("工具 / 提示词 / 资源");
    expect(container.textContent).toContain("已同步");
  });

  it("运行状态空态应引导到配置管理，而不是只显示旧式空白列表", async () => {
    useMcpMock.mockReturnValue(
      createMcpState({
        servers: [],
        tools: [],
        prompts: [],
        resources: [],
      }),
    );

    const container = await renderPanel({ hideHeader: true });

    expect(container.textContent).toContain("还没有 MCP 服务器");
    expect(container.textContent).toContain(
      "去“配置管理”添加或导入服务器后，这里会显示运行状态。",
    );
  });

  it("工具页继续复用统一容器，并能引导回运行状态", async () => {
    useMcpMock.mockReturnValue(
      createMcpState({
        servers: [createServer({ is_running: false, server_info: undefined })],
        tools: [],
        prompts: [],
        resources: [],
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    await act(async () => {
      findButton(container, "工具").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });

    expect(container.textContent).toContain(
      "已配置服务器，但当前没有运行中的 MCP 服务器",
    );

    await act(async () => {
      findButton(container, "去启动服务器").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });

    expect(container.textContent).toContain("服务器状态");
  });

  it("OAuth 需要登录时应打开授权页并传递 scopes", async () => {
    const loginOAuthServer = vi.fn(async () => ({
      authorizationUrl: "https://auth.example/authorize",
      state: "state-1",
    }));
    const windowOpen = vi.spyOn(window, "open").mockReturnValue(null);
    useMcpMock.mockReturnValue(
      createMcpState({
        servers: [
          createServer({
            id: "server-remote-docs",
            name: "remote-docs",
            config: {
              type: "streamable_http",
              url: "https://example.com/mcp",
              scopes: ["search.read"],
            },
            is_running: false,
            server_info: undefined,
            runtime_status: {
              name: "remote-docs",
              transport: "streamable_http",
              enabled: true,
              is_running: false,
              required: false,
              supports_parallel_tool_calls: false,
              startup_timeout: 30,
              tool_timeout: 30,
              disabled_tools: [],
              auth_status: {
                mode: "oauth",
                available: true,
                reason_code: "oauth_login_required",
                action_plan: {
                  kind: "oauth_login",
                  state: "login_required",
                  required_runtime: "mcp_server_oauth_login",
                  scopes: ["search.read"],
                },
              },
            },
          }),
        ],
        loginOAuthServer,
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    expect(container.textContent).toContain("需要授权");

    await act(async () => {
      findButton(container, "登录").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(loginOAuthServer).toHaveBeenCalledWith("remote-docs", {
      scopes: ["search.read"],
    });
    expect(openExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://auth.example/authorize",
    );
    expect(windowOpen).not.toHaveBeenCalled();
    expect(toastMock.success).toHaveBeenCalledWith(
      "已打开 remote-docs 授权页，请在浏览器完成授权。",
    );
  });

  it("OAuth 完成事件回流后应提示状态已刷新", async () => {
    useMcpMock.mockReturnValue(
      createMcpState({
        oauthCompletion: {
          serverName: "remote-docs",
          completedAt: 1,
        },
      }),
    );

    await renderPanel({ hideHeader: true });

    expect(toastMock.success).toHaveBeenCalledWith(
      "remote-docs 授权已完成，状态已刷新。",
    );
  });

  it("MCP event stream lifecycle 应在运行状态中保持稳定可见", async () => {
    useMcpMock.mockReturnValue(
      createMcpState({
        eventStreams: {
          "subscription-1": {
            subscriptionId: "subscription-1",
            phase: "active",
            lastEventMethod: "notifications/events/event",
            lastEventName: "issue.updated",
            eventCount: 2,
            activeCount: 2,
            reconnectCount: 1,
            updatedAt: 1,
          },
        },
      }),
    );

    const container = await renderPanel({ hideHeader: true });

    expect(
      container.querySelector('[data-testid="mcp-event-stream-status"]'),
    ).not.toBeNull();
    expect(
      container
        .querySelector(
          '[data-mcp-event-stream-subscription-id="subscription-1"]',
        )
        ?.getAttribute("data-mcp-event-stream-phase"),
    ).toBe("active");
    expect(container.textContent).toContain("已恢复连接");
    expect(container.textContent).toContain("issue.updated");
  });

  it("显式 OAuth 配置未接入运行时登录时应只显示不可用状态", async () => {
    useMcpMock.mockReturnValue(
      createMcpState({
        servers: [
          createServer({
            id: "server-explicit-oauth",
            name: "explicit-oauth",
            config: {
              type: "streamable_http",
              url: "https://example.com/mcp",
              oauth: { client_id: "lime-client" },
              oauth_resource: "https://example.com",
            },
            is_running: false,
            server_info: undefined,
            runtime_status: {
              name: "explicit-oauth",
              transport: "streamable_http",
              enabled: true,
              is_running: false,
              required: false,
              supports_parallel_tool_calls: false,
              startup_timeout: 30,
              tool_timeout: 30,
              disabled_tools: [],
              auth_status: {
                mode: "oauth",
                available: false,
                reason_code: "oauth_runtime_not_implemented",
                action_plan: {
                  kind: "oauth_login",
                  state: "runtime_not_connected",
                  required_runtime: "mcp_server_oauth_login",
                  oauth_resource: "https://example.com",
                  client_id: "lime-client",
                },
              },
            },
          }),
        ],
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    expect(container.textContent).toContain("OAuth 配置暂不支持登录");
    expect(container.textContent).not.toContain("需要授权");
    expect(
      Array.from(container.querySelectorAll("button")).some(
        (button) => button.textContent?.trim() === "登录",
      ),
    ).toBe(false);
  });

  it("OAuth 已授权时不应显示登录入口", async () => {
    useMcpMock.mockReturnValue(
      createMcpState({
        servers: [
          createServer({
            id: "server-authorized",
            name: "authorized",
            config: {
              type: "streamable_http",
              url: "https://example.com/mcp",
            },
            runtime_status: {
              name: "authorized",
              transport: "streamable_http",
              enabled: true,
              is_running: true,
              required: false,
              supports_parallel_tool_calls: false,
              startup_timeout: 30,
              tool_timeout: 30,
              disabled_tools: [],
              auth_status: {
                mode: "oauth",
                available: true,
              },
            },
          }),
        ],
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    expect(container.textContent).toContain("已授权");
    expect(container.textContent).not.toContain("需要授权");
  });

  it("资源预览打开、切换和关闭时应维护资源订阅生命周期", async () => {
    const subscribeResource = vi.fn(async () => undefined);
    const unsubscribeResource = vi.fn(async () => undefined);
    const readResource = vi.fn(async (_server: string, uri: string) => ({
      uri,
      text: uri.includes("readme") ? "README content" : "Guide content",
      mime_type: "text/markdown",
    }));
    useMcpMock.mockReturnValue(
      createMcpState({
        resources: [
          {
            uri: "file://demo/readme.md",
            name: "README",
            description: "项目说明",
            server_name: "demo",
          },
          {
            uri: "file://demo/guide.md",
            name: "GUIDE",
            description: "使用指南",
            server_name: "demo",
          },
        ],
        readResource,
        subscribeResource,
        unsubscribeResource,
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    await openResourcesTab(container);

    await act(async () => {
      findResourcePreviewButtons(container)[0]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(subscribeResource).toHaveBeenCalledWith(
      "demo",
      "file://demo/readme.md",
    );
    expect(readResource).toHaveBeenCalledWith("demo", "file://demo/readme.md");
    expect(container.textContent).toContain("README content");

    await act(async () => {
      findResourcePreviewButtons(container)[1]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(unsubscribeResource).toHaveBeenCalledWith(
      "demo",
      "file://demo/readme.md",
    );
    expect(subscribeResource).toHaveBeenCalledWith(
      "demo",
      "file://demo/guide.md",
    );
    expect(readResource).toHaveBeenCalledWith("demo", "file://demo/guide.md");
    expect(container.textContent).toContain("Guide content");

    await act(async () => {
      findResourcePreviewButtons(container)[1]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });

    expect(unsubscribeResource).toHaveBeenCalledWith(
      "demo",
      "file://demo/guide.md",
    );
  });

  it("同 URI 的跨服务器资源应按 server identity 切换预览", async () => {
    const subscribeResource = vi.fn(async () => undefined);
    const unsubscribeResource = vi.fn(async () => undefined);
    const readResource = vi.fn(async (server: string, uri: string) => ({
      uri,
      text: `${server} content`,
      mime_type: "text/plain",
    }));
    useMcpMock.mockReturnValue(
      createMcpState({
        servers: [
          createServer({ id: "server-a", name: "server-a" }),
          createServer({ id: "server-b", name: "server-b" }),
        ],
        resources: [
          { uri: "docs://shared", name: "Shared A", server_name: "server-a" },
          { uri: "docs://shared", name: "Shared B", server_name: "server-b" },
        ],
        readResource,
        subscribeResource,
        unsubscribeResource,
      }),
    );
    const container = await renderPanel({ hideHeader: true });
    await act(async () => {
      findButton(container, "资源").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });

    await act(async () => {
      findButton(container, "server-a").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });
    await act(async () => {
      findButton(container, "server-b").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });
    const previewButtons = findResourcePreviewButtons(container);

    await act(async () => {
      previewButtons[0]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });
    await act(async () => {
      previewButtons[1]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(unsubscribeResource).toHaveBeenCalledWith(
      "server-a",
      "docs://shared",
    );
    expect(subscribeResource).toHaveBeenLastCalledWith(
      "server-b",
      "docs://shared",
    );
    expect(readResource).toHaveBeenLastCalledWith("server-b", "docs://shared");
    expect(container.textContent).toContain("server-b content");
  });

  it("资源预览应截断超长文本，不渲染尾部内容", async () => {
    const hiddenTail = "TAIL_SHOULD_NOT_RENDER";
    const readResource = vi.fn(async () => ({
      uri: "file://demo/readme.md",
      text: "A".repeat(MCP_RESOURCE_TEXT_PREVIEW_CHAR_LIMIT) + hiddenTail,
      mime_type: "text/plain",
    }));
    useMcpMock.mockReturnValue(
      createMcpState({
        readResource,
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    await openResourcesTab(container);
    await act(async () => {
      findResourcePreviewButtons(container)[0]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    const preview = container.querySelector(
      '[data-testid="mcp-resource-text-preview"]',
    );
    expect(preview?.textContent).toHaveLength(
      MCP_RESOURCE_TEXT_PREVIEW_CHAR_LIMIT,
    );
    expect(container.textContent).toContain("已截断");
    expect(container.textContent).not.toContain(hiddenTail);
  });

  it("资源预览应只显示二进制摘要，不渲染 blob 正文", async () => {
    const readResource = vi.fn(async () => ({
      uri: "file://demo/readme.md",
      blob: "aGVsbG8=",
      mime_type: "application/octet-stream",
    }));
    useMcpMock.mockReturnValue(
      createMcpState({
        readResource,
      }),
    );
    const container = await renderPanel({ hideHeader: true });

    await openResourcesTab(container);
    await act(async () => {
      findResourcePreviewButtons(container)[0]?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="mcp-resource-blob-summary"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("二进制数据，5 字节");
    expect(container.textContent).not.toContain("aGVsbG8=");
  });
});
