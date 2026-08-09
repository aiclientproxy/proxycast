import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/desktop-runtime", () => ({
  hasDesktopHostInvokeCapability: vi.fn(() => false),
  hasDesktopHostRuntimeMarkers: vi.fn(() => false),
}));

import {
  __resetDevBridgeHttpStateForTests,
  healthCheck,
  invokeViaHttp,
  listenViaHttpEvent,
  resolveBridgeRequestTimeoutMs,
} from "./http-client";

type FetchInput = Parameters<typeof fetch>[0];
type FetchOptions = Parameters<typeof fetch>[1];

function createAbortablePendingFetch() {
  return vi.fn((_input: FetchInput, init?: FetchOptions) => {
    return new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal;
      signal?.addEventListener("abort", () => {
        reject(new DOMException("The operation was aborted.", "AbortError"));
      });
    });
  });
}

function createHardConnectionFailure(message = "Failed to fetch") {
  return vi.fn<typeof fetch>().mockRejectedValue(new TypeError(message));
}

function electronHostHealthResponse(): Response {
  return new Response(
    JSON.stringify({
      status: "ok",
      transport: "electron-host",
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
}

describe("http-client", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    __resetDevBridgeHttpStateForTests();
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    vi.spyOn(console, "debug").mockImplementation(() => undefined);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    __resetDevBridgeHttpStateForTests();
  });

  it("桥健康探测连续硬连接失败后，后续检查会在短退避窗口内快速失败", async () => {
    const fetchMock = createHardConnectionFailure();
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await expect(invokeViaHttp("get_api_key_providers")).rejects.toThrow(
      "Failed to fetch",
    );
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("桥首次健康探测硬连接失败后会立即重试，避免瞬时失败污染设置页", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: ["model-a"] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(true);
    await expect(invokeViaHttp("get_model_registry")).resolves.toEqual([
      "model-a",
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("短退避期间会主动探测恢复，避免 Electron 热重启污染后续设置页", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: { ok: true } }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(false);

    await expect(invokeViaHttp("get_environment_preview")).resolves.toEqual({
      ok: true,
    });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[3]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("桥健康探测必须校验 Electron Host 身份，不能误连旧 Tauri DevBridge", async () => {
    const tauriHealthResponse = () =>
      new Response(
        JSON.stringify({
          status: "ok",
          service: "DevBridge",
          version: "1.0.0",
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      );
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(tauriHealthResponse())
      .mockResolvedValueOnce(tauriHealthResponse());
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(false);
    await expect(invokeViaHttp("get_api_key_providers")).rejects.toThrow(
      "recovery probe failed",
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("工作区恢复命令应允许绕过短退避重新探测，避免恢复时卡在 cooldown", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            result: {
              id: "workspace-1",
              name: "当前工作区",
            },
          }),
          {
            status: 200,
            headers: {
              "Content-Type": "application/json",
            },
          },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(false);

    await expect(
      invokeViaHttp("workspace_get", { id: "workspace-1" }),
    ).resolves.toMatchObject({
      id: "workspace-1",
      name: "当前工作区",
    });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[3]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("工作区列表命令应允许绕过短退避重新探测，恢复首页和侧栏", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: [{ id: "session-1" }] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(false);

    await expect(invokeViaHttp("workspace_list")).resolves.toEqual([
      { id: "session-1" },
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[3]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("默认项目命令应允许绕过短退避重新探测，避免空 mock 触发重复错误", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: { id: "workspace-default" } }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(healthCheck()).resolves.toBe(false);

    await expect(
      invokeViaHttp<{ id: string }>("get_or_create_default_project"),
    ).resolves.toEqual({ id: "workspace-default" });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[3]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("桥首次健康探测超时后，后续调用应重新探测而不是进入 cooldown", async () => {
    const firstHealthTimeout = createAbortablePendingFetch();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(firstHealthTimeout)
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: ["project-a"] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const firstCheck = healthCheck();
    await vi.advanceTimersByTimeAsync(3200);
    await expect(firstCheck).resolves.toBe(false);

    await expect(invokeViaHttp<string[]>("workspace_list")).resolves.toEqual([
      "project-a",
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("首次 invoke 的健康探测超时后，后续调用应重新探测而不是进入 cooldown", async () => {
    const firstHealthTimeout = createAbortablePendingFetch();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(firstHealthTimeout)
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: { id: "default-project" } }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const firstInvoke = invokeViaHttp("workspace_list").then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    await vi.advanceTimersByTimeAsync(3200);
    await expect(firstInvoke).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("bridge health check failed"),
      }),
    });

    await expect(
      invokeViaHttp<{ id: string }>("workspace_get_default"),
    ).resolves.toEqual({ id: "default-project" });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("桥健康时会复用短期健康缓存，避免每次调用都重复探测", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: ["project-a"] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: { id: "default-project" } }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(invokeViaHttp<string[]>("workspace_list")).resolves.toEqual([
      "project-a",
    ]);
    await expect(
      invokeViaHttp<{ id: string }>("workspace_get_default"),
    ).resolves.toEqual({ id: "default-project" });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://127.0.0.1:3030/invoke");
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("桥已健康后，健康探测短暂超时不应立刻进入 cooldown", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: ["project-a"] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      )
      .mockImplementationOnce(createAbortablePendingFetch())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: { id: "default-project" } }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(invokeViaHttp<string[]>("workspace_list")).resolves.toEqual([
      "project-a",
    ]);

    await vi.advanceTimersByTimeAsync(11000);

    const secondInvoke = invokeViaHttp<{ id: string }>("workspace_get_default");
    await vi.advanceTimersByTimeAsync(3200);
    await expect(secondInvoke).resolves.toEqual({ id: "default-project" });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[3]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("App Server turn/start JSON-RPC 请求应保留真实运行时超时窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("app_server_handle_json_lines", {
      request: {
        lines: [
          JSON.stringify({
            id: 1,
            method: "turn/start",
            params: {},
          }),
        ],
      },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(60000);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(90000);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 150000ms"),
      }),
    });
  });

  it("App Server 会话列表读取应使用 current read 超时，避免冷启动和迁移期误判失败", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    const listSessionsPromise = invokeViaHttp("app_server_handle_json_lines", {
      request: {
        lines: [
          JSON.stringify({
            id: "list-sessions",
            method: "thread/list",
            params: { workspaceId: "workspace-1", limit: 21 },
          }),
        ],
      },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    await vi.advanceTimersByTimeAsync(30000);

    await expect(listSessionsPromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 30000ms"),
      }),
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("工作区读取命令硬连接失败后应强制健康探测并重试一次", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            result: {
              id: "workspace-1",
              name: "当前工作区",
            },
          }),
          {
            status: 200,
            headers: {
              "Content-Type": "application/json",
            },
          },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    const invokePromise = invokeViaHttp("workspace_get", {
      id: "workspace-1",
    });

    await expect(invokePromise).resolves.toMatchObject({
      id: "workspace-1",
      name: "当前工作区",
    });
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[2]?.[0]).toBe("http://127.0.0.1:3030/health");
    expect(fetchMock.mock.calls[3]?.[0]).toBe("http://127.0.0.1:3030/invoke");
  });

  it("App Server thread settings 更新应使用 current read 超时", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    const invokePromise = invokeViaHttp("app_server_handle_json_lines", {
      request: {
        lines: [
          JSON.stringify({
            id: "update-thread-settings",
            method: "thread/settings/update",
            params: {
              threadId: "thread-1",
              model: "gpt-5.4",
              modelProvider: "openai",
            },
          }),
        ],
      },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );

    await vi.advanceTimersByTimeAsync(30000);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 30000ms"),
      }),
    });
  });

  it("旧 agent 标题生成命令不再使用 agent 长超时窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("agent_generate_title", {
      sessionId: "session-1",
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(1800);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 1800ms"),
      }),
    });
    expect(settled).toBe(true);
  });

  it("Plugin v3 App Server 请求使用 current 读取窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("app_server_handle_json_lines", {
      request: {
        lines: [
          JSON.stringify({
            id: "plugin-install",
            method: "plugin/install",
            params: { sourcePath: "/tmp/plugin" },
          }),
        ],
      },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(1800);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(28200);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 30000ms"),
      }),
    });
  });

  it("bridge 真相命令应使用 5000ms 的请求超时窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("open_external_url", {
      request: { url: "https://example.com" },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(2000);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(2800);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(400);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 5000ms"),
      }),
    });
  });

  it("启动关键真相命令应保留更长超时窗口，避免冷启动误判后端不可用", () => {
    expect(resolveBridgeRequestTimeoutMs("workspace_ensure_ready")).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("workspace_ensure_default_ready"),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "read-session",
              method: "thread/read",
              params: { sessionId: "session-1" },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "save-provider",
              method: "modelProvider/create",
              params: {},
            }),
          ],
        },
      }),
    ).toBe(30000);
    for (const method of [
      "modelProvider/testConnection",
      "modelProvider/testChat",
      "modelProvider/fetchModels",
    ]) {
      expect(
        resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
          request: {
            lines: [
              JSON.stringify({
                id: `provider-network-${method}`,
                method,
                params: { providerId: "provider-1" },
              }),
            ],
          },
        }),
      ).toBe(60000);
    }
    for (const method of [
      "plugin/list",
      "plugin/search",
      "plugin/read",
      "plugin/install",
      "plugin/uninstall",
      "plugin/installed",
      "plugin/enabled/set",
    ]) {
      expect(
        resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
          request: {
            lines: [
              JSON.stringify({ id: `plugin-${method}`, method, params: {} }),
            ],
          },
        }),
      ).toBe(30000);
    }
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "list-sessions",
              method: "thread/list",
              params: { limit: 20 },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "read-codex-import-job",
              method: "conversationImport/job/read",
              params: { jobId: "import-job-1" },
            }),
          ],
        },
      }),
    ).toBe(120000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "scan-codex-import",
              method: "conversationImport/source/scan",
              params: { sourceClient: "codex" },
            }),
          ],
        },
      }),
    ).toBe(120000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "preview-codex-import",
              method: "conversationImport/thread/preview",
              params: { sourceClient: "codex", sourceThreadId: "thread-1" },
            }),
          ],
        },
      }),
    ).toBe(120000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "commit-codex-import",
              method: "conversationImport/thread/commit",
              params: {
                sourceClient: "codex",
                sourceThreadId: "codex-thread-1",
              },
            }),
          ],
        },
      }),
    ).toBe(120000);
    expect(resolveBridgeRequestTimeoutMs("plugin_start_ui_runtime")).toBe(1800);
    expect(resolveBridgeRequestTimeoutMs("plugin_runtime_get_task")).toBe(1800);
    expect(resolveBridgeRequestTimeoutMs("plugin_runtime_start_task")).toBe(
      1800,
    );
    expect(resolveBridgeRequestTimeoutMs("plugin_inspect_local_package")).toBe(
      1800,
    );
    expect(resolveBridgeRequestTimeoutMs("plugin_select_directory")).toBe(1800);
    expect(resolveBridgeRequestTimeoutMs("plugin_get_ui_runtime_status")).toBe(
      1800,
    );
    expect(resolveBridgeRequestTimeoutMs("open_external_url")).toBe(5000);
    expect(resolveBridgeRequestTimeoutMs("execute_skill")).toBe(1800);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 1,
              method: "turn/start",
              params: {},
            }),
          ],
        },
      }),
    ).toBe(150000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "automation-run-now",
              method: "automationJob/runNow",
              params: { id: "job-1" },
            }),
          ],
        },
      }),
    ).toBe(150000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "ui-runtime-start",
              method: "pluginUiRuntime/start",
              params: {},
            }),
          ],
        },
      }),
    ).toBe(5000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 2,
              method: "workspace/default/ensure",
              params: {},
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "automation-history",
              method: "automationJob/runHistory",
              params: { id: "job-1" },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 3,
              method: "workspace/ensureReady",
              params: { workspaceId: "default" },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 4,
              method: "thread/read",
              params: {},
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "session-files",
              method: "sessionFile/list",
              params: { sessionId: "session-1" },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "project-git-status",
              method: "projectGit/status",
              params: { rootPath: "/workspace" },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "knowledge",
              method: "knowledgePack/list",
              params: {},
            }),
            JSON.stringify({
              id: "providers",
              method: "modelProvider/list",
              params: {},
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "mcp-start",
              method: "mcpServer/start",
              params: { name: "context7" },
            }),
          ],
        },
      }),
    ).toBe(30000);
    expect(
      resolveBridgeRequestTimeoutMs("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "mcp-server-tool-call",
              method: "mcpServer/tool/call",
              params: {
                threadId: "thread-1",
                server: "context7",
                tool: "query-docs",
                arguments: { libraryId: "/openai/openai-agents-python" },
              },
            }),
          ],
        },
      }),
    ).toBe(30000);
  });

  it("图层设计工程落盘命令应使用长请求窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("save_layered_design_project_export", {
      request: {
        projectRootPath: "/tmp/lime-layered-design",
        documentId: "design-1",
        files: [],
      },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(5000);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(55000);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 60000ms"),
      }),
    });
  });

  it("App Server Knowledge Pack 整理应保留 Builder Skill 长请求窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("app_server_handle_json_lines", {
      request: {
        lines: [
          JSON.stringify({
            id: "knowledge-compile",
            method: "knowledgePack/compile",
            params: {
              workingDir: "/tmp/lime-knowledge-smoke",
              name: "content-ops-acceptance",
            },
          }),
        ],
      },
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(60_000);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(120_000);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 180000ms"),
      }),
    });
  });

  it("语音模型下载命令应保留长下载窗口", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockImplementationOnce(createAbortablePendingFetch());
    vi.stubGlobal("fetch", fetchMock);

    let settled = false;
    const invokePromise = invokeViaHttp("voice_models_download", {
      modelId: "sensevoice-small-int8-2024-07-17",
    }).then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );
    invokePromise.finally(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(60_000);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(29 * 60 * 1000);
    await expect(invokePromise).resolves.toMatchObject({
      ok: false,
      error: expect.objectContaining({
        message: expect.stringContaining("timeout after 1800000ms"),
      }),
    });
  });

  it("桥失败短退避期间，事件监听不应继续创建 EventSource 连接", async () => {
    const fetchMock = createHardConnectionFailure();
    const eventSourceMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      eventSourceMock as unknown as typeof EventSource,
    );

    await expect(healthCheck()).resolves.toBe(false);

    await expect(listenViaHttpEvent("config-changed", vi.fn())).rejects.toThrow(
      "Failed to fetch",
    );

    expect(eventSourceMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("多个浏览器事件监听应复用一条 multiplex SSE 连接，避免占满 invoke 连接槽", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 0;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }

      emitOpen() {
        this.readyState = 1;
        this.onopen?.();
      }

      emitMessage(payload: unknown) {
        this.onmessage?.({
          data: JSON.stringify(payload),
        } as MessageEvent);
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(electronHostHealthResponse());
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const configHandler = vi.fn();
    const taskHandler = vi.fn();
    const configUnlistenPromise = listenViaHttpEvent(
      "config-changed",
      configHandler,
    );
    const taskUnlistenPromise = listenViaHttpEvent(
      "lime://creation_task_submitted",
      taskHandler,
    );

    await vi.advanceTimersByTimeAsync(0);
    expect(MockEventSource.instances).toHaveLength(1);

    const source = MockEventSource.instances[0]!;
    const sourceUrl = new URL(source.url);
    expect(JSON.parse(sourceUrl.searchParams.get("events") ?? "[]")).toEqual([
      "config-changed",
      "lime://creation_task_submitted",
    ]);

    source.emitOpen();
    const [configUnlisten, taskUnlisten] = await Promise.all([
      configUnlistenPromise,
      taskUnlistenPromise,
    ]);

    source.emitMessage({
      event: "lime://creation_task_submitted",
      payload: { taskId: "task-1" },
    });
    expect(taskHandler).toHaveBeenCalledWith({
      payload: { taskId: "task-1" },
    });
    expect(configHandler).not.toHaveBeenCalled();

    configUnlisten();
    taskUnlisten();
  });

  it("事件连接建立中新增监听时应重建 SSE 并包含新增事件", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 0;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }

      emitOpen() {
        this.readyState = 1;
        this.onopen?.();
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(electronHostHealthResponse());
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const voiceUnlistenPromise = listenViaHttpEvent(
      "lime-open-voice-model-settings",
      vi.fn(),
    );
    await vi.advanceTimersByTimeAsync(0);
    expect(MockEventSource.instances).toHaveLength(1);

    const configUnlistenPromise = listenViaHttpEvent("config-changed", vi.fn());
    await vi.advanceTimersByTimeAsync(0);
    expect(MockEventSource.instances).toHaveLength(1);

    MockEventSource.instances[0]?.emitOpen();
    await vi.advanceTimersByTimeAsync(0);

    expect(MockEventSource.instances).toHaveLength(2);
    expect(MockEventSource.instances[0]?.close).toHaveBeenCalledTimes(1);
    const nextUrl = new URL(MockEventSource.instances[1]!.url);
    expect(JSON.parse(nextUrl.searchParams.get("events") ?? "[]")).toEqual([
      "config-changed",
      "lime-open-voice-model-settings",
    ]);

    MockEventSource.instances[1]?.emitOpen();
    const [voiceUnlisten, configUnlisten] = await Promise.all([
      voiceUnlistenPromise,
      configUnlistenPromise,
    ]);

    voiceUnlisten();
    configUnlisten();
  });

  it("事件流如果在绑定 onopen 前已经打开，也应立即完成监听注册", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 1;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(electronHostHealthResponse());
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const unlistenPromise = listenViaHttpEvent("agent_stream_test", vi.fn());
    await vi.advanceTimersByTimeAsync(0);

    await expect(unlistenPromise).resolves.toEqual(expect.any(Function));
    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0]?.close).not.toHaveBeenCalled();
  });

  it("事件流已打开但浏览器遗漏 onopen 时，超时检查应按已连接处理", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 0;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(electronHostHealthResponse());
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const unlistenPromise = listenViaHttpEvent("agent_stream_test", vi.fn());
    await vi.advanceTimersByTimeAsync(0);
    const source = MockEventSource.instances[0]!;
    source.readyState = 1;
    await vi.advanceTimersByTimeAsync(10_000);

    await expect(unlistenPromise).resolves.toEqual(expect.any(Function));
    expect(source.close).not.toHaveBeenCalled();
  });

  it("事件流本地冷启动超过 1.5 秒但未超出桥接窗口时不应误判失败", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 0;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }

      emitOpen() {
        this.readyState = 1;
        this.onopen?.();
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(electronHostHealthResponse());
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const unlistenPromise = listenViaHttpEvent("agent_stream_test", vi.fn());
    await vi.advanceTimersByTimeAsync(1_800);

    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0]?.close).not.toHaveBeenCalled();

    MockEventSource.instances[0]?.emitOpen();
    const unlisten = await unlistenPromise;

    unlisten();
  });

  it("事件流在已建立连接后断开时应关闭连接，避免浏览器自动重连风暴", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 0;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }

      emitOpen() {
        this.readyState = 1;
        this.onopen?.();
      }

      emitError(error = new Event("error")) {
        this.onerror?.(error);
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(electronHostHealthResponse());
    const debugSpy = vi.mocked(console.debug);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const unlistenPromise = listenViaHttpEvent("config-changed", vi.fn());
    await vi.advanceTimersByTimeAsync(0);
    expect(MockEventSource.instances).toHaveLength(1);

    const source = MockEventSource.instances[0]!;
    source.emitOpen();
    const unlisten = await unlistenPromise;

    source.emitError();
    source.emitError();

    const secondUnlistenPromise = listenViaHttpEvent("config-changed", vi.fn());
    await vi.advanceTimersByTimeAsync(0);
    expect(MockEventSource.instances).toHaveLength(2);
    const nextSource = MockEventSource.instances[1]!;
    nextSource.emitOpen();
    const secondUnlisten = await secondUnlistenPromise;

    expect(MockEventSource.instances).toHaveLength(2);
    expect(source.close).toHaveBeenCalledTimes(1);
    expect(debugSpy).toHaveBeenCalledTimes(1);

    unlisten();
    secondUnlisten();
    expect(source.close).toHaveBeenCalledTimes(1);
    expect(nextSource.close).toHaveBeenCalledTimes(1);
  });

  it("事件流在建立后结束不应把整个桥接误标记为 unavailable", async () => {
    class MockEventSource {
      static instances: MockEventSource[] = [];

      onopen: (() => void) | null = null;
      onerror: ((error: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      readyState = 0;
      readonly close = vi.fn(() => {
        this.readyState = 2;
      });

      constructor(readonly url: string) {
        MockEventSource.instances.push(this);
      }

      emitOpen() {
        this.readyState = 1;
        this.onopen?.();
      }

      emitError(error = new Event("error")) {
        this.onerror?.(error);
      }
    }

    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(electronHostHealthResponse())
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ result: ["project-a"] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "EventSource",
      MockEventSource as unknown as typeof EventSource,
    );

    const unlistenPromise = listenViaHttpEvent("agent_stream_test", vi.fn());
    await vi.advanceTimersByTimeAsync(0);
    expect(MockEventSource.instances).toHaveLength(1);

    const source = MockEventSource.instances[0]!;
    source.emitOpen();
    const unlisten = await unlistenPromise;

    source.emitError();

    await expect(invokeViaHttp<string[]>("workspace_list")).resolves.toEqual([
      "project-a",
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://127.0.0.1:3030/invoke");

    unlisten();
  });
});
