import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import {
  METHOD_MODEL_PROVIDER_LIST,
  METHOD_MODEL_PROVIDER_UI_STATE_READ,
  METHOD_MODEL_PROVIDER_UI_STATE_WRITE,
} from "../../../packages/app-server-client/src/protocol";

const { mockSafeInvoke } = vi.hoisted(() => ({
  mockSafeInvoke: vi.fn(),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: mockSafeInvoke,
}));

import { invalidateApiKeyProviderCache } from "@/lib/api/apiKeyProvider";
import { ApiKeyProviderSection } from "./ApiKeyProviderSection";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createProvider(
  overrides: Partial<ProviderWithKeysDisplay>,
): ProviderWithKeysDisplay {
  return {
    id: "openai",
    name: "OpenAI",
    type: "openai",
    api_host: "https://api.openai.com",
    is_system: true,
    group: "mainstream",
    enabled: true,
    sort_order: 1,
    api_key_count: 0,
    models: [],
    prompt_cache_mode: null,
    created_at: "2026-03-15T00:00:00.000Z",
    updated_at: "2026-03-15T00:00:00.000Z",
    api_keys: [],
    ...overrides,
  };
}

async function flushEffects(times = 6) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

function renderSection() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <ApiKeyProviderSection exposeOemLoginPrompt onOemLogin={vi.fn()} />,
    );
  });

  mountedRoots.push({ root, container });
  return container;
}

describe("ApiKeyProviderSection OEM 登录提示", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    vi.clearAllMocks();
    invalidateApiKeyProviderCache();

    const providers: ProviderWithKeysDisplay[] = [
      createProvider({
        id: "lime-hub",
        name: "Lime Hub",
        api_host: "https://hub.lime.test",
        group: "cloud",
        sort_order: 0,
      }),
      createProvider({
        id: "openai",
        name: "OpenAI",
        sort_order: 1,
      }),
    ];
    const uiStateWrites: unknown[] = [];

    mockSafeInvoke.mockImplementation(
      async (command: string, payload?: Record<string, unknown>) => {
        switch (command) {
          case "app_server_handle_json_lines":
            return handleAppServerJsonLines(payload, providers, uiStateWrites);
          default:
            throw new Error(`未处理的 safeInvoke 命令：${command}`);
        }
      },
    );
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
    invalidateApiKeyProviderCache();
    vi.clearAllMocks();
  });

  it("未登录 Lime Hub 可被选中为登录提示，不应被本地 Provider 重定向抢回", async () => {
    const container = renderSection();
    await flushEffects();

    expect(
      container.querySelector('[data-testid="provider-login-required"]'),
    ).not.toBeNull();
    expect(container.textContent ?? "").toContain("登录后会自动同步 Lime Hub");

    const persistedSelections = collectAppServerUiStateWrites().map(
      (params) => params.value,
    );

    expect(persistedSelections).toContain("lime-hub");
    expect(persistedSelections).not.toContain("openai");
  });
});

function collectAppServerUiStateWrites(): Array<{ value?: string }> {
  return mockSafeInvoke.mock.calls.flatMap(([, payload]) => {
    const request = (payload as { request?: { lines?: string[] } } | undefined)
      ?.request;
    return (request?.lines ?? [])
      .map((line) => JSON.parse(line))
      .filter(
        (message) => message.method === METHOD_MODEL_PROVIDER_UI_STATE_WRITE,
      )
      .map((message) => message.params ?? {});
  });
}

function handleAppServerJsonLines(
  payload: Record<string, unknown> | undefined,
  providers: ProviderWithKeysDisplay[],
  uiStateWrites: unknown[],
): { lines: string[] } {
  const request = payload?.request as { lines?: string[] } | undefined;
  const messages = request?.lines?.map((line) => JSON.parse(line)) ?? [];
  const lines = messages.map((message) => {
    if (message.method === METHOD_MODEL_PROVIDER_LIST) {
      return `${JSON.stringify({
        id: message.id,
        result: { providers },
      })}\n`;
    }
    if (message.method === METHOD_MODEL_PROVIDER_UI_STATE_READ) {
      return `${JSON.stringify({
        id: message.id,
        result: {
          value: message.params?.key === "selected_provider" ? "openai" : null,
        },
      })}\n`;
    }
    if (message.method === METHOD_MODEL_PROVIDER_UI_STATE_WRITE) {
      uiStateWrites.push(message.params);
      return `${JSON.stringify({
        id: message.id,
        result: {},
      })}\n`;
    }

    return `${JSON.stringify({
      id: message.id,
      error: {
        code: -32601,
        message: `未处理的 App Server 方法：${message.method}`,
      },
    })}\n`;
  });

  return { lines };
}
