import { describe, expect, it, vi } from "vitest";
import { createInventoryClient } from "./inventoryClient";

const emptyToolInventory = {
  request: {
    caller: "assistant",
    surface: {
      workbench: true,
      browser_assist: false,
    },
  },
  agent_initialized: true,
  warnings: [],
  mcp_servers: [],
  default_allowed_tools: [],
  counts: {
    catalog_total: 0,
    catalog_current_total: 0,
    catalog_compat_total: 0,
    catalog_deprecated_total: 0,
    default_allowed_total: 0,
    native_total: 0,
    native_visible_total: 0,
    native_catalog_unmapped_total: 0,
    extension_surface_total: 0,
    extension_mcp_bridge_total: 0,
    extension_runtime_total: 0,
    extension_tool_total: 0,
    extension_tool_visible_total: 0,
    mcp_server_total: 0,
    mcp_tool_total: 0,
    mcp_tool_visible_total: 0,
  },
  catalog_tools: [],
  native_tools: [],
  extension_surfaces: [],
  extension_tools: [],
  mcp_tools: [],
};

describe("agentRuntime inventoryClient", () => {
  it("workspace skill bindings 应通过 App Server JSON-RPC 获取 readiness", async () => {
    const appServerClient = {
      request: vi.fn().mockResolvedValueOnce({
        result: {
          bindings: {
            request: {
              workspace_root: "/tmp/work",
              caller: "assistant",
              surface: {
                workbench: true,
                browser_assist: false,
              },
            },
            warnings: [],
            counts: {
              registered_total: 1,
              ready_for_manual_enable_total: 1,
              blocked_total: 0,
              query_loop_visible_total: 0,
              tool_runtime_visible_total: 0,
              launch_enabled_total: 0,
            },
            bindings: [],
          },
        },
      }),
    };
    const client = createInventoryClient({ appServerClient });

    await expect(
      client.listWorkspaceSkillBindings({
        workspaceRoot: "  /tmp/work  ",
        caller: "assistant",
        workbench: true,
      }),
    ).resolves.toMatchObject({
      counts: {
        registered_total: 1,
        ready_for_manual_enable_total: 1,
      },
    });

    expect(appServerClient.request).toHaveBeenCalledWith(
      "workspaceSkillBindings/list",
      {
        workspaceRoot: "/tmp/work",
        caller: "assistant",
        workbench: true,
      },
    );
  });

  it("workspace skill bindings 缺少 workspaceRoot 时应 fail closed", async () => {
    const appServerClient = {
      request: vi.fn(),
    };
    const client = createInventoryClient({ appServerClient });

    await expect(
      client.listWorkspaceSkillBindings({ workspaceRoot: "   " }),
    ).rejects.toThrow(
      "workspaceRoot is required to list App Server workspace skill bindings",
    );

    expect(appServerClient.request).not.toHaveBeenCalled();
  });

  it("tool inventory 应通过 App Server current method 读取完整库存", async () => {
    const appServerClient = {
      request: vi.fn().mockResolvedValueOnce({
        result: {
          inventory: emptyToolInventory,
        },
      }),
    };
    const client = createInventoryClient({ appServerClient });

    await expect(
      client.getAgentRuntimeToolInventory({
        caller: "assistant",
        workbench: true,
      }),
    ).resolves.toMatchObject({
      counts: {
        catalog_total: 0,
        native_total: 0,
      },
    });

    expect(appServerClient.request).toHaveBeenCalledWith(
      "agentSession/toolInventory/read",
      {
        caller: "assistant",
        workbench: true,
      },
    );
  });

  it("tool inventory 应接受包含 runtime / extension / MCP 条目的真实库存", async () => {
    const inventory = {
      ...emptyToolInventory,
      counts: {
        ...emptyToolInventory.counts,
        runtime_total: 1,
        runtime_visible_total: 1,
      },
      catalog_tools: [
        {
          name: "site_run_adapter",
          profiles: [],
          capabilities: [],
          lifecycle: "compat",
          source: "runtime",
          permission_plane: "workspace",
          workspace_default_allow: false,
          execution_warning_policy: "warn",
          execution_warning_policy_source: "catalog",
          execution_restriction_profile: "standard",
          execution_restriction_profile_source: "catalog",
          execution_sandbox_profile: "workspace-write",
          execution_sandbox_profile_source: "catalog",
        },
      ],
      native_tools: [
        {
          name: "site_run_adapter",
          description: "Run site adapter",
          deferred_loading: false,
          always_visible: false,
          allowed_callers: ["assistant"],
          tags: ["site"],
          input_examples_count: 1,
          has_output_schema: false,
          caller_allowed: true,
          visible_in_context: true,
        },
      ],
      runtime_tools: [
        {
          name: "site_run_adapter",
          description: "Run site adapter",
          source_kind: "current_surface",
          deferred_loading: false,
          always_visible: false,
          allowed_callers: ["assistant"],
          tags: ["site"],
          input_examples_count: 1,
          has_output_schema: false,
          caller_allowed: true,
          visible_in_context: true,
        },
      ],
      extension_surfaces: [
        {
          extension_name: "mcp",
          description: "MCP bridge",
          source_kind: "mcp_bridge",
          deferred_loading: true,
          available_tools: ["mcp.search"],
          always_expose_tools: [],
          loaded_tools: [],
          searchable_tools: ["mcp.search"],
        },
      ],
      extension_tools: [
        {
          name: "mcp.search",
          description: "Search via MCP",
          extension_name: "mcp",
          source_kind: "mcp_bridge",
          deferred_loading: true,
          status: "available",
          caller_allowed: true,
          visible_in_context: true,
        },
      ],
      mcp_tools: [
        {
          server_name: "docs",
          name: "search",
          description: "Search docs",
          deferred_loading: false,
          always_visible: false,
          allowed_callers: ["assistant"],
          tags: [],
          input_examples_count: 0,
          has_output_schema: true,
          caller_allowed: true,
          visible_in_context: true,
        },
      ],
    };
    const appServerClient = {
      request: vi.fn().mockResolvedValueOnce({
        result: {
          inventory,
        },
      }),
    };
    const client = createInventoryClient({ appServerClient });

    await expect(client.getAgentRuntimeToolInventory()).resolves.toEqual(
      inventory,
    );
  });

  it("tool inventory 收到错误返回形态时应 fail closed", async () => {
    const appServerClient = {
      request: vi
        .fn()
        .mockResolvedValueOnce({ result: { inventory: { success: true } } })
        .mockResolvedValueOnce({
          result: {
            inventory: {
              ...emptyToolInventory,
              counts: { catalog_total: 0 },
            },
          },
        })
        .mockResolvedValueOnce({
          result: {
            inventory: {
              ...emptyToolInventory,
              native_tools: [{ name: "site_run_adapter" }],
            },
          },
        })
        .mockResolvedValueOnce({
          result: {
            inventory: {
              ...emptyToolInventory,
              runtime_tools: [{ name: "site_run_adapter" }],
            },
          },
        })
        .mockResolvedValueOnce({
          result: {
            inventory: {
              ...emptyToolInventory,
              mcp_tools: [{ name: "search" }],
            },
          },
        }),
    };
    const client = createInventoryClient({ appServerClient });

    await expect(client.getAgentRuntimeToolInventory()).rejects.toThrow(
      "App Server agentSession/toolInventory/read did not return tool inventory",
    );
    await expect(client.getAgentRuntimeToolInventory()).rejects.toThrow(
      "App Server agentSession/toolInventory/read did not return tool inventory",
    );
    await expect(client.getAgentRuntimeToolInventory()).rejects.toThrow(
      "App Server agentSession/toolInventory/read did not return tool inventory",
    );
    await expect(client.getAgentRuntimeToolInventory()).rejects.toThrow(
      "App Server agentSession/toolInventory/read did not return tool inventory",
    );
    await expect(client.getAgentRuntimeToolInventory()).rejects.toThrow(
      "App Server agentSession/toolInventory/read did not return tool inventory",
    );
  });
});
