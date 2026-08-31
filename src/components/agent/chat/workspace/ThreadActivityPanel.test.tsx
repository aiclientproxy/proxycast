import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThreadActivityPanel } from "./ThreadActivityPanel";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? key,
  }),
}));

describe("ThreadActivityPanel", () => {
  let container: HTMLDivElement;
  let root: Root;

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
  });

  function renderPanel(props: React.ComponentProps<typeof ThreadActivityPanel>) {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => root.render(<ThreadActivityPanel {...props} />));
  }

  it("把 canonical 子线程、MCP 和 skill inventory 汇总到同一面板", () => {
    renderPanel({
        canonicalChildren: [
          {
            name: "reviewer",
            parentThreadId: "parent",
            sessionId: "session-child",
            status: "running",
            threadId: "child-thread",
            updatedAtMs: 1,
          },
        ],
        threadItems: [
          {
            id: "activity-1",
            thread_id: "parent",
            turn_id: "turn-1",
            type: "subagent_activity",
            status: "in_progress",
            status_label: "running",
            title: "reviewer",
            updated_at: "2026-08-30T00:00:00Z",
          },
        ],
        toolInventory: {
          request: { caller: "agent", surface: { workbench: true } },
          agent_initialized: true,
          warnings: [],
          mcp_servers: ["docs"],
          default_allowed_tools: [],
          counts: {
            catalog_total: 1,
            catalog_current_total: 1,
            catalog_compat_total: 0,
            catalog_deprecated_total: 0,
            default_allowed_total: 1,
            native_total: 0,
            native_visible_total: 0,
            native_catalog_unmapped_total: 0,
            extension_surface_total: 0,
            extension_mcp_bridge_total: 0,
            extension_runtime_total: 0,
            extension_tool_total: 0,
            extension_tool_visible_total: 0,
            mcp_server_total: 1,
            mcp_tool_total: 2,
            mcp_tool_visible_total: 2,
          },
          catalog_tools: [
            {
              name: "skill_review",
              profiles: ["core"],
              capabilities: ["skill_execution"],
              lifecycle: "current",
              source: "agent_builtin",
              permission_plane: "session_allowlist",
              workspace_default_allow: true,
              execution_warning_policy: "none",
              execution_warning_policy_source: "default",
              execution_restriction_profile: "none",
              execution_restriction_profile_source: "default",
              execution_sandbox_profile: "none",
              execution_sandbox_profile_source: "default",
            },
          ],
          native_tools: [],
          extension_surfaces: [],
          extension_tools: [],
          mcp_tools: [],
        },
    });

    expect(container.querySelector('[data-testid="thread-activity-panel"]')).not.toBeNull();
    expect(container.querySelectorAll('[data-testid="thread-activity-subagent"]')).toHaveLength(1);
    expect(container.textContent).toContain("docs");
    expect(container.textContent).toContain("skill_review");
  });

  it("无活动时提供稳定空态", () => {
    renderPanel({});
    expect(container.querySelector('[data-testid="thread-activity-empty"]')).not.toBeNull();
  });
});
