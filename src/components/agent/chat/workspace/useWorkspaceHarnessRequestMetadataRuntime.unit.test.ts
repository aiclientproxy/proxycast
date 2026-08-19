import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  resolveWorkspaceHarnessRequestMetadata,
  useWorkspaceHarnessRequestMetadataRuntime,
} from "./useWorkspaceHarnessRequestMetadataRuntime";

const workspaceSkillBinding = {
  directory: "capability-report",
  name: "只读 CLI 报告",
  description: "把只读 CLI 输出整理成 Markdown 报告。",
  registered_skill_directory: "/tmp/project/.agents/skills/capability-report",
  binding_status: "ready_for_manual_enable",
  next_gate: "manual_runtime_enable",
  query_loop_visible: false,
  tool_runtime_visible: false,
  launch_enabled: false,
  permission_summary: ["Level 0 只读发现"],
  registration: {
    source_draft_id: "capdraft-1",
    source_verification_report_id: "capver-1",
  },
} as AgentRuntimeWorkspaceSkillBinding;

type HookProps = Parameters<
  typeof useWorkspaceHarnessRequestMetadataRuntime
>[0];

const mountedRoots: Array<{
  container: HTMLDivElement;
  root: ReturnType<typeof createRoot>;
}> = [];

function mountHook(initialProps: HookProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceHarnessRequestMetadataRuntime
  > | null = null;
  let currentProps = initialProps;

  function TestComponent() {
    latestValue = useWorkspaceHarnessRequestMetadataRuntime(currentProps);
    return null;
  }

  const render = (nextProps?: Partial<HookProps>) => {
    currentProps = {
      ...currentProps,
      ...nextProps,
    };
    act(() => {
      root.render(React.createElement(TestComponent));
    });
  };

  render();
  mountedRoots.push({ container, root });

  return {
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
    rerender: render,
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      continue;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
});

describe("workspace harness request metadata runtime", () => {
  it("应保持主组件原 harness metadata 映射语义", () => {
    const metadata = resolveWorkspaceHarnessRequestMetadata({
      agentResponseLanguage: "en-US",
      contentId: "content-1",
      currentGateKey: "write_mode",
      effectiveChatToolPreferences: {
        task: true,
        subagent: true,
      },
      isThemeWorkbench: true,
      mappedTheme: "general",
      themeWorkbenchActiveQueueTitle: "  修复导入渲染  ",
      workspaceSkillBindings: [workspaceSkillBinding],
      workspaceSkillRuntimeEnable: {
        workspaceRoot: "/tmp/project",
        bindings: [workspaceSkillBinding],
      },
    });

    expect(metadata).toMatchObject({
      theme: "general",
      preferences: {
        task: true,
        subagent: true,
      },
      session_mode: "general_workbench",
      gate_key: "write_mode",
      run_title: "修复导入渲染",
      content_id: "content-1",
      agent_response_language: "en-US",
      workspace_skill_bindings: {
        source: "p3c_runtime_binding",
        bindings: [
          expect.objectContaining({
            directory: "capability-report",
            source_draft_id: "capdraft-1",
          }),
        ],
      },
      workspace_skill_runtime_enable: {
        source: "manual_session_enable",
        approval: "manual",
        workspace_root: "/tmp/project",
        bindings: [
          expect.objectContaining({
            directory: "capability-report",
            skill: "project:capability-report",
          }),
        ],
      },
    });
    expect(metadata.browser_assist).toBeUndefined();
  });

  it("非主题工作台时不应写入 gate key", () => {
    const metadata = resolveWorkspaceHarnessRequestMetadata({
      effectiveChatToolPreferences: {
        task: false,
        subagent: false,
      },
      isThemeWorkbench: false,
      mappedTheme: "general",
      currentGateKey: "write_mode",
      workspaceSkillBindings: [],
    });

    expect(metadata.session_mode).toBe("default");
    expect(metadata.gate_key).toBeUndefined();
    expect(metadata.workspace_skill_bindings).toBeUndefined();
  });

  it("关闭时应返回稳定空 metadata，避免默认渲染组装 Harness 详情上下文", () => {
    const disabledMetadata = resolveWorkspaceHarnessRequestMetadata({
      enabled: false,
      effectiveChatToolPreferences: {
        task: true,
        subagent: true,
      },
      isThemeWorkbench: true,
      mappedTheme: "general",
      workspaceSkillBindings: [workspaceSkillBinding],
      workspaceSkillRuntimeEnable: {
        workspaceRoot: "/tmp/project",
        bindings: [workspaceSkillBinding],
      },
    });
    const disabledAgain = resolveWorkspaceHarnessRequestMetadata({
      enabled: false,
      effectiveChatToolPreferences: {
        task: true,
        subagent: true,
      },
      isThemeWorkbench: true,
      mappedTheme: "general",
      workspaceSkillBindings: [workspaceSkillBinding],
    });

    expect(disabledMetadata).toEqual({});
    expect(disabledAgain).toBe(disabledMetadata);
  });

  it("hook 关闭时应保持空 metadata 引用稳定，开启后再构造详情上下文", () => {
    const harness = mountHook({
      enabled: false,
      effectiveChatToolPreferences: {
        task: true,
        subagent: true,
      },
      isThemeWorkbench: true,
      mappedTheme: "general",
      workspaceSkillBindings: [workspaceSkillBinding],
    });
    const firstDisabled = harness.getValue();

    harness.rerender({
      contentId: "ignored-while-disabled",
    });
    expect(harness.getValue()).toBe(firstDisabled);
    expect(harness.getValue()).toEqual({});

    harness.rerender({ enabled: true });
    expect(harness.getValue()).toMatchObject({
      theme: "general",
      workspace_skill_bindings: {
        source: "p3c_runtime_binding",
      },
    });
  });
});
