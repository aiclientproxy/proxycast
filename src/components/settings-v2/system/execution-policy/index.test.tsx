import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const { mockGetConfig, mockSaveConfig } = vi.hoisted(() => ({
  mockGetConfig: vi.fn(),
  mockSaveConfig: vi.fn(),
}));

vi.mock("@/lib/api/appConfig", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api/appConfig")>(
    "@/lib/api/appConfig",
  );
  return {
    ...actual,
    getConfig: mockGetConfig,
    saveConfig: mockSaveConfig,
  };
});

import { ExecutionPolicySettings } from ".";

interface Mounted {
  container: HTMLDivElement;
  root: Root;
}

const mounted: Mounted[] = [];

function createConfig() {
  return {
    default_provider: "openai",
    providers: {},
    agent: {
      use_default_system_prompt: true,
      default_model: "claude-sonnet-4-20250514",
      temperature: 0.7,
      max_tokens: 4096,
      workspace_sandbox: {
        enabled: false,
        strict: false,
        notify_on_fallback: true,
      },
      tool_execution: {
        tool_overrides: {
          Bash: {
            warning_policy: "none",
            restriction_profile: "workspace_shell_command",
            sandbox_profile: "workspace_command",
          },
        },
        shell_command_rules: [
          {
            rule_id: "danger-rm",
            match_type: "prefix",
            pattern: "rm -rf",
            risk_level: "high",
            reason_code: "destructive_delete",
            reason: "delete workspace files",
          },
        ],
        network_rules: [
          {
            rule_id: "download-block",
            match_type: "regex",
            target: "url",
            pattern: "https://example.com/install.sh",
            risk_level: "high",
            reason_code: "request_download_url",
            reason: "external installer",
          },
        ],
      },
    },
  };
}

function renderComponent(
  props?: Parameters<typeof ExecutionPolicySettings>[0],
): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<ExecutionPolicySettings {...props} />);
  });

  mounted.push({ container, root });
  return container;
}

async function flushEffects(times = 6) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

async function waitForLoad() {
  await flushEffects();
}

function findButton(container: HTMLElement, text: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find((node) =>
    node.textContent?.includes(text),
  );
  if (!button) {
    throw new Error(`未找到按钮: ${text}`);
  }
  return button as HTMLButtonElement;
}

function findRuleInputByTestId(
  container: HTMLElement,
  testId: string,
  value: string,
): HTMLInputElement {
  const rule = container.querySelector(`[data-testid="${testId}"]`);
  const input = Array.from(rule?.querySelectorAll("input") ?? []).find(
    (node) => node.value === value,
  );
  if (!input) {
    throw new Error(`未找到规则 ${testId} 中的输入框值: ${value}`);
  }
  return input as HTMLInputElement;
}

function findInputByValue(
  container: HTMLElement,
  value: string,
): HTMLInputElement {
  const input = Array.from(container.querySelectorAll("input")).find(
    (node) => node.value === value,
  );
  if (!input) {
    throw new Error(`未找到输入框值: ${value}`);
  }
  return input as HTMLInputElement;
}

function findSelectByValue(
  container: HTMLElement,
  value: string,
): HTMLSelectElement {
  const select = Array.from(container.querySelectorAll("select")).find(
    (node) => node.value === value,
  );
  if (!select) {
    throw new Error(`未找到下拉值: ${value}`);
  }
  return select as HTMLSelectElement;
}

async function clickButton(button: HTMLButtonElement) {
  await act(async () => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushEffects(2);
  });
}

async function setSelectValue(select: HTMLSelectElement, value: string) {
  const nativeSetter = Object.getOwnPropertyDescriptor(
    window.HTMLSelectElement.prototype,
    "value",
  )?.set;
  if (!nativeSetter) {
    throw new Error("未找到 select value setter");
  }

  await act(async () => {
    nativeSetter.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
    await flushEffects(2);
  });
}

async function toggleSwitch(container: HTMLElement, ariaLabel: string) {
  const button = container.querySelector<HTMLButtonElement>(
    `button[aria-label='${ariaLabel}']`,
  );
  if (!button) {
    throw new Error(`未找到开关: ${ariaLabel}`);
  }
  await clickButton(button);
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  await changeLimeLocale("zh-CN");
  mockGetConfig.mockResolvedValue(createConfig());
  mockSaveConfig.mockResolvedValue(undefined);
});

afterEach(async () => {
  while (mounted.length > 0) {
    const current = mounted.pop();
    if (!current) {
      break;
    }
    act(() => {
      current.root.unmount();
    });
    current.container.remove();
  }

  await changeLimeLocale("zh-CN");
});

describe("ExecutionPolicySettings", () => {
  it("应从 current 配置读取沙箱、命令策略和网络规则", async () => {
    const container = renderComponent();
    await waitForLoad();

    expect(mockGetConfig).toHaveBeenCalledWith({ forceRefresh: true });
    expect(container.textContent).toContain("执行策略");
    expect(container.textContent).toContain("已自定义");
    expect(container.textContent).toContain("策略来源层级");
    expect(container.textContent).toContain("默认策略");
    expect(container.textContent).toContain("当前配置");
    expect(container.textContent).toContain("组织策略");
    expect(container.textContent).toContain("用户策略");
    expect(container.textContent).toContain("运行时策略");
    expect(container.textContent).toContain("本次请求策略");
    expect(container.textContent).toContain("当前可编辑");
    expect(container.textContent).toContain("运行期只读");
    expect(container.textContent).toContain(
      "本页保存 1 条命令规则、1 条网络规则、1 个工具覆盖",
    );
    expect(container.textContent).toContain("Shell 命令审批");
    expect(container.textContent).toContain("网络规则");
    expect(container.textContent).toContain("命令规则");
    expect(container.textContent).toContain("网络规则");
    expect(findInputByValue(container, "danger-rm")).toBeInstanceOf(
      HTMLInputElement,
    );
    expect(
      findInputByValue(container, "https://example.com/install.sh"),
    ).toBeInstanceOf(HTMLInputElement);
    expect(findSelectByValue(container, "none")).toBeInstanceOf(
      HTMLSelectElement,
    );
    expect(
      container.querySelector('[data-testid="execution-policy-source-layers"]'),
    ).not.toBeNull();
  });

  it("应根据诊断入口参数定位并高亮对应网络规则", async () => {
    const container = renderComponent({
      focus: {
        section: "network",
        ruleId: "download-block",
        target: "url",
        value: "https://example.com/install.sh",
        reasonCode: "request_download_url",
      },
    });
    await waitForLoad();

    const focusBanner = container.querySelector(
      '[data-testid="execution-policy-network-focus"]',
    );
    const focusedRule = container.querySelector(
      '[data-testid="network-rule-0"]',
    );

    expect(focusBanner?.textContent).toContain("download-block");
    expect(focusBanner?.textContent).toContain(
      "https://example.com/install.sh",
    );
    expect(focusedRule?.getAttribute("data-focused")).toBe("true");
  });

  it("诊断入口未命中现有网络规则时应可添加建议规则并随保存写回", async () => {
    const container = renderComponent({
      focus: {
        section: "network",
        ruleId: "blocked-docs-host",
        target: "host",
        value: "docs.example.com",
        reasonCode: "external_docs_host",
      },
    });
    await waitForLoad();

    const missingBanner = container.querySelector(
      '[data-testid="execution-policy-network-focus-missing"]',
    );
    expect(missingBanner?.textContent).toContain("docs.example.com");
    expect(
      container.querySelector(
        '[data-testid="execution-policy-network-suggestion"]',
      )?.textContent,
    ).toContain("添加建议规则");

    await clickButton(findButton(container, "添加建议规则"));

    expect(
      findRuleInputByTestId(container, "network-rule-1", "docs.example.com"),
    ).toBeInstanceOf(HTMLInputElement);
    expect(
      findRuleInputByTestId(container, "network-rule-1", "blocked-docs-host"),
    ).toBeInstanceOf(HTMLInputElement);
    await clickButton(findButton(container, "保存策略"));

    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        agent: expect.objectContaining({
          tool_execution: expect.objectContaining({
            network_rules: expect.arrayContaining([
              expect.objectContaining({
                rule_id: "blocked-docs-host",
                match_type: "exact",
                target: "host",
                pattern: "docs.example.com",
                risk_level: "high",
                reason_code: "external_docs_host",
              }),
            ]),
          }),
        }),
      }),
    );
  });

  it("保存时应写入 agent.workspace_sandbox 与 agent.tool_execution canonical bash 配置", async () => {
    const container = renderComponent();
    await waitForLoad();

    await toggleSwitch(container, "启用工作区沙箱");
    await toggleSwitch(container, "启用严格工作区沙箱");
    await setSelectValue(
      findSelectByValue(container, "none"),
      "shell_command_risk",
    );
    await clickButton(findButton(container, "保存策略"));

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        agent: expect.objectContaining({
          workspace_sandbox: {
            enabled: true,
            strict: true,
            notify_on_fallback: true,
          },
          tool_execution: expect.objectContaining({
            tool_overrides: {
              bash: expect.objectContaining({
                warning_policy: "shell_command_risk",
                restriction_profile: "workspace_shell_command",
                sandbox_profile: "workspace_command",
              }),
            },
            shell_command_rules: expect.arrayContaining([
              expect.objectContaining({
                rule_id: "danger-rm",
                pattern: "rm -rf",
                risk_level: "high",
              }),
            ]),
            network_rules: expect.arrayContaining([
              expect.objectContaining({
                rule_id: "download-block",
                target: "url",
                pattern: "https://example.com/install.sh",
              }),
            ]),
          }),
        }),
      }),
    );
    const savedConfig = mockSaveConfig.mock.calls[0]?.[0] as {
      agent?: {
        tool_execution?: {
          tool_overrides?: Record<string, unknown>;
        };
      };
    };
    expect(
      savedConfig.agent?.tool_execution?.tool_overrides,
    ).not.toHaveProperty("Bash");
    expect(container.textContent).toContain("执行策略已保存");
  });

  it("保存时应按 Codex opt-in 语义写入 update_plan 开关", async () => {
    const container = renderComponent();
    await waitForLoad();

    await toggleSwitch(container, "启用 update_plan 计划工具");
    await clickButton(findButton(container, "保存策略"));

    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        agent: expect.objectContaining({
          tool_execution: expect.objectContaining({
            update_plan_enabled: true,
          }),
        }),
      }),
    );
  });
});
