import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Palette, Brain, ShieldCheck } from "lucide-react";
import { changeLimeLocale } from "@/i18n/createI18n";
import { SettingsGroupKey, SettingsTabs } from "@/types/settings";

const { mockUseSettingsCategory } = vi.hoisted(() => ({
  mockUseSettingsCategory: vi.fn(),
}));

vi.mock("../hooks/useSettingsCategory", () => ({
  useSettingsCategory: () => mockUseSettingsCategory(),
}));

import { SettingsHomePage } from "./index";

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mounted: RenderResult[] = [];

function renderPage(
  onTabChange = vi.fn(),
  onTabPrefetch?: (tab: SettingsTabs) => void,
  onNavigate?: (page: string, params?: unknown) => void,
): RenderResult {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <SettingsHomePage
        onTabChange={onTabChange}
        onTabPrefetch={onTabPrefetch}
        onNavigate={onNavigate as any}
      />,
    );
  });

  const rendered = { container, root };
  mounted.push(rendered);
  return rendered;
}

function getBodyText() {
  return document.body.textContent ?? "";
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  await changeLimeLocale("en-US");
  mockUseSettingsCategory.mockReturnValue([
    {
      key: SettingsGroupKey.Overview,
      title: "Overview",
      items: [
        {
          key: SettingsTabs.Home,
          label: "Settings Home",
          icon: Palette,
        },
      ],
    },
    {
      key: SettingsGroupKey.General,
      title: "General",
      items: [
        {
          key: SettingsTabs.Appearance,
          label: "Appearance",
          icon: Palette,
        },
      ],
    },
    {
      key: SettingsGroupKey.Agent,
      title: "Agent",
      items: [
        {
          key: SettingsTabs.Providers,
          label: "AI Providers",
          icon: Brain,
        },
      ],
    },
    {
      key: SettingsGroupKey.System,
      title: "System",
      items: [
        {
          key: SettingsTabs.ChromeRelay,
          label: "Connector",
          icon: ShieldCheck,
        },
      ],
    },
  ]);
});

afterEach(async () => {
  mockUseSettingsCategory.mockReset();

  while (mounted.length > 0) {
    const target = mounted.pop();
    if (!target) {
      break;
    }
    act(() => {
      target.root.unmount();
    });
    target.container.remove();
  }

  await changeLimeLocale("zh-CN");
});

describe("SettingsHomePage", () => {
  it("应渲染设置首页总览与分组入口", () => {
    const { container } = renderPage();
    const text = container.textContent ?? "";

    expect(text).toContain("Settings Home");
    expect(text).toContain(
      "Quickly open common settings and review each group.",
    );
    expect(text).toContain("Quick Access");
    expect(text).toContain("General");
    expect(text).toContain("Agent");
    expect(text).toContain("System");
    expect(text).toContain("Appearance");
    expect(text).toContain("AI Providers");
    expect(text).not.toContain("SETTINGS OVERVIEW");
    expect(text).not.toContain("安全与性能");
    expect(text).not.toContain("权限、稳定性与运行开关");
    expect(text).not.toContain("settings.home");
  });

  it("渲染分组帮助按钮，但说明正文默认不常驻", () => {
    const { container } = renderPage();
    const generalHelpButton = container.querySelector<HTMLButtonElement>(
      "button[aria-label='General help']",
    );

    expect(generalHelpButton).not.toBeNull();
    expect(
      container.querySelector("button[aria-label='General说明']"),
    ).toBeNull();
    expect(getBodyText()).not.toContain(
      "Appearance, memory, and global experience settings.",
    );

    act(() => {
      generalHelpButton?.dispatchEvent(
        new MouseEvent("mouseover", { bubbles: true }),
      );
    });

    expect(getBodyText()).toContain(
      "Appearance, memory, and global experience settings.",
    );

    act(() => {
      generalHelpButton?.dispatchEvent(
        new MouseEvent("mouseout", { bubbles: true }),
      );
    });

    expect(getBodyText()).not.toContain(
      "Appearance, memory, and global experience settings.",
    );
  });

  it("点击常用入口时应触发 tab 切换", () => {
    const onTabChange = vi.fn();
    const { container } = renderPage(onTabChange);
    const button = Array.from(container.querySelectorAll("button")).find(
      (item) => item.textContent?.includes("Appearance"),
    );

    act(() => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onTabChange).toHaveBeenCalledWith(SettingsTabs.Appearance);
  });

  it("悬停常用入口时应触发对应 tab 预取", () => {
    const onTabPrefetch = vi.fn();
    const { container } = renderPage(vi.fn(), onTabPrefetch);
    const button = Array.from(container.querySelectorAll("button")).find(
      (item) => item.textContent?.includes("Appearance"),
    );

    act(() => {
      button?.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    });

    expect(onTabPrefetch).toHaveBeenCalledWith(SettingsTabs.Appearance);
  });

  it("应展示当前入口卡并支持跳到当前落点", () => {
    const onTabChange = vi.fn();
    const onNavigate = vi.fn();
    const { container } = renderPage(
      onTabChange,
      undefined,
      onNavigate,
    );
    const text = container.textContent ?? "";

    expect(text).toContain("Current Entrypoints");
    expect(text).not.toContain("All Skills");
    expect(text).toContain("Scheduled Tasks");
    expect(text).toContain("Message Channels");
    expect(text).toContain("Project Knowledge");

    const openScheduledTasksButton = Array.from(
      container.querySelectorAll("button"),
    ).find((item) => item.textContent?.includes("Open Scheduled Tasks"));
    const openChannelsButton = Array.from(
      container.querySelectorAll("button"),
    ).find((item) => item.textContent?.includes("Open Channels"));
    const openResourcesButton = Array.from(
      container.querySelectorAll("button"),
    ).find((item) => item.textContent?.includes("Open Project Knowledge"));

    act(() => {
      openScheduledTasksButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      openChannelsButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      openResourcesButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(onTabChange).not.toHaveBeenCalled();
    expect(onNavigate).toHaveBeenCalledWith("scheduled-tasks");
    expect(onNavigate).toHaveBeenCalledWith("channels");
    expect(onNavigate).toHaveBeenCalledWith("resources");
  });

  it("首页和常用入口说明只在 tip 交互时展示", () => {
    renderPage();
    const homeHelpButton = document.body.querySelector<HTMLButtonElement>(
      "button[aria-label='Settings home help']",
    );
    const appearanceHelpButton =
      document.body.querySelector<HTMLButtonElement>(
        "button[aria-label='Appearance help']",
      );

    expect(homeHelpButton).not.toBeNull();
    expect(appearanceHelpButton).not.toBeNull();
    expect(getBodyText()).not.toContain(
      "Quickly open common settings and review each group without digging through nested menus.",
    );
    expect(getBodyText()).not.toContain(
      "Theme, interface language, and reply language",
    );

    act(() => {
      homeHelpButton?.dispatchEvent(
        new MouseEvent("mouseover", { bubbles: true }),
      );
    });

    expect(getBodyText()).toContain(
      "Quickly open common settings and review each group without digging through nested menus.",
    );

    act(() => {
      homeHelpButton?.dispatchEvent(
        new MouseEvent("mouseout", { bubbles: true }),
      );
      appearanceHelpButton?.dispatchEvent(
        new MouseEvent("mouseover", { bubbles: true }),
      );
    });

    expect(getBodyText()).not.toContain(
      "Quickly open common settings and review each group without digging through nested menus.",
    );
    expect(getBodyText()).toContain(
      "Theme, interface language, and reply language",
    );
  });
});
