import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanupAppSidebarTest,
  flushEffects,
  mountSidebarContainer,
  openAccountMenu,
  resetAppSidebarTest,
} from "./AppSidebar.testFixtures";

describe("AppSidebar Plugins", () => {
  beforeEach(resetAppSidebarTest);
  afterEach(cleanupAppSidebarTest);

  it("插件中心入口应常驻主导航并进入插件页", async () => {
    const onNavigate = vi.fn();
    const container = mountSidebarContainer({ onNavigate });
    await flushEffects(2);

    expect(
      Array.from(
        container.querySelectorAll(
          '[data-testid="app-sidebar-main-nav"] button',
        ),
      ).map((button) => button.getAttribute("aria-label")),
    ).toEqual(["新建任务", "专家", "Skills", "插件"]);

    act(() => {
      container
        .querySelector<HTMLButtonElement>('button[aria-label="插件"]')
        ?.click();
    });

    expect(onNavigate).toHaveBeenCalledWith("plugins", undefined);
    await openAccountMenu(container);
    const accountMenu = container.querySelector(
      '[data-testid="app-sidebar-account-menu"]',
    );
    expect(accountMenu?.textContent).not.toContain("插件");
  });

  it("已安装 Plugin 不应作为左侧独立导航项显示", async () => {
    const container = mountSidebarContainer();
    await flushEffects(2);

    const mainNav = container.querySelector(
      '[data-testid="app-sidebar-main-nav"]',
    );
    expect(mainNav?.textContent).toContain("插件");
    expect(mainNav?.textContent).not.toContain("内容工厂");
    expect(mainNav?.textContent).not.toContain("发布应用");
    expect(
      Array.from(
        container.querySelectorAll(
          '[data-testid="app-sidebar-main-nav"] button',
        ),
      ).map((button) => button.getAttribute("aria-label")),
    ).toEqual(["新建任务", "专家", "Skills", "插件"]);
  });

  it("Plugin 变更事件不应影响侧栏聚合入口", async () => {
    const container = mountSidebarContainer();
    await flushEffects(2);

    await act(async () => {
      window.dispatchEvent(new Event("lime:plugins-changed"));
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(
      Array.from(
        container.querySelectorAll(
          '[data-testid="app-sidebar-main-nav"] button',
        ),
      ).map((button) => button.getAttribute("aria-label")),
    ).toEqual(["新建任务", "专家", "Skills", "插件"]);
  });

  it("安装态读取失败不应影响静态 Plugins 聚合入口", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    try {
      const container = mountSidebarContainer();
      await flushEffects(2);

      expect(
        container.querySelector<HTMLButtonElement>(
          'button[aria-label="插件"]',
        ),
      ).not.toBeNull();
      expect(container.textContent).not.toContain("内容工厂");
      expect(
        errorSpy.mock.calls.map(([message]) => String(message)),
      ).not.toEqual(
        expect.arrayContaining([
          expect.stringContaining("加载 Plugin 导航失败"),
        ]),
      );
      expect(
        warnSpy.mock.calls.map(([message]) => String(message)),
      ).not.toEqual(
        expect.arrayContaining([
          expect.stringContaining("加载 Plugin 导航失败"),
        ]),
      );
    } finally {
      errorSpy.mockRestore();
      warnSpy.mockRestore();
    }
  });
});
