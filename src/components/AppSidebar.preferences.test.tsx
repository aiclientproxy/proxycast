import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  APP_SIDEBAR_ENABLED_ITEMS_STORAGE_KEY,
  LIME_COLOR_SCHEME_STORAGE_KEY,
  LIME_THEME_STORAGE_KEY,
  act,
  changeLimeLocale,
  cleanupAppSidebarTest,
  flushEffects,
  mockCheckForUpdates,
  mockGetConfig,
  mockGetUpdateInstallSession,
  mockOpenUpdateWindow,
  mountSidebarContainer,
  resetAppSidebarTest,
} from "./AppSidebar.testFixtures";
import type { AgentPageParams } from "./AppSidebar.testFixtures";

describe("AppSidebar preferences", () => {
  beforeEach(resetAppSidebarTest);
  afterEach(cleanupAppSidebarTest);

  it("检测到新版本时应在账户区显示升级图标并点击打开更新专用窗口", async () => {
    mockCheckForUpdates.mockResolvedValue({
      current: "1.57.0",
      latest: "1.58.0",
      hasUpdate: true,
      downloadUrl: "https://example.com/lime",
      releaseNotesUrl: null,
      releaseNotes: null,
      pubDate: null,
      error: null,
    });

    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(3);

    expect(mockCheckForUpdates).toHaveBeenCalledWith({ automatic: true });

    const accountSlot = container.querySelector(
      '[data-testid="app-sidebar-account-slot"]',
    );
    const updateButton = accountSlot?.querySelector<HTMLButtonElement>(
      '[data-testid="app-sidebar-update-button"]',
    );

    expect(updateButton).not.toBeNull();
    expect(updateButton?.textContent).toBe("");
    expect(
      container.querySelector('[data-testid="app-sidebar-update-panel"]'),
    ).toBeNull();

    updateButton!.getBoundingClientRect = vi.fn(
      () =>
        ({
          x: 18,
          y: 816,
          width: 30,
          height: 30,
          top: 816,
          right: 48,
          bottom: 846,
          left: 18,
          toJSON: () => ({}),
        }) as DOMRect,
    );

    await act(async () => {
      updateButton?.click();
      await Promise.resolve();
    });

    expect(mockOpenUpdateWindow).toHaveBeenCalledWith({
      x: 18,
      y: 816,
      width: 30,
      height: 30,
    });
    expect(
      container.querySelector('[data-testid="app-sidebar-update-panel"]'),
    ).toBeNull();
    expect(
      accountSlot?.querySelector('[data-testid="app-sidebar-update-button"]'),
    ).not.toBeNull();
  });

  it("版本尚未确认的下载会话不应显示更新入口", async () => {
    mockGetUpdateInstallSession.mockResolvedValue({
      sessionId: "checking-update",
      stage: "downloading",
      currentVersion: "1.57.0",
      latestVersion: null,
      downloadUrl: null,
      downloadedBytes: 0,
      totalBytes: null,
      percent: 0,
      message: "正在下载并验证更新",
      error: null,
      startedAt: 1,
      updatedAt: 2,
      completedAt: null,
      canCloseWindow: true,
      isActive: true,
    });

    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(3);

    expect(
      container.querySelector('[data-testid="app-sidebar-update-button"]'),
    ).toBeNull();
  });

  it("旧 companion 配置不应复活可选系统入口", async () => {
    mockGetConfig.mockResolvedValue({
      navigation: {
        schema_version: 3,
        enabled_items: ["companion"],
      },
    });

    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(container.textContent).not.toContain("桌宠");

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-account-button"]',
        )
        ?.click();
      await Promise.resolve();
    });

    const accountMenu = container.querySelector(
      '[data-testid="app-sidebar-account-menu"]',
    );
    expect(accountMenu?.textContent).toContain("设置");
    expect(accountMenu?.textContent).toContain("消息渠道");
    expect(accountMenu?.textContent).not.toContain("已安排任务");
    expect(accountMenu?.textContent).not.toContain("桌宠");
  });

  it("配置变更后仍不应恢复旧 companion 入口", async () => {
    mockGetConfig
      .mockResolvedValueOnce({
        navigation: {
          schema_version: 3,
          enabled_items: [],
        },
      })
      .mockResolvedValueOnce({
        navigation: {
          schema_version: 3,
          enabled_items: ["companion"],
        },
      });

    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(container.textContent).not.toContain("桌宠");

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-account-button"]',
        )
        ?.click();
      await Promise.resolve();
    });

    let accountMenu = container.querySelector(
      '[data-testid="app-sidebar-account-menu"]',
    );
    expect(accountMenu?.textContent).toContain("设置");
    expect(accountMenu?.textContent).toContain("消息渠道");
    expect(accountMenu?.textContent).not.toContain("已安排任务");
    expect(accountMenu?.textContent).not.toContain("桌宠");

    await act(async () => {
      (
        globalThis as typeof globalThis & {
          __appConfigListener?: () => void;
        }
      ).__appConfigListener?.();
      await Promise.resolve();
    });
    await flushEffects(2);

    accountMenu = container.querySelector(
      '[data-testid="app-sidebar-account-menu"]',
    );
    expect(accountMenu?.textContent).not.toContain("桌宠");
  });

  it("旧 schema 中的桌宠入口不应默认显示", async () => {
    mockGetConfig.mockResolvedValue({
      navigation: {
        schema_version: 2,
        enabled_items: ["companion"],
      },
    });

    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-account-button"]',
        )
        ?.click();
      await Promise.resolve();
    });

    const accountMenu = container.querySelector(
      '[data-testid="app-sidebar-account-menu"]',
    );
    expect(accountMenu?.textContent).not.toContain("桌宠");
  });

  it.each(["plugins", "skills", "experts"] as const)(
    "插件入口在 %s 子页面应保持激活",
    async (currentPage) => {
      const container = mountSidebarContainer({ currentPage });
      await flushEffects();

      const button = container.querySelector<HTMLButtonElement>(
        'button[aria-label="插件"]',
      );

      expect(button).not.toBeNull();
      expect(button?.getAttribute("aria-current")).toBe("page");
      expect(
        container.querySelector('[data-testid="app-sidebar-nav-skills"]'),
      ).toBeNull();
      expect(
        container.querySelector('[data-testid="app-sidebar-nav-experts"]'),
      ).toBeNull();
    },
  );

  it("从 Skills 子页面点击侧栏插件入口应回到插件 Tab", async () => {
    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "skills",
      onNavigate,
    });
    await flushEffects();

    const button = container.querySelector<HTMLButtonElement>(
      'button[aria-label="插件"]',
    );

    expect(button).not.toBeNull();
    expect(button?.getAttribute("aria-current")).toBe("page");

    act(() => {
      button?.click();
    });

    expect(onNavigate).toHaveBeenCalledWith("plugins", undefined);
  });

  it("旧的 enabled-items 本地缓存不应再复活历史导航", async () => {
    localStorage.setItem(
      APP_SIDEBAR_ENABLED_ITEMS_STORAGE_KEY,
      JSON.stringify(["plugins", "companion", "video"]),
    );
    mockGetConfig.mockImplementation(() => new Promise(() => undefined));

    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects();

    expect(container.textContent).not.toContain("插件中心");
    expect(container.textContent).not.toContain("桌宠");
  });

  it("底部外观入口应弹出轻量快捷面板并同步主题与皮肤", async () => {
    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    const trigger = container.querySelector<HTMLButtonElement>(
      'button[aria-label="快速切换外观"]',
    );

    expect(trigger).not.toBeNull();

    await act(async () => {
      trigger?.click();
      await Promise.resolve();
    });

    const popover = container.querySelector(
      '[data-testid="app-sidebar-appearance-popover"]',
    );
    expect(popover).not.toBeNull();
    expect(popover?.textContent).toContain("浅色");
    expect(popover?.textContent).toContain("深色");
    expect(popover?.textContent).toContain("跟随系统");
    expect(popover?.textContent).toContain("随机");
    expect(popover?.textContent).toContain("梦樱花境");
    expect(popover?.textContent).toContain("森野秘境");
    expect(popover?.textContent).toContain("财神打工");
    expect(popover?.textContent).toContain("奥特曼守护");
    expect(popover?.textContent).toContain("东方国潮");
    expect(popover?.textContent).toContain("初音未来");
    expect(popover?.textContent).toContain("灵感少年");
    expect(popover?.textContent).toContain("黑金舞台");
    expect(popover?.textContent).toContain("极简未来");
    expect(popover?.textContent).toContain("爆燃涂鸦");
    expect(popover?.textContent).toContain("清透少年");
    expect(popover?.textContent).toContain("蓝紫星夜");
    expect(popover?.textContent).toContain("红白未来城");

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          'button[aria-label="切换皮肤为奥特曼守护"]',
        )
        ?.click();
      await Promise.resolve();
    });

    expect(localStorage.getItem(LIME_COLOR_SCHEME_STORAGE_KEY)).toBe(
      "lime-ocean",
    );
    expect(document.documentElement.dataset.limeColorScheme).toBe("lime-ocean");

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>('button[aria-label="切换主题为深色"]')
        ?.click();
      await Promise.resolve();
    });

    expect(localStorage.getItem(LIME_THEME_STORAGE_KEY)).toBe("dark");
    expect(document.documentElement.dataset.limeTheme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("外观快捷面板应使用 navigation 命名空间资源", async () => {
    await changeLimeLocale("en-US");
    const container = mountSidebarContainer({
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    const trigger = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Quick appearance switch"]',
    );

    expect(trigger).not.toBeNull();

    await act(async () => {
      trigger?.click();
      await Promise.resolve();
    });

    const popover = container.querySelector(
      '[data-testid="app-sidebar-appearance-popover"]',
    );
    expect(popover?.textContent).toContain("Appearance");
    expect(popover?.textContent).toContain("Follow system · Dream Blossom");
    expect(popover?.textContent).toContain("Theme");
    expect(popover?.textContent).toContain("Skin");
    expect(popover?.textContent).toContain("Light");
    expect(popover?.textContent).toContain("Dark");
    expect(popover?.textContent).toContain("Follow system");
    expect(popover?.textContent).toContain("Random");
    expect(popover?.textContent).toContain("Ultraman Guardian");
    expect(
      container.querySelector('button[aria-label="Switch theme to Dark"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        'button[aria-label="Switch skin to Ultraman Guardian"]',
      ),
    ).not.toBeNull();
  });

  it("外观弹层的随机皮肤应持久化到一个真实预设", async () => {
    const randomSpy = vi.spyOn(Math, "random").mockReturnValue(0);

    try {
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "new-task",
        } as AgentPageParams,
      });
      await flushEffects(2);

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>('button[aria-label="快速切换外观"]')
          ?.click();
        await Promise.resolve();
      });

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>('button[aria-label="随机切换皮肤"]')
          ?.click();
        await Promise.resolve();
      });

      expect(localStorage.getItem(LIME_COLOR_SCHEME_STORAGE_KEY)).toBe(
        "lime-classic",
      );
      expect(document.documentElement.dataset.limeColorScheme).toBe(
        "lime-classic",
      );
    } finally {
      randomSpy.mockRestore();
    }
  });
});
