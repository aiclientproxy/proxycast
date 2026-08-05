import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildWorkflowRuntimeCapabilityProfile } from "../testing/workflowRuntimeCapabilityProfile";
import type { AppManifest } from "../types";
import type { PluginHostLifecycleSnapshot } from "../host";
import {
  act,
  apiMocks,
  buildReadyState,
  buildReviewResult,
  buildStandaloneState,
  cleanupPluginsPageTest,
  contentFactoryFixture,
  expectInstallReviewDialog,
  flush,
  installedStates,
  LOCAL_APP_DIR,
  openAppDetail,
  REMOVED_MACHINE_PATH,
  renderPage,
  resetPluginsPageTest,
  setInputValue,
  toast,
} from "./PluginsPage.testFixtures";

describe("PluginsPage", () => {
  beforeEach(resetPluginsPageTest);
  afterEach(cleanupPluginsPageTest);

  it("应展示本地安装源和 Cloud catalog，并在审查确认后从本地安装第一个 Plugin", async () => {
    const container = await renderPage();
    await flush();

    expect(container.textContent).toContain("plugin.apps.center.title");
    expect(
      container
        .querySelector('[data-testid="plugins-search"]')
        ?.closest("header")?.textContent,
    ).not.toContain("plugin.apps.center.description");
    expect(
      container
        .querySelector('[data-testid="plugins-search"]')
        ?.getAttribute("placeholder"),
    ).toBe("plugin.apps.center.searchPlaceholder");
    expect(container.textContent).not.toContain(REMOVED_MACHINE_PATH);
    expect(
      container.querySelector(
        '[data-testid="plugins-list-row-content-factory-app"]',
      ),
    ).not.toBeNull();
    expect(apiMocks.listPluginHostLifecycleSnapshots).toHaveBeenCalledTimes(1);
    const fallbackIcon = container.querySelector(
      '[data-testid="plugins-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(fallbackIcon?.getAttribute("src")).toContain("data:image/svg+xml");
    expect(
      decodeURIComponent(fallbackIcon?.getAttribute("src") ?? ""),
    ).toContain("内容工厂");
    expect(
      container.querySelector(
        '[data-testid="plugins-open-detail-content-factory-app"]',
      )?.textContent,
    ).toContain("plugin.apps.center.action.details");
    expect(
      container.querySelector('[data-testid="plugins-detail"]'),
    ).toBeNull();
    expect(container.textContent).toContain("plugin.apps.center.source.cloud");
    expect(container.textContent).toContain(
      "plugin.apps.center.status.installable",
    );

    const installLocal = container.querySelector(
      '[data-testid="plugins-install-local"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installLocal?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.selectLocalPluginDirectory).toHaveBeenCalledWith({
      title: "plugin.apps.localSource.dialogTitle",
    });
    expect(apiMocks.reviewLocalPluginPackage).toHaveBeenCalledWith(
      expect.objectContaining({
        appDir: LOCAL_APP_DIR,
      }),
    );
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
    expectInstallReviewDialog(container);
    expect(
      container.querySelector(
        '[data-testid="plugins-install-review-release-evidence"]',
      ),
    ).toBeNull();

    const confirmInstall = container.querySelector(
      '[data-testid="plugins-install-review-confirm"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      confirmInstall?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.saveInstalledPluginState).toHaveBeenCalledWith({
      state: expect.objectContaining({
        appId: "content-factory-app",
      }),
    });
    expect(
      container.querySelector(
        '[data-testid="plugins-installed-content-factory-app"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("内容工厂");
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("正式入口页面壳和主按钮应接入 Lime 主题变量", async () => {
    const container = await renderPage();
    await flush();

    const page = container.querySelector('[data-testid="plugins-page"]');
    const installLocal = container.querySelector(
      '[data-testid="plugins-install-local"]',
    ) as HTMLButtonElement | null;

    expect(page?.className).toContain("lime-workbench-theme-scope");
    expect(page?.className).toContain("bg-[color:var(--lime-app-bg)]");
    expect(installLocal?.className).toContain(
      "bg-[color:var(--lime-text-strong)]",
    );
    expect(installLocal?.className).toContain("rounded-full");
    expect(
      container
        .querySelector('[data-testid="plugins-list"]')
        ?.closest("section")?.className,
    ).toContain("space-y-4");
    expect(
      container.querySelector('[data-testid="plugins-list"]')?.className,
    ).toContain("lg:grid-cols-3");
    expect(
      container.querySelector(
        '[data-testid="plugins-list-row-content-factory-app"]',
      )?.className,
    ).toContain("min-h-[188px]");
  });

  it("应用中心顶部操作区应避免压缩标题和拆分按钮文案", async () => {
    const container = await renderPage();
    await flush();

    const header = container
      .querySelector('[data-testid="plugins-search"]')
      ?.closest("header");
    expect(header?.className).toContain("xl:flex-row");
    expect(header?.className).not.toContain("lg:flex-row");

    const actionGroup = container.querySelector(
      '[data-testid="plugins-install-local"]',
    )?.parentElement;
    expect(actionGroup?.className).toContain("flex-wrap");

    for (const testId of ["plugins-install-local", "plugins-more-actions"]) {
      const button = container.querySelector(`[data-testid="${testId}"]`);
      expect(button?.className).toContain("shrink-0");
      expect(button?.className).toContain("whitespace-nowrap");
    }

    await act(async () => {
      (
        container.querySelector(
          '[data-testid="plugins-more-actions"]',
        ) as HTMLButtonElement
      )?.click();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugins-more-actions-menu"]'),
    ).not.toBeNull();
    for (const testId of [
      "plugins-open-publish",
      "plugins-open-release-review",
      "plugins-refresh",
    ]) {
      expect(
        container.querySelector(`[data-testid="${testId}"]`),
      ).not.toBeNull();
    }
  });

  it("应从应用中心打开内置发布工作台", async () => {
    const container = await renderPage();
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-publish-workbench"]'),
    ).toBeNull();

    await act(async () => {
      (
        container.querySelector(
          '[data-testid="plugins-more-actions"]',
        ) as HTMLButtonElement
      )?.click();
      await Promise.resolve();
    });
    await flush();
    await act(async () => {
      (
        container.querySelector(
          '[data-testid="plugins-open-publish"]',
        ) as HTMLButtonElement
      )?.click();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-publish-workbench"]'),
    ).not.toBeNull();
  });

  it("应从应用中心打开平台发布审核工作台", async () => {
    const container = await renderPage();
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-review-workbench"]'),
    ).toBeNull();

    await act(async () => {
      (
        container.querySelector(
          '[data-testid="plugins-more-actions"]',
        ) as HTMLButtonElement
      )?.click();
      await Promise.resolve();
    });
    await flush();
    await act(async () => {
      (
        container.querySelector(
          '[data-testid="plugins-open-release-review"]',
        ) as HTMLButtonElement
      )?.click();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-review-workbench"]'),
    ).not.toBeNull();
    expect(apiMocks.listPluginReleaseSubmissions).toHaveBeenCalledTimes(1);
  });

  it("已安装筛选应展示本地已安装应用", async () => {
    installedStates.push(buildReadyState());
    const container = await renderPage({ statusFilter: "installed" });
    await flush();

    expect(
      container.querySelector(
        '[data-testid="plugins-list-row-content-factory-app"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain(
      "plugin.apps.center.pagination.summary:1",
    );
  });

  it("搜索占位文案不应被当成真实搜索词隐藏已安装应用", async () => {
    installedStates.push(buildReadyState());
    const container = await renderPage();
    await flush();

    const search = container.querySelector(
      '[data-testid="plugins-search"]',
    ) as HTMLInputElement | null;
    expect(search).not.toBeNull();

    await act(async () => {
      if (search) {
        setInputValue(search, "plugin.apps.center.searchPlaceholder");
      }
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector(
        '[data-testid="plugins-list-row-content-factory-app"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain(
      "plugin.apps.center.pagination.summary:1",
    );
  });

  it("本地 App 卡片应优先展示 manifest 声明的图标", async () => {
    installedStates.push(
      buildReadyState({
        manifest: {
          ...(contentFactoryFixture as AppManifest),
          install: {
            branding: {
              name: "内容工厂",
              icon: "./assets/icon.svg",
              windowTitle: "内容工厂",
            },
          },
        },
      }),
    );
    const container = await renderPage();
    await flush();

    const icon = container.querySelector(
      '[data-testid="plugins-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(icon?.getAttribute("src")).toBe(
      `asset://${LOCAL_APP_DIR}/assets/icon.svg`,
    );

    await openAppDetail(container);

    const detailIcon = container.querySelector(
      '[data-testid="plugins-detail-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(detailIcon?.getAttribute("src")).toBe(
      `asset://${LOCAL_APP_DIR}/assets/icon.svg`,
    );
  });

  it("本地 App 卡片应解析 manifest presentation.icon 的包内 resources 路径", async () => {
    const manifest = structuredClone(contentFactoryFixture) as AppManifest;
    manifest.presentation = {
      ...(manifest.presentation ?? {}),
      icon: "./resources/icons/icon.svg",
    };
    const manifestInterface = manifest.interface as
      | Record<string, unknown>
      | undefined;
    if (manifestInterface) {
      delete manifestInterface.logo;
      delete manifestInterface.composerIcon;
    }
    installedStates.push(
      buildReadyState({
        manifest,
        profile: buildWorkflowRuntimeCapabilityProfile({
          realAdapterEnabled: true,
          uiRuntimeEnabled: true,
          workerRuntimeEnabled: true,
        }),
      }),
    );

    const container = await renderPage();
    await flush();

    const icon = container.querySelector(
      '[data-testid="plugins-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(icon?.getAttribute("src")).toBe(
      `asset://${LOCAL_APP_DIR}/resources/icons/icon.svg`,
    );
    expect(
      container.querySelector('img[src="./resources/icons/icon.svg"]'),
    ).toBeNull();
  });

  it("应用图标加载失败时应回退到名称 SVG", async () => {
    installedStates.push(
      buildReadyState({
        manifest: {
          ...(contentFactoryFixture as AppManifest),
          presentation: {
            logo: "https://lime.local/missing-content-factory-logo.svg",
          },
        },
      }),
    );

    const container = await renderPage();
    await flush();

    const icon = container.querySelector(
      '[data-testid="plugins-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(icon?.getAttribute("src")).toBe(
      "https://lime.local/missing-content-factory-logo.svg",
    );

    await act(async () => {
      icon?.dispatchEvent(new Event("error"));
      await Promise.resolve();
    });

    const fallbackSrc = icon?.getAttribute("src") ?? "";
    expect(fallbackSrc).toContain("data:image/svg+xml");
    expect(decodeURIComponent(fallbackSrc)).toContain("内容工厂");
  });

  it("详情主按钮应把 workflow 入口发送到 Agent current 主链", async () => {
    const manifest = structuredClone(contentFactoryFixture) as AppManifest;
    manifest.runtimePackage = {
      ...manifest.runtimePackage,
      ui: undefined,
    };
    manifest.entries = [
      {
        key: "content_article_generate",
        kind: "workflow",
        title: "写文章",
        requiredCapabilities: ["lime.agent", "lime.artifacts"],
      },
    ];
    installedStates.push(
      buildReadyState({
        manifest,
        profile: buildWorkflowRuntimeCapabilityProfile({
          realAdapterEnabled: true,
          uiRuntimeEnabled: true,
          workerRuntimeEnabled: true,
        }),
      }),
    );

    const onNavigate = vi.fn();
    const container = await renderPage(undefined, onNavigate);
    await flush();

    await openAppDetail(container);

    const primaryAction = container.querySelector(
      '[data-testid="plugins-detail-primary-action-content-factory-app"]',
    ) as HTMLButtonElement | null;
    expect(primaryAction).not.toBeNull();
    expect(primaryAction?.disabled).toBe(false);

    await act(async () => {
      primaryAction?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "new-task",
        initialUserPrompt: "@写文章 ",
        autoRunInitialPromptOnMount: false,
      }),
    );
    expect(
      container.querySelector('[data-testid="plugins-launch-summary"]'),
    ).toBeNull();
  });

  it("应用中心应展示 Plugin 宿主状态和 Right Surface 合同", async () => {
    const installed = buildReadyState({
      manifest: {
        ...(contentFactoryFixture as AppManifest),
        status: "ready",
        profiles: ["workbench"],
        workbench: {
          profile: "production",
          articleWorkspace: {
            scope: "session",
            primaryObjectKinds: ["articleDraft"],
          },
          productionObjects: [
            {
              kind: "articleDraft",
              title: "文章草稿",
              artifactKind: "markdown_document",
              defaultSurface: "documentCanvas",
              primary: true,
            },
          ],
          objectSurfaces: [
            {
              objectKind: "articleDraft",
              surfaceKind: "documentCanvas",
              renderer: "host_builtin",
            },
          ],
          historyRestore: {
            defaultSurface: "selectedObject",
            restoreSelection: true,
            restoreLayout: true,
          },
        },
      },
    });
    installedStates.push(installed);
    const hostSnapshot = {
      appId: installed.appId,
      displayName: "内容工厂",
      profiles: ["workbench"],
      appCenterStatus: "blocked",
      readinessStatus: "blocked",
      rightSurface: {
        dock: "right",
        physicalDockCount: 1,
        defaultActiveTab: "articleWorkspace",
        supportedTabs: ["articleWorkspace", "file"],
        articleWorkspace: {
          enabled: true,
          objects: [
            {
              kind: "articleDraft",
              title: "文章草稿",
              defaultPane: "documentCanvas",
              artifactKind: "markdown_document",
              primary: true,
            },
          ],
          panes: ["documentCanvas"],
          rendererKinds: ["host_builtin"],
        },
        historyRestore: {
          enabled: true,
          defaultTab: "articleWorkspace",
          defaultPane: "documentCanvas",
          restoreSelection: true,
          restoreLayout: true,
          fallback: "artifactPreview",
        },
      },
      taskRuntime: {
        enabled: true,
        packageRootPath: null,
        workerEntrypoint: null,
        contractPath: null,
        sampleRequestPath: null,
        outputArtifactKind: null,
        taskKinds: [],
        directProviderAccess: false,
        directFilesystemAccess: false,
        blockers: ["TASK_RUNTIME_WORKER_ENTRYPOINT_MISSING"],
        followUps: [],
      },
      functions: [],
      blockers: [
        "CLOUD_REGISTRATION_REQUIRED",
        "TASK_RUNTIME_WORKER_ENTRYPOINT_MISSING",
      ],
      followUps: [],
      generatedAt: installed.updatedAt,
    } satisfies PluginHostLifecycleSnapshot;
    apiMocks.listPluginHostLifecycleSnapshots.mockResolvedValue({
      snapshots: [hostSnapshot],
      issues: [],
    });
    const container = await renderPage();
    await flush();

    expect(
      container.querySelector(
        '[data-testid="plugins-host-status-content-factory-app"]',
      )?.textContent,
    ).toContain("plugin.apps.center.host.status.");
    expect(
      container.querySelector(
        '[data-testid="plugins-host-article-workspace-content-factory-app"]',
      )?.textContent,
    ).toContain("plugin.apps.center.host.articleWorkspace");

    await openAppDetail(container);

    const detail = container.querySelector(
      '[data-testid="plugins-host-lifecycle"]',
    );
    expect(detail?.textContent).toContain("plugin.apps.center.host.title");
    expect(detail?.textContent).toContain(
      "plugin.apps.center.host.rightSurface",
    );
    expect(detail?.textContent).toContain("plugin.apps.center.host.blockers");
    expect(detail?.textContent).toContain(
      "plugin.apps.center.host.issueSummary.title",
    );
    expect(detail?.textContent).toContain(
      "plugin.apps.center.host.issueCategory.cloud",
    );
    expect(detail?.textContent).toContain(
      "plugin.apps.center.host.issueCategory.taskRuntime",
    );
    expect(
      container.querySelector(
        '[data-testid="plugins-host-readiness-category-cloud"]',
      ),
    ).not.toBeNull();
  });

  it("安装确认弹层应展示待安装 App 的 manifest 图标", async () => {
    const stateWithIcon = buildReadyState({
      manifest: {
        ...(contentFactoryFixture as AppManifest),
        install: {
          branding: {
            name: "内容工厂",
            icon: "./assets/icon.svg",
            windowTitle: "内容工厂",
          },
        },
      },
    });
    apiMocks.reviewLocalPluginPackage.mockResolvedValue(
      buildReviewResult(stateWithIcon),
    );

    const container = await renderPage();
    await flush();

    const installLocal = container.querySelector(
      '[data-testid="plugins-install-local"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installLocal?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expectInstallReviewDialog(container);
    const reviewIcon = container.querySelector(
      '[data-testid="plugins-install-review-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(reviewIcon?.getAttribute("src")).toBe(
      `asset://${LOCAL_APP_DIR}/assets/icon.svg`,
    );
  });

  it("应用卡片应兼容 manifest presentation.logo 字段", async () => {
    installedStates.push(
      buildReadyState({
        manifest: {
          ...(contentFactoryFixture as AppManifest),
          presentation: {
            logo: "https://lime.local/content-factory-logo.png",
          },
        },
      }),
    );
    const container = await renderPage();
    await flush();

    const icon = container.querySelector(
      '[data-testid="plugins-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(icon?.getAttribute("src")).toBe(
      "https://lime.local/content-factory-logo.png",
    );

    await openAppDetail(container);

    const detailIcon = container.querySelector(
      '[data-testid="plugins-detail-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(detailIcon?.getAttribute("src")).toBe(
      "https://lime.local/content-factory-logo.png",
    );
  });

  it("本地图标不应把机器绝对路径作为浏览器路径输出到图片 src", async () => {
    installedStates.push(
      buildReadyState({
        manifest: {
          ...(contentFactoryFixture as AppManifest),
          install: {
            branding: {
              name: "内容工厂",
              icon: REMOVED_MACHINE_PATH,
              windowTitle: "内容工厂",
            },
          },
        },
      }),
    );
    const container = await renderPage();
    await flush();

    const icon = container.querySelector(
      '[data-testid="plugins-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(icon?.getAttribute("src")).not.toBe(REMOVED_MACHINE_PATH);
    expect(icon?.getAttribute("src")).toBe(`asset://${REMOVED_MACHINE_PATH}`);
    expect(
      container.querySelector('img[src^="/Users/"], img[src^="C:\\\\"]'),
    ).toBeNull();

    await openAppDetail(container);

    const detailIcon = container.querySelector(
      '[data-testid="plugins-detail-icon-content-factory-app"] img',
    ) as HTMLImageElement | null;
    expect(detailIcon?.getAttribute("src")).not.toBe(REMOVED_MACHINE_PATH);
    expect(detailIcon?.getAttribute("src")).toBe(
      `asset://${REMOVED_MACHINE_PATH}`,
    );
    expect(
      container.querySelector('img[src^="/Users/"], img[src^="C:\\\\"]'),
    ).toBeNull();
  });

  it("详情页面应支持返回卡片列表且不能渲染弹窗语义", async () => {
    const container = await renderPage();
    await flush();

    await openAppDetail(container);

    expect(
      container.querySelector('[data-testid="plugins-detail-overlay"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-detail"]'),
    ).not.toBeNull();
    expect(
      container
        .querySelector('[data-testid="plugins-detail"]')
        ?.getAttribute("role"),
    ).not.toBe("dialog");
    expect(
      container
        .querySelector('[data-testid="plugins-detail"]')
        ?.getAttribute("aria-modal"),
    ).toBeNull();
    expect(container.querySelector('[data-testid="plugins-list"]')).toBeNull();

    const closeDetail = container.querySelector(
      '[data-testid="plugins-close-detail"]',
    ) as HTMLButtonElement | null;
    expect(closeDetail?.getAttribute("aria-label")).toBe(
      "plugin.apps.center.detail.backToList",
    );
    expect(closeDetail?.textContent).toContain(
      "plugin.apps.center.detail.backToList",
    );

    await act(async () => {
      closeDetail?.click();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugins-detail"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-detail-overlay"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="plugins-list-row-content-factory-app"]',
      ),
    ).not.toBeNull();
  });

  it("详情应按本地 manifest 展示两个 Agent 入口、子智能体和技能", async () => {
    installedStates.push(buildReadyState());
    const onNavigate = vi.fn();
    const container = await renderPage(undefined, onNavigate);
    await flush();

    await openAppDetail(container);

    expect(
      container.querySelector('[data-testid="plugins-search"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-status-filter-all"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-launch-target-policy"]'),
    ).toBeNull();
    const agents = container.querySelector(
      '[data-testid="plugins-detail-agents"]',
    );
    expect(agents?.textContent).toContain("plugin.apps.center.detail.agents");
    expect(
      container.querySelectorAll('[data-testid^="plugins-detail-agent-"]'),
    ).toHaveLength(2);
    expect(
      container.querySelector(
        '[data-testid="plugins-detail-agent-content_article_generate"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="plugins-detail-agent-content_factory_generate"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="plugins-detail-subagent-content-researcher"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="plugins-detail-skill-article-writing"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-detail-summary"]'),
    ).not.toBeNull();

    const articleAgent = container.querySelector(
      '[data-testid="plugins-detail-agent-content_article_generate"]',
    ) as HTMLButtonElement | null;
    expect(articleAgent?.tagName).toBe("BUTTON");
    await act(async () => {
      articleAgent?.click();
      await Promise.resolve();
    });
    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        initialUserPrompt: "@写文章 ",
        autoRunInitialPromptOnMount: false,
      }),
    );
  });

  it("详情应兼容旧安装态缺少权限和能力数组", async () => {
    const installed = buildReadyState();
    delete (installed.manifest as Partial<AppManifest>).permissions;
    delete (
      installed.projection as {
        requiredCapabilities?: unknown;
      }
    ).requiredCapabilities;
    installedStates.push(installed);

    const container = await renderPage();
    await flush();

    await openAppDetail(container);

    expect(
      container.querySelector('[data-testid="plugins-detail"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-detail-authorizations"]')
        ?.textContent,
    ).toContain("plugin.apps.center.detail.noAuthorizations");
    expect(
      container.querySelector('[data-testid="plugins-detail-summary"]')
        ?.textContent,
    ).toContain("plugin.apps.center.detail.summary.capabilityCount:0");
  });

  it("取消选择本地目录时不应生成安装审查或写入 repository", async () => {
    apiMocks.selectLocalPluginDirectory.mockResolvedValue(null);
    const container = await renderPage();
    await flush();

    const installLocal = container.querySelector(
      '[data-testid="plugins-install-local"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installLocal?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.reviewLocalPluginPackage).not.toHaveBeenCalled();
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="plugins-install-review"]'),
    ).toBeNull();
  });

  it("本地 package 非法时不应写入 repository", async () => {
    apiMocks.reviewLocalPluginPackage.mockRejectedValue(
      new Error("APP.md invalid"),
    );
    const container = await renderPage();
    await flush();

    const installLocal = container.querySelector(
      '[data-testid="plugins-install-local"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installLocal?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.reviewLocalPluginPackage).toHaveBeenCalled();
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="plugins-install-review"]'),
    ).toBeNull();
  });

  it("本地企业定制包未激活注册码时应展示本地化阻断文案", async () => {
    const error = new Error("raw registration required");
    error.name = "PluginRegistrationRequiredError";
    apiMocks.reviewLocalPluginPackage.mockRejectedValue(error);
    const container = await renderPage();
    await flush();

    const installLocal = container.querySelector(
      '[data-testid="plugins-install-local"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installLocal?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.reviewLocalPluginPackage).toHaveBeenCalled();
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
    expect(toast.error).toHaveBeenCalledWith(
      "plugin.apps.toast.failed",
      expect.objectContaining({
        description: "plugin.apps.registration.localInstallBlocked",
      }),
    );
  });

  it("企业定制 Cloud App 未注册前应阻断安装并提交注册码", async () => {
    apiMocks.getPluginCloudCatalog.mockResolvedValue({
      source: "remote",
      payload: {
        schemaVersion: "plugin-cloud-bootstrap/v1",
        tenantId: "tenant-0001",
        generatedAt: "2026-05-15T00:00:00.000Z",
        apps: [
          {
            appId: "content-factory-app",
            displayName: "内容工厂",
            version: "0.3.0",
            registrationRequired: true,
            registrationState: "required",
            registrationHint: "请输入企业注册码",
            enabled: false,
            disabledReason: "registration required",
            packageUrl: "",
            packageHash: "",
            manifestHash: "",
            capabilityRequirements: {},
            defaultEntries: ["dashboard"],
            policyDefaults: {},
            toolAvailability: [],
          },
        ],
      },
    });

    const container = await renderPage();
    await flush();

    const installCloud = container.querySelector(
      '[data-testid="plugins-install-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    expect(installCloud?.disabled).toBe(true);
    expect(
      container.querySelector(
        '[data-testid="plugins-registration-content-factory-app"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-detail"]'),
    ).toBeNull();

    await openAppDetail(container);

    const input = container.querySelector(
      '[data-testid="plugins-registration-code-content-factory-app"]',
    ) as HTMLInputElement | null;
    await act(async () => {
      if (input) {
        input.value = "CF-REG-2026";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
      await Promise.resolve();
    });

    const submit = container.querySelector(
      '[data-testid="plugins-submit-registration-content-factory-app"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      submit?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.submitPluginRegistrationCode).toHaveBeenCalledWith(
      "content-factory-app",
      "CF-REG-2026",
    );
    expect(apiMocks.installCloudPluginRelease).not.toHaveBeenCalled();

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="plugins-close-detail"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector(
        '[data-testid="plugins-source-state-content-factory-app"]',
      )?.textContent,
    ).toContain("plugin.apps.sourceState.registrationActive");

    const enabledInstallCloud = container.querySelector(
      '[data-testid="plugins-install-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    expect(enabledInstallCloud?.disabled).toBe(false);
    await act(async () => {
      enabledInstallCloud?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.reviewCloudPluginRelease).toHaveBeenCalledWith(
      expect.objectContaining({
        app: expect.objectContaining({
          appId: "content-factory-app",
          registrationState: "active",
        }),
      }),
    );
    expectInstallReviewDialog(container);
  });

  it("已安装旧版 Cloud App 需要重新激活时，主按钮应提示输入激活码而不是假装一键更新", async () => {
    installedStates.push(
      buildReadyState({
        manifest: {
          ...(contentFactoryFixture as AppManifest),
          version: "0.2.0",
        },
      }),
    );
    apiMocks.getPluginCloudCatalog.mockResolvedValue({
      source: "remote",
      payload: {
        schemaVersion: "plugin-cloud-bootstrap/v1",
        tenantId: "tenant-0001",
        generatedAt: "2026-05-15T00:00:00.000Z",
        apps: [
          {
            appId: "content-factory-app",
            displayName: "内容工厂",
            version: "0.3.0",
            registrationRequired: true,
            registrationState: "required",
            registrationHint: "请输入企业注册码",
            enabled: false,
            disabledReason: "registration required",
            packageUrl: "",
            packageHash: "",
            manifestHash: "",
            capabilityRequirements: {},
            defaultEntries: ["dashboard"],
            policyDefaults: {},
            toolAvailability: [],
          },
        ],
      },
    });

    const container = await renderPage();
    await flush();

    expect(container.textContent).toContain(
      "plugin.apps.center.status.registration",
    );
    expect(container.textContent).toContain(
      "plugin.apps.center.action.activate",
    );
    expect(container.textContent).not.toContain(
      "plugin.apps.center.action.updateOneClick",
    );

    const activateButton = container.querySelector(
      '[data-testid="plugins-update-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    expect(activateButton?.disabled).toBe(false);
    await act(async () => {
      activateButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugins-detail"]'),
    ).not.toBeNull();
    expect(toast.error).toHaveBeenCalledWith(
      "plugin.apps.registration.codeRequired",
    );
    expect(apiMocks.reviewCloudPluginRelease).not.toHaveBeenCalled();
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
  });

  it("已安装旧版 Cloud App 可更新时应打开居中安装审查弹窗", async () => {
    installedStates.push(
      buildReadyState({
        manifest: {
          ...(contentFactoryFixture as AppManifest),
          version: "0.2.0",
        },
      }),
    );

    const container = await renderPage();
    await flush();

    expect(container.textContent).toContain("plugin.apps.center.status.update");
    const updateButton = container.querySelector(
      '[data-testid="plugins-update-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    expect(updateButton?.disabled).toBe(false);

    await act(async () => {
      updateButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.reviewCloudPluginRelease).toHaveBeenCalledWith(
      expect.objectContaining({
        app: expect.objectContaining({
          appId: "content-factory-app",
          version: "0.3.0",
        }),
      }),
    );
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
    expectInstallReviewDialog(container);
  });

  it("Cloud App 已满足注册条件时应先生成安装审查再写入 installed state", async () => {
    const container = await renderPage();
    await flush();

    expect(
      container.querySelector(
        '[data-testid="plugins-source-state-content-factory-app"]',
      )?.textContent,
    ).toContain("plugin.apps.sourceState.cloudDiscovered");

    const installCloud = container.querySelector(
      '[data-testid="plugins-install-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installCloud?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.reviewCloudPluginRelease).toHaveBeenCalledWith(
      expect.objectContaining({
        app: expect.objectContaining({
          appId: "content-factory-app",
        }),
        catalogSource: "seeded",
      }),
    );
    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
    expectInstallReviewDialog(container);
    const releaseEvidence = container.querySelector(
      '[data-testid="plugins-install-review-release-evidence"]',
    );
    expect(releaseEvidence).not.toBeNull();
    expect(releaseEvidence?.getAttribute("data-source-kind")).toBe(
      "fetched_package",
    );
    expect(releaseEvidence?.getAttribute("data-catalog-source")).toBe("seeded");
    expect(
      container.querySelector(
        '[data-testid="plugins-install-review-release-evidence-status"]',
      )?.textContent,
    ).toContain("plugin.apps.installReview.releaseEvidence.status.warning");
    expect(container.textContent).toContain(
      "plugin.apps.installReview.releaseEvidence.checkStatus.missing",
    );
    expect(
      container
        .querySelector(
          '[data-testid="plugins-install-review-release-audit-summary"]',
        )
        ?.getAttribute("data-can-install"),
    ).toBe("true");
    expect(container.textContent).toContain(
      "plugin.apps.installReview.releaseEvidence.audit.counts",
    );
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", {
      ...navigator,
      clipboard: { writeText },
    });
    const copyReport = container.querySelector(
      '[data-testid="plugins-install-review-release-audit-copy"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      copyReport?.click();
      await Promise.resolve();
    });
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining(
        "# Plugin Release Audit: content-factory-app@0.3.0",
      ),
    );
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("- Status: warning"),
    );
    expect(
      container.querySelector(
        '[data-testid="plugins-install-review-release-audit-copy-state"]',
      )?.textContent,
    ).toContain(
      "plugin.apps.installReview.releaseEvidence.audit.copyState.copied",
    );

    const confirmInstall = container.querySelector(
      '[data-testid="plugins-install-review-confirm"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      confirmInstall?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.saveInstalledPluginState).toHaveBeenCalledWith({
      state: expect.objectContaining({
        appId: "content-factory-app",
      }),
    });
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("Cloud App 发布包证据阻断时不允许确认安装", async () => {
    apiMocks.reviewCloudPluginRelease.mockImplementationOnce(async () => {
      const state = buildReadyState();
      return buildReviewResult(state, {
        sourceKind: "cloud_release",
        sourceUri:
          "https://lime.local/plugins/content-factory-app/releases/0.3.0/package.zip",
        packageUrl:
          "https://lime.local/plugins/content-factory-app/releases/0.3.0/package.zip",
        releaseEvidence: {
          appId: state.appId,
          version: state.identity.appVersion,
          catalogSource: "remote",
          sourceKind: "fetched_package",
          packageHashDeclared: true,
          manifestHashDeclared: true,
          signatureDeclared: true,
          declaredPackageHash: state.identity.packageHash,
          declaredManifestHash: state.identity.manifestHash,
          actualPackageHash: state.identity.packageHash,
          actualManifestHash: state.identity.manifestHash,
          packageHashMatched: true,
          manifestHashMatched: true,
          signatureRef: "sigstore:content-factory-app@0.3.0",
          signaturePolicy: "required",
          signatureVerificationStatus: "declared",
          packageVerificationStatus: "verified",
          status: "blocked",
          blockerCodes: ["signature_unverified"],
          warningCodes: [],
        },
      });
    });

    const container = await renderPage();
    await flush();

    const installCloud = container.querySelector(
      '[data-testid="plugins-install-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      installCloud?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    const confirmInstall = container.querySelector(
      '[data-testid="plugins-install-review-confirm"]',
    ) as HTMLButtonElement | null;
    expect(confirmInstall?.disabled).toBe(true);
    expect(
      container.querySelector(
        '[data-testid="plugins-install-review-release-evidence-status"]',
      )?.textContent,
    ).toContain("plugin.apps.installReview.releaseEvidence.status.blocked");
    expect(container.textContent).toContain(
      "plugin.apps.installReview.releaseEvidence.checkStatus.unverified",
    );
    expect(
      container
        .querySelector(
          '[data-testid="plugins-install-review-release-audit-summary"]',
        )
        ?.getAttribute("data-can-install"),
    ).toBe("false");
    expect(
      container
        .querySelector(
          '[data-testid="plugins-install-review-release-audit-signature"]',
        )
        ?.getAttribute("data-issue-codes"),
    ).toBe("CLOUD_SIGNATURE_UNVERIFIED");

    await act(async () => {
      confirmInstall?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.saveInstalledPluginState).not.toHaveBeenCalled();
  });

  it("Cloud App 缺少 hash 时应阻断安装审查", async () => {
    apiMocks.getPluginCloudCatalog.mockResolvedValue({
      source: "remote",
      payload: {
        schemaVersion: "plugin-cloud-bootstrap/v1",
        tenantId: "tenant-0001",
        generatedAt: "2026-05-15T00:00:00.000Z",
        apps: [
          {
            appId: "content-factory-app",
            displayName: "内容工厂",
            version: "0.3.0",
            registrationRequired: false,
            registrationState: "not_required",
            enabled: true,
            packageUrl:
              "https://lime.local/plugins/content-factory-app/releases/0.3.0/package.zip",
            packageHash: "",
            manifestHash: "",
            capabilityRequirements: {},
            defaultEntries: ["dashboard"],
            policyDefaults: {},
            toolAvailability: [],
          },
        ],
      },
    });
    const container = await renderPage();
    await flush();

    expect(
      container.querySelector(
        '[data-testid="plugins-source-state-content-factory-app"]',
      )?.textContent,
    ).toContain("plugin.apps.sourceState.hashMissing");
    const installCloud = container.querySelector(
      '[data-testid="plugins-install-cloud-content-factory-app"]',
    ) as HTMLButtonElement | null;
    expect(installCloud?.disabled).toBe(true);
    expect(apiMocks.reviewCloudPluginRelease).not.toHaveBeenCalled();
  });

  it("已安装 App 应支持启动 UI entry、禁用/启用、delete-data 阻断和 keep-data 卸载", async () => {
    installedStates.push(
      buildReadyState({
        profile: buildWorkflowRuntimeCapabilityProfile({
          realAdapterEnabled: true,
          uiRuntimeEnabled: true,
          workerRuntimeEnabled: true,
        }),
      }),
    );
    const container = await renderPage();
    await flush();

    await openAppDetail(container);

    const launchButton = container.querySelector(
      '[data-testid="plugins-launch-entry-dashboard"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      launchButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugins-launch-summary"]')
        ?.textContent,
    ).toContain("ui:项目首页:/dashboard");
    const lifecycleActions = container.querySelector(
      '[data-testid="plugins-lifecycle-actions"]',
    );
    expect(lifecycleActions).not.toBeNull();
    expect(
      lifecycleActions?.querySelector(
        '[data-testid="plugins-uninstall-keep-data"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-more-info-content"]'),
    ).toBeNull();

    const disableButton = container.querySelector(
      '[data-testid="plugins-disable"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      disableButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.setPluginDisabled).toHaveBeenCalledWith({
      appId: "content-factory-app",
      disabled: true,
      updatedAt: expect.any(String),
    });
    expect(container.textContent).toContain(
      "plugin.apps.center.status.disabled",
    );

    const enableButton = container.querySelector(
      '[data-testid="plugins-enable"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      enableButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.setPluginDisabled).toHaveBeenLastCalledWith({
      appId: "content-factory-app",
      disabled: false,
      updatedAt: expect.any(String),
    });

    const uninstallButton = container.querySelector(
      '[data-testid="plugins-uninstall-delete-data"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      uninstallButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.previewPluginUninstall).toHaveBeenCalledWith({
      appId: "content-factory-app",
      mode: "delete-data",
    });
    expect(
      container.querySelector('[data-testid="plugins-uninstall-preview"]')
        ?.textContent,
    ).toContain("delete:2 retain:1");
    expect(
      container.querySelector('[data-testid="plugins-cleanup-evidence"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-residual-audit"]')
        ?.textContent,
    ).toContain("plugin.lab.manager.evidence.residual.pendingDeletion");
    const evidenceJson = container.querySelector(
      '[data-testid="plugins-cleanup-evidence-json"]',
    )?.textContent;
    expect(evidenceJson).toContain('"namespaceKind": "overlay"');
    expect(evidenceJson).toContain('"category": "secret-ref"');
    expect(evidenceJson).not.toContain("sk-secret-value");

    const confirmButton = container.querySelector(
      '[data-testid="plugins-uninstall-confirm"]',
    ) as HTMLButtonElement | null;
    const phrase = container.querySelector(
      '[data-testid="plugins-delete-data-confirmation-phrase"]',
    )?.textContent;
    const confirmationInput = container.querySelector(
      '[data-testid="plugins-delete-data-confirmation-input"]',
    ) as HTMLInputElement | null;
    expect(phrase).toContain("DELETE_PLUGIN_DATA content-factory-app");
    expect(confirmationInput?.disabled).toBe(true);
    expect(confirmButton?.disabled).toBe(true);
    expect(
      container.querySelector(
        '[data-testid="plugins-delete-data-current-phase-gate"]',
      )?.textContent,
    ).toContain("plugin.apps.uninstallPreview.deleteDataGate.dryRunOnly");
    expect(
      container.querySelector(
        '[data-testid="plugins-delete-data-confirmation-status"]',
      )?.textContent,
    ).toContain("plugin.apps.uninstallPreview.deleteDataGate.dryRunOnly");
    const switchKeepDataButton = container.querySelector(
      '[data-testid="plugins-uninstall-switch-keep-data"]',
    ) as HTMLButtonElement | null;
    expect(switchKeepDataButton?.disabled).toBe(false);
    expect(switchKeepDataButton?.textContent).toContain(
      "plugin.apps.action.uninstallKeepData",
    );
    const readyConfirmButton = container.querySelector(
      '[data-testid="plugins-uninstall-confirm"]',
    ) as HTMLButtonElement | null;
    expect(readyConfirmButton?.disabled).toBe(true);

    await act(async () => {
      readyConfirmButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.uninstallPlugin).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="plugins-detail"]')?.textContent,
    ).toContain("plugin.apps.center.status.installed");

    await act(async () => {
      switchKeepDataButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.previewPluginUninstall).toHaveBeenLastCalledWith({
      appId: "content-factory-app",
      mode: "keep-data",
    });
    expect(
      container.querySelector('[data-testid="plugins-uninstall-preview"]')
        ?.textContent,
    ).toContain("delete:1 retain:2");

    const keepDataConfirmButton = container.querySelector(
      '[data-testid="plugins-uninstall-confirm"]',
    ) as HTMLButtonElement | null;
    expect(keepDataConfirmButton?.disabled).toBe(false);
    await act(async () => {
      keepDataConfirmButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.uninstallPlugin).toHaveBeenLastCalledWith({
      appId: "content-factory-app",
      mode: "keep-data",
    });
    const appRowAfterUninstall = container.querySelector(
      '[data-testid="plugins-list-row-content-factory-app"]',
    );
    expect(appRowAfterUninstall?.textContent).toContain(
      "plugin.apps.center.status.installable",
    );
    expect(
      container.querySelector('[data-testid="plugins-uninstall-preview"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugins-launch-summary"]')
        ?.textContent,
    ).toContain("plugin.apps.uninstall.completed");
  });

  it("standalone App 默认只打开独立窗口，不投递当前 Claw Right Surface", async () => {
    installedStates.push(buildStandaloneState());
    const container = await renderPage(undefined, undefined, {
      workspaceId: "workspace-main",
      sessionId: "session-main",
    });
    await flush();

    await openAppDetail(container);

    const launchButton = container.querySelector(
      '[data-testid="plugins-launch-entry-dashboard"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      launchButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.launchPluginShell).toHaveBeenCalledTimes(1);
    expect(apiMocks.requestWorkspaceRightSurface).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="plugins-launch-summary"]')
        ?.textContent,
    ).toContain("shell:项目首页:http://127.0.0.1:4199/dashboard");
  });

  it("选择当前 Claw 右侧后才把 standalone App surface 投递到 Right Surface", async () => {
    installedStates.push(buildStandaloneState());
    const container = await renderPage(undefined, undefined, {
      workspaceId: "workspace-main",
      sessionId: "session-main",
    });
    await flush();

    const rightSurfaceButton = container.querySelector(
      '[data-testid="plugins-launch-target-right-surface"]',
    ) as HTMLButtonElement | null;
    expect(rightSurfaceButton?.disabled).toBe(false);
    await act(async () => {
      rightSurfaceButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(rightSurfaceButton?.getAttribute("aria-pressed")).toBe("true");

    await openAppDetail(container);

    const launchButton = container.querySelector(
      '[data-testid="plugins-launch-entry-dashboard"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      launchButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.launchPluginShell).toHaveBeenCalledTimes(1);
    expect(apiMocks.requestWorkspaceRightSurface).toHaveBeenCalledWith(
      expect.objectContaining({
        workspaceId: "workspace-main",
        sessionId: "session-main",
        surfaceKind: "appSurface",
        origin: "plugin_center",
        reason: "plugin_shell_surface_ready",
        priority: "foreground",
        candidateId: "content-factory-app",
        metadata: expect.objectContaining({
          appId: "content-factory-app",
          title: "内容工厂",
          surface: expect.objectContaining({
            entryUrl: "http://127.0.0.1:4199/dashboard",
            containerId: "plugin-shell-content-factory-app-standalone",
            supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
            embedding: expect.objectContaining({
              rightSurfaceDock: true,
              iframe: false,
              browserView: false,
            }),
          }),
        }),
      }),
      {},
    );
    expect(
      container.querySelector('[data-testid="plugins-launch-summary"]')
        ?.textContent,
    ).toContain("shell:项目首页:http://127.0.0.1:4199/dashboard");
  });

  it("选择多个 Claw 目标时应把 standalone App surface 投递到选中的会话", async () => {
    installedStates.push(buildStandaloneState());
    const container = await renderPage(
      undefined,
      undefined,
      {
        workspaceId: "workspace-main",
        sessionId: "session-main",
        label: "主会话",
      },
      [
        {
          workspaceId: "workspace-main",
          sessionId: "session-main",
          label: "主会话",
        },
        {
          workspaceId: "workspace-main",
          sessionId: "session-review",
          label: "复盘会话",
        },
      ],
    );
    await flush();

    const rightSurfaceButton = container.querySelector(
      '[data-testid="plugins-launch-target-right-surface"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      rightSurfaceButton?.click();
      await Promise.resolve();
    });
    await flush();

    const targetSelect = container.querySelector(
      '[data-testid="plugins-launch-target-select"]',
    ) as HTMLSelectElement | null;
    expect(targetSelect).not.toBeNull();
    expect(targetSelect?.options).toHaveLength(2);
    await act(async () => {
      if (targetSelect?.options[1]) {
        targetSelect.value = targetSelect.options[1].value;
        targetSelect.dispatchEvent(new Event("change", { bubbles: true }));
      }
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugins-launch-target-current"]')
        ?.textContent,
    ).toContain("复盘会话");

    await openAppDetail(container);

    const launchButton = container.querySelector(
      '[data-testid="plugins-launch-entry-dashboard"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      launchButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await flush();

    expect(apiMocks.requestWorkspaceRightSurface).toHaveBeenCalledWith(
      expect.objectContaining({
        workspaceId: "workspace-main",
        sessionId: "session-review",
        surfaceKind: "appSurface",
      }),
      {},
    );
  });

  it("没有 Claw target 时应禁用右侧打开选项", async () => {
    const container = await renderPage();
    await flush();

    const rightSurfaceButton = container.querySelector(
      '[data-testid="plugins-launch-target-right-surface"]',
    ) as HTMLButtonElement | null;
    expect(rightSurfaceButton?.disabled).toBe(true);
    expect(rightSurfaceButton?.getAttribute("aria-pressed")).toBe("false");
    expect(
      container.querySelector(
        '[data-testid="plugins-launch-target-unavailable"]',
      )?.textContent,
    ).toContain("plugin.apps.launchTarget.rightSurfaceUnavailable");
  });
});
