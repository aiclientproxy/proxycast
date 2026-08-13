import { Suspense, useEffect } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  AgentPageParams,
  Page,
  PageParams,
  SettingsPageParams,
} from "@/types/page";
import { SettingsTabs } from "@/types/settings";
import { changeLimeLocale } from "@/i18n/createI18n";
import { AppPageContent } from "./AppPageContent";

const latestAgentChatProps = vi.hoisted(() => ({
  value: null as Record<string, unknown> | null,
}));
const agentChatLifecycle = vi.hoisted(() => ({ mounts: 0, unmounts: 0 }));
const latestSkillsWorkspaceProps = vi.hoisted(() => ({
  value: null as Record<string, unknown> | null,
}));
const latestKnowledgePageProps = vi.hoisted(() => ({
  value: null as Record<string, unknown> | null,
}));
const latestPluginsProps = vi.hoisted(() => ({
  value: null as Record<string, unknown> | null,
}));
const latestExpertPlazaProps = vi.hoisted(() => ({
  value: null as Record<string, unknown> | null,
}));
const latestSettingsPageProps = vi.hoisted(() => ({
  value: null as Record<string, unknown> | null,
}));
const pluginsLifecycle = vi.hoisted(() => ({ mounts: 0 }));
vi.mock("./agent/chat", () => ({
  AgentChatPage: (props: Record<string, unknown>) => {
    latestAgentChatProps.value = props;

    useEffect(() => {
      agentChatLifecycle.mounts += 1;

      return () => {
        agentChatLifecycle.unmounts += 1;
      };
    }, []);

    return <div data-testid="agent-chat-page" />;
  },
}));

vi.mock("./channels/ImConfigPage", () => ({
  ImConfigPage: () => <div data-testid="im-config-page" />,
}));

vi.mock("./settings-v2", () => ({
  SettingsPageV2: (props: Record<string, unknown>) => {
    latestSettingsPageProps.value = props;
    return <div data-testid="settings-page" />;
  },
}));

vi.mock("./automation", () => ({
  AutomationPage: () => <div data-testid="automation-page" />,
}));

vi.mock("./skills", () => ({
  SkillsWorkspacePage: (props: Record<string, unknown>) => {
    latestSkillsWorkspaceProps.value = props;
    return <div data-testid="skills-workspace-page" />;
  },
}));

vi.mock("@/features/knowledge", () => ({
  KnowledgePage: (props: Record<string, unknown>) => {
    latestKnowledgePageProps.value = props;
    return <div data-testid="knowledge-page" />;
  },
}));

vi.mock("@/features/plugin/ui/PluginCatalogPage", () => ({
  PluginCatalogPage: (props: Record<string, unknown>) => {
    latestPluginsProps.value = props;
    useEffect(() => {
      pluginsLifecycle.mounts += 1;
    }, []);

    return <div data-testid="plugins-page" />;
  },
}));

vi.mock("./experts", () => ({
  ExpertPlazaPage: (props: Record<string, unknown>) => {
    latestExpertPlazaProps.value = props;
    return <div data-testid="expert-plaza-page" />;
  },
}));

interface MountedContent {
  container: HTMLDivElement;
  root: Root;
}

const mountedContents: MountedContent[] = [];

interface RenderContentOptions {
  currentPage: Page;
  pageParams?: PageParams;
  requestedPage?: Page;
  requestedPageParams?: PageParams;
  onNavigate?: (page: Page, params?: PageParams) => void;
  onAgentSessionChange?: (sessionId: string | null) => void;
}

function renderContentWithNavigationState(options: RenderContentOptions) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const renderWithProps = (nextOptions: RenderContentOptions) => {
    act(() => {
      root.render(
        <Suspense fallback={<div data-testid="page-loading" />}>
          <AppPageContent
            currentPage={nextOptions.currentPage}
            pageParams={nextOptions.pageParams ?? {}}
            requestedPage={nextOptions.requestedPage}
            requestedPageParams={nextOptions.requestedPageParams}
            onNavigate={nextOptions.onNavigate ?? vi.fn()}
            onAgentHasMessagesChange={vi.fn()}
            onAgentSessionChange={nextOptions.onAgentSessionChange}
          />
        </Suspense>,
      );
    });
  };

  renderWithProps(options);

  mountedContents.push({ container, root });
  return {
    container,
    rerender(nextOptions: RenderContentOptions) {
      renderWithProps(nextOptions);
    },
  };
}

function renderContent(currentPage: Page, pageParams: PageParams = {}) {
  const rendered = renderContentWithNavigationState({
    currentPage,
    pageParams,
  });

  return {
    container: rendered.container,
    rerender(nextCurrentPage: Page, nextPageParams: PageParams = {}) {
      rendered.rerender({
        currentPage: nextCurrentPage,
        pageParams: nextPageParams,
      });
    },
  };
}

function expectTestId(container: HTMLElement, testId: string) {
  expect(container.querySelector(`[data-testid="${testId}"]`)).not.toBeNull();
}

async function flushEffects(times = 6) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("AppPageContent", () => {
  beforeEach(async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    await changeLimeLocale("zh-CN");
    latestAgentChatProps.value = null;
    agentChatLifecycle.mounts = 0;
    agentChatLifecycle.unmounts = 0;
    latestSkillsWorkspaceProps.value = null;
    latestKnowledgePageProps.value = null;
    latestExpertPlazaProps.value = null;
    latestSettingsPageProps.value = null;
    pluginsLifecycle.mounts = 0;
  });

  afterEach(() => {
    while (mountedContents.length > 0) {
      const mounted = mountedContents.pop();
      if (!mounted) {
        continue;
      }

      act(() => {
        mounted.root.unmount();
      });
      mounted.container.remove();
    }

    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("agent 页面应把 initialSiteSkillLaunch 透传给 AgentChatPage", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "claw",
      projectId: "project-1",
      contentId: "content-1",
      theme: "general",
      newChatAt: 1234567890,
      initialSiteSkillLaunch: {
        adapterName: "linux-do/categories",
        args: {
          limit: 10,
        },
        autoRun: true,
        profileKey: "attached-linux-do",
        targetId: "tab-linux-do",
        requireAttachedSession: true,
        saveTitle: "Linux.do 分类扫描",
        skillTitle: "Linux.do 分类扫描",
      },
    };

    renderContent("agent", pageParams);
    await flushEffects();

    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-1",
      contentId: "content-1",
      agentEntry: "claw",
      theme: "general",
      newChatAt: 1234567890,
      initialSiteSkillLaunch: {
        adapterName: "linux-do/categories",
        args: {
          limit: 10,
        },
        autoRun: true,
        profileKey: "attached-linux-do",
        targetId: "tab-linux-do",
        requireAttachedSession: true,
        saveTitle: "Linux.do 分类扫描",
        skillTitle: "Linux.do 分类扫描",
      },
    });
  });

  it("agent 页面应把 initialProjectFileOpenTarget 透传给 AgentChatPage", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "claw",
      projectId: "project-2",
      contentId: "content-2",
      theme: "general",
      initialProjectFileOpenTarget: {
        relativePath: "exports/x-article/google-cloud/index.md",
        requestKey: 20260408,
      },
    };

    renderContent("agent", pageParams);
    await flushEffects();

    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-2",
      contentId: "content-2",
      initialProjectFileOpenTarget: {
        relativePath: "exports/x-article/google-cloud/index.md",
        requestKey: 20260408,
      },
    });
  });

  it("AgentChatPage 的 onSessionChange 应回调到上层 onAgentSessionChange", async () => {
    const onAgentSessionChange = vi.fn();

    renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-route",
      } satisfies AgentPageParams,
      onAgentSessionChange,
    });
    await flushEffects();

    const onSessionChange = latestAgentChatProps.value?.onSessionChange;
    expect(typeof onSessionChange).toBe("function");

    await act(async () => {
      (onSessionChange as (sessionId: string | null) => void)(" session-live ");
      await Promise.resolve();
    });

    expect(onAgentSessionChange).toHaveBeenCalledWith("session-live");
    expect(onAgentSessionChange).toHaveBeenCalledTimes(1);
  });

  it("agent 页面应透传专家 Agent 身份，重复恢复同一专家不触发新 key", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "claw",
      projectId: "default",
      theme: "general",
      initialSessionId: "session-expert-1",
      expertAgentLaunch: {
        tenantId: "tenant-0001",
        expertId: "marketing-strategist",
        releaseId: "expert-release-0001",
        agentInstanceKey:
          "tenant-0001:marketing-strategist:expert-release-0001",
        launchMode: "new_thread",
      },
    };

    const mounted = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams,
    });
    await flushEffects();
    expect(agentChatLifecycle.mounts).toBe(1);
    expect(latestAgentChatProps.value).toMatchObject({
      initialSessionId: "session-expert-1",
      expertAgentLaunch: pageParams.expertAgentLaunch,
    });

    mounted.rerender({
      currentPage: "agent",
      pageParams: { ...pageParams },
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
  });

  it("agent 页面应把 initialSessionId 透传给 AgentChatPage", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "claw",
      initialSessionId: "session-sceneapp-1",
      entryBannerMessage: "已恢复 SceneApp 对应会话。",
    };

    renderContent("agent", pageParams);
    await flushEffects();

    expect(latestAgentChatProps.value).toMatchObject({
      agentEntry: "claw",
      initialSessionId: "session-sceneapp-1",
      entryBannerMessage: "已恢复 SceneApp 对应会话。",
    });
  });

  it("agent 页面切换 initialSessionId 应复用 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
        theme: "general",
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        theme: "general",
        initialSessionId: "session-opened-from-sidebar",
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      agentEntry: "claw",
      projectId: "project-1",
      initialSessionId: "session-opened-from-sidebar",
    });
  });

  it("agent 页面切换 agentEntry 应复用 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
        theme: "general",
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      agentEntry: "new-task",
      showChatPanel: false,
    });

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        theme: "general",
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      agentEntry: "claw",
      showChatPanel: true,
    });
  });

  it("agent 页面切换项目、文稿或主题边界也应复用 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        contentId: "content-1",
        theme: "general",
        lockTheme: false,
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-1",
      contentId: "content-1",
      theme: "general",
      lockTheme: false,
    });

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-2",
        contentId: "content-2",
        theme: "writing",
        lockTheme: true,
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-2",
      contentId: "content-2",
      theme: "writing",
      lockTheme: true,
    });
  });

  it("agent 页面 pending navigation 被 currentPage 追平时不应重建 AgentChatPage 实例", async () => {
    const agentParams = {
      agentEntry: "claw",
      projectId: "project-pending",
      theme: "general",
      initialSessionId: "session-pending-navigation",
    } satisfies AgentPageParams;
    const rendered = renderContentWithNavigationState({
      currentPage: "settings",
      pageParams: { tab: "general" },
      requestedPage: "agent",
      requestedPageParams: agentParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-pending",
      initialSessionId: "session-pending-navigation",
    });

    rendered.rerender({
      currentPage: "agent",
      pageParams: agentParams,
      requestedPage: "agent",
      requestedPageParams: agentParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-pending",
      initialSessionId: "session-pending-navigation",
    });
  });

  it("agent 页面切换 newChatAt 不应重建 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
        theme: "general",
        newChatAt: 111,
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      newChatAt: 111,
    });

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
        theme: "general",
        newChatAt: 222,
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      newChatAt: 222,
    });
  });

  it("agent 页面应把 initialPendingServiceSkillLaunch 透传给 AgentChatPage", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "new-task",
      projectId: "project-3",
      theme: "general",
      initialPendingServiceSkillLaunch: {
        skillId: "service-skill-1",
        requestKey: 20260409,
        initialSlotValues: {
          article_source: "参考摘要",
          target_duration: "90 秒",
        },
        prefillHint: "已根据最近一次创作自动预填。",
      },
    };

    renderContent("agent", pageParams);
    await flushEffects();

    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-3",
      agentEntry: "new-task",
      initialPendingServiceSkillLaunch: {
        skillId: "service-skill-1",
        requestKey: 20260409,
        initialSlotValues: {
          article_source: "参考摘要",
          target_duration: "90 秒",
        },
        prefillHint: "已根据最近一次创作自动预填。",
      },
    });
  });

  it("agent 页面应把 initialInputCapability 透传给 AgentChatPage", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "new-task",
      projectId: "project-3",
      theme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "local:writer",
          skillName: "写作助手",
        },
        requestKey: 20260418,
      },
    };

    renderContent("agent", pageParams);
    await flushEffects();

    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-3",
      agentEntry: "new-task",
      initialInputCapability: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "local:writer",
          skillName: "写作助手",
        },
        requestKey: 20260418,
      },
    });
  });

  it("agent 页面切换 initialInputCapability 时应复用 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-3",
        theme: "general",
        initialInputCapability: {
          capabilityRoute: {
            kind: "installed_skill",
            skillKey: "local:writer",
            skillName: "写作助手",
          },
          requestKey: 20260418,
        },
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-3",
        theme: "general",
        initialInputCapability: {
          capabilityRoute: {
            kind: "installed_skill",
            skillKey: "local:analyst",
            skillName: "分析助手",
          },
          requestKey: 20260419,
        },
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      initialInputCapability: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "local:analyst",
          skillName: "分析助手",
        },
        requestKey: 20260419,
      },
    });
  });

  it("agent 页面切换 Knowledge 协同资料时应复用 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-knowledge",
        initialKnowledgePackSelection: {
          enabled: true,
          packName: "content-calendar",
          workingDir: "/tmp/lime-project",
          label: "内容运营资料",
          status: "ready",
          companionPacks: [
            {
              name: "founder-persona",
              activation: "implicit",
            },
          ],
        },
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "claw",
        projectId: "project-knowledge",
        initialKnowledgePackSelection: {
          enabled: true,
          packName: "content-calendar",
          workingDir: "/tmp/lime-project",
          label: "内容运营资料",
          status: "ready",
          companionPacks: [
            {
              name: "founder-persona",
              activation: "implicit",
            },
            {
              name: "campaign-plan",
              activation: "explicit",
            },
          ],
        },
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      initialKnowledgePackSelection: {
        companionPacks: [
          {
            name: "founder-persona",
            activation: "implicit",
          },
          {
            name: "campaign-plan",
            activation: "explicit",
          },
        ],
      },
    });
  });

  it("agent 页面切换结果模板 initialInputCapability 时也应复用 AgentChatPage 实例", async () => {
    const rendered = renderContentWithNavigationState({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-3",
        theme: "general",
        initialInputCapability: {
          capabilityRoute: {
            kind: "curated_task",
            taskId: "daily-trend-briefing",
            taskTitle: "每日趋势摘要",
            prompt:
              "请先给我做一版每日趋势摘要：围绕当前主题梳理最近值得关注的趋势、热点内容方向、代表案例、用户正在关心的问题，以及最值得立即开工的 3 个选题。",
          },
          requestKey: 20260418,
        },
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);

    rendered.rerender({
      currentPage: "agent",
      pageParams: {
        agentEntry: "new-task",
        projectId: "project-3",
        theme: "general",
        initialInputCapability: {
          capabilityRoute: {
            kind: "curated_task",
            taskId: "social-post-starter",
            taskTitle: "内容主稿生成",
            prompt:
              "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
          },
          requestKey: 20260419,
        },
      } satisfies AgentPageParams,
    });
    await flushEffects();

    expect(agentChatLifecycle.mounts).toBe(1);
    expect(agentChatLifecycle.unmounts).toBe(0);
    expect(latestAgentChatProps.value).toMatchObject({
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "social-post-starter",
          taskTitle: "内容主稿生成",
          prompt:
            "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
        },
        requestKey: 20260419,
      },
    });
  });

  it("agent 页面应把做法执行摘要与自动发送 metadata 透传给 AgentChatPage", async () => {
    const pageParams: AgentPageParams = {
      agentEntry: "claw",
      projectId: "project-sceneapp",
      theme: "general",
      initialSceneAppExecutionSummary: {
        sceneappId: "story-video-suite",
        title: "短视频编排",
        summary: "把线框图、脚本与配乐压成同一条结果链。",
        businessLabel: "内容闭环",
        typeLabel: "多模态组合",
        executionChainLabel: "做法 -> 生成 -> Project Pack",
        deliveryContractLabel: "Project Pack",
        planningStatusLabel: "已就绪",
        planningSummary: "当前可直接进入生成。",
        activeLayers: [{ key: "skill", label: "Skill" }],
        referenceCount: 1,
        referenceItems: [
          {
            key: "ref-1",
            label: "品牌 KV",
            sourceLabel: "记忆参考",
            contentTypeLabel: "图片",
            selected: true,
          },
        ],
        tasteSummary: "偏好克制的科技蓝。",
        feedbackSummary: "最近反馈要求减少文案堆叠。",
        projectPackPlan: {
          packKindLabel: "短视频项目包",
          completionStrategyLabel: "按必含部件判断整包完成度",
          viewerLabel: "结果包查看器",
          primaryPart: "任务简报",
          requiredParts: [{ key: "brief", label: "任务简报" }],
          notes: ["完整度将按 1 个必含部件判断。"],
        },
        scorecardProfileRef: "story-video-scorecard",
        scorecardMetricKeys: [
          { key: "delivery_readiness", label: "交付就绪度" },
        ],
        scorecardFailureSignals: [
          { key: "publish_stalled", label: "发布卡点" },
        ],
        notes: ["已装配 1 条参考素材。"],
      },
      initialAutoSendRequestMetadata: {
        harness: {
          service_scene_launch: {
            kind: "local_service_skill",
            service_scene_run: {
              scene_key: "story-video-suite",
            },
          },
        },
      },
    };

    renderContent("agent", pageParams);
    await flushEffects();

    expect(latestAgentChatProps.value).toMatchObject({
      projectId: "project-sceneapp",
      initialSceneAppExecutionSummary: expect.objectContaining({
        sceneappId: "story-video-suite",
        title: "短视频编排",
      }),
      initialAutoSendRequestMetadata: {
        harness: {
          service_scene_launch: {
            kind: "local_service_skill",
            service_scene_run: {
              scene_key: "story-video-suite",
            },
          },
        },
      },
    });
  });

  it("channels 页面应渲染 IM 配置页", async () => {
    const { container } = renderContent("channels");
    await flushEffects();

    expectTestId(container, "im-config-page");
  });

  it("settings 页面应渲染设置页入口", async () => {
    const { container } = renderContent("settings");
    await flushEffects();

    expectTestId(container, "settings-page");
  });

  it("settings 页面应透传 executionPolicyFocus 给设置页", async () => {
    const pageParams: SettingsPageParams = {
      tab: SettingsTabs.ExecutionPolicy,
      executionPolicyFocus: {
        section: "network",
        ruleId: "download-block",
        target: "url",
        value: "https://example.com/install.sh",
        reasonCode: "request_download_url",
      },
    };

    renderContent("settings", pageParams);
    await flushEffects();

    expect(latestSettingsPageProps.value).toMatchObject({
      initialTab: "execution-policy",
      initialExecutionPolicyFocus: {
        section: "network",
        ruleId: "download-block",
        target: "url",
        value: "https://example.com/install.sh",
        reasonCode: "request_download_url",
      },
    });
  });

  it("settings 页面应透传 providerFocus 给设置页", async () => {
    const pageParams: SettingsPageParams = {
      tab: SettingsTabs.Providers,
      providerView: "settings",
      providerFocus: {
        providerId: "custom-coder",
        modelId: "coder-large",
        reasonCode: "missing_enabled_api_key",
        recoveryAction: "add_enabled_api_key",
      },
    };

    renderContent("settings", pageParams);
    await flushEffects();

    expect(latestSettingsPageProps.value).toMatchObject({
      initialTab: "providers",
      initialProviderView: "settings",
      initialProviderFocus: {
        providerId: "custom-coder",
        modelId: "coder-large",
        reasonCode: "missing_enabled_api_key",
        recoveryAction: "add_enabled_api_key",
      },
    });
  });

  it("skills 页面应把技能草稿参数透传给 SkillsWorkspacePage", async () => {
    const { container } = renderContent("skills", {
      initialScaffoldDraft: {
        target: "project",
        directory: "saved-skill-demo",
        name: "结果沉淀技能",
        description: "沉淀自一次成功结果",
      },
      initialScaffoldRequestKey: 20260408,
    });
    await flushEffects();

    expect(latestSkillsWorkspaceProps.value).toMatchObject({
      pageParams: {
        initialScaffoldDraft: {
          target: "project",
          directory: "saved-skill-demo",
          name: "结果沉淀技能",
          description: "沉淀自一次成功结果",
        },
        initialScaffoldRequestKey: 20260408,
      },
    });
    expect(
      container.querySelector('[data-testid="plugin-workspace-tabs"]'),
    ).not.toBeNull();
    expect(
      container
        .querySelector('[data-testid="plugin-workspace-tab-skills"]')
        ?.getAttribute("aria-selected"),
    ).toBe("true");
  });

  it("插件工作台应以插件、Skills、专家三个 Tab 切换现有页面", async () => {
    const onNavigate = vi.fn();
    const { container } = renderContentWithNavigationState({
      currentPage: "plugins",
      onNavigate,
    });
    await flushEffects();

    const tabs = Array.from(
      container.querySelectorAll(
        '[data-testid="plugin-workspace-tabs"] [role="tab"]',
      ),
    );
    expect(tabs.map((tab) => tab.textContent)).toEqual([
      "插件",
      "Skills",
      "专家",
    ]);
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="plugin-workspace-tab-skills"]',
        )
        ?.click();
    });

    expect(onNavigate).toHaveBeenCalledWith("skills", undefined);
  });

  it("Skills 与专家 Tab 切换时应保留当前项目作用域", async () => {
    const onNavigate = vi.fn();
    const { container } = renderContentWithNavigationState({
      currentPage: "skills",
      pageParams: { creationProjectId: "project-1" },
      onNavigate,
    });
    await flushEffects();

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="plugin-workspace-tab-experts"]',
        )
        ?.click();
    });

    expect(onNavigate).toHaveBeenCalledWith("experts", {
      currentProjectId: "project-1",
      projectId: "project-1",
    });
  });

  it("旧 memory 页面不再是可导航主页面", async () => {
    const { container } = renderContent("memory" as unknown as Page, {
      section: "home",
    });
    await flushEffects();

    const memoryPage = container.querySelector('[data-testid="memory-page"]');

    expect(memoryPage).toBeNull();
    expect(container.textContent ?? "").not.toContain("灵感库");
  });

  it("knowledge 页面应把工作区参数透传给 KnowledgePage", async () => {
    const { container } = renderContent("knowledge", {
      workingDir: "/tmp/project-knowledge",
      selectedPackName: "brand-product-demo",
    });
    await flushEffects();

    expectTestId(container, "knowledge-page");
    expect(latestKnowledgePageProps.value).toMatchObject({
      pageParams: {
        workingDir: "/tmp/project-knowledge",
        selectedPackName: "brand-product-demo",
      },
    });
  });

  it("experts 页面应挂载专家广场并透传导航和当前项目作用域", async () => {
    const { container } = renderContent("experts", {
      currentProjectId: "project-1",
      projectId: "project-1",
    });
    await flushEffects();

    expectTestId(container, "expert-plaza-page");
    expect(latestExpertPlazaProps.value).toMatchObject({
      currentProjectId: "project-1",
      onNavigate: expect.any(Function),
    });
  });

  it("plugins 页面带筛选参数时仍应渲染正式 Plugins 管理入口", async () => {
    const { container } = renderContent("plugins", {
      query: "research",
      statusFilter: "installable",
    });
    await flushEffects();

    expectTestId(container, "plugins-page");
    expect(pluginsLifecycle.mounts).toBe(1);
    expect(latestPluginsProps.value).toMatchObject({
      pageParams: {
        query: "research",
        statusFilter: "installable",
      },
    });
  });

  it("plugins 页面应渲染正式 Plugins 管理入口", async () => {
    const { container } = renderContent("plugins", {
      selectedPluginId: "content-factory-app",
    });
    await flushEffects();

    expectTestId(container, "plugins-page");
    expect(pluginsLifecycle.mounts).toBe(1);
    expect(latestPluginsProps.value).toMatchObject({
      pageParams: {
        selectedPluginId: "content-factory-app",
      },
    });
  });
});
