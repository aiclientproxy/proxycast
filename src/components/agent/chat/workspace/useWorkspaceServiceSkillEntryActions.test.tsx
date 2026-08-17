import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { ServiceSkillHomeItem } from "../service-skills/types";
import { useWorkspaceServiceSkillEntryActions } from "./useWorkspaceServiceSkillEntryActions";

const mockCreateScheduledTask = vi.fn();
const mockCreateContent = vi.fn();
const mockListProjects = vi.fn();
const mockGetOrCreateDefaultProject = vi.fn();
const mockRecordServiceSkillAutomationLink = vi.fn();
const mockSiteGetAdapterLaunchReadiness = vi.fn();
const mockToastSuccess = vi.fn();
const mockToastError = vi.fn();
const mockToastInfo = vi.fn();
const mockToastLoading = vi.fn();
const mockEnsureSessionForThreadLineage = vi.fn();

vi.mock("sonner", () => ({
  toast: {
    success: (...args: unknown[]) => mockToastSuccess(...args),
    error: (...args: unknown[]) => mockToastError(...args),
    info: (...args: unknown[]) => mockToastInfo(...args),
    loading: (...args: unknown[]) => mockToastLoading(...args),
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("@/lib/api/scheduledTasks", () => ({
  scheduledTasksApi: {
    create: (request: unknown) => mockCreateScheduledTask(request),
  },
}));

vi.mock("@/lib/api/project", () => ({
  createContent: (request: unknown) => mockCreateContent(request),
  getOrCreateDefaultProject: () => mockGetOrCreateDefaultProject(),
  listProjects: () => mockListProjects(),
  getDefaultContentTypeForProject: (projectType: string) => {
    switch (projectType) {
      case "general":
        return "post";
      case "video":
        return "episode";
      default:
        return "document";
    }
  },
}));

vi.mock("@/lib/webview-api", () => ({
  siteGetAdapterLaunchReadiness: (...args: unknown[]) =>
    mockSiteGetAdapterLaunchReadiness(...args),
}));

vi.mock("../service-skills/automationLinkStorage", () => ({
  recordServiceSkillAutomationLink: (input: unknown) =>
    mockRecordServiceSkillAutomationLink(input),
}));

type HookProps = Parameters<typeof useWorkspaceServiceSkillEntryActions>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];
const DEFAULT_CHAT_TOOL_PREFERENCES: ChatToolPreferences = {
  task: false,
  subagent: false,
};
function createProject(id = "project-1") {
  return {
    id,
    name: "项目一",
    workspaceType: "general",
    rootPath: "",
    isDefault: false,
    createdAt: 1,
    updatedAt: 1,
    isFavorite: false,
    isArchived: false,
    tags: [],
  };
}

function createBrowserServiceSkill(): ServiceSkillHomeItem {
  return {
    id: "github-repo-radar",
    title: "GitHub 仓库线索检索",
    summary:
      "复用你当前浏览器里的 GitHub 登录态，直接检索主题仓库并沉淀成结构化线索。",
    category: "情报研究",
    outputHint: "仓库列表 + 关键线索",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "browser_assist",
    executionLocation: "client_default",
    defaultArtifactKind: "analysis",
    themeTarget: "general",
    version: "seed-v1",
    readinessRequirements: {
      requiresBrowser: true,
      requiresProject: true,
    },
    siteCapabilityBinding: {
      adapterName: "github/search",
      autoRun: true,
      requireAttachedSession: true,
      saveMode: "current_content",
      slotArgMap: {
        repository_query: "query",
      },
      fixedArgs: {
        limit: 10,
      },
      suggestedTitleTemplate: "GitHub 仓库线索 · {{repository_query}}",
    },
    slotSchema: [
      {
        key: "repository_query",
        label: "检索主题",
        type: "text",
        required: true,
        placeholder: "例如 browser assist mcp",
      },
    ],
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "浏览器站点执行",
    runnerTone: "emerald",
    runnerDescription:
      "直接进入浏览器工作台，复用真实登录态执行站点脚本并沉淀结果。",
    actionLabel: "启动采集",
    automationStatus: null,
  };
}

function createScheduledServiceSkill(): ServiceSkillHomeItem {
  return {
    id: "daily-trend-briefing",
    title: "每日趋势摘要",
    summary: "围绕指定平台与关键词输出趋势摘要。",
    category: "内容运营",
    outputHint: "趋势摘要 + 调度建议",
    source: "cloud_catalog",
    runnerType: "scheduled",
    defaultExecutorBinding: "automation_job",
    executionLocation: "client_default",
    defaultArtifactKind: "analysis",
    themeTarget: "general",
    version: "seed-v1",
    slotSchema: [
      {
        key: "platform",
        label: "监测平台",
        type: "platform",
        required: true,
        placeholder: "选择平台",
        defaultValue: "x",
        options: [{ value: "x", label: "X / Twitter" }],
      },
      {
        key: "industry_keywords",
        label: "行业关键词",
        type: "textarea",
        required: true,
        placeholder: "输入关键词",
      },
      {
        key: "schedule_time",
        label: "推送时间",
        type: "schedule_time",
        required: false,
        placeholder: "例如 每天 09:00",
        defaultValue: "每天 09:00",
      },
    ],
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "本地计划任务",
    runnerTone: "sky",
    runnerDescription: "可直接创建本地定时任务，并回流到生成与工作区。",
    actionLabel: "创建任务",
    automationStatus: null,
  };
}

function createLegacyCompatServiceSkill(): ServiceSkillHomeItem {
  return {
    id: "cloud-video-dubbing",
    skillKey: "campaign-launch",
    title: "视频配音",
    summary: "围绕视频文案与素材整理一版可继续加工的配音稿。",
    category: "视频创作",
    outputHint: "配音文案 + 结果摘要",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    defaultArtifactKind: "brief",
    themeTarget: "general",
    version: "seed-v1",
    slotSchema: [
      {
        key: "reference_video",
        label: "参考视频链接/素材",
        type: "url",
        required: true,
        placeholder: "输入视频链接",
      },
    ],
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "立即开始",
    runnerTone: "slate",
    runnerDescription: "直接在当前工作区整理首版配音稿。",
    actionLabel: "对话内补参",
    automationStatus: null,
  };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let latestValue: ReturnType<
    typeof useWorkspaceServiceSkillEntryActions
  > | null = null;

  const defaultProps: HookProps = {
    activeTheme: "general",
    creationMode: "guided",
    projectId: "project-1",
    contentId: "content-current",
    sessionId: "session-current",
    threadId: "thread-current",
    ensureSessionForThreadLineage: mockEnsureSessionForThreadLineage,
    input: "请结合当前上下文继续",
    chatToolPreferences: DEFAULT_CHAT_TOOL_PREFERENCES,
    onNavigate: vi.fn(),
    recordServiceSkillUsage: vi.fn(),
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceServiceSkillEntryActions(currentProps);
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    await act(async () => {
      root.render(<Probe {...defaultProps} {...props} {...nextProps} />);
      await Promise.resolve();
    });
  };

  mountedRoots.push({ root, container });

  return {
    render,
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  mockSiteGetAdapterLaunchReadiness.mockResolvedValue({
    status: "ready",
    adapter: "github/search",
    domain: "github.com",
    profile_key: "attached-github",
    target_id: "tab-github",
    message:
      "已检测到 github.com 的真实浏览器页面，Claw 可以直接复用当前会话执行。",
  });
  mockCreateScheduledTask.mockResolvedValue({
    id: "automation-job-1",
    title: "每日趋势摘要｜定时执行",
  });
  mockCreateContent.mockResolvedValue({
    id: "content-created-by-service-skill",
  });
  mockListProjects.mockResolvedValue([createProject()]);
  mockGetOrCreateDefaultProject.mockResolvedValue(
    createProject("project-default"),
  );
  mockRecordServiceSkillAutomationLink.mockReset();
  mockToastSuccess.mockReset();
  mockToastError.mockReset();
  mockToastInfo.mockReset();
  mockToastLoading.mockReset();
  mockToastLoading.mockImplementation(() => "toast-loading");
  mockEnsureSessionForThreadLineage.mockReset();
  mockEnsureSessionForThreadLineage.mockResolvedValue("session-ensured");
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
  vi.clearAllMocks();
});

describe("useWorkspaceServiceSkillEntryActions", () => {
  it("选择需要补参的技能时应在当前对话挂起 A2UI 表单，而不是打开弹窗", async () => {
    const onNavigate = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
    });
    await render();

    act(() => {
      getValue().handleServiceSkillSelect(createBrowserServiceSkill());
    });

    expect(getValue().pendingServiceSkillLaunchForm).toMatchObject({
      id: expect.stringContaining("service-skill-launch:github-repo-radar"),
      submitAction: expect.objectContaining({
        label: "继续当前结果",
      }),
    });
    expect(getValue().pendingServiceSkillLaunchForm?.components).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: expect.stringContaining(":title"),
          text: "继续「GitHub 仓库线索检索」前，先补齐做法所需信息",
        }),
      ]),
    );
    expect(getValue().pendingServiceSkillLaunchSource).toEqual(
      expect.objectContaining({
        kind: "service_skill",
        skillId: "github-repo-radar",
      }),
    );
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("外部透传的 slotValues 和提示文案应优先用于当前对话 A2UI", async () => {
    const { render, getValue } = renderHook({
      creationReplay: {
        kind: "skill_scaffold",
        source: {
          page: "skills",
        },
        data: {
          name: "AI Agent 行业复盘",
          description: "旧草稿说明",
          outputs: ["聚焦 AI Agent 协作趋势"],
          source_excerpt: "旧草稿线索",
        },
      } as HookProps["creationReplay"],
    });
    await render();

    act(() => {
      getValue().handleServiceSkillSelect(createScheduledServiceSkill(), {
        requestKey: 20260409,
        initialSlotValues: {
          industry_keywords: "",
          schedule_time: "每天 10:00",
        },
        prefillHint: "已根据 Skills 页入口推荐自动预填。",
      });
    });

    expect(getValue().pendingServiceSkillLaunchForm).toMatchObject({
      id: expect.stringContaining("service-skill-launch:daily-trend-briefing"),
    });
    expect(getValue().pendingServiceSkillLaunchSource).toEqual(
      expect.objectContaining({
        kind: "service_skill",
        skillId: "daily-trend-briefing",
        requestKey: "daily-trend-briefing:20260409",
      }),
    );
    expect(getValue().pendingServiceSkillLaunchForm?.components).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: expect.stringContaining(":prefill-hint"),
          text: "已根据 Skills 页入口推荐自动预填。",
        }),
        expect.objectContaining({
          id: "service-skill-slot-industry_keywords",
          value: "",
        }),
        expect.objectContaining({
          id: "service-skill-slot-schedule_time",
          value: "每天 10:00",
        }),
      ]),
    );
  });

  it("提交聊天内技能补参表单后应继续走现有启动链路", async () => {
    const onNavigate = vi.fn();
    const recordServiceSkillUsage = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage,
    });
    await render();

    act(() => {
      getValue().handleServiceSkillSelect(createBrowserServiceSkill());
    });

    await act(async () => {
      await getValue().handlePendingServiceSkillLaunchSubmit({
        "service-skill-slot-repository_query": "browser assist mcp",
      });
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        initialUserPrompt:
          "你帮我在 GitHub 找一下和“browser assist mcp”相关的项目。",
      }),
    );
    expect(recordServiceSkillUsage).toHaveBeenCalledWith(
      expect.objectContaining({
        skillId: "github-repo-radar",
        runnerType: "instant",
        slotValues: {
          repository_query: "browser assist mcp",
        },
      }),
    );
    expect(getValue().pendingServiceSkillLaunchForm).toBeNull();
  });

  it("清理当前入口状态时应移除挂起的技能补参表单", async () => {
    const { render, getValue } = renderHook();
    await render();

    act(() => {
      getValue().handleServiceSkillSelect(createBrowserServiceSkill());
    });
    expect(getValue().pendingServiceSkillLaunchForm).not.toBeNull();

    act(() => {
      getValue().clearPendingServiceSkillLaunch();
    });

    expect(getValue().pendingServiceSkillLaunchForm).toBeNull();
    expect(getValue().pendingServiceSkillLaunchSource).toBeNull();
  });

  it("站点型技能主按钮应进入 Claw 工作区并复用当前主稿", async () => {
    const onNavigate = vi.fn();
    const recordServiceSkillUsage = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage,
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(createBrowserServiceSkill(), {
        repository_query: "browser assist mcp",
      });
    });

    expect(mockCreateContent).not.toHaveBeenCalled();
    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        projectId: "project-1",
        contentId: "content-current",
        theme: "general",
        lockTheme: true,
        initialRequestMetadata: undefined,
        initialCreationMode: "guided",
        newChatAt: expect.any(Number),
        autoRunInitialPromptOnMount: true,
        initialUserPrompt:
          "你帮我在 GitHub 找一下和“browser assist mcp”相关的项目。",
        initialAutoSendRequestMetadata: {
          harness: {
            browser_requirement: "required",
            browser_requirement_reason:
              expect.stringContaining("真实浏览器页面"),
            browser_assist: {
              enabled: true,
              profile_key: "attached-github",
              preferred_backend: "lime_extension_bridge",
              auto_launch: false,
              stream_mode: "both",
            },
            service_skill_launch: expect.objectContaining({
              adapter_name: "github/search",
              skill_title: "GitHub 仓库线索检索",
              content_id: "content-current",
              project_id: "project-1",
              save_mode: "current_content",
              args: {
                query: "browser assist mcp",
                limit: 10,
              },
              launch_readiness: expect.objectContaining({
                status: "ready",
                profile_key: "attached-github",
                target_id: "tab-github",
              }),
            }),
          },
        },
      }),
    );
    const firstSiteSkillLaunchPayload = onNavigate.mock.calls.find(
      ([route]) => route === "agent",
    )?.[1];
    expect(firstSiteSkillLaunchPayload?.initialUserPrompt).not.toContain(
      "[站点技能启动上下文]",
    );
    expect(firstSiteSkillLaunchPayload?.initialUserPrompt).not.toContain(
      "adapter_name",
    );
    expect(
      firstSiteSkillLaunchPayload?.initialAutoSendRequestMetadata,
    ).not.toHaveProperty("artifact");
    expect(recordServiceSkillUsage).toHaveBeenCalledWith({
      skillId: "github-repo-radar",
      runnerType: "instant",
      slotValues: {
        repository_query: "browser assist mcp",
      },
    });
  });

  it("站点型技能进入工作区时应保留真实 service skill metadata", async () => {
    const onNavigate = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage: vi.fn(),
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(createBrowserServiceSkill(), {
        repository_query: "browser assist mcp",
      });
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        initialAutoSendRequestMetadata: {
          harness: expect.objectContaining({
            service_skill_launch: expect.objectContaining({
              adapter_name: "github/search",
            }),
          }),
        },
      }),
    );
  });

  it("站点型技能缺少当前项目时应提示进入项目，不再回退默认项目", async () => {
    const onNavigate = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      projectId: null,
      contentId: null,
      recordServiceSkillUsage: vi.fn(),
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(createBrowserServiceSkill(), {
        repository_query: "browser assist mcp",
      });
    });

    expect(mockGetOrCreateDefaultProject).not.toHaveBeenCalled();
    expect(mockCreateContent).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
    expect(mockToastError).toHaveBeenCalledWith(
      expect.stringContaining("当前技能需要项目工作区，请先进入项目工作。"),
    );
  });

  it("站点型技能次级动作才应跳到浏览器工作台", async () => {
    const onNavigate = vi.fn();
    const recordServiceSkillUsage = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage,
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillBrowserRuntimeLaunch(
        createBrowserServiceSkill(),
        {
          repository_query: "browser assist mcp",
        },
      );
    });

    expect(onNavigate).toHaveBeenCalledWith("browser-runtime", {
      projectId: "project-1",
      contentId: "content-current",
      initialProfileKey: "attached-github",
      initialTargetId: "tab-github",
      initialAdapterName: "github/search",
      initialArgs: {
        query: "browser assist mcp",
        limit: 10,
      },
      initialAutoRun: true,
      initialRequireAttachedSession: true,
      initialSaveTitle: undefined,
    });
    expect(recordServiceSkillUsage).toHaveBeenCalledWith({
      skillId: "github-repo-radar",
      runnerType: "instant",
      slotValues: {
        repository_query: "browser assist mcp",
      },
    });
  });

  it("显式透传的 launchUserInput 应随 recent usage 一起记录下来", async () => {
    const onNavigate = vi.fn();
    const recordServiceSkillUsage = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage,
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(
        createBrowserServiceSkill(),
        {
          repository_query: "browser assist mcp",
        },
        {
          launchUserInput: "优先看最近 30 天仍在活跃更新的仓库",
        },
      );
    });

    expect(recordServiceSkillUsage).toHaveBeenCalledWith({
      skillId: "github-repo-radar",
      runnerType: "instant",
      slotValues: {
        repository_query: "browser assist mcp",
      },
      launchUserInput: "优先看最近 30 天仍在活跃更新的仓库",
    });
  });

  it("站点型技能缺少附着会话时应留在入口层并提示先准备浏览器", async () => {
    const onNavigate = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage: vi.fn(),
    });
    mockSiteGetAdapterLaunchReadiness.mockResolvedValueOnce({
      status: "requires_browser_runtime",
      adapter: "github/search",
      domain: "github.com",
      message: "当前没有检测到已附着到真实浏览器的 github.com 页面。",
      report_hint: "请先去浏览器工作台连接真实浏览器。",
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(createBrowserServiceSkill(), {
        repository_query: "browser assist mcp",
      });
    });

    expect(onNavigate).not.toHaveBeenCalledWith("agent", expect.anything());
    expect(mockToastInfo).toHaveBeenCalledWith(
      expect.stringContaining("请先去浏览器工作台连接真实浏览器"),
    );
    expect(mockToastError).not.toHaveBeenCalled();
  });

  it("legacy cloud_required 服务型技能当前也应按本地执行主链进入工作区", async () => {
    const onNavigate = vi.fn();
    const recordServiceSkillUsage = vi.fn();

    const { render, getValue } = renderHook({
      activeTheme: "general",
      onNavigate,
      recordServiceSkillUsage,
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(
        createLegacyCompatServiceSkill(),
        {
          reference_video: "https://example.com/cloud-video",
        },
      );
    });

    expect(mockCreateContent).not.toHaveBeenCalled();
    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        projectId: "project-1",
        contentId: "content-current",
        theme: "general",
        initialCreationMode: "guided",
        initialRequestMetadata: {
          artifact: {
            artifact_mode: "draft",
            artifact_kind: "brief",
            workbench_surface: "right_panel",
          },
        },
      }),
    );
    expect(recordServiceSkillUsage).toHaveBeenCalledWith({
      skillId: "cloud-video-dubbing",
      runnerType: "instant",
      slotValues: {
        reference_video: "https://example.com/cloud-video",
      },
    });
    expect(mockToastLoading).not.toHaveBeenCalled();
    expect(mockToastSuccess).not.toHaveBeenCalled();
  });

  it("普通技能进入工作区时应保留 seed metadata", async () => {
    const onNavigate = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage: vi.fn(),
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillLaunch(createScheduledServiceSkill(), {
        platform: "x",
        industry_keywords: "AI Agent",
        schedule_time: "每天 09:00",
      });
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        theme: "general",
        initialRequestMetadata: {
          artifact: {
            artifact_mode: "draft",
            artifact_kind: "analysis",
            workbench_surface: "right_panel",
          },
        },
      }),
    );
  });

  it("本地自动化型技能在已有 contentId 时应复用当前主稿创建任务并进入工作区", async () => {
    const onNavigate = vi.fn();
    const recordServiceSkillUsage = vi.fn();
    const { render, getValue } = renderHook({
      onNavigate,
      recordServiceSkillUsage,
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillAutomationSetup(
        createScheduledServiceSkill(),
        {
          platform: "x",
          industry_keywords: "AI Agent，创作者工具",
          schedule_time: "每天 09:00",
        },
      );
    });

    expect(getValue().automationDialogOpen).toBe(true);

    await act(async () => {
      await getValue().handleAutomationDialogSubmit({
        ...getValue().automationDialogInitialValues!,
        title: "每日趋势摘要｜定时执行",
        prompt: "定时任务 prompt",
      });
    });

    expect(mockCreateContent).not.toHaveBeenCalled();
    expect(mockCreateScheduledTask).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "每日趋势摘要｜定时执行",
        schedule: {
          type: "daily",
          time: "09:00",
          timezone: "Asia/Shanghai",
        },
        execution: expect.objectContaining({
          threadMode: "continue_thread",
          sourceThreadId: "thread-current",
          projectId: "project-1",
          requestMetadata: expect.objectContaining({
            service_skill: expect.objectContaining({
              id: "daily-trend-briefing",
              title: "每日趋势摘要",
              runner_type: "scheduled",
              slot_values: [
                {
                  key: "platform",
                  label: "监测平台",
                  value: "X / Twitter",
                },
                {
                  key: "industry_keywords",
                  label: "行业关键词",
                  value: "AI Agent，创作者工具",
                },
                {
                  key: "schedule_time",
                  label: "推送时间",
                  value: "每天 09:00",
                },
              ],
              slot_summary: [
                "监测平台: X / Twitter",
                "行业关键词: AI Agent，创作者工具",
                "推送时间: 每天 09:00",
              ],
              user_input: "请结合当前上下文继续",
            }),
            harness: expect.objectContaining({
              theme: "general",
              session_mode: "general_workbench",
              content_id: "content-current",
            }),
          }),
        }),
      }),
    );
    expect(mockEnsureSessionForThreadLineage).not.toHaveBeenCalled();
    expect(mockRecordServiceSkillAutomationLink).toHaveBeenCalledWith({
      skillId: "daily-trend-briefing",
      jobId: "automation-job-1",
      jobName: "每日趋势摘要｜定时执行",
    });
    expect(recordServiceSkillUsage).toHaveBeenCalledWith({
      skillId: "daily-trend-briefing",
      runnerType: "scheduled",
      slotValues: {
        platform: "x",
        industry_keywords: "AI Agent，创作者工具",
        schedule_time: "每天 09:00",
      },
    });
    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        projectId: "project-1",
        contentId: "content-current",
        theme: "general",
        initialCreationMode: "guided",
        initialUserPrompt: expect.stringContaining("[技能任务] 每日趋势摘要"),
        autoRunInitialPromptOnMount: true,
      }),
    );
  });

  it("本地自动化型技能缺少当前 session 时应先物化 Thread 再创建任务", async () => {
    const { render, getValue } = renderHook({
      sessionId: null,
      threadId: null,
      contentId: null,
    });
    await render();

    await act(async () => {
      await getValue().handleServiceSkillAutomationSetup(
        createScheduledServiceSkill(),
        {
          platform: "x",
          industry_keywords: "AI Agent，创作者工具",
          schedule_time: "每天 09:00",
        },
      );
    });

    expect(mockEnsureSessionForThreadLineage).toHaveBeenCalledTimes(1);
    expect(getValue().automationDialogOpen).toBe(true);

    await act(async () => {
      await getValue().handleAutomationDialogSubmit({
        ...getValue().automationDialogInitialValues!,
        title: "每日趋势摘要｜定时执行",
        prompt: "定时任务 prompt",
      });
    });

    expect(mockCreateContent).toHaveBeenCalledWith(
      expect.objectContaining({
        project_id: "project-1",
      }),
    );
    expect(mockCreateScheduledTask).toHaveBeenCalledWith(
      expect.objectContaining({
        execution: expect.objectContaining({
          threadMode: "continue_thread",
          sourceThreadId: "session-ensured",
          projectId: "project-1",
          requestMetadata: expect.objectContaining({
            harness: expect.objectContaining({
              content_id: "content-created-by-service-skill",
            }),
          }),
        }),
      }),
    );
  });
});
