import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GeneralWorkbenchSidebar } from "./GeneralWorkbenchSidebar";
import { recordCuratedTaskRecommendationSignal } from "../utils/curatedTaskRecommendationSignals";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];
const mockWriteClipboardText = vi.fn();
const {
  mockRevealSessionFileInFinder,
  mockOpenSessionFileWithDefaultApp,
  mockToastError,
  mockToastSuccess,
  mockOpenDialog,
} = vi.hoisted(() => ({
  mockRevealSessionFileInFinder: vi.fn(),
  mockOpenSessionFileWithDefaultApp: vi.fn(),
  mockToastError: vi.fn(),
  mockToastSuccess: vi.fn(),
  mockOpenDialog: vi.fn(),
}));

vi.mock("@/lib/api/session-files", () => ({
  revealFileInFinder: mockRevealSessionFileInFinder,
  openFileWithDefaultApp: mockOpenSessionFileWithDefaultApp,
}));

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: mockOpenDialog,
}));

vi.mock("@/lib/desktop-host/shell", () => ({
  open: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("sonner", () => ({
  toast: {
    error: mockToastError,
    success: mockToastSuccess,
  },
}));

vi.mock("react-i18next", async () => {
  const { agentZhCNResource } = await import("@/i18n/agentResources");
  const agentZhCN = agentZhCNResource as Record<string, string>;

  const interpolate = (template: string, values: Record<string, unknown>) =>
    template.replace(/{{\s*([^}]+?)\s*}}/g, (_, name: string) => {
      const value = values[name.trim()];
      return value == null ? "" : String(value);
    });

  return {
    useTranslation: () => ({
      i18n: { language: "zh-CN" },
      t: (key: string, options?: Record<string, unknown>) => {
        const template = agentZhCN[key];
        if (!template) {
          return key;
        }
        return options ? interpolate(template, options) : template;
      },
    }),
  };
});

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  if (!HTMLElement.prototype.scrollIntoView) {
    HTMLElement.prototype.scrollIntoView = () => {};
  }

  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: {
      writeText: mockWriteClipboardText,
    },
  });
  mockWriteClipboardText.mockResolvedValue(undefined);
  mockRevealSessionFileInFinder.mockResolvedValue(undefined);
  mockOpenSessionFileWithDefaultApp.mockResolvedValue(undefined);
  mockOpenDialog.mockResolvedValue(null);
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
  window.localStorage.clear();
});

function renderSidebar(
  props?: Partial<React.ComponentProps<typeof GeneralWorkbenchSidebar>>,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const defaultProps: React.ComponentProps<typeof GeneralWorkbenchSidebar> = {
    onNewTopic: vi.fn(),
    onSwitchTopic: vi.fn(),
    onDeleteTopic: vi.fn(),
    branchItems: [
      {
        id: "topic-a",
        title: "话题 A",
        status: "in_progress",
        isCurrent: true,
      },
    ],
    onSetBranchStatus: vi.fn(),
    workflowSteps: [
      { id: "brief", title: "明确需求", status: "completed" },
      { id: "create", title: "创作内容", status: "active" },
    ],
    contextSearchQuery: "品牌",
    onContextSearchQueryChange: vi.fn(),
    contextSearchMode: "web",
    onContextSearchModeChange: vi.fn(),
    contextSearchLoading: false,
    contextSearchError: null,
    onSubmitContextSearch: vi.fn(),
    contextItems: [
      {
        id: "search:web:brand",
        name: "品牌话题观察",
        source: "search",
        searchMode: "web",
        query: "品牌 2026",
        previewText: "品牌讨论聚焦产品定位、渠道节奏与转化质量。",
        citations: [{ title: "官方博客", url: "https://example.com/blog" }],
        active: true,
      },
    ],
    onToggleContextActive: vi.fn(),
    contextBudget: {
      activeCount: 1,
      activeCountLimit: 12,
      estimatedTokens: 600,
      tokenLimit: 32000,
    },
    activityLogs: [
      {
        id: "log-1",
        name: "content_post_with_cover",
        status: "completed",
        timeLabel: "10:30",
        applyTarget: "封面/插图",
        contextIds: ["material:1"],
        gateKey: "write_mode",
        runId: "run-abcdef123456",
        source: "skill",
        sourceRef: "content_post_with_cover",
      },
    ],
    skillDetailMap: {
      content_post_with_cover: {
        name: "content_post_with_cover",
        display_name: "社媒主稿与封面",
        description: "生成内容主稿，并补齐封面素材。",
        execution_mode: "prompt",
        has_workflow: true,
        workflow_steps: [
          {
            id: "outline",
            name: "提炼内容主线",
            dependencies: [],
          },
          {
            id: "cover",
            name: "生成封面提示词",
            dependencies: ["outline"],
          },
        ],
        allowed_tools: ["read_file", "generate_image"],
        when_to_use: "适合需要主稿与封面同时产出的社媒场景。",
        markdown_content: "",
      },
    },
    onViewRunDetail: vi.fn(),
    activeRunDetail: null,
    activeRunDetailLoading: false,
  };

  act(() => {
    root.render(<GeneralWorkbenchSidebar {...defaultProps} {...props} />);
  });

  mountedRoots.push({ root, container });
  return { container, props: { ...defaultProps, ...props } };
}

describe("GeneralWorkbenchSidebar", () => {
  it("当前进展面板应突出当前任务并按状态优先级排序", () => {
    const { container } = renderSidebar({
      workflowSteps: [
        { id: "done", title: "完成提纲", status: "completed" },
        { id: "pending", title: "等待补充案例", status: "pending" },
        { id: "active", title: "撰写主稿", status: "active" },
        { id: "error", title: "封面生成失败", status: "error" },
      ],
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    expect(workflowTab).toBeTruthy();
    expect(workflowTab?.textContent).toContain("进展");
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    expect(container.textContent).toContain("生成侧栏");
    expect(container.textContent).toContain(
      "聚焦当前结果、下一步与可继续的版本。",
    );
    expect(container.textContent).toContain("当前进展");
    expect(container.textContent).toContain("当前焦点");
    expect(container.textContent).toContain("撰写主稿");
    expect(container.textContent).toContain("结果去向");
    expect(container.textContent).toContain("产出记录 / 执行经过");
    expect(container.textContent).toContain("继续上次做法");
    expect(container.textContent).toMatch(/可继续稿件|可继续版本/);

    const stepNodes = Array.from(
      container.querySelectorAll('[data-testid="workflow-sidebar-step"]'),
    );
    const taskSection = container.querySelector(
      '[data-testid="workflow-sidebar-task-section"]',
    ) as HTMLElement | null;
    const branchSection = container.querySelector(
      '[data-testid="workflow-sidebar-branch-section"]',
    ) as HTMLElement | null;

    expect(taskSection).toBeTruthy();
    expect(branchSection).toBeTruthy();
    expect(stepNodes).toHaveLength(0);
    expect(taskSection?.textContent).toContain("当前焦点");
    expect(container.textContent).not.toContain("后续任务");
    expect(container.textContent).not.toContain("已完成 1/4");
    expect(container.textContent).not.toContain("已完成 1 项");
    expect(
      taskSection?.querySelector(
        '[data-testid="workflow-sidebar-result-destination-hint"]',
      ),
    ).toBeTruthy();
    const taskSectionOrder =
      taskSection && branchSection
        ? taskSection.compareDocumentPosition(branchSection) &
          Node.DOCUMENT_POSITION_FOLLOWING
        : 0;
    expect(taskSectionOrder).toBeTruthy();
  });

  it("Workflow 控制项应在当前进展里渲染并触发回调", () => {
    const onTriggerWorkflowControl = vi.fn();
    const workflowControlItem = {
      id: "workflow-run-1-retry-draft",
      kind: "retry" as const,
      workflowRunId: "workflow-run-1",
      stepId: "draft",
      requestId: null,
      actionType: null,
      labelKey: "generalWorkbench.workflow.control.retry",
      ariaLabelKey: "generalWorkbench.workflow.control.retryAria",
      tone: "warning" as const,
    };
    const { container } = renderSidebar({
      workflowControlItems: [workflowControlItem],
      onTriggerWorkflowControl,
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    expect(workflowTab).toBeTruthy();
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const retryButton = container.querySelector(
      '[data-testid="workflow-control-retry"]',
    ) as HTMLButtonElement | null;
    expect(retryButton).toBeTruthy();
    expect(retryButton?.textContent).toContain("重试失败步骤");
    if (retryButton) {
      act(() => {
        retryButton.click();
      });
    }

    expect(onTriggerWorkflowControl).toHaveBeenCalledWith(workflowControlItem);
  });

  it("传入折叠回调时应显示折叠按钮并可触发", () => {
    const onRequestCollapse = vi.fn();
    const { container } = renderSidebar({ onRequestCollapse });

    const collapseButton = container.querySelector(
      'button[aria-label="折叠上下文侧栏"]',
    ) as HTMLButtonElement | null;
    expect(collapseButton).toBeTruthy();
    if (collapseButton) {
      act(() => {
        collapseButton.click();
      });
    }
    expect(onRequestCollapse).toHaveBeenCalledTimes(1);
  });

  it("点击添加上下文应打开添加弹窗", () => {
    const { container } = renderSidebar();

    const addContextButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("添加上下文"));
    expect(addContextButton).toBeTruthy();
    if (addContextButton) {
      act(() => {
        addContextButton.click();
      });
    }

    expect(container.textContent).toContain("添加新上下文");
    expect(container.textContent).toContain("上传文件");
    expect(container.textContent).toContain("网站链接");
    expect(container.textContent).toContain("输入文本");
  });

  it("输入文本上下文后确认应触发回调", async () => {
    const onAddTextContext = vi.fn().mockResolvedValue(undefined);
    const { container } = renderSidebar({ onAddTextContext });

    const addContextButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("添加上下文"));
    expect(addContextButton).toBeTruthy();
    if (addContextButton) {
      act(() => {
        addContextButton.click();
      });
    }

    const textButton = container.querySelector(
      'button[aria-label="输入文本上下文"]',
    ) as HTMLButtonElement | null;
    expect(textButton).toBeTruthy();
    if (textButton) {
      act(() => {
        textButton.click();
      });
    }

    const textarea = container.querySelector(
      'textarea[placeholder="在此粘贴或输入文本..."]',
    ) as HTMLTextAreaElement | null;
    expect(textarea).toBeTruthy();
    if (textarea) {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value",
      )?.set;
      act(() => {
        setter?.call(textarea, "这是一段用于测试的上下文内容");
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
      });
    }

    const confirmButton = container.querySelector(
      'button[aria-label="确认添加文本上下文"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      await Promise.resolve();
    });
    if (confirmButton) {
      await act(async () => {
        confirmButton.click();
        await Promise.resolve();
      });
    }

    expect(onAddTextContext).toHaveBeenCalledTimes(1);
    expect(onAddTextContext).toHaveBeenCalledWith({
      content: "这是一段用于测试的上下文内容",
    });
  });

  it("应展示新的双 tab 与紧凑上下文列表结构", () => {
    const { container } = renderSidebar();
    const shell = container.querySelector(
      '[data-testid="general-workbench-sidebar"]',
    ) as HTMLElement | null;
    const header = container.querySelector(
      '[data-testid="general-workbench-sidebar-header"]',
    ) as HTMLElement | null;
    const tabs = container.querySelector(
      '[data-testid="general-workbench-sidebar-tabs"]',
    ) as HTMLElement | null;

    expect(container.textContent).toContain("上下文管理");
    expect(
      container.querySelector('button[aria-label="打开上下文管理"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('button[aria-label="打开当前进展"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('button[aria-label="打开执行日志"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("搜索上下文");
    expect(container.textContent).toContain("上下文列表");
    expect(container.textContent).not.toContain("上下文概览");
    expect(container.textContent).not.toContain("项目资料");
    expect(shell?.className).toContain("bg-slate-50");
    expect(shell?.className).not.toContain("linear-gradient");
    expect(header?.className).toContain("bg-white");
    expect(header?.className).not.toContain("backdrop-blur");
    expect(tabs?.className).toContain("bg-slate-100");
  });

  it("应展示新的搜索上下文输入区", () => {
    const { container } = renderSidebar();
    expect(container.textContent).toContain("添加上下文");

    const searchInput = container.querySelector(
      'input[placeholder="搜索网络添加新上下文"]',
    ) as HTMLInputElement | null;
    expect(searchInput).toBeTruthy();
    expect(searchInput?.value).toBe("品牌");
  });

  it("日志存在更多历史时应显示加载按钮并可触发", () => {
    const onLoadMoreHistory = vi.fn();
    const { container } = renderSidebar({
      historyHasMore: true,
      onLoadMoreHistory,
    });

    const logTabButton = container.querySelector(
      'button[aria-label="打开执行日志"]',
    ) as HTMLButtonElement | null;
    expect(logTabButton).toBeTruthy();
    if (logTabButton) {
      act(() => {
        logTabButton.click();
      });
    }

    const loadMoreButton = container.querySelector(
      'button[aria-label="加载更早历史日志"]',
    ) as HTMLButtonElement | null;
    expect(loadMoreButton).toBeTruthy();
    if (loadMoreButton) {
      act(() => {
        loadMoreButton.click();
      });
    }

    expect(onLoadMoreHistory).toHaveBeenCalledTimes(1);
  });

  it("执行日志应展示技能显示名与技能描述", () => {
    const { container } = renderSidebar();

    const logTabButton = container.querySelector(
      'button[aria-label="打开执行日志"]',
    ) as HTMLButtonElement | null;
    expect(logTabButton).toBeTruthy();
    if (logTabButton) {
      act(() => {
        logTabButton.click();
      });
    }

    expect(container.textContent).toContain("技能：社媒主稿与封面");
    expect(container.textContent).toContain("生成内容主稿，并补齐封面素材。");
    expect(container.textContent).toContain(
      "技能标识：content_post_with_cover",
    );
  });

  it("执行日志应支持展开技能详情", () => {
    const { container } = renderSidebar();

    const logTabButton = container.querySelector(
      'button[aria-label="打开执行日志"]',
    ) as HTMLButtonElement | null;
    expect(logTabButton).toBeTruthy();
    if (logTabButton) {
      act(() => {
        logTabButton.click();
      });
    }

    const detailButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("查看技能详情"),
    );
    expect(detailButton).toBeTruthy();
    if (detailButton) {
      act(() => {
        detailButton.click();
      });
    }

    expect(container.textContent).toContain("工作流步骤");
    expect(container.textContent).toContain("1. 提炼内容主线");
    expect(container.textContent).toContain("2. 生成封面提示词");
    expect(container.textContent).toContain("允许工具");
    expect(container.textContent).toContain("查看文件");
    expect(container.textContent).toContain("生成图片");
    expect(container.textContent).toContain("适用场景");
    expect(container.textContent).toContain(
      "适合需要主稿与封面同时产出的社媒场景。",
    );
  });

  it("执行日志应支持展开工具详情", () => {
    const { container } = renderSidebar({
      messages: [
        {
          id: "assistant-tool-detail",
          role: "assistant",
          content: "",
          timestamp: new Date("2026-03-12T10:35:00.000Z"),
          toolCalls: [
            {
              id: "tool-detail-1",
              name: "read_file",
              arguments: JSON.stringify({ path: "/tmp/a.txt", limit: 50 }),
              status: "failed",
              result: {
                success: false,
                output: "",
                error: "文件不存在",
              },
              startTime: new Date("2026-03-12T10:35:00.000Z"),
            },
          ],
        },
      ],
    });

    const logTabButton = container.querySelector(
      'button[aria-label="打开执行日志"]',
    ) as HTMLButtonElement | null;
    expect(logTabButton).toBeTruthy();
    if (logTabButton) {
      act(() => {
        logTabButton.click();
      });
    }

    const detailButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("查看工具详情"),
    );
    expect(detailButton).toBeTruthy();
    if (detailButton) {
      act(() => {
        detailButton.click();
      });
    }

    expect(container.textContent).toContain("请求参数");
    expect(container.textContent).toContain('"path": "/tmp/a.txt"');
    expect(container.textContent).toContain('"limit": 50');
    expect(container.textContent).toContain("错误信息");
    expect(container.textContent).toContain("文件不存在");
  });

  it("执行日志应支持清空全部记录", () => {
    const { container } = renderSidebar();

    const logTabButton = container.querySelector(
      'button[aria-label="打开执行日志"]',
    ) as HTMLButtonElement | null;
    expect(logTabButton).toBeTruthy();
    if (logTabButton) {
      act(() => {
        logTabButton.click();
      });
    }

    const clearButton = container.querySelector(
      'button[aria-label="清空全部日志"]',
    ) as HTMLButtonElement | null;
    expect(clearButton).toBeTruthy();
    if (clearButton) {
      act(() => {
        clearButton.click();
      });
    }

    expect(container.textContent).toContain("日志已清空，等待新的运行记录");
    expect(container.textContent).not.toContain("执行技能 社媒主稿与封面");
  });

  it("执行日志应支持按失败项筛选", () => {
    const { container } = renderSidebar({
      messages: [
        {
          id: "assistant-1",
          role: "assistant",
          content: "",
          timestamp: new Date("2026-03-12T10:32:00.000Z"),
          toolCalls: [
            {
              id: "tool-1",
              name: "read_file",
              arguments: JSON.stringify({ path: "/tmp/a.txt" }),
              status: "failed",
              result: {
                success: false,
                output: "",
                error: "文件不存在",
              },
              startTime: new Date("2026-03-12T10:32:00.000Z"),
            },
          ],
        },
      ],
    });

    const logTabButton = container.querySelector(
      'button[aria-label="打开执行日志"]',
    ) as HTMLButtonElement | null;
    expect(logTabButton).toBeTruthy();
    if (logTabButton) {
      act(() => {
        logTabButton.click();
      });
    }

    const failedFilterButton = container.querySelector(
      'button[aria-label="筛选执行日志-失败"]',
    ) as HTMLButtonElement | null;
    expect(failedFilterButton).toBeTruthy();
    if (failedFilterButton) {
      act(() => {
        failedFilterButton.click();
      });
    }

    expect(container.textContent).toContain("查看文件");
    expect(container.textContent).toContain("文件不存在");
    expect(container.textContent).not.toContain("执行技能 社媒主稿与封面");
  });

  it("应支持触发上下文搜索与切换来源", () => {
    const onSubmitContextSearch = vi.fn();
    const onContextSearchModeChange = vi.fn();
    const { container } = renderSidebar({
      onSubmitContextSearch,
      onContextSearchModeChange,
    });

    const submitButton = container.querySelector(
      'button[aria-label="提交上下文搜索"]',
    ) as HTMLButtonElement | null;
    expect(submitButton).toBeTruthy();
    if (submitButton) {
      act(() => {
        submitButton.click();
      });
    }
    expect(onSubmitContextSearch).toHaveBeenCalledTimes(1);

    const triggerButton = container.querySelector(
      'button[aria-label="选择上下文搜索来源"]',
    ) as HTMLButtonElement | null;
    expect(triggerButton).toBeTruthy();
    if (triggerButton) {
      act(() => {
        triggerButton.click();
      });
    }

    const socialMenuText = Array.from(container.querySelectorAll("span")).find(
      (node) => node.textContent === "社交媒体",
    );
    const socialMenuItem = socialMenuText?.closest("div");
    expect(socialMenuItem).toBeTruthy();
    if (socialMenuItem) {
      act(() => {
        socialMenuItem.click();
      });
    }

    expect(onContextSearchModeChange).toHaveBeenCalledWith("social");
  });

  it("应按标题列表展示搜索结果，并支持进入详情查看来源", () => {
    const { container } = renderSidebar();

    expect(container.textContent).toContain("上下文列表");
    expect(container.textContent).toContain("品牌话题观察");
    expect(container.textContent).not.toContain("检索词：品牌 2026");
    expect(container.textContent).not.toContain("品牌讨论聚焦产品定位");
    expect((container.textContent?.match(/品牌话题观察/g) || []).length).toBe(
      1,
    );

    const openButton = container.querySelector(
      'button[aria-label="查看搜索结果 品牌话题观察"]',
    ) as HTMLButtonElement | null;
    expect(openButton).toBeTruthy();
    if (openButton) {
      act(() => {
        openButton.click();
      });
    }

    expect(container.textContent).toContain("搜索结果详情");
    expect(container.textContent).toContain("检索词：品牌 2026");
    expect(container.textContent).toContain("品牌讨论聚焦产品定位");

    const citationLink = container.querySelector(
      'a[href="https://example.com/blog"]',
    ) as HTMLAnchorElement | null;
    expect(citationLink).toBeTruthy();

    const backButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("返回列表"),
    );
    expect(backButton).toBeTruthy();
    if (backButton) {
      act(() => {
        backButton.click();
      });
    }

    expect(container.textContent).toContain("搜索结果");
    expect(container.textContent).not.toContain("检索词：品牌 2026");
  });

  it("搜索被阻塞时应展示原因并禁用提交", () => {
    const { container } = renderSidebar({
      contextSearchQuery: "品牌",
      contextSearchBlockedReason: "请先选择可用模型后再搜索",
    });

    expect(container.textContent).toContain("请先选择可用模型后再搜索");
    const submitButton = container.querySelector(
      'button[aria-label="提交上下文搜索"]',
    ) as HTMLButtonElement | null;
    expect(submitButton?.disabled).toBe(true);
  });

  it("应支持分支状态操作", () => {
    const onSetBranchStatus = vi.fn();
    const { container } = renderSidebar({
      branchMode: "topic",
      onSetBranchStatus,
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    expect(
      container.querySelector("button[aria-label='切换可继续记录']"),
    ).toBeTruthy();
    expect(container.querySelector("button[aria-label='删除分支']")).toBeNull();
    expect(container.textContent).toContain("当前焦点落在");

    const branchToggle = container.querySelector(
      "button[aria-label='切换可继续记录']",
    ) as HTMLButtonElement | null;
    if (branchToggle) {
      act(() => {
        branchToggle.click();
      });
    }

    const mergeButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "收进主稿",
    );
    expect(mergeButton).toBeTruthy();
    if (mergeButton) {
      act(() => {
        mergeButton.click();
      });
    }
    expect(onSetBranchStatus).toHaveBeenCalledWith("topic-a", "merged");
  });

  it("点击运行标识应触发详情回调", () => {
    const onViewRunDetail = vi.fn();
    const { container } = renderSidebar({ onViewRunDetail });
    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const activityToggle = container.querySelector(
      "button[aria-label='切换执行经过']",
    ) as HTMLButtonElement | null;
    if (activityToggle) {
      act(() => {
        activityToggle.click();
      });
    }

    const runButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("查看运行 run-abcd…"),
    );
    expect(runButton).toBeTruthy();
    if (runButton) {
      act(() => {
        runButton.click();
      });
    }

    expect(onViewRunDetail).toHaveBeenCalledWith("run-abcdef123456");
  });

  it("复盘结果带着 结果基线时，应在当前进展里显影当前结果基线", () => {
    const { container } = renderSidebar({
      activeRunDetail: {
        id: "run-result-baseline-workflow",
        source: "skill",
        source_ref: "account-project-review",
        session_id: "session-result-baseline-workflow",
        status: "success",
        started_at: "2026-03-06T01:12:03Z",
        finished_at: "2026-03-06T01:12:10Z",
        duration_ms: 7000,
        error_code: null,
        error_message: null,
        metadata: JSON.stringify({
          harness: {
            curated_task: {
              task_id: "account-project-review",
              task_title: "复盘这个账号/项目",
              reference_entries: [
                {
                  id: "memory:ai-weekly:run:1",
                  source_kind: "memory",
                  title: "AI 内容周报",
                  summary: "已有一轮可继续放量的结果。",
                  category: "experience",
                  tags: ["复盘", "增长"],
                  task_prefill_by_task_id: {
                    "account-project-review": {
                      project_goal: "AI 内容周报",
                      existing_results:
                        "当前判断：适合继续放量 经营动作：保留品牌联名方向 更适合去向：内容主稿生成 / 渠道改写",
                    },
                  },
                },
              ],
            },
          },
        }),
        created_at: "2026-03-06T01:12:03Z",
        updated_at: "2026-03-06T01:12:10Z",
      },
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const baselineCard = container.querySelector(
      '[data-testid="workflow-sidebar-result-baseline-card"]',
    ) as HTMLElement | null;
    expect(baselineCard).toBeTruthy();
    expect(baselineCard?.textContent).toContain("当前结果基线");
    expect(baselineCard?.textContent).toContain("AI 内容周报");
    expect(baselineCard?.textContent).toContain("当前判断：适合继续放量");
    expect(baselineCard?.textContent).toContain("经营动作：保留品牌联名方向");
    expect(baselineCard?.textContent).toContain(
      "更适合去向：内容主稿生成 / 渠道改写",
    );
  });

  it("点击当前进展里的复盘建议时，应把结果基线 continuation 回传给工作区", () => {
    const onApplyFollowUpAction = vi.fn();
    recordCuratedTaskRecommendationSignal({
      source: "review_feedback",
      category: "experience",
      title: "AI 内容周报 · 转成主稿",
      summary: "这轮判断已经清楚，建议直接回到内容主稿生成继续往下做。",
      tags: ["复盘", "主稿"],
      preferredTaskIds: ["social-post-starter", "account-project-review"],
      createdAt: Date.now(),
      projectId: "project-review-feedback-sidebar",
      sessionId: "session-review-feedback-sidebar",
    });

    const { container } = renderSidebar({
      projectId: "project-review-feedback-sidebar",
      onApplyFollowUpAction,
      activeRunDetail: {
        id: "run-review-feedback-sidebar",
        source: "skill",
        source_ref: "account-project-review",
        session_id: "session-review-feedback-sidebar",
        status: "success",
        started_at: "2026-03-06T01:12:03Z",
        finished_at: "2026-03-06T01:12:10Z",
        duration_ms: 7000,
        error_code: null,
        error_message: null,
        metadata: JSON.stringify({
          harness: {
            curated_task: {
              task_id: "account-project-review",
              task_title: "复盘这个账号/项目",
              reference_entries: [
                {
                  id: "memory:ai-weekly:run:3",
                  source_kind: "memory",
                  title: "AI 内容周报",
                  summary: "当前已有一轮结果，可直接进入下游主稿。",
                  category: "experience",
                  tags: ["复盘", "周报"],
                  task_prefill_by_task_id: {
                    "account-project-review": {
                      project_goal: "AI 内容周报",
                      existing_results:
                        "当前判断：适合继续放量 当前卡点：封面信息过密 经营动作：保留品牌联名方向 更适合去向：内容主稿生成",
                    },
                  },
                },
              ],
            },
          },
        }),
        created_at: "2026-03-06T01:12:03Z",
        updated_at: "2026-03-06T01:12:10Z",
      },
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const actionButton = container.querySelector(
      '[data-testid="workflow-sidebar-review-feedback-banner-action"]',
    ) as HTMLButtonElement | null;
    expect(actionButton?.textContent).toContain("继续去「内容主稿生成」");

    if (actionButton) {
      act(() => {
        actionButton.click();
      });
    }

    expect(onApplyFollowUpAction).toHaveBeenCalledWith(
      expect.objectContaining({
        bannerMessage:
          "已切到“内容主稿生成”这条下一步，并带着当前结果继续进入生成。",
        capabilityRoute: expect.objectContaining({
          kind: "curated_task",
          taskId: "social-post-starter",
          taskTitle: "内容主稿生成",
          prompt: expect.stringContaining("当前结果基线：AI 内容周报"),
          referenceEntries: [
            expect.objectContaining({
              sourceKind: "memory",
              title: "AI 内容周报",
            }),
          ],
        }),
        prompt: expect.stringContaining("继续沿这轮项目结果基线推进"),
      }),
    );
  });

  it("点击建议下一步应把 continuation prompt 回传给工作区", () => {
    const onApplyFollowUpAction = vi.fn();
    const { container } = renderSidebar({
      onApplyFollowUpAction,
      activeRunDetail: {
        id: "run-curated-task-follow-up",
        source: "skill",
        source_ref: "daily-trend-briefing",
        session_id: "session-curated-task-follow-up",
        status: "success",
        started_at: "2026-03-06T01:12:03Z",
        finished_at: "2026-03-06T01:12:10Z",
        duration_ms: 7000,
        error_code: null,
        error_message: null,
        metadata: JSON.stringify({
          harness: {
            curated_task: {
              task_id: "daily-trend-briefing",
              task_title: "每日趋势摘要",
              launch_input_values: {
                theme_target: "AI 内容创作",
                platform_region: "X 与 TikTok 北美区",
              },
              reference_entries: [
                {
                  id: "memory-1",
                  title: "品牌风格样本",
                  summary: "保留轻盈但专业的表达。",
                  category: "context",
                  tags: ["品牌", "语气"],
                },
              ],
            },
          },
        }),
        created_at: "2026-03-06T01:12:03Z",
        updated_at: "2026-03-06T01:12:10Z",
      },
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const followUpButton = container.querySelector(
      'button[aria-label="应用建议下一步-继续展开其中一个选题"]',
    ) as HTMLButtonElement | null;
    expect(followUpButton).toBeTruthy();
    if (followUpButton) {
      act(() => {
        followUpButton.click();
      });
    }

    expect(onApplyFollowUpAction).toHaveBeenCalledWith({
      prompt: "请基于「每日趋势摘要」这轮结果继续：继续展开其中一个选题",
      bannerMessage:
        "已按“继续展开其中一个选题”接着推进「每日趋势摘要」，可继续改写后发送。",
      capabilityRoute: {
        kind: "curated_task",
        taskId: "daily-trend-briefing",
        taskTitle: "每日趋势摘要",
        prompt: "请基于「每日趋势摘要」这轮结果继续：继续展开其中一个选题",
        launchInputValues: {
          theme_target: "AI 内容创作",
          platform_region: "X 与 TikTok 北美区",
        },
        referenceMemoryIds: ["memory-1"],
        referenceEntries: [
          {
            id: "memory-1",
            sourceKind: "memory",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });
  });

  it("运行详情应展示阶段与产物路径，并支持复制产物路径", () => {
    const { container } = renderSidebar({
      activeRunDetail: {
        id: "run-artifact-1",
        source: "skill",
        source_ref: "content_post_with_cover",
        session_id: "session-artifact",
        status: "success",
        started_at: "2026-03-06T02:00:03Z",
        finished_at: "2026-03-06T02:00:08Z",
        duration_ms: 5000,
        error_code: null,
        error_message: null,
        metadata: JSON.stringify({
          workflow: "social_content_pipeline_v1",
          execution_id: "exec-artifact-1",
          version_id: "ver-artifact-1",
          stages: ["topic_select", "write_mode", "publish_confirm"],
          artifact_paths: [
            "content-posts/demo.md",
            "content-posts/demo.publish-pack.json",
          ],
        }),
        created_at: "2026-03-06T02:00:03Z",
        updated_at: "2026-03-06T02:00:08Z",
      },
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const activityToggle = container.querySelector(
      "button[aria-label='切换执行经过']",
    ) as HTMLButtonElement | null;
    if (activityToggle) {
      act(() => {
        activityToggle.click();
      });
    }

    expect(container.textContent).toContain(
      "工作流 social_content_pipeline_v1",
    );
    expect(container.textContent).toContain("选题闸门 → 写作闸门 → 发布闸门");
    expect(container.textContent).toContain("content-posts/demo.md");
    expect(container.textContent).toContain(
      "content-posts/demo.publish-pack.json",
    );

    const copyArtifactButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) =>
      button
        .getAttribute("aria-label")
        ?.startsWith("复制产物路径-content-posts/demo.md"),
    );
    expect(copyArtifactButton).toBeTruthy();
    if (copyArtifactButton) {
      act(() => {
        copyArtifactButton.click();
      });
    }

    expect(mockWriteClipboardText).toHaveBeenCalledWith(
      "content-posts/demo.md",
    );

    const revealArtifactButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) =>
      button
        .getAttribute("aria-label")
        ?.startsWith("定位产物路径-content-posts/demo.md"),
    );
    expect(revealArtifactButton).toBeTruthy();
    if (revealArtifactButton) {
      act(() => {
        revealArtifactButton.click();
      });
    }
    expect(mockRevealSessionFileInFinder).toHaveBeenCalledWith(
      "session-artifact",
      "content-posts/demo.md",
    );

    const openArtifactButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) =>
      button
        .getAttribute("aria-label")
        ?.startsWith("打开产物路径-content-posts/demo.md"),
    );
    expect(openArtifactButton).toBeTruthy();
    if (openArtifactButton) {
      act(() => {
        openArtifactButton.click();
      });
    }
    expect(mockOpenSessionFileWithDefaultApp).toHaveBeenCalledWith(
      "session-artifact",
      "content-posts/demo.md",
    );
  });

  it("活动日志分组应支持直接定位与打开产物", () => {
    const { container } = renderSidebar({
      activityLogs: [
        {
          id: "log-run-artifact-1",
          name: "content_post_with_cover",
          status: "completed",
          timeLabel: "11:20",
          applyTarget: "主稿内容",
          contextIds: ["material:1"],
          sessionId: "session-group",
          artifactPaths: ["content-posts/group.md"],
          gateKey: "write_mode",
          source: "skill",
        },
      ],
    });

    const workflowTab = container.querySelector(
      'button[aria-label="打开当前进展"]',
    ) as HTMLButtonElement | null;
    if (workflowTab) {
      act(() => {
        workflowTab.click();
      });
    }

    const activityToggle = container.querySelector(
      "button[aria-label='切换执行经过']",
    ) as HTMLButtonElement | null;
    if (activityToggle) {
      act(() => {
        activityToggle.click();
      });
    }

    const revealArtifactButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) =>
      button
        .getAttribute("aria-label")
        ?.startsWith("定位活动产物路径-content-posts/group.md"),
    );
    expect(revealArtifactButton).toBeTruthy();
    if (revealArtifactButton) {
      act(() => {
        revealArtifactButton.click();
      });
    }
    expect(mockRevealSessionFileInFinder).toHaveBeenCalledWith(
      "session-group",
      "content-posts/group.md",
    );

    const openArtifactButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) =>
      button
        .getAttribute("aria-label")
        ?.startsWith("打开活动产物路径-content-posts/group.md"),
    );
    expect(openArtifactButton).toBeTruthy();
    if (openArtifactButton) {
      act(() => {
        openArtifactButton.click();
      });
    }
    expect(mockOpenSessionFileWithDefaultApp).toHaveBeenCalledWith(
      "session-group",
      "content-posts/group.md",
    );
  });

  it("定位产物失败时应透传后端错误信息", async () => {
    const consoleWarnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => {});
    mockRevealSessionFileInFinder.mockRejectedValueOnce(
      new Error("文件不存在"),
    );
    try {
      const { container } = renderSidebar({
        activeRunDetail: {
          id: "run-error-path",
          source: "skill",
          source_ref: "content_post_with_cover",
          session_id: "session-error",
          status: "success",
          started_at: "2026-03-06T02:10:03Z",
          finished_at: "2026-03-06T02:10:08Z",
          duration_ms: 5000,
          error_code: null,
          error_message: null,
          metadata: JSON.stringify({
            artifact_paths: ["content-posts/error.md"],
          }),
          created_at: "2026-03-06T02:10:03Z",
          updated_at: "2026-03-06T02:10:08Z",
        },
      });

      const workflowTab = container.querySelector(
        'button[aria-label="打开当前进展"]',
      ) as HTMLButtonElement | null;
      if (workflowTab) {
        act(() => {
          workflowTab.click();
        });
      }

      const activityToggle = container.querySelector(
        "button[aria-label='切换执行经过']",
      ) as HTMLButtonElement | null;
      if (activityToggle) {
        act(() => {
          activityToggle.click();
        });
      }

      const revealArtifactButton = Array.from(
        container.querySelectorAll("button"),
      ).find((button) =>
        button
          .getAttribute("aria-label")
          ?.startsWith("定位产物路径-content-posts/error.md"),
      );
      expect(revealArtifactButton).toBeTruthy();
      if (revealArtifactButton) {
        await act(async () => {
          revealArtifactButton.click();
          await Promise.resolve();
        });
      }

      expect(mockToastError).toHaveBeenCalledWith(
        expect.stringContaining("文件不存在"),
      );
    } finally {
      consoleWarnSpy.mockRestore();
    }
  });
});
