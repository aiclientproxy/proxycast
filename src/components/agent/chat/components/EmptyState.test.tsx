import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EmptyState } from "./EmptyState";
import type { Character } from "@/lib/api/projectMemory";
import type { ProjectGitStatus } from "@/lib/api/projectGit";
import type { Skill } from "@/lib/api/skills";
import type { ServiceSkillHomeItem } from "../service-skills/types";
import type { InputCapabilitySelection } from "../skill-selection/inputCapabilitySelection";
import type { InputbarSendPayload } from "./Inputbar/inputbarSendPayload";
import { recordSlashEntryUsage } from "../skill-selection/slashEntryUsage";
import {
  buildCuratedTaskLaunchPrompt,
  findCuratedTaskTemplateById,
  recordCuratedTaskTemplateUsage,
} from "../utils/curatedTaskTemplates";
import { changeLimeLocale } from "@/i18n/createI18n";
import {
  recordCuratedTaskRecommendationSignalFromMemoryReference,
  recordCuratedTaskRecommendationSignalFromReviewDecision,
} from "../utils/curatedTaskRecommendationSignals";
import { DEFAULT_LIME_COLOR_SCHEME_ID } from "@/lib/appearance/colorSchemes";
import { getLimeSkinCopy } from "@/lib/appearance/skinContent";

const {
  mockGetConfig,
  mockProjectSelector,
  mockReadProjectGitStatus,
  mockCheckoutProjectGitBranch,
  mockCreateProjectGitBranch,
  mockCreateProjectGitWorktree,
} = vi.hoisted(() => ({
  mockGetConfig: vi.fn(async () => ({})),
  mockProjectSelector: vi.fn(),
  mockReadProjectGitStatus: vi.fn(
    async (): Promise<ProjectGitStatus> => ({
      rootPath: "/workspace/lime",
      repositoryRoot: undefined,
      hasGitRepository: false,
      currentBranch: undefined,
      branches: [],
      uncommittedFileCount: 0,
    }),
  ),
  mockCheckoutProjectGitBranch: vi.fn(async () => ({
    rootPath: "/workspace/lime",
    repositoryRoot: "/workspace/lime",
    hasGitRepository: true,
    currentBranch: "main",
    branches: ["feature/demo", "main"],
    uncommittedFileCount: 0,
  })),
  mockCreateProjectGitBranch: vi.fn(async () => ({
    rootPath: "/workspace/lime",
    repositoryRoot: "/workspace/lime",
    hasGitRepository: true,
    currentBranch: "feature/new",
    branches: ["feature/demo", "feature/new", "main"],
    uncommittedFileCount: 0,
  })),
  mockCreateProjectGitWorktree: vi.fn(async () => ({
    worktreePath: "/workspace/lime-worktree-test",
    branch: "main",
    status: {
      rootPath: "/workspace/lime-worktree-test",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "abcdef0",
      branches: ["feature/demo", "main"],
      uncommittedFileCount: 0,
    },
  })),
}));

const mockGetSkillCatalog = vi.hoisted(() =>
  vi.fn(async () => ({
    version: "test",
    tenantId: "tenant-0001",
    syncedAt: "2026-04-30T00:00:00Z",
    groups: [],
    items: [],
    entries: [],
  })),
);

const {
  mockGatewayChannelStatus,
  mockGetBrowserConnectorSettings,
  mockGetChromeBridgeStatus,
} = vi.hoisted(() => ({
  mockGatewayChannelStatus: vi.fn(),
  mockGetBrowserConnectorSettings: vi.fn(),
  mockGetChromeBridgeStatus: vi.fn(),
}));


const mockCharacterMention = vi.fn<
  (props: {
    characters?: Character[];
    skills?: Skill[];
    serviceSkills?: ServiceSkillHomeItem[];
    onSelectSkill?: (skill: Skill) => void;
    onSelectInputCapability?: (
      capability: InputCapabilitySelection,
      options?: { replayText?: string },
    ) => void;
    onSelectServiceSkill?: (skill: ServiceSkillHomeItem) => void;
    defaultCuratedTaskReferenceMemoryIds?: string[];
    defaultCuratedTaskReferenceEntries?: Array<{
      id: string;
      title: string;
      summary: string;
      category: string;
      categoryLabel: string;
      tags: string[];
    }>;
    value: string;
    onChange: (value: string) => void;
  }) => React.ReactNode
>();

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
}));

vi.mock("@/lib/api/channelsRuntime", () => ({
  gatewayChannelStatus: mockGatewayChannelStatus,
}));

vi.mock("@/lib/api/skillCatalog", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/api/skillCatalog")>();
  return {
    ...actual,
    getSkillCatalog: mockGetSkillCatalog,
    listSkillCatalogSceneEntries: () => [],
    subscribeSkillCatalogChanged: () => () => undefined,
  };
});

vi.mock("@/lib/webview-api", () => ({
  getBrowserConnectorSettings: mockGetBrowserConnectorSettings,
  getChromeBridgeStatus: mockGetChromeBridgeStatus,
}));

vi.mock("@/components/projects/ProjectSelector", () => ({
  ProjectSelector: (props: {
    value: string | null;
    placeholder?: string;
    onChange?: (projectId: string) => void;
  }) => {
    mockProjectSelector(props);
    return (
      <div data-testid="project-selector-stub">
        {props.value ?? props.placeholder ?? "选择项目"}
      </div>
    );
  },
}));

vi.mock("@/lib/api/projectGit", () => ({
  readProjectGitStatus: mockReadProjectGitStatus,
  checkoutProjectGitBranch: mockCheckoutProjectGitBranch,
  createProjectGitBranch: mockCreateProjectGitBranch,
  createProjectGitWorktree: mockCreateProjectGitWorktree,
}));

vi.mock("./ChatModelSelector", () => ({
  ChatModelSelector: () => <div data-testid="chat-model-selector" />,
}));

vi.mock("../utils/contextualRecommendations", () => ({
  buildRecommendationPrompt: vi.fn((fullPrompt: string) => fullPrompt),
  getContextualRecommendations: vi.fn(() => []),
}));

vi.mock("../skill-selection/CharacterMention", () => ({
  CharacterMention: (props: {
    characters?: Character[];
    skills?: Skill[];
    serviceSkills?: ServiceSkillHomeItem[];
    onSelectSkill?: (skill: Skill) => void;
    onSelectInputCapability?: (
      capability: InputCapabilitySelection,
      options?: { replayText?: string },
    ) => void;
    onSelectServiceSkill?: (skill: ServiceSkillHomeItem) => void;
    defaultCuratedTaskReferenceMemoryIds?: string[];
    defaultCuratedTaskReferenceEntries?: Array<{
      id: string;
      title: string;
      summary: string;
      category: string;
      categoryLabel: string;
      tags: string[];
    }>;
    value: string;
    onChange: (value: string) => void;
  }) => {
    mockCharacterMention(props);
    return <div data-testid="character-mention-stub" />;
  },
}));

vi.mock("../skill-selection/SkillSelector", () => ({
  SkillSelector: () => <div data-testid="skill-selector-stub" />,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    ...rest
  }: {
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
    [key: string]: unknown;
  }) => (
    <button type="button" onClick={onClick} disabled={disabled} {...rest}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/input", () => ({
  Input: ({
    value,
    onChange,
    placeholder,
    ...rest
  }: {
    value?: string;
    onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
    placeholder?: string;
    [key: string]: unknown;
  }) => (
    <input
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      {...rest}
    />
  ),
}));

vi.mock("@/components/ui/textarea", () => {
  const Textarea = React.forwardRef<
    HTMLTextAreaElement,
    React.TextareaHTMLAttributes<HTMLTextAreaElement>
  >((props, ref) => <textarea ref={ref} {...props} />);
  return { Textarea };
});

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  DialogHeader: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  DialogFooter: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  DialogTitle: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  DialogDescription: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("@/components/ui/select", () => ({
  Select: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  SelectContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  SelectItem: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  SelectTrigger: ({ children }: { children: React.ReactNode }) => (
    <button type="button">{children}</button>
  ),
  SelectValue: () => null,
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  PopoverContent: ({
    children,
    align: _align,
    side: _side,
    sideOffset: _sideOffset,
    onCloseAutoFocus: _onCloseAutoFocus,
    ...rest
  }: {
    children: React.ReactNode;
    align?: string;
    side?: string;
    sideOffset?: number;
    onCloseAutoFocus?: unknown;
    [key: string]: unknown;
  }) => (
    <div {...(rest as React.HTMLAttributes<HTMLDivElement>)}>{children}</div>
  ),
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({
    children,
    onClick,
  }: {
    children: React.ReactNode;
    onClick?: () => void;
  }) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  await changeLimeLocale("zh-CN");
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  mockGetConfig.mockImplementation(async () => ({}));
  mockGatewayChannelStatus.mockResolvedValue({
    status: {
      running_accounts: 0,
    },
  });
  mockGetBrowserConnectorSettings.mockResolvedValue({
    enabled: true,
    system_connectors: [
      {
        connector_type: "chrome",
        available: true,
        authorization_status: "authorized",
      },
    ],
  });
  mockGetChromeBridgeStatus.mockResolvedValue({
    observer_count: 1,
    control_count: 0,
  });
  mockReadProjectGitStatus.mockResolvedValue({
    rootPath: "/workspace/lime",
    repositoryRoot: undefined,
    hasGitRepository: false,
    currentBranch: undefined,
    branches: [],
    uncommittedFileCount: 0,
  });
  mockCheckoutProjectGitBranch.mockResolvedValue({
    rootPath: "/workspace/lime",
    repositoryRoot: "/workspace/lime",
    hasGitRepository: true,
    currentBranch: "main",
    branches: ["feature/demo", "main"],
    uncommittedFileCount: 0,
  });
  mockCreateProjectGitBranch.mockResolvedValue({
    rootPath: "/workspace/lime",
    repositoryRoot: "/workspace/lime",
    hasGitRepository: true,
    currentBranch: "feature/new",
    branches: ["feature/demo", "feature/new", "main"],
    uncommittedFileCount: 0,
  });
  mockCreateProjectGitWorktree.mockResolvedValue({
    worktreePath: "/workspace/lime-worktree-test",
    branch: "main",
    status: {
      rootPath: "/workspace/lime-worktree-test",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "abcdef0",
      branches: ["feature/demo", "main"],
      uncommittedFileCount: 0,
    },
  });
  window.localStorage.clear();
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
  window.localStorage.clear();
  vi.clearAllMocks();
});

function renderEmptyState(
  props?: Partial<React.ComponentProps<typeof EmptyState>>,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const defaultProps: React.ComponentProps<typeof EmptyState> = {
    input: "",
    setInput: vi.fn(),
    onSend: vi.fn(),
    providerType: "openai",
    setProviderType: vi.fn(),
    model: "gpt-4.1",
    setModel: vi.fn(),
  };

  act(() => {
    root.render(<EmptyState {...defaultProps} {...props} />);
  });

  mountedRoots.push({ root, container });
  return container;
}

async function flushAsyncEffects(turns = 4): Promise<void> {
  await act(async () => {
    for (let index = 0; index < turns; index += 1) {
      await Promise.resolve();
    }
  });
}

async function waitForProjectContextSelector(
  container: HTMLElement,
  selector: string,
  turns = 12,
): Promise<Element | null> {
  for (let index = 0; index < turns; index += 1) {
    const element = container.querySelector(selector);
    if (element) {
      return element;
    }
    await flushAsyncEffects(1);
  }
  return container.querySelector(selector);
}

function expectEmptyStateSend(
  onSend: ReturnType<typeof vi.fn>,
  payload: Pick<
    InputbarSendPayload,
    "images" | "textOverride" | "sendOptions"
  > = {},
) {
  const actual = onSend.mock.calls.at(-1)?.[0] as
    | InputbarSendPayload
    | undefined;
  expect(actual).toBeTruthy();
  expect(actual?.images).toEqual(payload.images);
  expect(actual?.textOverride).toEqual(payload.textOverride);
  expect(actual?.sendOptions).toEqual(payload.sendOptions);
}

function updateFieldValue(
  element: HTMLInputElement | HTMLTextAreaElement | null,
  value: string,
) {
  expect(element).toBeTruthy();
  if (!element) {
    return;
  }

  const prototype =
    element instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
  setter?.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

function findLauncherConfirmButton() {
  return (
    (document.body.querySelector(
      '[data-testid="curated-task-launcher-confirm"]',
    ) as HTMLButtonElement | null) ??
    (Array.from(document.body.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("开始生成"),
    ) as HTMLButtonElement | undefined)
  );
}

async function openHomeMoreSkillsDrawer(
  container: HTMLElement,
): Promise<HTMLElement> {
  const trigger = container.querySelector(
    '[data-testid="home-more-skills-trigger"]',
  ) as HTMLButtonElement | null;
  expect(trigger).toBeTruthy();

  await act(async () => {
    trigger?.click();
    await Promise.resolve();
  });

  const drawer = container.querySelector(
    '[data-testid="home-more-skills-drawer"]',
  ) as HTMLElement | null;
  expect(drawer).toBeTruthy();
  if (!drawer) {
    throw new Error("home more skills drawer should be visible");
  }
  return drawer;
}

function queryHomeDrawerButton(
  drawer: HTMLElement,
  testId: string,
): HTMLButtonElement | null {
  return drawer.querySelector(
    `[data-testid="${testId}"]`,
  ) as HTMLButtonElement | null;
}

function expectHomeDrawerButton(
  drawer: HTMLElement,
  testId: string,
): HTMLButtonElement {
  const button = queryHomeDrawerButton(drawer, testId);
  expect(button).toBeTruthy();
  if (!button) {
    throw new Error(`expected home drawer button ${testId}`);
  }
  return button;
}

function createRepositoryResearchServiceSkill(): ServiceSkillHomeItem {
  return {
    id: "repository-research",
    title: "仓库线索研究",
    summary: "围绕仓库主题整理项目线索。",
    category: "情报研究",
    outputHint: "仓库列表 + 关键线索",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "先做这一轮",
    runnerTone: "emerald",
    runnerDescription: "先在当前工作区整理一版仓库研究结果。",
    actionLabel: "补齐这一步",
    automationStatus: null,
    slotSchema: [
      {
        key: "repository_query",
        label: "检索主题",
        type: "text",
        required: true,
        placeholder: "例如 AI Agent",
      },
    ],
  };
}

function createSceneBoundServiceSkill(): ServiceSkillHomeItem {
  return {
    id: "project-insight-flow",
    title: "项目线索整理",
    summary: "围绕当前项目整理线索、结论和下一步动作。",
    category: "研究与方案",
    outputHint: "线索清单 + 下一步建议",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "立即开始",
    runnerTone: "emerald",
    runnerDescription: "围绕当前项目继续整理线索和结果。",
    actionLabel: "继续整理",
    automationStatus: null,
    slotSchema: [],
    sceneBinding: {
      sceneKey: "project-insight-flow",
      commandPrefix: "/project-insight-flow",
      title: "项目线索整理",
      summary: "围绕当前项目整理线索和下一步动作。",
    },
  };
}

describe("EmptyState", () => {
  it("首页应以 slogan 作为主视觉并保留创作语义", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("创作");
    expect(container.textContent).toContain("青柠一下，灵感即来");
    expect(
      container.querySelector('[data-testid="empty-state-hero-eyebrow-badge"]')
        ?.textContent,
    ).toBe("创作");
    expect(container.textContent).toContain("说一句目标，Lime 就接着帮你做。");
    expect(container.textContent).not.toContain(
      "文案、图片、视频、搜索和网页任务围绕同一目标持续推进，并沉淀上下文、偏好和做法。",
    );
    expect(container.textContent).not.toContain("通用对话");
    expect(container.textContent).not.toContain("新建任务");
  });

  it("通用首页应挂载轻量起手区与下滑第二屏，无 Git 仓库时不显示分支", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist: vi.fn(),
      onSelectServiceSkill: vi.fn(),
      projectId: "lime",
      openedProjects: [
        {
          id: "lime",
          name: "lime",
          rootPath: "/workspace/lime",
        },
      ],
    });

    await flushAsyncEffects();

    expect(
      container.querySelector('[data-testid="home-start-surface"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="home-starter-chips"]'),
    ).toBeTruthy();
    const primaryStack = container.querySelector(
      '[data-testid="empty-state-primary-stack"]',
    );
    const inputbarCore = primaryStack?.querySelector(
      '[data-testid="inputbar-core-container"]',
    );
    const homeStartSurface = primaryStack?.querySelector(
      '[data-testid="home-start-surface"]',
    );
    const projectContextSlot = primaryStack?.querySelector(
      '[data-testid="inputbar-context-bar-slot"]',
    );
    const connectedComposer = primaryStack?.querySelector(
      '[data-testid="inputbar-connected-composer"]',
    );
    const projectContextBar = primaryStack?.querySelector(
      '[data-testid="inputbar-project-context-bar"]',
    );
    expect(primaryStack).toBeTruthy();
    expect(inputbarCore).toBeTruthy();
    expect(connectedComposer).toBeTruthy();
    expect(projectContextSlot).toBeTruthy();
    expect(projectContextBar).toBeTruthy();
    expect(connectedComposer?.contains(inputbarCore ?? null)).toBe(true);
    expect(connectedComposer?.contains(projectContextSlot ?? null)).toBe(true);
    expect(homeStartSurface).toBeTruthy();
    expect(
      Boolean(
        inputbarCore &&
        projectContextSlot &&
        (inputbarCore.compareDocumentPosition(projectContextSlot) &
          Node.DOCUMENT_POSITION_FOLLOWING) !==
          0,
      ),
    ).toBe(true);
    expect(
      Boolean(
        projectContextSlot &&
        homeStartSurface &&
        (projectContextSlot.compareDocumentPosition(homeStartSurface) &
          Node.DOCUMENT_POSITION_FOLLOWING) !==
          0,
      ),
    ).toBe(true);
    expect(projectContextBar?.textContent).toContain("lime");
    expect(projectContextBar?.textContent).toContain("本地模式");
    expect(projectContextBar?.textContent).not.toContain("main");
    expect(
      projectContextBar?.querySelector(
        '[data-testid="inputbar-project-context-branch"]',
      ),
    ).toBeNull();
    expect(container.textContent).toContain("引导帮助");
    expect(container.textContent).toContain("写作");
    expect(container.textContent).toContain("添加资料");
    expect(container.textContent).toContain("PPT");
    expect(container.textContent).toContain("调研报告");
    expect(container.textContent).toContain("更多做法");
    expect(
      container.querySelector('[data-testid="home-guide-cards"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="entry-home-knowledge-import"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="home-scroll-cue"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain(
      "向下滑，看看 Lime 可以帮你做什么",
    );
    expect(container.textContent).not.toContain(
      "往下看更多任务样例；真正执行仍会回到生成里继续补充。",
    );
    expect(
      container.querySelector('[data-testid="home-second-screen"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="home-skill-gallery"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("你可以从这些任务开始");
    expect(container.textContent).toContain("每日趋势摘要");
    expect(container.textContent).not.toContain("先开始这一轮");
    expect(container.textContent).not.toContain("其他起手结果");
    expect(container.textContent).not.toContain("也可以直接按做法开工");
    expect(
      container.querySelector('[data-testid^="entry-continuation-"]'),
    ).toBeNull();
  });

  it("通用首页项目包含 Git 目录时应显示启动模式和分支菜单", async () => {
    mockReadProjectGitStatus.mockResolvedValue({
      rootPath: "/workspace/lime",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "main",
      branches: ["feature/demo", "main"],
      uncommittedFileCount: 1,
    });

    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist: vi.fn(),
      onSelectServiceSkill: vi.fn(),
      projectId: "lime",
      openedProjects: [
        {
          id: "lime",
          name: "lime",
          rootPath: "/workspace/lime",
        },
      ],
    });

    await flushAsyncEffects(6);
    await waitForProjectContextSelector(
      container,
      '[data-testid="inputbar-project-context-branch"]',
    );

    const projectContextBar = container.querySelector(
      '[data-testid="inputbar-project-context-bar"]',
    );

    expect(projectContextBar).toBeTruthy();
    expect(
      projectContextBar?.querySelector(
        '[data-testid="inputbar-project-context-mode-menu"]',
      ),
    ).toBeTruthy();
    expect(
      projectContextBar?.querySelector(
        '[data-testid="inputbar-project-context-branch"]',
      ),
    ).toBeTruthy();
    expect(projectContextBar?.textContent).toContain("启动模式");
    expect(projectContextBar?.textContent).toContain("在本地处理");
    expect(projectContextBar?.textContent).toContain("新工作树");
    expect(projectContextBar?.textContent).toContain("未提交：1 个文件");
    expect(projectContextBar?.textContent).toContain("main");
    expect(projectContextBar?.textContent).toContain("feature/demo");
  });

  it("通用首页项目分支菜单应通过后端切换分支", async () => {
    mockReadProjectGitStatus.mockResolvedValue({
      rootPath: "/workspace/lime",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "main",
      branches: ["feature/demo", "main"],
      uncommittedFileCount: 0,
    });
    mockCheckoutProjectGitBranch.mockResolvedValue({
      rootPath: "/workspace/lime",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "feature/demo",
      branches: ["feature/demo", "main"],
      uncommittedFileCount: 0,
    });

    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist: vi.fn(),
      onSelectServiceSkill: vi.fn(),
      projectId: "lime",
      openedProjects: [
        {
          id: "lime",
          name: "lime",
          rootPath: "/workspace/lime",
        },
      ],
    });

    await flushAsyncEffects(6);

    const featureBranch = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("feature/demo"),
    );
    expect(featureBranch).toBeTruthy();

    await act(async () => {
      featureBranch?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(mockCheckoutProjectGitBranch).toHaveBeenCalledWith(
      "/workspace/lime",
      "feature/demo",
    );
    expect(
      container.querySelector('[data-testid="inputbar-project-context-bar"]')
        ?.textContent,
    ).toContain("feature/demo");
  });

  it("通用首页项目分支搜索回车应通过后端创建并检出新分支", async () => {
    mockReadProjectGitStatus.mockResolvedValue({
      rootPath: "/workspace/lime",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "main",
      branches: ["feature/demo", "main"],
      uncommittedFileCount: 0,
    });
    mockCreateProjectGitBranch.mockResolvedValue({
      rootPath: "/workspace/lime",
      repositoryRoot: "/workspace/lime",
      hasGitRepository: true,
      currentBranch: "feature/new",
      branches: ["feature/demo", "feature/new", "main"],
      uncommittedFileCount: 0,
    });

    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist: vi.fn(),
      onSelectServiceSkill: vi.fn(),
      projectId: "lime",
      openedProjects: [
        {
          id: "lime",
          name: "lime",
          rootPath: "/workspace/lime",
        },
      ],
    });

    await flushAsyncEffects(6);

    const branchSearch = container.querySelector(
      'input[placeholder="搜索分支"]',
    ) as HTMLInputElement | null;
    updateFieldValue(branchSearch, "feature/new");

    await act(async () => {
      branchSearch?.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          bubbles: true,
        }),
      );
      await Promise.resolve();
    });

    expect(mockCreateProjectGitBranch).toHaveBeenCalledWith(
      "/workspace/lime",
      "feature/new",
    );
    expect(
      container.querySelector('[data-testid="inputbar-project-context-bar"]')
        ?.textContent,
    ).toContain("feature/new");
  });

  it("第二屏向上回首屏应使用非 passive wheel listener", async () => {
    const addEventListenerSpy = vi.spyOn(
      HTMLElement.prototype,
      "addEventListener",
    );
    const originalScrollTo = HTMLElement.prototype.scrollTo;
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      value: scrollTo,
    });

    try {
      const container = renderEmptyState({
        activeTheme: "general",
        onLaunchBrowserAssist: vi.fn(),
        onSelectServiceSkill: vi.fn(),
      });

      await act(async () => {
        await Promise.resolve();
      });

      expect(addEventListenerSpy).toHaveBeenCalledWith(
        "wheel",
        expect.any(Function),
        { passive: false },
      );

      const secondScreen = container.querySelector(
        '[data-testid="home-second-screen"]',
      ) as HTMLElement | null;
      expect(secondScreen).toBeTruthy();

      act(() => {
        secondScreen?.dispatchEvent(
          new WheelEvent("wheel", {
            bubbles: true,
            cancelable: true,
            deltaY: -24,
          }),
        );
      });

      expect(scrollTo).toHaveBeenCalledWith({ top: 0, behavior: "smooth" });
    } finally {
      addEventListenerSpy.mockRestore();
      if (originalScrollTo) {
        Object.defineProperty(HTMLElement.prototype, "scrollTo", {
          configurable: true,
          value: originalScrollTo,
        });
      } else {
        Object.defineProperty(HTMLElement.prototype, "scrollTo", {
          configurable: true,
          value: undefined,
        });
      }
    }
  });

  it("通用首页静态起手与引导文案应跟随 en-US 资源", async () => {
    await changeLimeLocale("en-US");
    const skinCopy = getLimeSkinCopy(DEFAULT_LIME_COLOR_SCHEME_ID, "en-US");
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain(skinCopy.brandSubtitle);
    expect(container.textContent).toContain(skinCopy.tagline);
    expect(container.textContent).toContain(skinCopy.eyebrow);
    expect(container.textContent).toContain(skinCopy.slogan);
    expect(container.textContent).toContain(skinCopy.description);
    expect(container.textContent).toContain(skinCopy.status);
    expect(container.textContent).toContain("Guide help");
    expect(container.textContent).toContain("Writing");
    expect(container.textContent).toContain("Add knowledge");
    expect(container.textContent).toContain("More methods");
    expect(container.textContent).toContain("Start from these tasks");
    expect(
      container.querySelector('[data-testid="home-scroll-cue"]'),
    ).toBeNull();

    const guideTrigger = container.querySelector(
      '[data-testid="home-guide-help-trigger"]',
    ) as HTMLButtonElement | null;
    act(() => {
      guideTrigger?.click();
    });

    expect(container.textContent).toContain(
      "How do I add and use project knowledge?",
    );
    expect(container.querySelector("textarea")?.placeholder).toContain(
      "What would you like to learn?",
    );
  });

  it("非通用首页快速启动面板应跟随 en-US 资源", async () => {
    await changeLimeLocale("en-US");
    const container = renderEmptyState({
      activeTheme: "image",
      serviceSkills: [],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("Quick start");
    expect(container.textContent).toContain(
      "Choose a task template first, then keep adding details and follow-ups in this session.",
    );
    expect(container.textContent).toContain("Generate image");
    expect(container.textContent).toContain("Organize as Notebook");
    expect(container.textContent).toContain("Enter research mode");
  });

  it("仅发送路径引用时应使用当前 locale 的兜底 prompt", async () => {
    await changeLimeLocale("en-US");
    const onSend = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      input: "",
      onSend,
      pathReferences: [
        {
          id: "path-1",
          path: "/tmp/report.md",
          name: "report.md",
          isDir: false,
          source: "file_manager",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      'button[aria-label="Send"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "Please review these files or folders.",
      sendOptions: expect.objectContaining({
        requestMetadata: expect.objectContaining({
          path_references: [
            expect.objectContaining({
              path: "/tmp/report.md",
              name: "report.md",
            }),
          ],
        }),
      }),
    });
    const payload = onSend.mock.calls.at(-1)?.[0] as
      | InputbarSendPayload
      | undefined;
    expect(payload?.triggerSource).toBe("button");
    expect(payload?.triggeredAt).toEqual(expect.any(Number));
  });

  it("启用 pointerdown 发送时首页发送按钮不应在 click 重复发送", async () => {
    const nowSpy = vi.spyOn(Date, "now").mockReturnValue(1_780_000_200_000);
    const onSend = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      input: "你好",
      onSend,
      sendOnPointerDown: true,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.dispatchEvent(new Event("pointerdown", { bubbles: true }));
      sendButton?.click();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectEmptyStateSend(onSend, {
      textOverride: "你好",
    });
    const payload = onSend.mock.calls.at(-1)?.[0] as
      | InputbarSendPayload
      | undefined;
    expect(payload?.triggerSource).toBe("button");
    expect(payload?.triggeredAt).toBe(1_780_000_200_000);
    nowSpy.mockRestore();
  });

  it("从 Skills 页带回的技能应显示在首页输入框内的 @ 标签", async () => {
    const onSend = vi.fn();
    const skill = {
      key: "local:writer",
      name: "写作助手",
      description: "本地补充技能",
      directory: "writer",
      installed: true,
      sourceKind: "other",
    } as Skill;
    const container = renderEmptyState({
      activeTheme: "general",
      input: "整理最近发布计划",
      onSend,
      skills: [skill],
      initialInputCapability: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: skill.key,
          skillName: skill.name,
        },
        requestKey: 20260512,
      },
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const badge = container.querySelector('[data-testid="input-skill-badge"]');
    expect(badge).toBeTruthy();
    expect(badge?.textContent).toContain("@");
    expect(badge?.textContent).toContain("写作助手");

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "整理最近发布计划",
      sendOptions: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "writer",
          skillName: "写作助手",
        },
        displayContent: "整理最近发布计划",
      },
    });
  });

  it("首页中断恢复请求应由 EmptyState 完整恢复文本、图片、路径和技能 route", async () => {
    const onSend = vi.fn();
    const onInputRestoreRequestHandled = vi.fn();
    const pathReference = {
      id: "file:/tmp/report.md",
      path: "/tmp/report.md",
      name: "report.md",
      isDir: false,
      source: "file_manager" as const,
    };
    const image = {
      data: "image-data",
      mediaType: "image/png",
    };
    const capabilityRoute = {
      kind: "installed_skill" as const,
      skillKey: "draft",
      skillName: "起草",
    };
    const skill = {
      key: "local:draft",
      name: "起草",
      description: "恢复输入用技能",
      directory: "draft",
      installed: true,
      sourceKind: "other",
    } as Skill;
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedRoots.push({ root, container });

    function RestoreHarness() {
      const [input, setInput] = React.useState("");
      const [pathReferences, setPathReferences] = React.useState<
        (typeof pathReference)[]
      >([]);
      const [restoreRequest, setRestoreRequest] = React.useState<
        React.ComponentProps<typeof EmptyState>["inputRestoreRequest"]
      >({
        requestId: "restore-empty-state-1",
        reason: "output_free_interrupted_turn",
        draft: {
          text: "继续生成提纲",
          images: [image],
          pathReferences: [pathReference],
          inputCapabilityRoute: capabilityRoute,
        },
      });

      return (
        <EmptyState
          input={input}
          setInput={setInput}
          onSend={onSend}
          providerType="openai"
          setProviderType={vi.fn()}
          model="gpt-4.1"
          setModel={vi.fn()}
          activeTheme="general"
          skills={[skill]}
          pathReferences={pathReferences}
          onClearPathReferences={() => setPathReferences([])}
          onAddPathReferences={(references) =>
            setPathReferences(references as (typeof pathReference)[])
          }
          inputRestoreRequest={restoreRequest}
          onInputRestoreRequestHandled={(requestId) => {
            onInputRestoreRequestHandled(requestId);
            setRestoreRequest(null);
          }}
        />
      );
    }

    await act(async () => {
      root.render(<RestoreHarness />);
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushAsyncEffects();

    expect(container.querySelector("textarea")?.value).toBe("继续生成提纲");
    expect(
      container.querySelector('[data-testid="input-skill-badge"]'),
    ).toBeTruthy();
    expect(onInputRestoreRequestHandled).toHaveBeenCalledWith(
      "restore-empty-state-1",
    );

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      images: [image],
      textOverride: "继续生成提纲",
      sendOptions: expect.objectContaining({
        capabilityRoute,
        inputRestoreDraft: expect.objectContaining({
          text: "继续生成提纲",
          images: [image],
          pathReferences: [pathReference],
          inputCapabilityRoute: capabilityRoute,
        }),
      }),
    });
  });

  it("首页添加资料入口应打开输入框资料中枢，而不是预填一段说明", async () => {
    const setInput = vi.fn();
    const onToggleKnowledgePack = vi.fn<(enabled: boolean) => void>();
    const container = renderEmptyState({
      setInput,
      knowledgePackSelection: {
        enabled: false,
        packName: "team-notes",
        workingDir: "workspace-root",
        label: "团队资料",
        status: "ready",
      },
      knowledgePackOptions: [
        {
          packName: "team-notes",
          label: "团队资料",
          status: "ready",
          defaultForWorkspace: true,
        },
      ],
      onToggleKnowledgePack,
      onSelectKnowledgePack: vi.fn(),
      onStartKnowledgeOrganize: vi.fn(),
      onManageKnowledgePacks: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    const starter = container.querySelector(
      '[data-testid="entry-home-knowledge-import"]',
    ) as HTMLButtonElement | null;
    expect(starter?.textContent).toContain("添加资料");

    act(() => {
      starter?.click();
    });

    expect(setInput).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="inputbar-knowledge-hub"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("团队资料");

    const useKnowledgeButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("使用这份资料"));

    act(() => {
      useKnowledgeButton?.click();
    });

    expect(onToggleKnowledgePack).toHaveBeenCalledWith(true);
  });

  it("点击引导帮助后进入可关闭的帮助模式，关闭后恢复默认起手入口", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const guideTrigger = container.querySelector(
      '[data-testid="home-guide-help-trigger"]',
    ) as HTMLButtonElement | null;
    expect(guideTrigger).toBeTruthy();

    act(() => {
      guideTrigger?.click();
    });

    expect(
      container.querySelector('[data-testid="home-guide-help-active-badge"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="home-guide-cards"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("项目资料怎么添加和使用？");
    expect(
      container.querySelector('[data-testid="home-starter-chips"]'),
    ).toBeNull();
    expect(container.querySelector("textarea")?.placeholder).toContain(
      "想了解什么？试试",
    );

    const closeGuide = container.querySelector(
      '[data-testid="home-guide-help-active-badge"] button',
    ) as HTMLButtonElement | null;

    act(() => {
      closeGuide?.click();
    });

    expect(
      container.querySelector('[data-testid="home-guide-help-active-badge"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="home-guide-cards"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="home-starter-chips"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="home-input-tab-suggestion"]'),
    ).toBeNull();
  });

  it("点击首屏起手任务只挂载输入区能力，不弹出启动表单", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const starter = container.querySelector(
      '[data-testid="entry-recommended-daily-trend-briefing"]',
    ) as HTMLButtonElement | null;
    expect(starter).toBeTruthy();

    act(() => {
      starter?.click();
    });

    expect(document.body.textContent).not.toContain(
      "开始这一步前，我先确认几件事。",
    );
    expect(
      container.querySelector('[data-testid="curated-task-badge"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("每日趋势摘要");
  });

  it("保存判断结论后不应打乱首屏固定起手入口", async () => {
    const projectId = "project-review-recommendation";
    const container = renderEmptyState({
      activeTheme: "general",
      projectId,
      onLaunchBrowserAssist: vi.fn(),
      onSelectServiceSkill: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("围绕最近判断");
    expect(
      container.querySelector('[data-testid="entry-review-feedback-banner"]'),
    ).toBeNull();
    const initialStarterItems = Array.from(
      container.querySelectorAll('[data-testid^="entry-recommended-"]'),
    ).map((element) => element.getAttribute("data-testid"));

    await act(async () => {
      recordCuratedTaskRecommendationSignalFromReviewDecision(
        {
          session_id: "session-review-needs-evidence",
          decision_status: "needs_more_evidence",
          decision_summary:
            "这轮结果还缺证据，需要回到账号表现和爆款样本继续补证据。",
          chosen_fix_strategy: "先补账号数据复盘，再拆一轮高表现内容做对照。",
          risk_level: "medium",
          risk_tags: ["证据不足", "需要复盘"],
          followup_actions: ["补账号数据复盘", "拆解一条高表现内容"],
        },
        {
          projectId,
          sceneTitle: "短视频编排",
        },
      );
      await Promise.resolve();
    });

    const starterItems = Array.from(
      container.querySelectorAll('[data-testid^="entry-recommended-"]'),
    ).map((element) => element.getAttribute("data-testid"));

    expect(starterItems).toEqual(initialStarterItems);
    expect(starterItems[0]).toBe("entry-recommended-social-post-starter");
    expect(
      container.querySelector('[data-testid="entry-review-feedback-banner"]'),
    ).toBeNull();
  });

  it("无浏览器入口且无可续接动作时，不应默认渲染补充续接条", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="entry-supplemental-panel"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="entry-connect-browser"]'),
    ).toBeNull();
  });

  it("只有未形成直接收益的方法目录时，不应默认渲染补充续接条", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [createSceneBoundServiceSkill()],
      onSelectServiceSkill: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="entry-supplemental-panel"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="entry-connect-browser"]'),
    ).toBeNull();
  });

  it("首页快捷做法应复用 service skill 的统一合同缩略", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [
        {
          id: "daily-trend-briefing",
          title: "每日趋势摘要",
          summary: "围绕指定平台与关键词输出趋势摘要。",
          entryHint: "把平台和关键词给我，我先整理一份趋势报告。",
          category: "内容运营",
          outputHint: "趋势摘要 + 调度建议",
          source: "cloud_catalog",
          runnerType: "scheduled",
          defaultExecutorBinding: "automation_job",
          executionLocation: "client_default",
          slotSchema: [],
          version: "seed-v1",
          badge: "云目录",
          recentUsedAt: null,
          isRecent: false,
          runnerLabel: "本地计划任务",
          runnerTone: "sky",
          runnerDescription:
            "当前先进入工作区生成首版任务方案，后续再接本地自动化。",
          actionLabel: "先做方案",
          automationStatus: null,
        },
        {
          ...createSceneBoundServiceSkill(),
          id: "project-insight-flow",
          title: "项目线索整理",
          entryHint: "给我一个项目目标，我先把线索和下一步整理出来。",
          slotSchema: [
            {
              key: "project_goal",
              label: "项目目标",
              type: "text",
              required: true,
              placeholder: "例如 新品发布前的竞品和内容线索",
            },
          ],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("每日趋势摘要");
    expect(container.textContent).not.toContain("云目录");
    expect(container.textContent).not.toContain("最近使用");

    const projectCard = container.querySelector(
      '[data-testid="home-gallery-entry-service-skill-project-insight-flow"]',
    );
    expect(projectCard?.textContent).toContain("项目线索整理");
    expect(projectCard?.textContent).toContain(
      "围绕当前项目整理线索、结论和下一步动作。",
    );
  });

  it("runtime tool surface 告警不应再透传到首页空态输入区", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      subagentEnabled: true,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector(
        '[data-testid="empty-state-runtime-tool-warning"]',
      ),
    ).toBeNull();
    expect(container.textContent).not.toContain("当前 runtime tool surface");
    expect(container.textContent).not.toContain("联网搜索偏好本轮可能不会生效");
    expect(container.textContent).not.toContain(
      "Subagents 本轮可能不会完全生效",
    );
  });

  it("有最近记录时应在更多做法抽屉里展示最近入口，并允许继续模板与方法", async () => {
    recordCuratedTaskTemplateUsage({
      templateId: "social-post-starter",
      launchInputValues: {
        subject_or_product: "上次沉淀的主线升级",
        target_audience: "品牌内容负责人",
      },
      referenceMemoryIds: ["memory-idea-1"],
      referenceEntries: [
        {
          id: "memory-idea-1",
          title: "上次主稿参考",
          summary: "保留品牌调性与平台拆分思路",
          category: "context",
          categoryLabel: "参考",
          tags: ["品牌", "拆分"],
        },
      ],
    });
    const onSelectServiceSkill = vi.fn<(skill: ServiceSkillHomeItem) => void>();
    const setInput = vi.fn<(value: string) => void>();
    const template = findCuratedTaskTemplateById("social-post-starter");
    expect(template).toBeTruthy();
    const recentMethod: ServiceSkillHomeItem = {
      ...createSceneBoundServiceSkill(),
      id: "content-iteration-flow",
      title: "内容迭代整理",
      recentUsedAt: 1_000,
      isRecent: true,
    };

    const container = renderEmptyState({
      activeTheme: "general",
      setInput,
      onSelectServiceSkill,
      serviceSkills: [recentMethod],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const drawer = await openHomeMoreSkillsDrawer(container);
    const recentTemplateButton = expectHomeDrawerButton(
      drawer,
      "home-drawer-entry-recommended-social-post-starter",
    );

    await act(async () => {
      recentTemplateButton?.click();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("开始这一步前，我先确认几件事。");
    expect(setInput).not.toHaveBeenCalled();

    const subjectInput = document.body.querySelector(
      "#curated-task-social-post-starter-subject_or_product",
    ) as HTMLTextAreaElement | null;
    const audienceInput = document.body.querySelector(
      "#curated-task-social-post-starter-target_audience",
    ) as HTMLInputElement | null;

    expect(subjectInput?.value).toBe("上次沉淀的主线升级");
    expect(audienceInput?.value).toBe("品牌内容负责人");
    expect(document.body.textContent).toContain(
      "已选择 1 条参考对象，本轮会一起带入生成。",
    );

    await act(async () => {
      updateFieldValue(subjectInput, "Lime 的内容创作主链升级");
      updateFieldValue(audienceInput, "内容负责人");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith(
      buildCuratedTaskLaunchPrompt({
        task: template!,
        inputValues: {
          subject_or_product: "Lime 的内容创作主链升级",
          target_audience: "内容负责人",
        },
        referenceEntries: [
          {
            id: "memory-idea-1",
            title: "上次主稿参考",
            summary: "保留品牌调性与平台拆分思路",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "拆分"],
          },
        ],
      }),
    );

    const recentMethodButton = expectHomeDrawerButton(
      drawer,
      "home-drawer-entry-service-skill-content-iteration-flow",
    );

    act(() => {
      recentMethodButton?.click();
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "content-iteration-flow",
        title: "内容迭代整理",
      }),
    );
  });

  it("scene 最近使用记录也应把对应做法带回更多做法抽屉", async () => {
    recordSlashEntryUsage({
      kind: "scene",
      entryId: "project-insight-flow",
      usedAt: 1_800_000_000_000,
      replayText: "继续帮我整理这个项目的关键信息和后续动作",
    });
    const onSelectServiceSkill = vi.fn<(skill: ServiceSkillHomeItem) => void>();
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [createSceneBoundServiceSkill()],
      onSelectServiceSkill,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const drawer = await openHomeMoreSkillsDrawer(container);
    const sceneContinuationButton = expectHomeDrawerButton(
      drawer,
      "home-drawer-entry-service-skill-project-insight-flow",
    );

    act(() => {
      sceneContinuationButton?.click();
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "project-insight-flow",
        title: "项目线索整理",
      }),
    );
  });

  it("最近使用的本地已安装 skill 也应回到更多做法抽屉，并恢复上次目标", async () => {
    recordSlashEntryUsage({
      kind: "skill",
      entryId: "content-playbook",
      usedAt: 1_900_000_000_000,
      replayText: "继续优化这套内容主稿",
    });
    const setInput = vi.fn<(value: string) => void>();
    const installedSkill: Skill = {
      key: "content-playbook",
      name: "内容主稿方法",
      description: "本地补充技能",
      directory: "content-playbook",
      installed: true,
      sourceKind: "other",
      metadata: {
        lime_when_to_use: "当你需要继续复用这套内容主稿方法时使用。",
        lime_argument_hint: "主题、受众与复盘约束",
        lime_output_hint: "带着这套内容主稿方法进入生成",
      },
    };

    const container = renderEmptyState({
      activeTheme: "general",
      setInput,
      skills: [installedSkill],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const drawer = await openHomeMoreSkillsDrawer(container);
    const installedSkillButton = expectHomeDrawerButton(
      drawer,
      "home-drawer-entry-installed-skill-content-playbook",
    );

    await act(async () => {
      installedSkillButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith("继续优化这套内容主稿");
    expect(container.textContent).not.toContain("已挂载 内容主稿方法");
  });

  it("同页内新增本地 skill 使用记录后，更多做法抽屉应即时刷新", async () => {
    const installedSkill: Skill = {
      key: "content-playbook",
      name: "内容主稿方法",
      description: "本地补充技能",
      directory: "content-playbook",
      installed: true,
      sourceKind: "other",
    };

    const container = renderEmptyState({
      activeTheme: "general",
      skills: [installedSkill],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const drawer = await openHomeMoreSkillsDrawer(container);
    const installedSkillButton = expectHomeDrawerButton(
      drawer,
      "home-drawer-entry-installed-skill-content-playbook",
    );
    expect(installedSkillButton.title).toContain("本地补充技能");

    await act(async () => {
      recordSlashEntryUsage({
        kind: "skill",
        entryId: "content-playbook",
        usedAt: 1_900_000_000_100,
        replayText: "继续完善这套内容方法",
      });
      await Promise.resolve();
    });

    const updatedInstalledSkillButton = expectHomeDrawerButton(
      drawer,
      "home-drawer-entry-installed-skill-content-playbook",
    );
    expect(updatedInstalledSkillButton.title).toContain("继续完善这套内容方法");
  });

  it("首页主体不再重复展示项目选择器入口", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      projectId: "project-brand",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="project-selector-stub"]'),
    ).toBeNull();
    expect(mockProjectSelector).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="home-start-surface"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid^="entry-project-continuation-"]'),
    ).toBeNull();
  });

  it("通用首页不应把浏览器连接提前放到首屏补充入口", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="home-supplemental-actions"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="entry-connect-browser"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("连接浏览器");
    expect(container.textContent).not.toContain(
      "需要网页登录时，也可以先把浏览器接上。",
    );
    expect(
      container.querySelector('[data-testid="entry-capability-toggle"]'),
    ).toBeNull();
  });

  it("通用首页不再渲染 runtime 总览层", async () => {
    const onOpenMemoryWorkbench = vi.fn();
    const onOpenChannels = vi.fn();
    const onOpenChromeRelay = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      runtimeToolAvailability: {
        source: "runtime_tools",
        known: true,
        agentInitialized: true,
        availableToolCount: 9,
        webSearch: true,
        subagentCore: true,
        subagentTeamTools: true,
        subagentRuntime: true,
        planRuntime: true,
        missingSubagentCoreTools: [],
        missingSubagentTeamTools: [],
        missingPlanTools: [],
      },
      runtimeTaskCard: {
        taskId: "task-1",
        title: "当前任务",
        summary: "正在执行",
        status: "running",
        statusLabel: "进行中",
        phase: "tool_batch",
        phaseLabel: "工具批次处理中",
        detail: "正在执行工具批次",
        supportingLines: [],
        batchDescriptor: null,
        queuedTurnCount: 0,
        pendingRequestCount: 0,
        subtaskStats: null,
      },
      onOpenMemoryWorkbench,
      onOpenChannels,
      onOpenChromeRelay,
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="empty-state-runtime-overview"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("当前能力、执行态和远程入口");
    expect(container.textContent).not.toContain("Runtime Tool Surface");
    expect(container.textContent).not.toContain("Thread Runtime");
    expect(container.textContent).not.toContain("消息渠道 + 浏览器连接器");
    expect(container.textContent).not.toContain(
      "当前回合 / 长期 / Team / 压缩",
    );
    expect(container.textContent).not.toContain("频道入口");
    expect(container.textContent).not.toContain("浏览器连接器");
    expect(container.textContent).not.toContain("打开记忆工作台");

    expect(onOpenChannels).not.toHaveBeenCalled();
    expect(onOpenChromeRelay).not.toHaveBeenCalled();
    expect(onOpenMemoryWorkbench).not.toHaveBeenCalled();
  });

  it("浏览器接入不再通过首屏补充入口直接触发", async () => {
    const onLaunchBrowserAssist = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const mediaButton = container.querySelector(
      '[data-testid="entry-connect-browser"]',
    ) as HTMLButtonElement | null;
    expect(mediaButton).toBeNull();

    expect(onLaunchBrowserAssist).not.toHaveBeenCalled();
  });

  it("点击每日趋势摘要应打开模板确认，不再切换联网搜索前置开关", async () => {
    const setInput = vi.fn<(value: string) => void>();
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();
    const container = renderEmptyState({
      activeTheme: "general",
      setInput,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const card = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLDivElement | null;
    expect(card).toBeTruthy();

    act(() => {
      card?.click();
    });

    expect(container.textContent).toContain("开始这一步前，我先确认几件事。");
    expect(setInput).not.toHaveBeenCalled();

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(themeInput, "AI 内容创作");
      updateFieldValue(platformInput, "X 与 TikTok 北美区");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();
    expect(confirmButton?.disabled).toBe(false);

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith(
      buildCuratedTaskLaunchPrompt({
        task: template!,
        inputValues: {
          theme_target: "AI 内容创作",
          platform_region: "X 与 TikTok 北美区",
        },
      }),
    );
  });

  it("当前激活模板不是最近判断首选时，输入区 badge 应可直接切到推荐模板", async () => {
    const projectId = "project-active-badge-review-switch";
    const container = renderEmptyState({
      activeTheme: "general",
      projectId,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const card = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLDivElement | null;
    expect(card).toBeTruthy();

    act(() => {
      card?.click();
    });

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(themeInput, "AI 内容创作");
      updateFieldValue(platformInput, "X 与 TikTok 北美区");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton?.disabled).toBe(false);

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    await act(async () => {
      recordCuratedTaskRecommendationSignalFromReviewDecision(
        {
          session_id: "session-active-badge-review-switch",
          decision_status: "needs_more_evidence",
          decision_summary:
            "这轮结果还缺证据，需要回到账号表现和爆款样本继续补证据。",
          chosen_fix_strategy: "先补账号数据复盘，再拆一轮高表现内容做对照。",
          risk_level: "medium",
          risk_tags: ["证据不足", "需要复盘"],
          followup_actions: ["补账号数据复盘", "拆解一条高表现内容"],
        },
        {
          projectId,
          sceneTitle: "短视频编排",
        },
      );
      await Promise.resolve();
    });

    const badgeAction = container.querySelector(
      '[data-testid="curated-task-badge-review-action"]',
    ) as HTMLButtonElement | null;
    expect(badgeAction?.textContent).toContain("改用「复盘这个账号/项目」");

    await act(async () => {
      badgeAction?.click();
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("复盘这个账号/项目");
    expect(document.body.textContent).toContain(
      "已按最近判断切到更适合的结果模板",
    );
  });

  it("点击内容主稿生成应直接写入起始动作，不再切换旧主题", async () => {
    const setInput = vi.fn<(value: string) => void>();
    const onThemeChange = vi.fn<(theme: string) => void>();
    const template = findCuratedTaskTemplateById("social-post-starter");
    expect(template).toBeTruthy();
    const container = renderEmptyState({
      activeTheme: "general",
      setInput,
      onThemeChange,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const card = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-social-post-starter"]',
    ) as HTMLDivElement | null;
    expect(card).toBeTruthy();

    act(() => {
      card?.click();
    });

    const subjectInput = document.body.querySelector(
      "#curated-task-social-post-starter-subject_or_product",
    ) as HTMLTextAreaElement | null;
    const audienceInput = document.body.querySelector(
      "#curated-task-social-post-starter-target_audience",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(subjectInput, "Lime 的 AI 内容创作闭环升级");
      updateFieldValue(audienceInput, "内容团队负责人");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(onThemeChange).not.toHaveBeenCalled();
    expect(setInput).toHaveBeenCalledWith(
      buildCuratedTaskLaunchPrompt({
        task: template!,
        inputValues: {
          subject_or_product: "Lime 的 AI 内容创作闭环升级",
          target_audience: "内容团队负责人",
        },
      }),
    );
  });

  it("点击长文转多平台发布稿应直接写入起始动作", async () => {
    const setInput = vi.fn<(value: string) => void>();
    const template = findCuratedTaskTemplateById(
      "longform-multiplatform-rewrite",
    );
    expect(template).toBeTruthy();
    const container = renderEmptyState({
      activeTheme: "general",
      setInput,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const card = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-longform-multiplatform-rewrite"]',
    ) as HTMLDivElement | null;
    expect(card).toBeTruthy();

    act(() => {
      card?.click();
    });

    const sourceInput = document.body.querySelector(
      "#curated-task-longform-multiplatform-rewrite-source_article",
    ) as HTMLTextAreaElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-longform-multiplatform-rewrite-target_platform",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(sourceInput, "这是一篇关于品牌内容中台升级的长文。");
      updateFieldValue(platformInput, "X、LinkedIn");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith(
      buildCuratedTaskLaunchPrompt({
        task: template!,
        inputValues: {
          source_article: "这是一篇关于品牌内容中台升级的长文。",
          target_platform: "X、LinkedIn",
        },
      }),
    );
  });

  it("首页注入 current service skill 时仍应把 seeded 目录拼接到精选区尾部", async () => {
    const onSelectServiceSkill = vi.fn<(skill: ServiceSkillHomeItem) => void>();
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [createRepositoryResearchServiceSkill()],
      onSelectServiceSkill,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("复制轮播帖");
    expect(container.textContent).toContain("仓库线索研究");

    const card = container.querySelector(
      '[data-testid="home-gallery-entry-service-skill-carousel-post-replication"]',
    ) as HTMLButtonElement | null;
    expect(card).toBeTruthy();

    act(() => {
      card?.click();
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "carousel-post-replication",
        title: "复制轮播帖",
      }),
    );
  });

  it("应挂载 CharacterMention，并透传角色与技能", async () => {
    const characters: Character[] = [
      {
        id: "char-1",
        project_id: "project-1",
        name: "角色A",
        aliases: [],
        relationships: [],
        is_main: true,
        order: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ];
    const skills: Skill[] = [
      {
        key: "skill-1",
        name: "技能A",
        description: "desc",
        directory: "skill-a",
        installed: true,
        sourceKind: "builtin",
      },
    ];
    const setInput = vi.fn<(value: string) => void>();
    const onSend = vi.fn();

    const container = renderEmptyState({
      input: "@",
      setInput,
      onSend,
      characters,
      skills,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const mention = container.querySelector(
      '[data-testid="character-mention-stub"]',
    );
    expect(mention).toBeTruthy();
    expect(mockCharacterMention.mock.calls.length).toBeGreaterThan(0);
    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];
    expect(latestCall.characters).toEqual(characters);
    expect(latestCall.skills).toEqual(skills);

    act(() => {
      latestCall.onChange("@技能A");
    });

    expect(setInput).not.toHaveBeenCalled();

    const sendButton = container.querySelector(
      'button[title="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton?.disabled).toBe(false);

    act(() => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expectEmptyStateSend(onSend, {
      textOverride: "@技能A",
    });
  });

  it("应把服务型技能与选择回调透传给 CharacterMention", async () => {
    const serviceSkills: ServiceSkillHomeItem[] = [
      {
        id: "daily-trend-briefing",
        title: "每日趋势摘要",
        summary: "围绕指定平台与关键词输出趋势摘要。",
        entryHint: "把平台和关键词给我，我先整理一份趋势报告。",
        aliases: ["趋势报告"],
        category: "内容运营",
        outputHint: "趋势摘要 + 调度建议",
        source: "cloud_catalog",
        runnerType: "scheduled",
        defaultExecutorBinding: "automation_job",
        executionLocation: "client_default",
        slotSchema: [],
        surfaceScopes: ["home", "mention", "workspace"],
        promptTemplateKey: "trend_briefing",
        version: "seed-v1",
        badge: "云目录",
        recentUsedAt: null,
        isRecent: false,
        runnerLabel: "本地计划任务",
        runnerTone: "sky",
        runnerDescription:
          "当前先进入工作区生成首版任务方案，后续再接本地自动化。",
        actionLabel: "先做方案",
        automationStatus: null,
      },
    ];
    const onSelectServiceSkill = vi.fn<(skill: ServiceSkillHomeItem) => void>();

    renderEmptyState({
      input: "@",
      serviceSkills,
      onSelectServiceSkill,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];
    expect(latestCall.serviceSkills).toEqual(serviceSkills);
    expect(typeof latestCall.onSelectInputCapability).toBe("function");

    act(() => {
      latestCall.onSelectInputCapability?.({
        kind: "service_skill",
        skill: serviceSkills[0]!,
      });
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(serviceSkills[0]!);
  });

  it("首页安装技能选择应走统一 capability 回调，且发送后清除激活态", async () => {
    const onSend = vi.fn();
    const skill: Skill = {
      key: "canvas-design",
      name: "canvas-design",
      description: "desc",
      directory: "canvas-design",
      installed: true,
      sourceKind: "builtin",
    };

    const container = renderEmptyState({
      input: "帮我设计封面",
      onSend,
      skills: [skill],
    });
    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];
    expect(typeof latestCall.onSelectInputCapability).toBe("function");

    act(() => {
      latestCall.onSelectInputCapability?.({
        kind: "installed_skill",
        skill,
      });
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("canvas-design");
    expect(container.textContent).not.toContain("已挂载 canvas-design");

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });
    expectEmptyStateSend(onSend, {
      textOverride: "帮我设计封面",
      sendOptions: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "canvas-design",
          skillName: "canvas-design",
        },
        displayContent: "帮我设计封面",
      },
    });

    await act(async () => {
      await Promise.resolve();
    });
    act(() => {
      updateFieldValue(container.querySelector("textarea"), "帮我设计封面");
    });
    await act(async () => {
      await Promise.resolve();
    });

    const sendButtonAfterCapabilityCleared = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButtonAfterCapabilityCleared).toBeTruthy();

    act(() => {
      sendButtonAfterCapabilityCleared?.click();
    });
    const secondPayload = onSend.mock.calls[1]?.[0] as
      | InputbarSendPayload
      | undefined;
    expect(secondPayload).toMatchObject({
      textOverride: "帮我设计封面",
    });
    expect(secondPayload?.images).toBeUndefined();
    expect(secondPayload?.sendOptions).toBeUndefined();
  });

  it("首页选择 builtin command 后发送应透传结构化 capability route", async () => {
    const onSend = vi.fn();
    const command = {
      key: "image-compose",
      label: "配图",
      mentionLabel: "配图",
      commandPrefix: "@配图",
      description: "输出配图方向。",
      aliases: [],
    };
    const container = renderEmptyState({
      input: "帮我整理这篇文章的配图方向",
      onSend,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];

    act(() => {
      latestCall.onSelectInputCapability?.({
        kind: "builtin_command",
        command,
      });
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "帮我整理这篇文章的配图方向",
      sendOptions: {
        capabilityRoute: {
          kind: "builtin_command",
          commandKey: "image-compose",
          commandPrefix: "@配图",
        },
        displayContent: "帮我整理这篇文章的配图方向",
      },
    });
  });

  it("首页选择 @资料 兼容入口时应打开资料中枢而不是普通命令标签", async () => {
    const onToggleKnowledgePack = vi.fn<(enabled: boolean) => void>();
    const onStartKnowledgeOrganize = vi.fn();
    const command = {
      key: "knowledge_pack",
      label: "资料",
      mentionLabel: "资料",
      commandPrefix: "@资料",
      description: "查看、添加、选择或使用当前项目资料。",
      aliases: [],
    };
    const container = renderEmptyState({
      input: "按项目资料写一版介绍",
      knowledgePackSelection: {
        enabled: false,
        packName: "team-notes",
        workingDir: "workspace-root",
        label: "团队资料",
        status: "ready",
      },
      knowledgePackOptions: [
        {
          packName: "team-notes",
          label: "团队资料",
          status: "ready",
          defaultForWorkspace: true,
        },
      ],
      onToggleKnowledgePack,
      onSelectKnowledgePack: vi.fn(),
      onStartKnowledgeOrganize,
      onManageKnowledgePacks: vi.fn(),
    });
    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];

    await act(async () => {
      latestCall.onSelectInputCapability?.({
        kind: "builtin_command",
        command,
      });
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-builtin-command-badge"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-knowledge-hub"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("团队资料");
    expect(container.textContent).toContain("使用这份资料");

    const useKnowledgeButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("使用这份资料"));

    act(() => {
      useKnowledgeButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(onToggleKnowledgePack).toHaveBeenCalledWith(true);
    expect(onStartKnowledgeOrganize).not.toHaveBeenCalled();
  });

  it("首页启用项目资料后发送应携带资料引用 metadata", async () => {
    const onSend = vi.fn();
    const container = renderEmptyState({
      input: "按项目资料写一版介绍",
      onSend,
      knowledgePackSelection: {
        enabled: true,
        packName: "team-notes",
        workingDir: "workspace-root",
        label: "团队资料",
        status: "ready",
      },
      knowledgePackOptions: [
        {
          packName: "team-notes",
          label: "团队资料",
          status: "ready",
          defaultForWorkspace: true,
        },
      ],
      onToggleKnowledgePack: vi.fn(),
      onSelectKnowledgePack: vi.fn(),
      onStartKnowledgeOrganize: vi.fn(),
      onManageKnowledgePacks: vi.fn(),
    });
    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "按项目资料写一版介绍",
      sendOptions: {
        requestMetadata: {
          knowledge_pack: {
            pack_name: "team-notes",
            working_dir: "workspace-root",
            source: "inputbar",
          },
        },
      },
    });
  });

  it("首页点击结果模板后发送时，应透传 curated_task capability route", async () => {
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();
    const prompt = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: {
        theme_target: "AI 内容创作",
        platform_region: "X 与 TikTok 北美区",
      },
    });
    const onSend = vi.fn();
    const setInput = vi.fn();
    const container = renderEmptyState({
      input: prompt,
      setInput,
      onSend,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const templateButton = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLButtonElement | null;
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(themeInput, "AI 内容创作");
      updateFieldValue(platformInput, "X 与 TikTok 北美区");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith(prompt);

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: prompt,
      sendOptions: expect.objectContaining({
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt,
          launchInputValues: {
            theme_target: "AI 内容创作",
            platform_region: "X 与 TikTok 北美区",
          },
        },
        displayContent: prompt,
        requestMetadata: {
          harness: {
            curated_task: expect.objectContaining({
              task_id: "daily-trend-briefing",
              launch_input_values: {
                theme_target: "AI 内容创作",
                platform_region: "X 与 TikTok 北美区",
              },
            }),
          },
        },
      }),
    });
  });

  it("首页结果模板带着记忆参考发送时，应附带引用 route 与 request metadata", async () => {
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();
    const promptWithReference = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: {
        theme_target: "AI 内容创作",
        platform_region: "X 与 TikTok 北美区",
      },
      referenceEntries: [
        {
          id: "memory-1",
          title: "品牌风格样本",
          summary: "轻盈但专业的品牌语气参考。",
          category: "context",
          categoryLabel: "参考",
          tags: ["品牌", "语气"],
        },
      ],
    });
    const onSend = vi.fn();
    const setInput = vi.fn();
    const container = renderEmptyState({
      input: promptWithReference,
      setInput,
      onSend,
      creationReplaySurface: {
        kind: "memory_entry",
        eyebrow: "当前带入记忆参考",
        badgeLabel: "参考",
        title: "品牌风格样本",
        summary: "轻盈但专业的品牌语气参考。",
        hint: "后续结果模板会默认把它一起带入。",
        defaultReferenceMemoryIds: ["memory-1"],
        defaultReferenceEntries: [
          {
            id: "memory-1",
            sourceKind: "memory",
            title: "品牌风格样本",
            summary: "轻盈但专业的品牌语气参考。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });

    const templateButton = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLButtonElement | null;
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    await act(async () => {
      await Promise.resolve();
    });

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;
    const referenceButton = document.body.querySelector(
      '[data-testid="curated-task-reference-option-memory-1"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      updateFieldValue(themeInput, "AI 内容创作");
      updateFieldValue(platformInput, "X 与 TikTok 北美区");
      await Promise.resolve();
    });
    expect(referenceButton).toBeTruthy();
    expect(referenceButton?.disabled).toBe(false);

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: expect.stringContaining("本轮可优先参考这些参考对象"),
      sendOptions: expect.objectContaining({
        capabilityRoute: expect.objectContaining({
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          referenceMemoryIds: ["memory-1"],
          referenceEntries: [
            expect.objectContaining({
              id: "memory-1",
              sourceKind: "memory",
            }),
          ],
        }),
        displayContent: expect.stringContaining("本轮可优先参考这些参考对象"),
        requestMetadata: expect.objectContaining({
          harness: expect.objectContaining({
            creation_replay: expect.objectContaining({
              kind: "memory_entry",
            }),
            curated_task: expect.objectContaining({
              task_id: "daily-trend-briefing",
              reference_memory_ids: ["memory-1"],
              reference_entries: [
                expect.objectContaining({
                  id: "memory-1",
                }),
              ],
            }),
          }),
        }),
      }),
    });
  });

  it("首页结果模板启动时，应默认沿用当前带入的记忆参考", async () => {
    const container = renderEmptyState({
      creationReplaySurface: {
        kind: "memory_entry",
        eyebrow: "当前带入记忆参考",
        badgeLabel: "参考",
        title: "品牌风格样本",
        summary: "保留轻盈但专业的表达。",
        hint: "后续结果模板会默认把它一起带入。",
        defaultReferenceMemoryIds: ["memory-1"],
        defaultReferenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });

    const templateButton = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLButtonElement | null;
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain(
      "已选择 1 条参考对象，本轮会一起带入生成。",
    );
  });

  it("首页复盘模板启动时，应默认带入当前项目结果基线", async () => {
    const onSend = vi.fn();
    const setInput = vi.fn();
    const container = renderEmptyState({
      onSend,
      setInput,
      defaultCuratedTaskReferenceEntries: [
        {
          id: "sceneapp:content-pack:run:1",
          sourceKind: "sceneapp_execution_summary",
          title: "AI 内容周报",
          summary: "当前已有一轮运行结果，可直接作为复盘基线。",
          category: "experience",
          categoryLabel: "成果",
          tags: ["复盘", "项目结果"],
          taskPrefillByTaskId: {
            "account-project-review": {
              project_goal: "AI 内容周报",
              existing_results:
                "这轮运行已产出项目结果，但仍需补齐复核意见。 当前卡点：复核阻塞 当前判断：先补复核与修复 经营动作：优先准备结果对齐包，再决定是否继续放大。 更适合去向：结果对齐",
            },
          },
        },
      ],
    });

    const templateButton = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-account-project-review"]',
    ) as HTMLButtonElement | null;
    expect(templateButton).toBeTruthy();
    expect(container.textContent).not.toContain("当前判断：先补复核与修复");
    expect(container.textContent).not.toContain("当前卡点：复核阻塞");
    expect(container.textContent).not.toContain("更适合去向：结果对齐");

    act(() => {
      templateButton?.click();
    });

    await act(async () => {
      await Promise.resolve();
    });

    const goalInput = document.body.querySelector(
      "#curated-task-account-project-review-project_goal",
    ) as HTMLInputElement | null;
    const resultsInput = document.body.querySelector(
      "#curated-task-account-project-review-existing_results",
    ) as HTMLTextAreaElement | null;

    expect(goalInput?.value).toBe("AI 内容周报");
    expect(resultsInput?.value).toContain("这轮运行已产出项目结果");

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();
    expect(confirmButton?.disabled).toBe(false);

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith(
      expect.stringContaining("AI 内容周报"),
    );
    expect(onSend).not.toHaveBeenCalled();
  });

  it("首页下游结果模板卡片不应把项目结果基线提前摊开", async () => {
    const container = renderEmptyState({
      defaultCuratedTaskReferenceEntries: [
        {
          id: "sceneapp:content-pack:run:1",
          sourceKind: "sceneapp_execution_summary",
          title: "AI 内容周报",
          summary: "当前已有一轮运行结果，可直接作为后续生成基线。",
          category: "experience",
          categoryLabel: "成果",
          tags: ["复盘", "项目结果"],
          taskPrefillByTaskId: {
            "account-project-review": {
              project_goal: "AI 内容周报",
              existing_results:
                "这轮运行已产出项目结果，但仍需补齐复核意见。 当前卡点：复核阻塞 当前判断：先补复核与修复 经营动作：优先准备结果对齐包，再决定是否继续放大。 更适合去向：结果对齐",
            },
          },
        },
      ],
    });

    const templateButton = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      await Promise.resolve();
    });

    expect(templateButton).toBeTruthy();
    expect(templateButton?.textContent).toContain("每日趋势摘要");
    expect(container.textContent).not.toContain("当前结果基线：AI 内容周报");
    expect(container.textContent).not.toContain("当前判断：先补复核与修复");
    expect(container.textContent).not.toContain("当前卡点：复核阻塞");
    expect(container.textContent).not.toContain("更适合去向：结果对齐");
  });

  it("当前带入参考灵感时，结果模板推荐应显式标记为围绕当前参考", async () => {
    const container = renderEmptyState({
      creationReplaySurface: {
        kind: "memory_entry",
        eyebrow: "当前带入灵感",
        badgeLabel: "参考",
        title: "品牌风格样本",
        summary: "保留轻盈但专业的表达。",
        hint: "后续结果模板会默认把它一起带入。",
        defaultReferenceMemoryIds: ["memory-1"],
        defaultReferenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("参考");
    expect(container.textContent).toContain("品牌风格样本");
    expect(container.textContent).toContain("内容主稿生成");
  });

  it("当前带入 Skill 草稿时，首页应显影更明确的连续性横幅", async () => {
    const container = renderEmptyState({
      creationReplaySurface: {
        kind: "skill_scaffold",
        eyebrow: "当前带入 Skill 草稿",
        badgeLabel: "Skill 草稿",
        title: "账号复盘方法",
        summary: "把结果复盘成下一轮增长方案。",
        hint: "这轮会先沿着这份 Skill 草稿继续生成，跑顺后可回到 Skills 继续整理。",
        defaultReferenceMemoryIds: [],
        defaultReferenceEntries: [],
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("Skill 草稿");
    expect(container.textContent).toContain("账号复盘方法");
    expect(container.textContent).not.toContain("沿着当前上下文继续");
    expect(container.textContent).not.toContain("先沿着当前做法开工");
  });

  it("最近记忆参考成果信号应影响首页结果模板推荐", async () => {
    recordCuratedTaskRecommendationSignalFromMemoryReference(
      {
        id: "memory-review-1",
        session_id: "session-review-1",
        memory_type: "project",
        category: "experience",
        title: "账号复盘结论",
        summary:
          "最近两次反馈都提示封面信息过密，需要继续复盘增长数据并优化结构。",
        content:
          "最近两次反馈都提示封面信息过密，需要继续复盘增长数据并优化结构。",
        tags: ["复盘", "反馈", "增长"],
        metadata: {
          confidence: 0.92,
          importance: 8,
          access_count: 2,
          last_accessed_at: null,
          source: "manual",
          embedding: null,
        },
        created_at: 1_712_345_670_000,
        updated_at: 1_712_345_678_000,
        archived: false,
      },
      {
        projectId: "project-review",
      },
    );

    const container = renderEmptyState({
      projectId: "project-review",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("成果：账号复盘结论");
    expect(container.textContent).toContain("复盘这个账号/项目");
  });

  it("已激活结果模板后重新编辑启动信息时，应回填并更新输入", async () => {
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();
    const initialInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const editedInputValues = {
      theme_target: "品牌内容中台",
      platform_region: "LinkedIn 与 X（海外）",
    };
    const initialPrompt = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: initialInputValues,
    });
    const editedPrompt = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: editedInputValues,
    });
    const setInput = vi.fn();
    const container = renderEmptyState({
      input: initialPrompt,
      setInput,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const templateButton = container.querySelector(
      '[data-testid="home-gallery-entry-recommended-daily-trend-briefing"]',
    ) as HTMLButtonElement | null;
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    const initialThemeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const initialPlatformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(initialThemeInput, initialInputValues.theme_target);
      updateFieldValue(
        initialPlatformInput,
        initialInputValues.platform_region,
      );
      await Promise.resolve();
    });

    const firstConfirmButton = findLauncherConfirmButton();
    expect(firstConfirmButton).toBeTruthy();

    await act(async () => {
      firstConfirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenLastCalledWith(initialPrompt);

    const editButton = Array.from(container.querySelectorAll("button")).find(
      (button) =>
        button.getAttribute("aria-label") === "编辑 每日趋势摘要 启动信息",
    ) as HTMLButtonElement | undefined;
    expect(editButton).toBeTruthy();

    await act(async () => {
      editButton?.click();
      await Promise.resolve();
    });

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    expect(themeInput?.value).toBe(initialInputValues.theme_target);
    expect(platformInput?.value).toBe(initialInputValues.platform_region);

    await act(async () => {
      updateFieldValue(themeInput, editedInputValues.theme_target);
      updateFieldValue(platformInput, editedInputValues.platform_region);
      await Promise.resolve();
    });

    const secondConfirmButton = findLauncherConfirmButton();
    expect(secondConfirmButton).toBeTruthy();

    await act(async () => {
      secondConfirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenLastCalledWith(editedPrompt);
  });

  it("首页不再显式暴露做法数量文案", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist: vi.fn(),
      skills: [
        {
          key: "writer",
          name: "写作助手",
          description: "desc",
          directory: "writer",
          installed: true,
          sourceKind: "builtin",
        },
      ],
      serviceSkills: [
        {
          id: "trend-briefing",
          title: "趋势情报",
          summary: "输出趋势摘要",
          category: "研究",
          outputHint: "摘要",
          source: "cloud_catalog",
          runnerType: "instant",
          defaultExecutorBinding: "agent_turn",
          executionLocation: "client_default",
          slotSchema: [],
          version: "seed-v1",
          badge: "云目录",
          recentUsedAt: null,
          isRecent: false,
          runnerLabel: "先做这一轮",
          runnerTone: "emerald",
          runnerDescription: "先在当前工作区完成情报任务。",
          actionLabel: "开始执行",
          automationStatus: null,
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("2 套做法可直接复用");
    expect(container.textContent).not.toContain("按需挂上常用做法");
  });

  it("通用对话存在 current service skill 时仍应展示稳定占位文案", async () => {
    const container = renderEmptyState({
      serviceSkills: [createRepositoryResearchServiceSkill()],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const textarea = container.querySelector("textarea");
    expect(textarea?.getAttribute("placeholder")).toBe(
      "先说这轮要做什么，目标、对象或限制都可以。",
    );
  });

  it("通用对话默认不展示 Tab 起手建议", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
      serviceSkills: [],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="home-input-tab-suggestion"]'),
    ).toBeNull();
  });

  it("高级设置不再渲染联网搜索前置开关", async () => {
    const container = renderEmptyState({ activeTheme: "general" });
    await act(async () => {
      await Promise.resolve();
    });

    const plusTrigger = container.querySelector(
      '[data-testid="inputbar-plus-trigger"]',
    ) as HTMLButtonElement | null;
    expect(plusTrigger).toBeTruthy();
    const globeToggle = container.querySelector(
      'button[title="联网搜索已关闭"]',
    ) as HTMLButtonElement | null;
    expect(globeToggle).toBeNull();

    act(() => {
      plusTrigger?.click();
    });

    const expandedGlobeToggle = container.querySelector(
      'button[title="联网搜索已关闭"]',
    ) as HTMLButtonElement | null;
    expect(expandedGlobeToggle).toBeNull();
  });

  it("通用主题默认只保留最小主路径，展开高级设置后才显示进阶控制", async () => {
    const onSubagentEnabledChange = vi.fn<(enabled: boolean) => void>();
    const setAccessMode =
      vi.fn<(mode: "read-only" | "current" | "full-access") => void>();
    const container = renderEmptyState({
      activeTheme: "general",
      subagentEnabled: false,
      onSubagentEnabledChange,
      accessMode: "current",
      setAccessMode,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const attachButton = container.querySelector(
      'button[title="添加图片"]',
    ) as HTMLButtonElement | null;
    expect(attachButton).toBeTruthy();

    const plusTrigger = container.querySelector(
      '[data-testid="inputbar-plus-trigger"]',
    ) as HTMLButtonElement | null;
    expect(plusTrigger).toBeTruthy();

    expect(
      container.querySelector('button[title="深度思考已关闭"]'),
    ).toBeNull();
    expect(
      container.querySelector('button[title="联网搜索已关闭"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-plan-toggle"]'),
    ).toBeNull();
    expect(
      container.querySelector('button[title="Subagents 已关闭"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-access-mode-select"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="chat-model-selector"]'),
    ).toBeTruthy();
    expect(container.textContent).not.toContain("通用任务上下文");
    expect(container.textContent).not.toContain("当前模型");

    act(() => {
      plusTrigger?.click();
    });

    const thinkingButton = container.querySelector(
      'button[title="深度思考已关闭"]',
    ) as HTMLButtonElement | null;
    expect(thinkingButton).toBeNull();
    const globeButton = container.querySelector(
      'button[title="联网搜索已关闭"]',
    ) as HTMLButtonElement | null;
    expect(globeButton).toBeNull();
    const planButton = container.querySelector(
      '[data-testid="inputbar-plan-toggle"]',
    ) as HTMLButtonElement | null;
    expect(planButton).toBeNull();
    const subagentButton = document.body.querySelector(
      '[data-testid="inputbar-plus-subagent-mode"]',
    ) as HTMLButtonElement | null;
    expect(subagentButton).toBeTruthy();
    const accessModeSelect = container.querySelector(
      '[data-testid="inputbar-access-mode-select"]',
    ) as HTMLSelectElement | null;
    expect(accessModeSelect).toBeTruthy();
    expect(
      container.querySelector('[data-testid="chat-model-selector"]'),
    ).toBeTruthy();
    expect(container.textContent).not.toContain("通用任务上下文");

    act(() => {
      subagentButton?.click();
    });
    act(() => {
      accessModeSelect!.value = "full-access";
      accessModeSelect?.dispatchEvent(new Event("change", { bubbles: true }));
    });

    expect(onSubagentEnabledChange).toHaveBeenCalledWith(true);
    expect(setAccessMode).toHaveBeenCalledWith("full-access");
  });

  it("首页默认不再把执行模式和联网状态作为 hero 徽标暴露", async () => {
    const container = renderEmptyState({
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("编排模式已开启");
    expect(container.textContent).not.toContain("联网搜索已开启");
    expect(
      container.querySelector('[data-testid="home-start-surface"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("调研报告");
  });

  it("通用首页发送时不应自动注入任何历史默认 skill", async () => {
    const onSend = vi.fn();
    const container = renderEmptyState({
      input: "请输出一篇新品发布文案",
      activeTheme: "general",
      onSend,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "请输出一篇新品发布文案",
    });
  });

  it("首页开启 plan 和 goal 后发送应绑定当前 thread goal", async () => {
    const onSend = vi.fn();
    const container = renderEmptyState({
      input: "请先规划再持续推进这个任务",
      activeTheme: "general",
      sessionId: "thread-empty-state-plan-goal",
      taskEnabled: true,
      onSend,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const plusTrigger = container.querySelector(
      '[data-testid="inputbar-plus-trigger"]',
    ) as HTMLButtonElement | null;
    expect(plusTrigger).toBeTruthy();

    act(() => {
      plusTrigger?.click();
    });

    const objectiveButton = document.body.querySelector(
      '[data-testid="inputbar-plus-objective"]',
    ) as HTMLButtonElement | null;
    expect(objectiveButton).toBeTruthy();

    act(() => {
      objectiveButton?.click();
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "请先规划再持续推进这个任务",
      sendOptions: expect.objectContaining({
        collaborationMode: "plan",
        threadGoal: { objective: "请先规划再持续推进这个任务" },
        toolPreferencesOverride: {
          task: true,
          subagent: false,
        },
      }),
    });
  });

  it("即使存在历史配置字段，通用首页也不再自动注入默认 skill", async () => {
    mockGetConfig.mockImplementation(async () => ({
      chat_appearance: {},
    }));

    const onSend = vi.fn();
    const container = renderEmptyState({
      input: "请输出一篇用户访谈纪要",
      activeTheme: "general",
      onSend,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "请输出一篇用户访谈纪要",
    });
  });

  it("首页手动选择安装技能后发送时仍应优先使用 capability route", async () => {
    const onSend = vi.fn();
    const skill: Skill = {
      key: "custom-writing-skill",
      name: "custom-writing-skill",
      description: "desc",
      directory: "custom-writing-skill",
      installed: true,
      sourceKind: "builtin",
    };

    const container = renderEmptyState({
      input: "请输出一篇品牌故事",
      activeTheme: "general",
      onSend,
      skills: [skill],
    });
    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];
    act(() => {
      latestCall.onSelectInputCapability?.({
        kind: "installed_skill",
        skill,
      });
    });

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "请输出一篇品牌故事",
      sendOptions: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "custom-writing-skill",
          skillName: "custom-writing-skill",
        },
        displayContent: "请输出一篇品牌故事",
      },
    });
  });

  it("首页先选 capability 再切到服务技能入口时，不应继续残留旧 capability route", async () => {
    const onSend = vi.fn();
    const onSelectServiceSkill = vi.fn<(skill: ServiceSkillHomeItem) => void>();
    const installedSkill: Skill = {
      key: "canvas-design",
      name: "canvas-design",
      description: "desc",
      directory: "canvas-design",
      installed: true,
      sourceKind: "builtin",
    };
    const serviceSkill = createRepositoryResearchServiceSkill();

    const container = renderEmptyState({
      input: "整理最近发布计划",
      onSend,
      skills: [installedSkill],
      serviceSkills: [serviceSkill],
      onSelectServiceSkill,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];

    act(() => {
      latestCall.onSelectInputCapability?.({
        kind: "installed_skill",
        skill: installedSkill,
      });
    });

    act(() => {
      latestCall.onSelectInputCapability?.({
        kind: "service_skill",
        skill: serviceSkill,
      });
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(serviceSkill);

    const sendButton = container.querySelector(
      'button[aria-label="发送"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    act(() => {
      sendButton?.click();
    });

    expectEmptyStateSend(onSend, {
      textOverride: "整理最近发布计划",
    });
  });

  it("通用主题不应在首屏直接展示浏览器协助入口", async () => {
    const onLaunchBrowserAssist = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      onLaunchBrowserAssist,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const launchButton = container.querySelector(
      '[data-testid="entry-connect-browser"]',
    ) as HTMLButtonElement | null;
    expect(launchButton).toBeNull();

    expect(onLaunchBrowserAssist).not.toHaveBeenCalled();
  });

  it("存在最近会话时应提供继续最近会话入口", async () => {
    const onResumeRecentSession = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      recentSessionTitle: "品牌发布节奏整理",
      recentSessionSummary: "已整理到待确认发布标题这一步。",
      recentSessionActionLabel: "继续最近会话",
      onResumeRecentSession,
    });
    await act(async () => {
      await Promise.resolve();
    });

    const resumeButton = container.querySelector(
      '[data-testid="entry-recent-session-resume"]',
    ) as HTMLButtonElement | null;
    expect(resumeButton).toBeTruthy();
    expect(resumeButton?.textContent).toContain("继续最近会话");
    expect(resumeButton?.textContent).toContain("品牌发布节奏整理");

    act(() => {
      resumeButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onResumeRecentSession).toHaveBeenCalledTimes(1);
  });

  it("存在项目会话目录时应替换最近会话入口并打开对应会话", async () => {
    const onResumeRecentSession = vi.fn();
    const onOpenProjectConversation = vi.fn();
    const container = renderEmptyState({
      activeTheme: "general",
      recentSessionTitle: "品牌发布节奏整理",
      recentSessionSummary: "已整理到待确认发布标题这一步。",
      recentSessionActionLabel: "继续最近会话",
      onResumeRecentSession,
      projectConversationGroups: [
        {
          projectId: "project-1",
          projectName: "内容项目",
          conversations: [
            {
              id: "topic-1",
              title: "发布计划",
              summary: "已记录 6 条消息。",
              statusReason: "workspace_error",
            },
          ],
        },
      ],
      onOpenProjectConversation,
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="home-project-conversations"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="home-supplemental-actions"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="entry-recent-session-resume"]'),
    ).toBeNull();
    expect(container.textContent).toContain("更多 1 个对话");
    expect(container.textContent).not.toContain("发布计划");
    expect(container.textContent).not.toContain("内容项目");
    expect(container.textContent).not.toContain("已记录 6 条消息。");

    const moreButton = container.querySelector(
      '[data-testid="home-project-conversation-more"] button',
    ) as HTMLButtonElement | null;
    act(() => {
      moreButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(container.textContent).toContain("发布计划");

    const conversationButton = container.querySelector(
      '[data-testid="home-project-conversation"]',
    ) as HTMLButtonElement | null;
    expect(conversationButton?.textContent).toContain("发布计划");

    act(() => {
      conversationButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(onOpenProjectConversation).toHaveBeenCalledWith(
      "topic-1",
      "workspace_error",
    );
    expect(onResumeRecentSession).not.toHaveBeenCalled();
  });
});
