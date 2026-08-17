import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, vi } from "vitest";
import type {
  ServiceSkillGroup,
  ServiceSkillHomeItem,
} from "@/components/agent/chat/service-skills/types";
import type {
  SkillMarketplaceBundle,
  SkillMarketplaceItem,
} from "@/lib/api/officialSkillMarketplace";
import type { CreateSkillScaffoldRequest, Skill } from "@/lib/api/skills";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { SkillsPageParams } from "@/types/page";
import { SkillsWorkspacePage } from "./SkillsWorkspacePage";

const hoisted = vi.hoisted(() => ({
  mocks: {
    refreshServiceSkills: vi.fn(),
    refreshOfficialMarketplace: vi.fn(),
    recordServiceSkillUsage: vi.fn(),
    refreshLocalSkills: vi.fn(),
    uninstallLocalSkill: vi.fn(),
    installOfficialMarketplaceSkill: vi.fn(),
    getOfficialSkillMarketplaceBundle: vi.fn(),
    toastSuccess: vi.fn(),
    toastError: vi.fn(),
    openDialog: vi.fn(),
    saveDialog: vi.fn(),
    createSkillScaffold: vi.fn(),
    importLocalSkill: vi.fn(),
    inspectLocalSkill: vi.fn(),
    inspectLocalSkillDetail: vi.fn(),
    revealLocalSkill: vi.fn(),
    renameLocalSkill: vi.fn(),
    replaceLocalSkillPackage: vi.fn(),
    exportLocalSkillPackage: vi.fn(),
    inspectLocalSkillPackage: vi.fn(),
    installLocalSkillPackage: vi.fn(),
    getProject: vi.fn(),
    listRegisteredSkills: vi.fn(),
    listWorkspaceSkillBindings: vi.fn(),
    listScheduledTasks: vi.fn(),
    setScheduledTaskEnabled: vi.fn(),
    serviceSkills: [] as ServiceSkillHomeItem[],
    officialMarketplaceSkills: [] as SkillMarketplaceItem[],
    officialMarketplaceError: null as string | null,
    officialMarketplaceLoading: false,
    skillGroups: [] as ServiceSkillGroup[],
    localSkills: [] as Skill[],
  },
}));

export const mocks = hoisted.mocks;

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: (...args: unknown[]) => mocks.openDialog(...args),
  save: (...args: unknown[]) => mocks.saveDialog(...args),
}));

vi.mock("sonner", () => ({
  toast: {
    success: (...args: unknown[]) => mocks.toastSuccess(...args),
    error: (...args: unknown[]) => mocks.toastError(...args),
    info: vi.fn(),
    loading: vi.fn(() => "toast-id"),
  },
}));

vi.mock("@/components/agent/chat/service-skills/useServiceSkills", () => ({
  useServiceSkills: () => ({
    skills: mocks.serviceSkills,
    groups: mocks.skillGroups,
    catalogMeta: {
      tenantId: "tenant-demo",
      version: "catalog-v2",
      syncedAt: "2026-03-29T08:00:00Z",
      itemCount: mocks.serviceSkills.length,
      groupCount: mocks.skillGroups.length,
      sourceLabel: "租户技能目录",
      isSeeded: false,
    },
    isLoading: false,
    error: null,
    refresh: mocks.refreshServiceSkills,
    recordUsage: mocks.recordServiceSkillUsage,
  }),
}));

vi.mock("@/hooks/useOfficialSkillMarketplace", () => ({
  useOfficialSkillMarketplace: () => ({
    skills: mocks.officialMarketplaceSkills,
    isLoading: mocks.officialMarketplaceLoading,
    error: mocks.officialMarketplaceError,
    refresh: mocks.refreshOfficialMarketplace,
  }),
}));

vi.mock("@/hooks/useSkills", () => ({
  useSkills: () => ({
    skills: mocks.localSkills,
    repos: [],
    loading: false,
    remoteLoading: false,
    error: null,
    refresh: mocks.refreshLocalSkills,
    install: vi.fn(),
    uninstall: mocks.uninstallLocalSkill,
    addRepo: vi.fn(),
    removeRepo: vi.fn(),
  }),
}));

vi.mock("@/lib/api/skills", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api/skills")>();
  return {
    ...actual,
    skillsApi: {
      ...actual.skillsApi,
      createSkillScaffold: (...args: unknown[]) =>
        mocks.createSkillScaffold(...args),
      importLocalSkill: (...args: unknown[]) => mocks.importLocalSkill(...args),
      inspectLocalSkill: (...args: unknown[]) =>
        mocks.inspectLocalSkill(...args),
      inspectLocalSkillDetail: (...args: unknown[]) =>
        mocks.inspectLocalSkillDetail(...args),
      revealLocalSkill: (...args: unknown[]) => mocks.revealLocalSkill(...args),
      renameLocalSkill: (...args: unknown[]) => mocks.renameLocalSkill(...args),
      replaceLocalSkillPackage: (...args: unknown[]) =>
        mocks.replaceLocalSkillPackage(...args),
      exportLocalSkillPackage: (...args: unknown[]) =>
        mocks.exportLocalSkillPackage(...args),
      inspectLocalSkillPackage: (...args: unknown[]) =>
        mocks.inspectLocalSkillPackage(...args),
      installLocalSkillPackage: (...args: unknown[]) =>
        mocks.installLocalSkillPackage(...args),
    },
  };
});

vi.mock("@/lib/api/officialSkillMarketplace", () => ({
  installOfficialMarketplaceSkill: (...args: unknown[]) =>
    mocks.installOfficialMarketplaceSkill(...args),
  getOfficialSkillMarketplaceBundle: (...args: unknown[]) =>
    mocks.getOfficialSkillMarketplaceBundle(...args),
}));

vi.mock("@/lib/api/project", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api/project")>();
  return {
    ...actual,
    getProject: (...args: unknown[]) => mocks.getProject(...args),
  };
});

vi.mock("@/lib/api/capabilityDrafts", () => ({
  capabilityDraftsApi: {
    listRegisteredSkills: (...args: unknown[]) =>
      mocks.listRegisteredSkills(...args),
  },
}));

vi.mock("@/lib/api/agentRuntime/inventoryClient", () => ({
  listWorkspaceSkillBindings: (...args: unknown[]) =>
    mocks.listWorkspaceSkillBindings(...args),
}));

vi.mock("@/lib/api/scheduledTasks", () => ({
  scheduledTasksApi: {
    listDetailed: (...args: unknown[]) => mocks.listScheduledTasks(...args),
    setEnabled: (...args: unknown[]) =>
      mocks.setScheduledTaskEnabled(...args),
  },
}));

vi.mock("./SkillScaffoldDialog", () => ({
  SkillScaffoldDialog: ({
    open,
    initialValues,
    onCreate,
  }: {
    open: boolean;
    initialValues?: {
      target?: "project" | "user";
      directory?: string;
      name?: string;
      description?: string;
    } | null;
    onCreate?: (request: CreateSkillScaffoldRequest) => Promise<void>;
  }) =>
    open ? (
      <div data-testid="skill-scaffold-dialog">
        <span>{initialValues?.directory}</span>
        <span>{initialValues?.name}</span>
        <span>{initialValues?.description}</span>
        <button
          type="button"
          data-testid="skill-scaffold-create"
          onClick={() =>
            void onCreate?.({
              target: initialValues?.target ?? "user",
              directory: initialValues?.directory ?? "project-report",
              name: initialValues?.name ?? "项目报告",
              description:
                initialValues?.description ?? "沉淀为可注册的工作区技能。",
            })
          }
        >
          创建
        </button>
      </div>
    ) : null,
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: ReactNode }) =>
    open ? <div>{children}</div> : null,
  DialogContent: ({
    children,
    className,
  }: {
    children: ReactNode;
    className?: string;
  }) => <div className={className}>{children}</div>,
  DialogHeader: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  DialogFooter: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  DialogTitle: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
}));

export function createDefaultLocalSkills(): Skill[] {
  return [
    {
      key: "builtin:aspnet-core",
      name: "ASP.NET Core",
      description: "构建 ASP.NET Core Web 应用。",
      directory: "aspnet-core",
      installed: true,
      sourceKind: "builtin",
      catalogSource: "user",
    },
    {
      key: "local:writer",
      name: "写作助手",
      description: "本地补充技能",
      directory: "writer",
      installed: true,
      sourceKind: "other",
      catalogSource: "user",
      metadata: {
        lime_when_to_use: "当你需要复用本地写作 Skill 时使用。",
        lime_argument_hint: "主题、受众与语气要求",
      },
    },
  ];
}

function createDefaultServiceSkills(): ServiceSkillHomeItem[] {
  return [
    {
      id: "service-skill-research",
      title: "深度研究",
      summary: "综合多来源信息并给出归纳后的结论。",
      category: "调研",
      outputHint: "研究摘要",
      source: "cloud_catalog",
      runnerType: "instant",
      defaultExecutorBinding: "agent_turn",
      executionLocation: "client_default",
      slotSchema: [],
      version: "2026-03-29",
      badge: "官方",
      recentUsedAt: null,
      isRecent: false,
      runnerLabel: "立即开始",
      runnerTone: "sky",
      runnerDescription: "会先给出这一轮结果，接着就能继续改。",
      actionLabel: "开始这一步",
      automationStatus: null,
      groupKey: "general",
    },
    {
      id: "site-skill:github/search",
      title: "GitHub 仓库检索",
      summary: "围绕关键词采集 GitHub 仓库搜索结果。",
      category: "GitHub",
      outputHint: "仓库列表",
      source: "cloud_catalog",
      runnerType: "instant",
      defaultExecutorBinding: "browser_assist",
      executionLocation: "client_default",
      slotSchema: [],
      version: "2026-03-29",
      badge: "官方",
      recentUsedAt: null,
      isRecent: false,
      runnerLabel: "接着浏览器继续",
      runnerTone: "emerald",
      runnerDescription: "复用浏览器登录态执行。",
      actionLabel: "补齐这一步",
      automationStatus: null,
      groupKey: "github",
    },
  ];
}

function createDefaultMarketplaceSkills(): SkillMarketplaceItem[] {
  return [
    {
      id: "official:analysis",
      name: "analysis",
      aliases: ["data-analysis"],
      title: "数据分析",
      summary: "整理数据、提炼结论，并输出可继续追问的分析摘要。",
      category: "数据",
      outputHint: "分析摘要",
      version: "2026.05",
      sort: 10,
      icon: {
        kind: "svg",
        svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="12" fill="#e0f2fe"/></svg>',
      },
      cover: {
        kind: "svg",
        svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180"><rect width="320" height="180" rx="20" fill="#f0f9ff"/></svg>',
      },
      bundle: {
        name: "analysis",
        description: "标准 AgentSkills 数据分析包。",
        resourceSummary: {
          hasReferences: true,
          hasScripts: false,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      },
    },
  ];
}

function createDefaultMarketplaceBundle(): SkillMarketplaceBundle {
  return {
    manifestVersion: "agentskills.v1",
    name: "analysis",
    aliases: ["data-analysis"],
    version: "2026.05",
    contentHash: "sha256:bundle",
    fileCount: 1,
    files: [
      {
        path: "SKILL.md",
        content:
          "---\nname: Analysis\n---\n# Deep Research\n\n## Core Purpose\n\nDeliver **citation-backed** reports with `SKILL.md` guidance.\n\n> CRITICAL - Phase 0 is mandatory.\n\n## Decision Tree\n\n| Step | Output |\n| --- | --- |\n| Phase 0 | Scope |\n\n```\nRequest received\n```",
      },
    ],
  };
}

function createDefaultSkillGroups(): ServiceSkillGroup[] {
  return [
    {
      key: "github",
      title: "GitHub",
      summary: "围绕仓库与 Issue 的只读研究技能。",
      sort: 10,
      itemCount: 1,
    },
    {
      key: "general",
      title: "通用技能",
      summary: "不依赖站点登录态的创作技能。",
      sort: 90,
      itemCount: 1,
    },
  ];
}

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: RenderResult[] = [];

export function renderPage(pageParams?: SkillsPageParams) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const onNavigate = vi.fn<(page: string, params?: unknown) => void>();

  act(() => {
    root.render(
      <SkillsWorkspacePage onNavigate={onNavigate} pageParams={pageParams} />,
    );
  });

  mountedRoots.push({ container, root });
  return { container, onNavigate };
}

export function findButton(container: HTMLElement, label: string) {
  return Array.from(container.querySelectorAll("button")).find((button) =>
    button.textContent?.trim().includes(label),
  ) as HTMLButtonElement | undefined;
}

export function findMarketplaceCard(container: HTMLElement, title: string) {
  return Array.from(
    container.querySelectorAll('[data-testid="skills-marketplace-card"]'),
  ).find((card) => card.textContent?.includes(title)) as
    | HTMLElement
    | undefined;
}

export function findButtonIn(container: HTMLElement, label: string) {
  return Array.from(container.querySelectorAll("button")).find((button) =>
    button.textContent?.trim().includes(label),
  ) as HTMLButtonElement | undefined;
}

function findMenuItem(container: HTMLElement, label: string) {
  return Array.from(container.querySelectorAll('[role="menuitem"]')).find(
    (item) => item.textContent?.trim().includes(label),
  ) as HTMLElement | undefined;
}

export function clickMenuItem(container: HTMLElement, label: string) {
  const item = findMenuItem(container, label);
  expect(item).toBeTruthy();
  item?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

export function findLocalSkillRow(container: HTMLElement, directory: string) {
  return Array.from(
    container.querySelectorAll('[data-testid="skills-local-skill-row"]'),
  ).find((row) => row.getAttribute("data-skill-directory") === directory) as
    | HTMLElement
    | undefined;
}

export function openLocalSkillMenu(
  container: HTMLElement,
  directory = "writer",
) {
  const row = findLocalSkillRow(container, directory);
  expect(row).toBeTruthy();
  const skillName = directory === "writer" ? "写作助手" : directory;
  const button = Array.from(row?.querySelectorAll("button") ?? []).find(
    (candidate) =>
      candidate.getAttribute("aria-label") === `更多操作：${skillName}`,
  ) as HTMLButtonElement | undefined;
  expect(button).toBeTruthy();
  act(() => {
    button?.click();
  });
  return row!;
}

export function getLatestNavigationPayload(
  onNavigate: ReturnType<typeof vi.fn>,
) {
  return onNavigate.mock.calls.at(-1)?.[1] as
    | Record<string, unknown>
    | undefined;
}

export function useSkillsWorkspacePageTestLifecycle() {
  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;

    mocks.serviceSkills = createDefaultServiceSkills();
    mocks.officialMarketplaceSkills = createDefaultMarketplaceSkills();
    mocks.officialMarketplaceError = null;
    mocks.officialMarketplaceLoading = false;
    mocks.skillGroups = createDefaultSkillGroups();
    mocks.localSkills = createDefaultLocalSkills();
    mocks.refreshServiceSkills.mockReset();
    mocks.refreshServiceSkills.mockResolvedValue(undefined);
    mocks.refreshOfficialMarketplace.mockReset();
    mocks.refreshOfficialMarketplace.mockResolvedValue(undefined);
    mocks.recordServiceSkillUsage.mockReset();
    mocks.refreshLocalSkills.mockReset();
    mocks.refreshLocalSkills.mockResolvedValue(undefined);
    mocks.uninstallLocalSkill.mockReset();
    mocks.uninstallLocalSkill.mockResolvedValue(undefined);
    mocks.installOfficialMarketplaceSkill.mockReset();
    mocks.installOfficialMarketplaceSkill.mockResolvedValue({
      directory: "analysis",
      inspection: {
        content: "# Analysis",
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: false,
          hasReferences: true,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      },
    });
    mocks.getOfficialSkillMarketplaceBundle.mockReset();
    mocks.getOfficialSkillMarketplaceBundle.mockResolvedValue(
      createDefaultMarketplaceBundle(),
    );
    mocks.toastSuccess.mockReset();
    mocks.toastError.mockReset();
    mocks.openDialog.mockReset();
    mocks.openDialog.mockResolvedValue(null);
    mocks.saveDialog.mockReset();
    mocks.saveDialog.mockResolvedValue(null);
    mocks.createSkillScaffold.mockReset();
    mocks.importLocalSkill.mockReset();
    mocks.inspectLocalSkill.mockReset();
    mocks.inspectLocalSkill.mockImplementation((directory: string) =>
      Promise.resolve({
        content: `---\nname: ${directory}\n---\n\n# ${directory}\n\nDetail for **${directory}** with \`SKILL.md\`.\n\n| Section | Status |\n| --- | --- |\n| Preview | Ready |`,
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: false,
          hasReferences: false,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      }),
    );
    mocks.inspectLocalSkillDetail.mockReset();
    mocks.inspectLocalSkillDetail.mockImplementation((directory: string) =>
      Promise.resolve({
        directory,
        inspection: {
          content: `---\nname: ${directory}\n---\n\n# ${directory}\n\nDetail for **${directory}** with \`SKILL.md\`.\n\n| Section | Status |\n| --- | --- |\n| Preview | Ready |`,
          metadata: {},
          allowedTools: [],
          resourceSummary: {
            hasScripts: false,
            hasReferences: true,
            hasAssets: false,
          },
          standardCompliance: {
            isStandard: true,
            validationErrors: [],
            deprecatedFields: [],
          },
        },
        files: [
          {
            path: "SKILL.md",
            isDirectory: false,
            size: 128,
            content: `# ${directory}\n\nDetail for **${directory}** with \`SKILL.md\`.\n\n| Section | Status |\n| --- | --- |\n| Preview | Ready |`,
          },
          { path: "references", isDirectory: true, size: 0 },
          {
            path: "references/guide.md",
            isDirectory: false,
            size: 48,
            content: "# Reference Guide\n\nFile tree detail.",
          },
        ],
      }),
    );
    mocks.revealLocalSkill.mockReset();
    mocks.revealLocalSkill.mockResolvedValue(true);
    mocks.renameLocalSkill.mockReset();
    mocks.renameLocalSkill.mockImplementation(
      (_directory: string, newDirectory: string) =>
        Promise.resolve({ directory: newDirectory }),
    );
    mocks.replaceLocalSkillPackage.mockReset();
    mocks.replaceLocalSkillPackage.mockResolvedValue({
      directory: "writer",
      inspection: {
        content: "# Replaced Writer",
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: false,
          hasReferences: true,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      },
    });
    mocks.exportLocalSkillPackage.mockReset();
    mocks.exportLocalSkillPackage.mockResolvedValue({
      directory: "writer",
      outputPath: "/Users/demo/writer.skills",
      fileCount: 2,
      bytesWritten: 512,
    });
    mocks.inspectLocalSkillPackage.mockReset();
    mocks.inspectLocalSkillPackage.mockResolvedValue({
      directory: "article-typesetting-master",
      inspection: {
        content: "---\nname: Article Typesetting\n---\n\n# Article Typesetting",
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: false,
          hasReferences: true,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      },
      files: [
        { path: "SKILL.md", isDirectory: false, size: 128 },
        { path: "references", isDirectory: true, size: 0 },
      ],
    });
    mocks.installLocalSkillPackage.mockReset();
    mocks.installLocalSkillPackage.mockResolvedValue({
      directory: "article-typesetting-master",
      inspection: {
        content: "# Article Typesetting",
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: false,
          hasReferences: true,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      },
    });
    mocks.getProject.mockReset();
    mocks.getProject.mockResolvedValue({
      id: "default-workspace",
      name: "默认工作区",
      workspaceType: "general",
      rootPath: "/Users/demo/Lime/default-workspace",
      isDefault: true,
      createdAt: 0,
      updatedAt: 0,
      isFavorite: false,
      isArchived: false,
      tags: [],
    });
    mocks.listRegisteredSkills.mockReset();
    mocks.listRegisteredSkills.mockResolvedValue([]);
    mocks.listWorkspaceSkillBindings.mockReset();
    mocks.listWorkspaceSkillBindings.mockResolvedValue({ bindings: [] });
    mocks.listScheduledTasks.mockReset();
    mocks.listScheduledTasks.mockResolvedValue([]);
    mocks.setScheduledTaskEnabled.mockReset();
    mocks.setScheduledTaskEnabled.mockResolvedValue({});
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
}
