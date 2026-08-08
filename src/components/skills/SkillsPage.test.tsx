import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { SkillsPage } from "./SkillsPage";
import {
  filterSkillsByQueryAndStatus,
  groupSkillsBySourceKind,
} from "./skillsUtils";
import type { Skill } from "@/lib/api/skills";

const mockUseSkills = vi.fn();
const mockInspectLocalSkill = vi.fn();
const mockInspectRemoteSkill = vi.fn();
const mockCreateSkillScaffold = vi.fn();
const mockImportLocalSkill = vi.fn();
const mockOpenDialog = vi.fn();

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: (...args: unknown[]) => mockOpenDialog(...args),
}));

vi.mock("@/hooks/useSkills", () => ({
  useSkills: (...args: unknown[]) => mockUseSkills(...args),
}));

vi.mock("@/lib/api/skills", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/api/skills")>(
      "@/lib/api/skills",
    );

  return {
    ...actual,
    skillsApi: {
      ...actual.skillsApi,
      inspectLocalSkill: (...args: unknown[]) => mockInspectLocalSkill(...args),
      inspectRemoteSkill: (...args: unknown[]) =>
        mockInspectRemoteSkill(...args),
      createSkillScaffold: (...args: unknown[]) =>
        mockCreateSkillScaffold(...args),
      importLocalSkill: (...args: unknown[]) => mockImportLocalSkill(...args),
    },
  };
});

vi.mock("./SkillContentDialog", () => ({
  SkillContentDialog: () => null,
}));

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: RenderResult[] = [];

function createSkill(overrides: Partial<Skill> = {}): Skill {
  return {
    key: "skill:test",
    name: "Test Skill",
    description: "A test skill",
    directory: "test-skill",
    installed: true,
    sourceKind: "other",
    ...overrides,
  };
}

function renderSkillsPage(
  props: Partial<ComponentProps<typeof SkillsPage>> = {},
): RenderResult {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<SkillsPage hideHeader {...props} />);
  });

  const rendered = { container, root };
  mountedRoots.push(rendered);
  return rendered;
}

function getBodyText() {
  return document.body.textContent ?? "";
}

async function hoverTip(ariaLabel: string) {
  const trigger = document.body.querySelector(
    `button[aria-label='${ariaLabel}']`,
  );
  expect(trigger).toBeInstanceOf(HTMLButtonElement);

  await act(async () => {
    trigger?.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    await Promise.resolve();
  });

  return trigger as HTMLButtonElement;
}

async function leaveTip(trigger: HTMLButtonElement | null) {
  await act(async () => {
    trigger?.dispatchEvent(new MouseEvent("mouseout", { bubbles: true }));
    await Promise.resolve();
  });
}

function fillField(
  element: HTMLInputElement | HTMLTextAreaElement | null,
  value: string,
) {
  if (!element) {
    throw new Error("field not found");
  }

  const prototype =
    element instanceof HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
  setter?.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");

  mockUseSkills.mockReturnValue({
    skills: [],
    repos: [],
    loading: false,
    remoteLoading: false,
    error: null,
    refresh: vi.fn(),
    install: vi.fn(),
    uninstall: vi.fn(),
    addRepo: vi.fn(),
    removeRepo: vi.fn(),
  });

  mockInspectLocalSkill.mockReset();
  mockInspectRemoteSkill.mockReset();
  mockCreateSkillScaffold.mockReset();
  mockImportLocalSkill.mockReset();
  mockOpenDialog.mockReset();
  mockInspectLocalSkill.mockResolvedValue({
    content: "# Test",
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
  });
  mockInspectRemoteSkill.mockResolvedValue({
    content: "# Remote",
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
  });
  mockCreateSkillScaffold.mockResolvedValue({
    content: "# Scaffold",
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
  });
  mockImportLocalSkill.mockResolvedValue({ directory: "imported-skill" });
  mockOpenDialog.mockResolvedValue("/tmp/imported-skill");
});

afterEach(async () => {
  mockUseSkills.mockReset();

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
  await changeLimeLocale("zh-CN");
});

describe("filterSkillsByQueryAndStatus", () => {
  it("应同时按搜索词和安装状态过滤技能", () => {
    const skills = [
      createSkill({ name: "Video Skill", installed: true }),
      createSkill({
        key: "skill:draft",
        name: "Draft Writer",
        directory: "draft-writer",
        installed: false,
      }),
    ];

    const result = filterSkillsByQueryAndStatus(skills, "draft", "uninstalled");
    expect(result).toHaveLength(1);
    expect(result[0]?.name).toBe("Draft Writer");
  });
});

describe("groupSkillsBySourceKind", () => {
  it("应将技能分为内置、本地和远程三组", () => {
    const sections = groupSkillsBySourceKind([
      createSkill({
        key: "builtin:video_generate",
        name: "Video Generate",
        directory: "video_generate",
        sourceKind: "builtin",
      }),
      createSkill({
        key: "local:custom",
        name: "Custom Skill",
        directory: "custom-skill",
        sourceKind: "other",
      }),
      createSkill({
        key: "repo:remote",
        name: "Remote Skill",
        directory: "remote-skill",
        sourceKind: "other",
        catalogSource: "remote",
      }),
    ]);

    expect(sections[0]?.key).toBe("builtin");
    expect(sections[0]?.skills).toHaveLength(1);
    expect(sections[1]?.key).toBe("local");
    expect(sections[1]?.skills).toHaveLength(1);
    expect(sections[2]?.key).toBe("remote");
    expect(sections[2]?.skills).toHaveLength(1);
  });
});

describe("SkillsPage", () => {
  it("高级入口 chrome 应通过 agent namespace 渲染英文资源", async () => {
    await changeLimeLocale("en-US");
    renderSkillsPage({ hideHeader: false });

    expect(getBodyText()).toContain("Manage and use AI skill extensions");
    expect(getBodyText()).toContain("Refresh");
    expect(getBodyText()).toContain("New Skill");
    expect(getBodyText()).toContain("Import Skill");
    expect(getBodyText()).toContain("Repositories");
    expect(
      document.body.querySelector("input")?.getAttribute("placeholder"),
    ).toBe("Search Skills...");
    expect(getBodyText()).not.toContain("管理和使用 AI 技能扩展");

    const workspaceTip = await hoverTip("Skills workspace info");
    expect(getBodyText()).toContain(
      "View installation status, repository sources, and readable content in one place",
    );
    await leaveTip(workspaceTip);
  });

  it("应将首屏说明与使用规则收进 tips", async () => {
    renderSkillsPage({ hideHeader: false });

    expect(getBodyText()).not.toContain(
      "统一查看安装状态、仓库来源与可读内容，减少在不同入口之间来回切换。",
    );
    expect(getBodyText()).not.toContain(
      "Built-in Skills 为应用内置技能，默认可用且不可卸载。",
    );

    const workspaceTip = await hoverTip("技能工作台说明");
    expect(getBodyText()).toContain(
      "统一查看安装状态、仓库来源与可读内容，减少在不同入口之间来回切换。",
    );
    await leaveTip(workspaceTip);

    const usageTip = await hoverTip("技能使用规则");
    expect(getBodyText()).toContain(
      "Built-in Skills 为应用内置技能，默认可用且不可卸载。",
    );
    await leaveTip(usageTip);
  });

  it("应按 Built-in / Local / Remote Skills 分组渲染，并隐藏内置技能卸载入口", () => {
    mockUseSkills.mockReturnValue({
      skills: [
        createSkill({
          key: "builtin:video_generate",
          name: "Video Generate",
          directory: "video_generate",
          sourceKind: "builtin",
        }),
        createSkill({
          key: "local:custom",
          name: "Local Skill",
          directory: "local-skill",
          sourceKind: "other",
          installed: false,
        }),
        createSkill({
          key: "repo:custom",
          name: "Remote Skill",
          directory: "remote-skill",
          catalogSource: "remote",
          repoOwner: "lime",
          repoName: "skills",
          sourceKind: "other",
        }),
      ],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: null,
      refresh: vi.fn(),
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    const { container } = renderSkillsPage();
    const text = container.textContent ?? "";

    expect(text).toContain("内置 Skills");
    expect(text).toContain("本地 Skills");
    expect(text).toContain("远程 Skills");
    expect(text.indexOf("内置 Skills")).toBeLessThan(
      text.indexOf("本地 Skills"),
    );
    expect(text.indexOf("本地 Skills")).toBeLessThan(
      text.indexOf("远程 Skills"),
    );

    const buttonTexts = Array.from(container.querySelectorAll("button")).map(
      (button) => button.textContent?.trim() ?? "",
    );
    const uninstallButtons = buttonTexts.filter((textContent) =>
      textContent.includes("卸载"),
    );

    expect(uninstallButtons).toHaveLength(1);
    expect(
      buttonTexts.some((textContent) => textContent.includes("执行")),
    ).toBe(false);
  });

  it("远程缓存为空时仍应显示远程分组和刷新提示", () => {
    mockUseSkills.mockReturnValue({
      skills: [
        createSkill({
          key: "builtin:video_generate",
          name: "Video Generate",
          directory: "video_generate",
          sourceKind: "builtin",
        }),
        createSkill({
          key: "local:custom",
          name: "Local Skill",
          directory: "local-skill",
          sourceKind: "other",
        }),
      ],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: null,
      refresh: vi.fn(),
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    const { container } = renderSkillsPage();
    const text = container.textContent ?? "";

    expect(text).toContain("远程 Skills");
    expect(text).toContain("暂无远程缓存");
    expect(text).toContain('点击"刷新"同步已启用仓库');
  });

  it("skills 读取错误时应使用浅色错误横幅", () => {
    mockUseSkills.mockReturnValue({
      skills: [],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: "远程仓库读取失败",
      refresh: vi.fn(),
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    const { container } = renderSkillsPage();
    const errorBanner = Array.from(container.querySelectorAll("div")).find(
      (element) =>
        element.textContent?.includes("远程仓库读取失败") &&
        element.className.includes("bg-red-50/90"),
    );

    expect(errorBanner).toBeTruthy();
    expect(errorBanner?.className).toContain("bg-red-50/90");
    expect(errorBanner?.className).not.toContain("dark:bg-red-950/30");
  });

  it("点击本地 skill 的查看内容应调用本地 inspection", async () => {
    mockUseSkills.mockReturnValue({
      skills: [
        createSkill({
          key: "local:custom",
          name: "Local Skill",
          directory: "local-skill",
          sourceKind: "other",
          installed: true,
        }),
      ],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: null,
      refresh: vi.fn(),
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    const { container } = renderSkillsPage();
    const button = Array.from(container.querySelectorAll("button")).find(
      (item) => item.textContent?.includes("查看内容"),
    );

    await act(async () => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(mockInspectLocalSkill).toHaveBeenCalledWith("local-skill", "lime");
    expect(mockInspectRemoteSkill).not.toHaveBeenCalled();
  });

  it("点击远程 skill 的检查详情应调用远程 inspection", async () => {
    mockUseSkills.mockReturnValue({
      skills: [
        createSkill({
          key: "repo:remote",
          name: "Remote Skill",
          directory: "remote-skill",
          installed: false,
          sourceKind: "other",
          catalogSource: "remote",
          repoOwner: "lime",
          repoName: "skills",
          repoBranch: "main",
        }),
      ],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: null,
      refresh: vi.fn(),
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    const { container } = renderSkillsPage();
    const button = Array.from(container.querySelectorAll("button")).find(
      (item) => item.textContent?.includes("检查详情"),
    );

    await act(async () => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(mockInspectRemoteSkill).toHaveBeenCalledWith({
      owner: "lime",
      name: "skills",
      branch: "main",
      directory: "remote-skill",
    });
    expect(mockInspectLocalSkill).not.toHaveBeenCalled();
  });

  it("创建标准 Skill 脚手架后应调用创建 API 并刷新列表", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    mockUseSkills.mockReturnValue({
      skills: [],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: null,
      refresh,
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    renderSkillsPage();

    const openButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((item) => item.textContent?.includes("新建 Skill"));

    await act(async () => {
      openButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    const directoryInput = document.body.querySelector(
      "#skill-scaffold-directory",
    ) as HTMLInputElement | null;
    const nameInput = document.body.querySelector(
      "#skill-scaffold-name",
    ) as HTMLInputElement | null;
    const descriptionInput = document.body.querySelector(
      "#skill-scaffold-description",
    ) as HTMLTextAreaElement | null;

    await act(async () => {
      fillField(directoryInput, "draft-skill");
      fillField(nameInput, "Draft Skill");
      fillField(descriptionInput, "Create a standard scaffold");
    });

    const createButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((item) => item.textContent?.trim() === "创建 Skill");

    await act(async () => {
      createButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(mockCreateSkillScaffold).toHaveBeenCalledWith(
      {
        target: "project",
        directory: "draft-skill",
        name: "Draft Skill",
        description: "Create a standard scaffold",
      },
      "lime",
    );
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("带着结果草稿进入时应自动打开预填的脚手架对话框", async () => {
    renderSkillsPage({
      initialScaffoldDraft: {
        target: "project",
        directory: "saved-skill-draft",
        name: "沉淀后的技能",
        description: "沉淀自一次成功结果",
        whenToUse: ["当你需要继续复用这类结果时使用。"],
        inputs: ["目标与主题：一段来自聊天结果的摘要"],
        outputs: ["交付一份可直接复用的完整结果。"],
        steps: ["先确认目标，再沿用结构。"],
        fallbackStrategy: ["信息不足时先补问。"],
        sourceExcerpt: "一段来自聊天结果的摘要",
      },
      initialScaffoldRequestKey: 20260408,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      (
        document.body.querySelector(
          "#skill-scaffold-directory",
        ) as HTMLInputElement | null
      )?.value,
    ).toBe("saved-skill-draft");
    expect(
      (
        document.body.querySelector(
          "#skill-scaffold-name",
        ) as HTMLInputElement | null
      )?.value,
    ).toBe("沉淀后的技能");
    expect(document.body.textContent).toContain(
      "来源结果：一段来自聊天结果的摘要",
    );
  });

  it("带着结果草稿进入时应支持带回创作输入", async () => {
    const onBringScaffoldToCreation = vi.fn();
    renderSkillsPage({
      initialScaffoldDraft: {
        target: "project",
        directory: "saved-skill-draft",
        name: "沉淀后的技能",
        description: "沉淀自一次成功结果",
        whenToUse: ["当你需要继续复用这类结果时使用。"],
        inputs: ["目标与主题：一段来自聊天结果的摘要"],
        outputs: ["交付一份可直接复用的完整结果。"],
        steps: ["先确认目标，再沿用结构。"],
        fallbackStrategy: ["信息不足时先补问。"],
        sourceExcerpt: "一段来自聊天结果的摘要",
        sourceMessageId: "msg-1",
      },
      initialScaffoldRequestKey: 20260410,
      onBringScaffoldToCreation,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const backButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((item) => item.textContent?.trim() === "带回创作输入");

    await act(async () => {
      backButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onBringScaffoldToCreation).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "沉淀后的技能",
        sourceExcerpt: "一段来自聊天结果的摘要",
      }),
    );
  });

  it("带着结构化草稿进入后创建时应把隐藏骨架一并传给创建 API", async () => {
    renderSkillsPage({
      initialScaffoldDraft: {
        target: "project",
        directory: "saved-skill-draft",
        name: "沉淀后的技能",
        description: "沉淀自一次成功结果",
        whenToUse: ["当你需要继续复用这类结果时使用。"],
        inputs: ["目标与主题：一段来自聊天结果的摘要"],
        outputs: ["交付一份可直接复用的完整结果。"],
        steps: ["先确认目标，再沿用结构。"],
        fallbackStrategy: ["信息不足时先补问。"],
      },
      initialScaffoldRequestKey: 20260409,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const createButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((item) => item.textContent?.trim() === "创建 Skill");

    await act(async () => {
      createButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(mockCreateSkillScaffold).toHaveBeenCalledWith(
      {
        target: "project",
        directory: "saved-skill-draft",
        name: "沉淀后的技能",
        description: "沉淀自一次成功结果",
        whenToUse: ["当你需要继续复用这类结果时使用。"],
        inputs: ["目标与主题：一段来自聊天结果的摘要"],
        outputs: ["交付一份可直接复用的完整结果。"],
        steps: ["先确认目标，再沿用结构。"],
        fallbackStrategy: ["信息不足时先补问。"],
      },
      "lime",
    );
  });

  it("创建脚手架成功后应把新 Skill 回调给外层工作台", async () => {
    const onScaffoldCreated = vi.fn();

    renderSkillsPage({
      initialScaffoldDraft: {
        target: "project",
        directory: "saved-skill-draft",
        name: "沉淀后的技能",
        description: "沉淀自一次成功结果",
      },
      initialScaffoldRequestKey: 20260411,
      onScaffoldCreated,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const createButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((item) => item.textContent?.trim() === "创建 Skill");

    await act(async () => {
      createButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onScaffoldCreated).toHaveBeenCalledWith(
      expect.objectContaining({
        key: "local:saved-skill-draft",
        name: "沉淀后的技能",
        directory: "saved-skill-draft",
        installed: true,
        catalogSource: "project",
      }),
    );
  });

  it("点击导入 Skill 应调用导入 API 并刷新列表", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    mockUseSkills.mockReturnValue({
      skills: [],
      repos: [],
      loading: false,
      remoteLoading: false,
      error: null,
      refresh,
      install: vi.fn(),
      uninstall: vi.fn(),
      addRepo: vi.fn(),
      removeRepo: vi.fn(),
    });

    const { container } = renderSkillsPage();
    const importButton = Array.from(container.querySelectorAll("button")).find(
      (item) => item.textContent?.includes("导入 Skill"),
    );

    await act(async () => {
      importButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(mockOpenDialog).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
      title: "选择一个包含 SKILL.md 的技能目录",
    });
    expect(mockImportLocalSkill).toHaveBeenCalledWith(
      "/tmp/imported-skill",
      "lime",
    );
    expect(refresh).toHaveBeenCalledTimes(1);
  });
});
