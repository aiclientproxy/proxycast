/* eslint-disable react-refresh/only-export-components */
import React, { useRef, useState } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, vi } from "vitest";
import { CharacterMention } from "./CharacterMention";
import type { Character } from "@/lib/api/projectMemory";
import type { Skill } from "@/lib/api/skills";
import {
  clearSkillCatalogCache,
  getSeededSkillCatalog,
  type SkillCatalog,
} from "@/lib/api/skillCatalog";
import type {
  ServiceSkillGroup,
  ServiceSkillHomeItem,
} from "@/components/agent/chat/service-skills/types";
import type { InputCapabilitySelection } from "./inputCapabilitySelection";
import type { CuratedTaskReferenceEntry } from "../utils/curatedTaskReferenceSelection";
import { changeLimeLocale } from "@/i18n/createI18n";
import type {
  InputbarPluginCapability,
  InputbarPluginSelectionOptions,
  InputbarPluginSkillCapability,
} from "../components/Inputbar/pluginInputCapability";

const hoistedMocks = vi.hoisted(() => ({
  mockListServiceSkills: vi.fn(),
  mockGetProject: vi.fn(),
  mockSearchProjectFiles: vi.fn(),
}));

export const mockListServiceSkills = hoistedMocks.mockListServiceSkills;
export const mockGetProject = hoistedMocks.mockGetProject;
export const mockSearchProjectFiles = hoistedMocks.mockSearchProjectFiles;

vi.mock("sonner", () => ({
  toast: {
    info: vi.fn(),
  },
}));

vi.mock("@/lib/api/serviceSkills", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/api/serviceSkills")>();

  return {
    ...actual,
    listServiceSkills: () => hoistedMocks.mockListServiceSkills(),
  };
});

vi.mock("@/lib/api/project", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api/project")>();
  return {
    ...actual,
    getProject: (...args: unknown[]) => hoistedMocks.mockGetProject(...args),
  };
});

vi.mock("@/lib/api/fuzzyFileSearch", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/api/fuzzyFileSearch")>();
  return {
    ...actual,
    searchProjectFiles: (...args: unknown[]) =>
      hoistedMocks.mockSearchProjectFiles(...args),
  };
});

vi.mock("@/components/ui/popover", () => {
  const Popover = ({
    open,
    children,
  }: {
    open?: boolean;
    children: React.ReactNode;
  }) => (open ? <div data-testid="mention-popover">{children}</div> : null);

  const PopoverTrigger = ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  );

  const PopoverContent = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement> & {
      side?: string;
      align?: string;
      avoidCollisions?: boolean;
      sideOffset?: number;
      onOpenAutoFocus?: (event: Event) => void;
    }
  >(
    (
      {
        children,
        className,
        style,
        side,
        align,
        avoidCollisions,
        sideOffset: _sideOffset,
        onOpenAutoFocus: _onOpenAutoFocus,
        ...props
      },
      ref,
    ) => (
      <div
        ref={ref}
        className={className}
        style={style}
        data-side={side}
        data-align={align}
        data-avoid-collisions={String(avoidCollisions)}
        {...props}
      >
        {children}
      </div>
    ),
  );

  return { Popover, PopoverTrigger, PopoverContent };
});

vi.mock("@/components/ui/command", () => {
  const Command = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement> & { shouldFilter?: boolean }
  >(({ children, shouldFilter = true, ...props }, ref) => (
    <div
      ref={ref}
      data-testid="mention-command-root"
      data-should-filter={String(shouldFilter)}
      {...props}
    >
      {children}
    </div>
  ));

  const CommandInput = ({
    value,
    onValueChange,
    placeholder,
  }: {
    value?: string;
    onValueChange?: (value: string) => void;
    placeholder?: string;
  }) => (
    <input
      data-testid="mention-command-input"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onValueChange?.(e.target.value)}
    />
  );

  const CommandList = ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  );

  const CommandGroup = ({
    heading,
    children,
  }: {
    heading?: string;
    children: React.ReactNode;
  }) => (
    <section>
      {heading && <div>{heading}</div>}
      {children}
    </section>
  );

  const CommandItem = ({
    children,
    onSelect,
    disabled,
  }: {
    children: React.ReactNode;
    onSelect?: () => void;
    disabled?: boolean;
  }) => (
    <button
      type="button"
      disabled={disabled}
      onClick={() => {
        if (!disabled) {
          onSelect?.();
        }
      }}
    >
      {children}
    </button>
  );

  const CommandEmpty = ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  );

  return {
    Command,
    CommandInput,
    CommandList,
    CommandGroup,
    CommandItem,
    CommandEmpty,
  };
});

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

vi.mock("@/components/ui/label", () => ({
  Label: ({
    children,
    htmlFor,
    className,
  }: {
    children: React.ReactNode;
    htmlFor?: string;
    className?: string;
  }) => (
    <label htmlFor={htmlFor} className={className}>
      {children}
    </label>
  ),
}));

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

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  await changeLimeLocale("zh-CN");

  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  clearSkillCatalogCache();
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
  clearSkillCatalogCache();
  vi.clearAllMocks();
});

beforeEach(() => {
  mockListServiceSkills.mockResolvedValue([]);
  mockGetProject.mockResolvedValue(null);
  mockSearchProjectFiles.mockResolvedValue([]);
});

export interface HarnessProps {
  characters?: Character[];
  skills?: Skill[];
  serviceSkills?: ServiceSkillHomeItem[];
  serviceSkillGroups?: ServiceSkillGroup[];
  syncValue?: boolean;
  onNavigateToSettings?: () => void;
  onChangeSpy?: (value: string) => void;
  onSelectInputCapability?: (
    capability: InputCapabilitySelection,
    options?: { replayText?: string },
  ) => void;
  pluginSuggestions?: readonly InputbarPluginCapability[];
  onSelectPlugin?: (
    plugin: InputbarPluginCapability,
    skill?: InputbarPluginSkillCapability,
    options?: InputbarPluginSelectionOptions,
  ) => void;
  projectId?: string | null;
  sessionId?: string | null;
  defaultCuratedTaskReferenceMemoryIds?: string[];
  defaultCuratedTaskReferenceEntries?: CuratedTaskReferenceEntry[];
  inputCompletionEnabled?: boolean;
}

const Harness: React.FC<HarnessProps> = ({
  characters = [],
  skills = [],
  serviceSkills = [],
  serviceSkillGroups = [],
  syncValue = true,
  onNavigateToSettings,
  onChangeSpy,
  onSelectInputCapability,
  pluginSuggestions = [],
  onSelectPlugin,
  projectId = null,
  sessionId = null,
  defaultCuratedTaskReferenceMemoryIds = [],
  defaultCuratedTaskReferenceEntries = [],
  inputCompletionEnabled = true,
}) => {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  return (
    <div>
      <textarea
        ref={inputRef}
        data-testid="mention-input"
        defaultValue=""
        onChange={(event) => {
          if (syncValue) {
            setValue(event.target.value);
          }
        }}
      />
      <CharacterMention
        characters={characters}
        skills={skills}
        serviceSkills={serviceSkills}
        serviceSkillGroups={serviceSkillGroups}
        inputRef={inputRef}
        value={value}
        onChange={(next) => {
          onChangeSpy?.(next);
          if (syncValue) {
            setValue(next);
          }
        }}
        onSelectInputCapability={onSelectInputCapability}
        pluginSuggestions={pluginSuggestions}
        onSelectPlugin={onSelectPlugin}
        projectId={projectId}
        sessionId={sessionId}
        defaultCuratedTaskReferenceMemoryIds={
          defaultCuratedTaskReferenceMemoryIds
        }
        defaultCuratedTaskReferenceEntries={defaultCuratedTaskReferenceEntries}
        onNavigateToSettings={onNavigateToSettings}
        inputCompletionEnabled={inputCompletionEnabled}
      />
    </div>
  );
};

export function renderHarness(props: HarnessProps = {}) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<Harness {...props} />);
  });

  mountedRoots.push({ root, container });
  return container;
}

export function getTextarea(container: HTMLElement): HTMLTextAreaElement {
  const textarea = container.querySelector(
    '[data-testid="mention-input"]',
  ) as HTMLTextAreaElement | null;
  if (!textarea) {
    throw new Error("未找到输入框");
  }
  return textarea;
}

export function getMentionPopoverContent(): Element | null {
  return document.body.querySelector('[data-testid="mention-popover-content"]');
}

export function findButtonContaining(
  ...texts: string[]
): HTMLButtonElement | undefined {
  return Array.from(document.body.querySelectorAll("button")).find((button) =>
    texts.every((text) => button.textContent?.includes(text)),
  );
}

export function getButtonsContaining(...texts: string[]): HTMLButtonElement[] {
  return Array.from(document.body.querySelectorAll("button")).filter((button) =>
    texts.every((text) => button.textContent?.includes(text)),
  );
}

export function typeAt(textarea: HTMLTextAreaElement) {
  act(() => {
    textarea.focus();
    textarea.value = "@";
    textarea.setSelectionRange(1, 1);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

export function typeMention(textarea: HTMLTextAreaElement, value: string) {
  act(() => {
    textarea.focus();
    textarea.value = value;
    textarea.setSelectionRange(value.length, value.length);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

export function typeSlash(textarea: HTMLTextAreaElement, value = "/") {
  act(() => {
    textarea.focus();
    textarea.value = value;
    textarea.setSelectionRange(value.length, value.length);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

export async function typeAtAndWait(textarea: HTMLTextAreaElement) {
  await act(async () => {
    await import("./CharacterMentionPanel");
  });

  typeAt(textarea);
  await act(async () => {
    await Promise.resolve();
  });
}

export async function typeMentionAndWait(
  textarea: HTMLTextAreaElement,
  value: string,
) {
  await act(async () => {
    await import("./CharacterMentionPanel");
  });

  typeMention(textarea, value);
  await act(async () => {
    await Promise.resolve();
  });
}

export async function typeSlashAndWait(
  textarea: HTMLTextAreaElement,
  value = "/",
) {
  await act(async () => {
    await import("./CharacterMentionPanel");
  });

  typeSlash(textarea, value);
  await act(async () => {
    await Promise.resolve();
  });
}

export function updateFieldValue(
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

export function findLauncherConfirmButton() {
  return (
    (document.body.querySelector(
      '[data-testid="curated-task-launcher-confirm"]',
    ) as HTMLButtonElement | null) ??
    (Array.from(document.body.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("开始生成"),
    ) as HTMLButtonElement | undefined)
  );
}

export function createSkill(
  name: string,
  key: string,
  installed: boolean,
  overrides: Partial<Skill> = {},
): Skill {
  return {
    key,
    name,
    description: "测试技能",
    directory: `${key}-dir`,
    installed,
    sourceKind: "builtin",
    ...overrides,
  };
}

export function createCharacter(name: string): Character {
  const now = new Date().toISOString();
  return {
    id: "char-1",
    project_id: "project-1",
    name,
    aliases: [],
    description: "测试角色",
    personality: undefined,
    background: undefined,
    appearance: undefined,
    relationships: [],
    avatar_url: undefined,
    is_main: true,
    order: 0,
    extra: undefined,
    created_at: now,
    updated_at: now,
  };
}

export function createServiceSkill(
  overrides: Partial<ServiceSkillHomeItem> = {},
): ServiceSkillHomeItem {
  return {
    id: "daily-trend-briefing",
    title: "每日趋势摘要",
    summary: "围绕指定平台与关键词输出趋势摘要。",
    entryHint: "把平台和关键词给我，我先整理一份趋势报告。",
    aliases: ["趋势报告", "热点摘要"],
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
    runnerDescription: "当前先进入工作区生成首版任务方案，后续再接本地自动化。",
    actionLabel: "先做方案",
    automationStatus: null,
    groupKey: "general",
    ...overrides,
  };
}

export function createStoryVideoSceneServiceSkill(
  overrides: Partial<ServiceSkillHomeItem> = {},
): ServiceSkillHomeItem {
  return createServiceSkill({
    id: "story-video-suite",
    skillKey: "story-video-suite",
    title: "短视频编排",
    summary: "把活动介绍、脚本、分镜和发布节奏串成短视频执行稿。",
    category: "视频创作",
    outputHint: "短视频结构 + 分镜 + 执行清单",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    slotSchema: [
      {
        key: "source_material",
        label: "活动 / 内容素材",
        type: "textarea",
        required: true,
        placeholder: "粘贴活动介绍、产品卖点、文章摘要或脚本素材",
      },
    ],
    sceneBinding: {
      sceneKey: "story-video-suite",
      commandPrefix: "/story-video-suite",
      title: "短视频编排",
      summary: "把文本、分镜、配乐和执行步骤串成一条本地场景链。",
      aliases: ["story-video", "短视频编排"],
    },
    runnerLabel: "先做这一轮",
    runnerTone: "emerald",
    runnerDescription: "先在当前工作区产出一版可继续加工的执行稿。",
    actionLabel: "补齐这一步",
    ...overrides,
  });
}

export function buildCatalogWithSceneEntry(): SkillCatalog {
  const seeded = getSeededSkillCatalog();

  return {
    ...seeded,
    tenantId: "tenant-scene-demo",
    version: "tenant-scene-demo-2026-04-05",
    syncedAt: "2026-04-05T12:00:00.000Z",
    entries: [
      ...seeded.entries,
      {
        id: "scene:campaign-launch",
        kind: "scene",
        title: "新品发布场景",
        summary: "把链接解析、配图和封面串成一条产品链路。",
        sceneKey: "campaign-launch",
        commandPrefix: "/campaign-launch",
        aliases: ["launch", "campaign"],
        executionKind: "scene",
        renderContract: {
          resultKind: "tool_timeline",
          detailKind: "scene_detail",
          supportsStreaming: true,
          supportsTimeline: true,
        },
      },
    ],
  };
}

export function buildCatalogWithStoryVideoSceneEntry(): SkillCatalog {
  const seeded = getSeededSkillCatalog();

  return {
    ...seeded,
    tenantId: "tenant-story-video-scene-demo",
    version: "tenant-story-video-scene-demo-2026-04-07",
    syncedAt: "2026-04-07T12:00:00.000Z",
  };
}
