/* eslint-disable react-refresh/only-export-components */
import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, vi } from "vitest";
import { EmptyStateComposerPanel } from "./EmptyStateComposerPanel";
import {
  buildSkillSelectionProps,
  type SkillSelectionProps,
} from "../skill-selection/skillSelectionBindings";
import { agentEnUSResource, agentZhCNResource } from "@/i18n/agentResources";
import {
  buildHomeSurfaceCopy,
  type HomeSurfaceCopyKey,
} from "../home/homeSurfaceCopy";
import {
  buildInputbarCoreCopy,
  type InputbarCoreCopyKey,
} from "./Inputbar/components/inputbarCoreCopy";
import type { InputbarWorkflowCopyKey } from "./Inputbar/inputbarWorkflowCopy";
import { changeLimeLocale } from "@/i18n/createI18n";
import type {
  InputbarPluginCapability,
  InputbarPluginSelectionOptions,
  InputbarPluginSkillCapability,
} from "./Inputbar/pluginInputCapability";

vi.mock("./ChatModelSelector", () => ({
  ChatModelSelector: () => <div data-testid="empty-state-model-selector" />,
}));

export const mockEmptyStateCharacterMention = vi.fn<
  (props: {
    onSelectPlugin?: (
      plugin: InputbarPluginCapability,
      skill?: InputbarPluginSkillCapability,
      options?: InputbarPluginSelectionOptions,
    ) => void;
  }) => React.ReactNode
>();

vi.mock("../skill-selection/CharacterMention", () => ({
  CharacterMention: (props: {
    onSelectPlugin?: (
      plugin: InputbarPluginCapability,
      skill?: InputbarPluginSkillCapability,
      options?: InputbarPluginSelectionOptions,
    ) => void;
  }) => {
    mockEmptyStateCharacterMention(props);
    return <div data-testid="empty-state-character-mention" />;
  },
}));

vi.mock("../skill-selection/SkillBadge", () => ({
  SkillBadge: () => <div data-testid="empty-state-skill-badge" />,
}));

vi.mock("../skill-selection/CuratedTaskBadge", () => ({
  CuratedTaskBadge: (props: {
    referenceEntries?: Array<{ id: string; sourceKind?: string }>;
  }) => (
    <div
      data-testid="empty-state-curated-task-badge"
      data-reference-count={String(props.referenceEntries?.length ?? 0)}
      data-first-source-kind={props.referenceEntries?.[0]?.sourceKind ?? ""}
    />
  ),
}));

vi.mock("../skill-selection/SkillSelector", () => ({
  SkillSelector: () => <div data-testid="empty-state-skill-selector" />,
}));

function translateResource(
  resource: Partial<
    Record<
      HomeSurfaceCopyKey | InputbarCoreCopyKey | InputbarWorkflowCopyKey,
      string
    >
  >,
  key: HomeSurfaceCopyKey | InputbarCoreCopyKey | InputbarWorkflowCopyKey,
  values?: Record<string, number | string>,
) {
  return Object.entries(values ?? {}).reduce(
    (text, [name, value]) => text.split(`{{${name}}}`).join(String(value)),
    resource[key] ?? key,
  );
}

export const TEST_COMPOSER_COPY = buildHomeSurfaceCopy((key, values) =>
  translateResource(agentZhCNResource, key, values),
).composer;

export const TEST_EN_COMPOSER_COPY = buildHomeSurfaceCopy((key, values) =>
  translateResource(agentEnUSResource, key, values),
).composer;

export const TEST_INPUTBAR_CORE_COPY = buildInputbarCoreCopy((key, values) =>
  translateResource(agentZhCNResource, key, values),
);

export const TEST_EN_INPUTBAR_CORE_COPY = buildInputbarCoreCopy((key, values) =>
  translateResource(agentEnUSResource, key, values),
);

export const mockSelectedTeam = {
  id: "frontend-triage-team",
  source: "builtin" as const,
  label: "前端联调子代理组",
  description: "分析、实现、验证三段式推进。",
  roles: [
    {
      id: "analysis",
      label: "分析",
      summary: "负责拆解问题。",
    },
  ],
};

export function createGithubSearchServiceSkill() {
  return {
    id: "github-repo-radar",
    title: "GitHub 仓库线索检索",
    summary: "复用 GitHub 登录态检索项目。",
    category: "情报研究",
    outputHint: "仓库列表 + 关键线索",
    source: "cloud_catalog" as const,
    runnerType: "instant" as const,
    defaultExecutorBinding: "browser_assist" as const,
    executionLocation: "client_default" as const,
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "浏览器站点执行",
    runnerTone: "emerald" as const,
    runnerDescription: "直接复用浏览器登录态执行。",
    actionLabel: "启动采集",
    automationStatus: null,
    slotSchema: [
      {
        key: "repository_query",
        label: "检索主题",
        type: "text" as const,
        required: true,
        placeholder: "例如 AI Agent",
      },
    ],
    siteCapabilityBinding: {
      adapterName: "github/search",
      autoRun: true,
      requireAttachedSession: true,
      saveMode: "current_content" as const,
      slotArgMap: {
        repository_query: "query",
      },
      fixedArgs: {
        limit: 10,
      },
    },
  };
}

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
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
  vi.useRealTimers();
  vi.clearAllMocks();
});

export function createSkillSelection(
  overrides: Partial<SkillSelectionProps> = {},
): SkillSelectionProps {
  return buildSkillSelectionProps({
    skills: [],
    onSelectInputCapability: vi.fn(),
    onClearSkill: vi.fn(),
    onNavigateToSettings: vi.fn(),
    onImportSkill: vi.fn(),
    onRefreshSkills: vi.fn(),
    ...overrides,
  });
}

export function renderPanel(
  props?: Partial<React.ComponentProps<typeof EmptyStateComposerPanel>>,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const defaultProps: React.ComponentProps<typeof EmptyStateComposerPanel> = {
    input: "",
    setInput: vi.fn(),
    placeholder: "输入内容",
    onSend: vi.fn(),
    activeTheme: "general",
    providerType: "openai",
    setProviderType: vi.fn(),
    model: "gpt-4.1",
    setModel: vi.fn(),
    onManageProviders: vi.fn(),
    isGeneralTheme: false,
    characters: [],
    skillSelection: createSkillSelection(),
    copy: TEST_COMPOSER_COPY,
    inputbarCopy: TEST_INPUTBAR_CORE_COPY,
    showCreationModeSelector: false,
    creationMode: "guided",
    onCreationModeChange: vi.fn(),
    subagentEnabled: false,
    onSubagentEnabledChange: vi.fn(),
    pendingImages: [],
    onFileSelect: vi.fn(),
    onPaste: vi.fn(),
    onRemoveImage: vi.fn(),
  };

  const panelProps = { ...defaultProps, ...props };
  const ControlledPanel = () => {
    const [value, setValue] = React.useState(panelProps.input);
    return (
      <EmptyStateComposerPanel
        {...panelProps}
        input={value}
        setInput={setValue}
      />
    );
  };

  act(() => {
    root.render(<ControlledPanel />);
  });

  mountedRoots.push({ root, container });
  return container;
}

export function renderStatefulPanel(
  props?: Partial<React.ComponentProps<typeof EmptyStateComposerPanel>>,
  initialSubagentEnabled = false,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const initialInput = props?.input ?? "";
  const {
    input: _ignoredInput,
    setInput: _ignoredSetInput,
    subagentEnabled: _ignoredSubagentEnabled,
    onSubagentEnabledChange: _ignoredOnSubagentEnabledChange,
    ...restProps
  } = props || {};

  const StatefulPanel = () => {
    const [input, setInput] = React.useState(initialInput);
    const [subagentEnabled, setSubagentEnabled] = React.useState(
      initialSubagentEnabled,
    );
    return (
      <EmptyStateComposerPanel
        input={input}
        setInput={setInput}
        placeholder="输入内容"
        onSend={vi.fn()}
        activeTheme="general"
        providerType="openai"
        setProviderType={vi.fn()}
        model="gpt-4.1"
        setModel={vi.fn()}
        onManageProviders={vi.fn()}
        isGeneralTheme
        characters={[]}
        skillSelection={createSkillSelection()}
        copy={TEST_COMPOSER_COPY}
        inputbarCopy={TEST_INPUTBAR_CORE_COPY}
        showCreationModeSelector={false}
        creationMode="guided"
        onCreationModeChange={vi.fn()}
        subagentEnabled={subagentEnabled}
        onSubagentEnabledChange={setSubagentEnabled}
        pendingImages={[]}
        onFileSelect={vi.fn()}
        onPaste={vi.fn()}
        onRemoveImage={vi.fn()}
        {...restProps}
      />
    );
  };

  act(() => {
    root.render(<StatefulPanel />);
  });

  mountedRoots.push({ root, container });
  return container;
}

export function openPlusMenu(container: HTMLDivElement) {
  const toggleButton = container.querySelector(
    '[data-testid="inputbar-plus-trigger"]',
  ) as HTMLButtonElement | null;

  expect(toggleButton).toBeTruthy();

  act(() => {
    toggleButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  return toggleButton;
}

export function openPlusMenuPanel(
  container: HTMLDivElement,
  panel: "knowledge" | "objective" | "plugins" | "skills",
) {
  openPlusMenu(container);

  const row = document.body.querySelector(
    `[data-testid="inputbar-plus-${panel === "knowledge" ? "knowledge" : panel}"]`,
  ) as HTMLButtonElement | null;

  expect(row).toBeTruthy();

  act(() => {
    row?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  return document.body.querySelector(
    `[data-testid="inputbar-plus-panel-${panel}"]`,
  );
}

export const expandAdvancedControls = openPlusMenu;

export function updateTextareaValue(
  textarea: HTMLTextAreaElement | null,
  value: string,
) {
  expect(textarea).toBeTruthy();

  const valueSetter = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype,
    "value",
  )?.set;

  act(() => {
    valueSetter?.call(textarea, value);
    textarea?.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
