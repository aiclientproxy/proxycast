import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Inputbar } from "./index";
import { changeLimeLocale } from "@/i18n/createI18n";
import { toast } from "sonner";
import type { Character } from "@/lib/api/projectMemory";
import type { Skill } from "@/lib/api/skills";
import type { ServiceSkillHomeItem } from "@/components/agent/chat/service-skills/types";
import type {
  BuiltinInputCommand,
  RuntimeSceneSlashCommand,
} from "../../skill-selection/builtinCommands";
import type { InputCapabilitySelection } from "../../skill-selection/inputCapabilitySelection";
import {
  buildCuratedTaskLaunchPrompt,
  findCuratedTaskTemplateById,
  resolveCuratedTaskTemplateLaunchPrefill,
} from "../../utils/curatedTaskTemplates";
import type { InputbarSendPayload } from "./inputbarSendPayload";
import type { BaseComposerSendMetadata } from "@/components/input-kit";
import type {
  InputbarPluginCapability,
  InputbarPluginSelectionOptions,
  InputbarPluginSkillCapability,
} from "./pluginInputCapability";

const mockCharacterMention = vi.fn<
  (props: {
    characters?: Character[];
    skills?: Skill[];
    serviceSkills?: ServiceSkillHomeItem[];
    onSelectInputCapability?: (
      capability: InputCapabilitySelection,
      options?: { replayText?: string },
    ) => void;
    onSelectPlugin?: (
      plugin: InputbarPluginCapability,
      skill?: InputbarPluginSkillCapability,
      options?: InputbarPluginSelectionOptions,
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
    inputCompletionEnabled?: boolean;
  }) => React.ReactNode
>();
type MockInputbarPlusPanelId = "knowledge" | "plugins" | "skills";

interface MockInputbarPlusMenuConfig {
  labels: {
    open: string;
    addFiles: string;
    attachKnowledge: string;
    planMode: string;
    subagent: string;
    objective: string;
    plugins: string;
    skills: string;
  };
  taskEnabled: boolean;
  knowledgeOpenRequestKey?: number;
  subagentEnabled?: boolean;
  pluginsActive?: boolean;
  knowledgePanel?: React.ReactNode;
  pluginsPanel?: React.ReactNode;
  skillsPanel?: React.ReactNode;
  onAddFiles: () => void;
  onToggleTask: () => void;
  onToggleObjective: () => void;
  onToggleSubagent?: () => void;
  onPanelOpen?: (panelId: MockInputbarPlusPanelId) => void;
}

interface MockInputbarCoreProps {
  text?: string;
  setText?: (value: string) => void;
  disabled?: boolean;
  onToolClick?: (tool: string) => void;
  activeTools?: Record<string, boolean>;
  onSend?: (metadata?: BaseComposerSendMetadata) => void;
  pendingImages?: Array<{
    data: string;
    mediaType: string;
  }>;
  pathReferences?: Array<{
    id: string;
    path: string;
    name: string;
    isDir: boolean;
  }>;
  leftExtra?: React.ReactNode;
  trailingMeta?: React.ReactNode;
  topExtra?: React.ReactNode;
  placeholder?: string;
  toolMode?: "default" | "attach-only";
  showMetaTools?: boolean;
  plusMenu?: MockInputbarPlusMenuConfig;
}

const mockInputbarCore = vi.fn((props: MockInputbarCoreProps) => props);

function MockInputbarCoreView(props: MockInputbarCoreProps) {
  mockInputbarCore(props);
  const [activePanel, setActivePanel] =
    React.useState<MockInputbarPlusPanelId | null>(null);

  React.useEffect(() => {
    if (!props.plusMenu?.knowledgeOpenRequestKey) {
      return;
    }
    setActivePanel("knowledge");
  }, [props.plusMenu?.knowledgeOpenRequestKey]);

  const activePlusPanel =
    activePanel === "knowledge"
      ? props.plusMenu?.knowledgePanel
      : activePanel === "plugins"
        ? props.plusMenu?.pluginsPanel
        : activePanel === "skills"
          ? props.plusMenu?.skillsPanel
          : null;

  return (
    <div data-testid="inputbar-core">
      <div data-testid="top-extra">{props.topExtra}</div>
      {props.showMetaTools ? (
        <>
          <button
            type="button"
            data-testid="toggle-subagent-mode"
            onClick={() => props.onToolClick?.("subagent_mode")}
          >
            打开 Subagents
          </button>
          <span data-testid="subagent-state">
            {props.activeTools?.subagent_mode ? "on" : "off"}
          </span>
        </>
      ) : null}
      {props.plusMenu ? (
        <div>
          <button
            type="button"
            aria-label={props.plusMenu.labels.open}
            title={props.plusMenu.labels.open}
            data-testid="inputbar-plus-trigger"
          >
            +
          </button>
          <div data-testid="inputbar-plus-menu">
            <button
              type="button"
              data-testid="inputbar-plus-add-files"
              onClick={props.plusMenu.onAddFiles}
            >
              {props.plusMenu.labels.addFiles}
            </button>
            <button
              type="button"
              data-testid="inputbar-plus-knowledge"
              disabled={!props.plusMenu.knowledgePanel}
              onClick={() => {
                props.plusMenu?.onPanelOpen?.("knowledge");
                setActivePanel("knowledge");
              }}
            >
              {props.plusMenu.labels.attachKnowledge}
            </button>
            <button
              type="button"
              data-testid="inputbar-plus-plan-mode"
              onClick={props.plusMenu.onToggleTask}
            >
              {props.plusMenu.labels.planMode}
              {props.plusMenu.taskEnabled ? " on" : " off"}
            </button>
            {props.plusMenu.onToggleSubagent ? (
              <button
                type="button"
                data-testid="inputbar-plus-subagent-mode"
                onClick={props.plusMenu.onToggleSubagent}
              >
                {props.plusMenu.labels.subagent}
                {props.plusMenu.subagentEnabled ? " on" : " off"}
              </button>
            ) : null}
            <button
              type="button"
              data-testid="inputbar-plus-objective"
              onClick={props.plusMenu.onToggleObjective}
            >
              {props.plusMenu.labels.objective}
            </button>
            <button
              type="button"
              data-testid="inputbar-plus-plugins"
              disabled={!props.plusMenu.pluginsPanel}
              onClick={() => {
                props.plusMenu?.onPanelOpen?.("plugins");
                setActivePanel("plugins");
              }}
            >
              {props.plusMenu.labels.plugins}
              {props.plusMenu.pluginsActive ? " on" : " off"}
            </button>
            <button
              type="button"
              data-testid="inputbar-plus-skills"
              disabled={!props.plusMenu.skillsPanel}
              onClick={() => {
                props.plusMenu?.onPanelOpen?.("skills");
                setActivePanel("skills");
              }}
            >
              {props.plusMenu.labels.skills}
            </button>
            {activePlusPanel ? (
              <div data-testid={`inputbar-plus-panel-${activePanel}`}>
                {activePlusPanel}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      <button
        type="button"
        data-testid="send-btn"
        onClick={() =>
          props.onSend?.({
            triggeredAt: Date.now(),
            triggerSource: "button",
          })
        }
      >
        发送
      </button>
      <div data-testid="left-extra">{props.leftExtra}</div>
      <div data-testid="trailing-meta">{props.trailingMeta}</div>
    </div>
  );
}

vi.mock("./components/InputbarCore", () => ({
  InputbarCore: (props: MockInputbarCoreProps) => (
    <MockInputbarCoreView {...props} />
  ),
}));

vi.mock("../ThreadGoalPanel", () => ({
  ThreadGoalPanel: (props: {
    threadId?: string | null;
    threadGoal?: { objective: string } | null;
    runtimeBusy?: boolean;
  }) => (
    <div
      data-testid="thread-goal-inline-panel"
      data-thread-id={props.threadId ?? ""}
      data-objective={props.threadGoal?.objective ?? ""}
      data-runtime-busy={String(Boolean(props.runtimeBusy))}
      data-variant="inline"
    />
  ),
}));

vi.mock("../../skill-selection/CharacterMention", () => ({
  CharacterMention: (props: {
    characters?: Character[];
    skills?: Skill[];
    serviceSkills?: ServiceSkillHomeItem[];
    onSelectInputCapability?: (
      capability: InputCapabilitySelection,
      options?: { replayText?: string },
    ) => void;
    onSelectPlugin?: (
      plugin: InputbarPluginCapability,
      skill?: InputbarPluginSkillCapability,
      options?: InputbarPluginSelectionOptions,
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
    inputCompletionEnabled?: boolean;
  }) => {
    mockCharacterMention(props);
    return <div data-testid="character-mention-stub" />;
  },
}));

vi.mock("../../skill-selection/SkillBadge", () => ({
  SkillBadge: () => <div data-testid="skill-badge" />,
}));

vi.mock("../../skill-selection/CuratedTaskBadge", () => ({
  CuratedTaskBadge: (props: {
    task?: {
      id?: string;
      title?: string;
      followUpActions?: string[];
    };
    projectId?: string | null;
    sessionId?: string | null;
    referenceEntries?: Array<{
      id: string;
      sourceKind?: string;
    }>;
    onEdit?: () => void;
    onApplyReviewSuggestion?: (task: {
      id: string;
      title: string;
      prompt: string;
      requiredInputFields: Array<{
        key: string;
        label: string;
        placeholder: string;
        type: "text" | "textarea";
      }>;
      outputContract: string[];
      followUpActions: string[];
    }) => void;
    onClear?: () => void;
  }) => (
    <div
      data-testid="curated-task-badge"
      data-task-id={props.task?.id ?? ""}
      data-project-id={props.projectId ?? ""}
      data-session-id={props.sessionId ?? ""}
      data-reference-count={String(props.referenceEntries?.length ?? 0)}
      data-first-source-kind={props.referenceEntries?.[0]?.sourceKind ?? ""}
    >
      <span>{props.task?.title}</span>
      <span>{props.task?.followUpActions?.join("、")}</span>
      <button
        type="button"
        data-testid="curated-task-badge-edit"
        onClick={props.onEdit}
      >
        编辑
      </button>
      <button
        type="button"
        data-testid="curated-task-badge-review-action"
        onClick={() => {
          const suggestedTask = findCuratedTaskTemplateById(
            "account-project-review",
          );
          if (suggestedTask) {
            props.onApplyReviewSuggestion?.(suggestedTask);
          }
        }}
      >
        改用复盘
      </button>
      <button
        type="button"
        data-testid="curated-task-badge-clear"
        onClick={props.onClear}
      >
        清除
      </button>
    </div>
  ),
}));

vi.mock("../CuratedTaskLauncherDialog", () => ({
  CuratedTaskLauncherDialog: (props: {
    open: boolean;
    task: {
      id: string;
      requiredInputFields: Array<{
        key: string;
        label: string;
      }>;
    } | null;
    projectId?: string | null;
    sessionId?: string | null;
    initialInputValues?: Record<string, string> | null;
    initialReferenceMemoryIds?: string[] | null;
    initialReferenceEntries?: Array<{
      id: string;
      title: string;
      summary: string;
      category: string;
      categoryLabel: string;
      tags: string[];
    }> | null;
    prefillHint?: string | null;
    onOpenChange?: (open: boolean) => void;
    onApplyReviewSuggestion?: (
      task: {
        id: string;
        requiredInputFields: Array<{
          key: string;
          label: string;
        }>;
        prompt: string;
        outputContract: string[];
        followUpActions: string[];
      },
      options: {
        inputValues: Record<string, string>;
        referenceSelection: {
          referenceMemoryIds: string[];
          referenceEntries: Array<{
            id: string;
            title: string;
            summary: string;
            category: string;
            categoryLabel: string;
            tags: string[];
          }>;
        };
      },
    ) => void;
    onConfirm?: (
      task: {
        id: string;
        requiredInputFields: Array<{
          key: string;
          label: string;
        }>;
      },
      inputValues: Record<string, string>,
      referenceSelection: {
        referenceMemoryIds: string[];
        referenceEntries: Array<{
          id: string;
          title: string;
          summary: string;
          category: string;
          categoryLabel: string;
          tags: string[];
        }>;
      },
    ) => void;
  }) => {
    const [inputValues, setInputValues] = React.useState<
      Record<string, string>
    >({});
    const [referenceMemoryIds, setReferenceMemoryIds] = React.useState<
      string[]
    >([]);

    React.useEffect(() => {
      if (!props.open || !props.task) {
        setInputValues({});
        setReferenceMemoryIds([]);
        return;
      }

      setInputValues(
        Object.fromEntries(
          props.task.requiredInputFields.map((field) => [
            field.key,
            String(props.initialInputValues?.[field.key] ?? ""),
          ]),
        ),
      );
      setReferenceMemoryIds(props.initialReferenceMemoryIds ?? []);
    }, [
      props.initialInputValues,
      props.initialReferenceMemoryIds,
      props.open,
      props.task,
    ]);

    if (!props.open || !props.task) {
      return null;
    }

    return (
      <div
        data-testid="curated-task-launcher-dialog"
        data-task-id={props.task.id}
        data-project-id={props.projectId ?? ""}
        data-session-id={props.sessionId ?? ""}
        data-reference-memory-count={String(referenceMemoryIds.length)}
      >
        {props.prefillHint ? (
          <div data-testid="curated-task-launcher-prefill-hint">
            {props.prefillHint}
          </div>
        ) : null}
        {props.task.requiredInputFields.map((field) => (
          <label key={field.key}>
            {field.label}
            <input
              data-testid={`curated-task-dialog-field-${field.key}`}
              value={inputValues[field.key] ?? ""}
              onChange={(event) =>
                setInputValues((current) => ({
                  ...current,
                  [field.key]: event.target.value,
                }))
              }
            />
          </label>
        ))}
        <button
          type="button"
          data-testid="curated-task-dialog-reference-toggle"
          onClick={() =>
            setReferenceMemoryIds((current) =>
              current.includes("memory-1") ? [] : ["memory-1"],
            )
          }
        >
          切换引用
        </button>
        <button
          type="button"
          data-testid="curated-task-dialog-review-action"
          onClick={() => {
            const suggestedTask = findCuratedTaskTemplateById(
              "account-project-review",
            );
            if (!suggestedTask) {
              return;
            }

            props.onApplyReviewSuggestion?.(suggestedTask, {
              inputValues,
              referenceSelection: {
                referenceMemoryIds,
                referenceEntries: referenceMemoryIds.includes("memory-1")
                  ? [
                      {
                        id: "memory-1",
                        title: "品牌风格样本",
                        summary: "保留轻盈但专业的表达。",
                        category: "context",
                        categoryLabel: "参考",
                        tags: ["品牌", "语气"],
                      },
                    ]
                  : [],
              },
            });
          }}
        >
          改用复盘
        </button>
        <button
          type="button"
          data-testid="curated-task-dialog-confirm"
          onClick={() =>
            props.onConfirm?.(props.task!, inputValues, {
              referenceMemoryIds,
              referenceEntries: referenceMemoryIds.includes("memory-1")
                ? [
                    {
                      id: "memory-1",
                      title: "品牌风格样本",
                      summary: "保留轻盈但专业的表达。",
                      category: "context",
                      categoryLabel: "参考",
                      tags: ["品牌", "语气"],
                    },
                  ]
                : [],
            })
          }
        >
          确认
        </button>
        <button
          type="button"
          data-testid="curated-task-dialog-close"
          onClick={() => props.onOpenChange?.(false)}
        >
          关闭
        </button>
      </div>
    );
  },
}));

vi.mock("./components/RuntimeSceneBadge", () => ({
  RuntimeSceneBadge: () => <div data-testid="runtime-scene-badge" />,
}));

vi.mock("../ChatModelSelector", () => ({
  ChatModelSelector: ({ disabled }: { disabled?: boolean }) => (
    <button type="button" data-testid="model-selector" disabled={disabled} />
  ),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(async () => []),
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
}));

vi.mock("@/components/input-kit", () => ({
  createAgentInputAdapter: (options: {
    text: string;
    setText: (value: string) => void;
    isSending: boolean;
    disabled?: boolean;
    attachments?: unknown[];
    providerType: string;
    model: string;
    setProviderType: (providerType: string) => void;
    setModel: (model: string) => void;
    stop?: () => void;
  }) => ({
    state: {
      text: options.text,
      isSending: options.isSending,
      disabled: options.disabled,
      attachments: options.attachments,
    },
    model: {
      providerType: options.providerType,
      model: options.model,
    },
    actions: {
      setText: options.setText,
      send: vi.fn(),
      stop: options.stop,
      setProviderType: options.setProviderType,
      setModel: options.setModel,
    },
    ui: {
      showModelSelector: true,
      showToolBar: true,
    },
  }),
}));

vi.mock("sonner", () => ({
  toast: {
    info: vi.fn(),
    success: vi.fn(),
    error: vi.fn(),
  },
}));

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
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

function renderInputbar(
  props?: Partial<React.ComponentProps<typeof Inputbar>>,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const defaultProps: React.ComponentProps<typeof Inputbar> = {
    input: "",
    setInput: vi.fn(),
    onSend: vi.fn(),
    isLoading: false,
    characters: [],
    skills: [],
  };

  const render = (
    nextProps?: Partial<React.ComponentProps<typeof Inputbar>>,
  ) => {
    act(() => {
      root.render(<Inputbar {...defaultProps} {...props} {...nextProps} />);
    });
  };

  render();

  mountedRoots.push({ root, container });
  return {
    container,
    rerender: render,
  };
}

function expectInputbarSend(
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
  if (payload.sendOptions) {
    expect(actual?.sendOptions).toMatchObject(payload.sendOptions);
    return;
  }
  expect(actual?.sendOptions).toBe(payload.sendOptions);
}

function expandAdvancedControls(container: HTMLDivElement) {
  const toggleButton = container.querySelector(
    '[data-testid="inputbar-plus-trigger"]',
  ) as HTMLButtonElement | null;

  expect(toggleButton).toBeTruthy();

  act(() => {
    toggleButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  return toggleButton;
}

function openPlusMenuPanel(
  container: HTMLDivElement,
  panel: "knowledge" | "plugins" | "skills",
) {
  expandAdvancedControls(container);
  const trigger = document.body.querySelector(
    `[data-testid="inputbar-plus-${panel}"]`,
  ) as HTMLButtonElement | null;
  expect(trigger).toBeTruthy();

  act(() => {
    trigger?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  const panelElement = document.body.querySelector(
    `[data-testid="inputbar-plus-panel-${panel}"]`,
  );
  expect(panelElement).toBeTruthy();
  return panelElement as HTMLElement;
}

function openKnowledgePanel(container: HTMLDivElement) {
  return openPlusMenuPanel(container, "knowledge");
}

function openPluginsPanel(container: HTMLDivElement) {
  return openPlusMenuPanel(container, "plugins");
}

function openSkillsPanel(container: HTMLDivElement) {
  return openPlusMenuPanel(container, "skills");
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

function dispatchInputbarDrop(
  container: HTMLDivElement,
  dataTransfer: {
    files: File[];
    getData: (format: string) => string;
  },
): Event {
  const surface = container.firstElementChild;
  expect(surface).toBeTruthy();

  const event = new Event("drop", {
    bubbles: true,
    cancelable: true,
  });
  Object.defineProperty(event, "dataTransfer", {
    value: dataTransfer,
  });

  act(() => {
    surface?.dispatchEvent(event);
  });
  return event;
}

describe("Inputbar", () => {
  it("收到中断恢复请求时即使仍在 loading 也应回填文本、图片、路径和技能 chip", async () => {
    const onInputRestoreRequestHandled = vi.fn();
    const pathReference = {
      id: "file:/tmp/report.md",
      path: "/tmp/report.md",
      name: "report.md",
      isDir: false,
      source: "file_manager" as const,
    };
    const skill: Skill = {
      key: "local:draft",
      name: "起草",
      description: "恢复输入用技能",
      directory: "draft",
      installed: true,
      sourceKind: "other",
    };
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
        React.ComponentProps<typeof Inputbar>["inputRestoreRequest"]
      >({
        requestId: "restore-1",
        reason: "output_free_interrupted_turn",
        draft: {
          text: "继续生成提纲",
          images: [
            {
              data: "image-data",
              mediaType: "image/png",
            },
          ],
          pathReferences: [pathReference],
          inputCapabilityRoute: {
            kind: "installed_skill",
            skillKey: "draft",
            skillName: "起草",
          },
        },
      });

      return (
        <Inputbar
          input={input}
          setInput={setInput}
          onSend={vi.fn()}
          isLoading={true}
          characters={[]}
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
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 0));
      await Promise.resolve();
    });

    const latestCoreProps =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCoreProps?.text).toBe("继续生成提纲");
    expect(latestCoreProps?.pathReferences).toEqual([pathReference]);
    expect(latestCoreProps?.pendingImages).toEqual([
      {
        data: "image-data",
        mediaType: "image/png",
      },
    ]);
    expect(container.querySelector('[data-testid="skill-badge"]')).toBeTruthy();
    expect(onInputRestoreRequestHandled).toHaveBeenCalledWith("restore-1");
  });

  it("发送 Promise 未结束时收到恢复请求，不应让旧发送清理刚恢复的富输入", async () => {
    const onInputRestoreRequestHandled = vi.fn();
    const pathReference = {
      id: "file:/tmp/report.md",
      path: "/tmp/report.md",
      name: "report.md",
      isDir: false,
      source: "file_manager" as const,
    };
    const skill: Skill = {
      key: "local:draft",
      name: "起草",
      description: "恢复输入用技能",
      directory: "draft",
      installed: true,
      sourceKind: "other",
    };
    let resolveSend: ((value: boolean) => void) | null = null;
    let triggerRestore: (() => void) | null = null;
    const onSend = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          resolveSend = resolve;
        }),
    );
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedRoots.push({ root, container });

    function RestoreDuringSendHarness() {
      const [input, setInput] = React.useState("请结合现有资料生成提纲");
      const [pathReferences, setPathReferences] = React.useState<
        (typeof pathReference)[]
      >([]);
      const [restoreRequest, setRestoreRequest] =
        React.useState<
          React.ComponentProps<typeof Inputbar>["inputRestoreRequest"]
        >(null);

      triggerRestore = () =>
        setRestoreRequest({
          requestId: "restore-during-send",
          reason: "output_free_interrupted_turn",
          draft: {
            text: "继续生成提纲",
            images: [
              {
                data: "image-data",
                mediaType: "image/png",
              },
            ],
            pathReferences: [pathReference],
            inputCapabilityRoute: {
              kind: "installed_skill",
              skillKey: "draft",
              skillName: "起草",
            },
          },
        });

      return (
        <Inputbar
          input={input}
          setInput={setInput}
          onSend={onSend}
          isLoading={false}
          characters={[]}
          skills={[skill]}
          pathReferences={pathReferences}
          onClearPathReferences={() => setPathReferences([])}
          onAddPathReferences={(references) =>
            setPathReferences(references as (typeof pathReference)[])
          }
          initialInputCapability={{
            requestKey: 1,
            capabilityRoute: {
              kind: "installed_skill",
              skillKey: "draft",
              skillName: "起草",
            },
          }}
          inputRestoreRequest={restoreRequest}
          onInputRestoreRequestHandled={(requestId) => {
            onInputRestoreRequestHandled(requestId);
            setRestoreRequest(null);
          }}
        />
      );
    }

    await act(async () => {
      root.render(<RestoreDuringSendHarness />);
      await Promise.resolve();
      await Promise.resolve();
    });

    const firstCoreProps =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    let sendPromise: Promise<unknown> | undefined;
    await act(async () => {
      sendPromise = Promise.resolve(firstCoreProps?.onSend?.());
      await Promise.resolve();
    });
    expect(onSend).toHaveBeenCalledTimes(1);

    await act(async () => {
      triggerRestore?.();
      await Promise.resolve();
    });
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 0));
      await Promise.resolve();
    });

    let latestCoreProps =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCoreProps?.text).toBe("继续生成提纲");
    expect(latestCoreProps?.pathReferences).toEqual([pathReference]);
    expect(latestCoreProps?.pendingImages).toEqual([
      {
        data: "image-data",
        mediaType: "image/png",
      },
    ]);
    expect(container.querySelector('[data-testid="skill-badge"]')).toBeTruthy();

    await act(async () => {
      resolveSend?.(true);
      await sendPromise;
      await Promise.resolve();
    });

    latestCoreProps =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCoreProps?.text).toBe("继续生成提纲");
    expect(latestCoreProps?.pathReferences).toEqual([pathReference]);
    expect(latestCoreProps?.pendingImages).toEqual([
      {
        data: "image-data",
        mediaType: "image/png",
      },
    ]);
    expect(container.querySelector('[data-testid="skill-badge"]')).toBeTruthy();
    expect(onInputRestoreRequestHandled).toHaveBeenCalledWith(
      "restore-during-send",
    );
  });

  it("即使角色和技能为空，也应挂载 CharacterMention", async () => {
    const { container } = renderInputbar();
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
    expect(latestCall.characters).toEqual([]);
    expect(latestCall.skills).toEqual([]);
  });

  it("应把输入自动补全开关透传给现有 CharacterMention 主链", async () => {
    renderInputbar({
      inputCompletionEnabled: false,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];

    expect(latestCall.inputCompletionEnabled).toBe(false);
  });

  it("应把当前带入的灵感引用默认值透传给 CharacterMention", async () => {
    renderInputbar({
      defaultCuratedTaskReferenceMemoryIds: ["memory-1"],
      defaultCuratedTaskReferenceEntries: [
        {
          id: "memory-1",
          title: "品牌风格样本",
          summary: "保留轻盈但专业的表达。",
          category: "context",
          categoryLabel: "参考",
          tags: ["品牌", "语气"],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];

    expect(latestCall.defaultCuratedTaskReferenceMemoryIds).toEqual([
      "memory-1",
    ]);
    expect(latestCall.defaultCuratedTaskReferenceEntries).toEqual([
      expect.objectContaining({
        id: "memory-1",
        title: "品牌风格样本",
      }),
    ]);
  });

  it("应将服务型技能目录透传给 CharacterMention，并经统一 capability 触发启动", async () => {
    const serviceSkills = [
      {
        id: "daily-trend-briefing",
        title: "每日趋势摘要",
      },
    ] as ServiceSkillHomeItem[];
    const onSelectServiceSkill = vi.fn();

    renderInputbar({
      serviceSkills,
      onSelectServiceSkill,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(mockCharacterMention.mock.calls.length).toBeGreaterThan(0);
    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ][0];

    expect(latestCall.serviceSkills).toBe(serviceSkills);
    expect(latestCall.onSelectInputCapability).toBeTypeOf("function");

    await act(async () => {
      latestCall.onSelectInputCapability?.({
        kind: "service_skill",
        skill: serviceSkills[0] as ServiceSkillHomeItem,
      });
      await Promise.resolve();
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(serviceSkills[0]);
  });

  it("@资料兼容入口应打开资料中枢而不是直接启用资料或创建新的 Agent", async () => {
    const onToggleKnowledgePack = vi.fn();
    const onKnowledgePacksNeeded = vi.fn();
    renderInputbar({
      knowledgePackSelection: {
        enabled: false,
        packName: "brand-guide",
        workingDir: "/workspace/demo",
        label: "品牌资料",
        status: "ready",
      },
      onKnowledgePacksNeeded,
      onToggleKnowledgePack,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];
    expect(latestCall?.onSelectInputCapability).toBeTypeOf("function");

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "builtin_command",
        command: {
          key: "knowledge_pack",
          label: "资料",
          mentionLabel: "资料",
          commandPrefix: "@资料",
          description: "使用项目资料",
          aliases: [],
        },
      });
      await Promise.resolve();
    });

    expect(onToggleKnowledgePack).not.toHaveBeenCalled();
    expect(onKnowledgePacksNeeded).toHaveBeenCalledTimes(1);
    expect(
      document.body.querySelector('[data-testid="inputbar-knowledge-hub"]')
        ?.textContent,
    ).toContain("使用这份资料");

    const useKnowledgeButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("使用这份资料"));

    act(() => {
      useKnowledgeButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(onToggleKnowledgePack).toHaveBeenCalledWith(true);
  });

  it("@沉淀资料应复用输入框资料整理动作", async () => {
    const onStartKnowledgeOrganize = vi.fn();
    renderInputbar({
      onStartKnowledgeOrganize,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "builtin_command",
        command: {
          key: "knowledge_settle",
          label: "沉淀资料",
          mentionLabel: "沉淀资料",
          commandPrefix: "@沉淀资料",
          description: "沉淀项目资料",
          aliases: [],
        },
      });
      await Promise.resolve();
    });

    expect(onStartKnowledgeOrganize).toHaveBeenCalledTimes(1);
  });

  it("初始路由带 @资料 时应兼容打开资料中枢而不是渲染命令 badge", async () => {
    const onToggleKnowledgePack = vi.fn();
    const onKnowledgePacksNeeded = vi.fn();
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: false,
        packName: "brand-guide",
        workingDir: "/workspace/demo",
        label: "品牌资料",
        status: "ready",
      },
      onKnowledgePacksNeeded,
      onToggleKnowledgePack,
      initialInputCapability: {
        capabilityRoute: {
          kind: "builtin_command",
          commandKey: "knowledge_pack",
          commandPrefix: "@资料",
        },
        requestKey: 20260505,
      },
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="top-extra"]')?.textContent,
    ).not.toContain("资料");
    expect(
      document.body.querySelector('[data-testid="inputbar-knowledge-hub"]')
        ?.textContent,
    ).toContain("使用这份资料");
    expect(onToggleKnowledgePack).not.toHaveBeenCalled();
    expect(onKnowledgePacksNeeded).toHaveBeenCalledTimes(1);
  });

  it("先选内建命令再选技能时，应以后一次 capability 为准发送", async () => {
    const onSend = vi.fn();
    const skill = {
      key: "writer",
      name: "写作助手",
      description: "用于写作",
      directory: "writer",
      installed: true,
      sourceKind: "builtin",
    } as Skill;
    renderInputbar({
      input: "整理最近发布计划",
      onSend,
      activeTheme: "general",
      skills: [skill],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const builtinCommand: BuiltinInputCommand = {
      key: "research",
      label: "搜索",
      mentionLabel: "搜索",
      commandPrefix: "@搜索",
      description: "搜索资料",
      aliases: [],
    };
    const firstCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];
    expect(firstCall?.onSelectInputCapability).toBeTypeOf("function");

    await act(async () => {
      firstCall?.onSelectInputCapability?.({
        kind: "builtin_command",
        command: builtinCommand,
      });
      await Promise.resolve();
    });

    const secondCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      secondCall?.onSelectInputCapability?.({
        kind: "installed_skill",
        skill,
      });
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectInputbarSend(onSend, {
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

  it("加号菜单开启 plan 和 goal 后发送应下传 typed mode", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input: "继续执行当前任务",
      onSend,
      activeTheme: "general",
      projectId: "project-inputbar-plan-goal",
      sessionId: "thread-inputbar-plan-goal",
    });

    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      document
        .querySelector('[data-testid="inputbar-plus-plan-mode"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    await act(async () => {
      document
        .querySelector('[data-testid="inputbar-plus-objective"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    await act(async () => {
      document
        .querySelector('[data-testid="send-btn"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "继续执行当前任务",
      sendOptions: {
        collaborationMode: "plan",
        displayContent: "继续执行当前任务",
        threadGoal: { objective: "继续执行当前任务" },
        toolPreferencesOverride: {
          task: true,
          subagent: false,
        },
      },
    });
  });

  it("canonical Goal 应按 thread identity 常显，不由发送模式开关拥有", async () => {
    const { container } = renderInputbar({
      activeTheme: "general",
      projectId: "project-goal",
      sessionId: "thread-inputbar-goal-panel",
      threadId: "thread-canonical-goal-panel",
      threadGoal: {
        createdAt: 10,
        objective: "完成 canonical Goal GUI",
        status: "active",
        threadId: "thread-canonical-goal-panel",
        timeUsedSeconds: 20,
        tokenBudget: null,
        tokensUsed: 100,
        updatedAt: 20,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const objectivePanel = container.querySelector(
      '[data-testid="thread-goal-inline-panel"]',
    );
    expect(objectivePanel).toBeTruthy();
    expect(objectivePanel?.getAttribute("data-thread-id")).toBe(
      "thread-canonical-goal-panel",
    );
    expect(objectivePanel?.getAttribute("data-objective")).toBe(
      "完成 canonical Goal GUI",
    );

    await act(async () => {
      container
        .querySelector('[data-testid="inputbar-plus-objective"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    await act(async () => {
      container
        .querySelector('[data-testid="inputbar-objective-status"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="thread-goal-inline-panel"]'),
    ).toBeTruthy();
  });

  it("Plan 和 Goal 开启后应在左侧显示可单独关闭的状态标签，模型放在右侧", async () => {
    const { container } = renderInputbar({
      input: "继续执行当前任务",
      activeTheme: "general",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
      accessMode: "full-access",
      setAccessMode: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      container
        .querySelector('[data-testid="inputbar-plus-plan-mode"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => {
      container
        .querySelector('[data-testid="inputbar-plus-objective"]')
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    const leftExtra = container.querySelector('[data-testid="left-extra"]');
    const trailingMeta = container.querySelector(
      '[data-testid="trailing-meta"]',
    );
    const planChip = leftExtra?.querySelector(
      '[data-testid="inputbar-task-mode-status"]',
    ) as HTMLButtonElement | null;
    const objectiveChip = leftExtra?.querySelector(
      '[data-testid="inputbar-objective-status"]',
    ) as HTMLButtonElement | null;
    const accessSelect = leftExtra?.querySelector(
      '[data-testid="inputbar-access-mode-select"]',
    ) as HTMLSelectElement | null;

    expect(accessSelect).toBeTruthy();
    expect(planChip?.textContent).toContain("计划");
    expect(objectiveChip?.textContent).toContain("追求目标");
    expect(
      getComputedStyle(
        planChip?.querySelector(
          '[data-testid="inputbar-task-mode-status-remove-mark"]',
        ) as HTMLElement,
      ).opacity,
    ).toBe("0");
    expect(
      getComputedStyle(
        objectiveChip?.querySelector(
          '[data-testid="inputbar-objective-status-remove-mark"]',
        ) as HTMLElement,
      ).opacity,
    ).toBe("0");
    expect(
      accessSelect!.compareDocumentPosition(planChip!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      accessSelect!.compareDocumentPosition(objectiveChip!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      leftExtra?.querySelector('[data-testid="model-selector"]'),
    ).toBeNull();
    expect(
      trailingMeta?.querySelector('[data-testid="model-selector"]'),
    ).toBeTruthy();

    await act(async () => {
      planChip?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-task-mode-status"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-objective-status"]'),
    ).toBeTruthy();

    const nextObjectiveChip = container.querySelector(
      '[data-testid="inputbar-objective-status"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      nextObjectiveChip?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-objective-status"]'),
    ).toBeNull();
  });

  it("工作台输入区也应把模型切换器放在右侧", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      activeTheme: "general",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-5.2",
      setModel: vi.fn(),
      accessMode: "full-access",
      setAccessMode: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    const leftExtra = container.querySelector('[data-testid="left-extra"]');
    const trailingMeta = container.querySelector(
      '[data-testid="trailing-meta"]',
    );

    expect(
      leftExtra?.querySelector('[data-testid="inputbar-access-mode-select"]'),
    ).toBeTruthy();
    expect(
      leftExtra?.querySelector('[data-testid="model-selector"]'),
    ).toBeNull();
    expect(
      trailingMeta?.querySelector('[data-testid="model-selector"]'),
    ).toBeTruthy();
  });

  it("parent-owned 禁用态应锁定输入、工具与模型设置", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      disabled: true,
      disabledPlaceholder: "此子线程由父线程管理，无法直接输入",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-5.4",
      setModel: vi.fn(),
      accessMode: "default",
      setAccessMode: vi.fn(),
      toolStates: { plan: true },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const coreProps = mockInputbarCore.mock.calls.at(-1)?.[0];
    const accessSelect = container.querySelector(
      '[data-testid="inputbar-access-mode-select"]',
    ) as HTMLSelectElement;
    const planChip = container.querySelector(
      '[data-testid="inputbar-task-mode-status"]',
    ) as HTMLButtonElement;
    const modelTrigger = container.querySelector(
      '[data-testid="trailing-meta"] [data-testid="model-selector"]',
    ) as HTMLButtonElement;

    expect(coreProps?.disabled).toBe(true);
    expect(coreProps?.placeholder).toBe("此子线程由父线程管理，无法直接输入");
    expect(accessSelect.disabled).toBe(true);
    expect(planChip.disabled).toBe(true);
    expect(modelTrigger.disabled).toBe(true);
  });

  it("选择本地安装技能发送时，应把 local:* key 归一到目录 slash key", async () => {
    const onSend = vi.fn();
    const skill = {
      key: "local:writer",
      name: "写作助手",
      description: "用于写作",
      directory: "writer",
      installed: true,
      sourceKind: "other",
    } as Skill;
    renderInputbar({
      input: "整理最近发布计划",
      onSend,
      activeTheme: "general",
      skills: [skill],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "installed_skill",
        skill,
      });
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
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

  it("仅添加路径引用时应允许发送并写入 request metadata", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input: "",
      onSend,
      pathReferences: [
        {
          id: "dir:/Users/demo/Downloads",
          path: "/Users/demo/Downloads",
          name: "Downloads",
          isDir: true,
          source: "file_manager",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "请查看这些文件或文件夹。",
      sendOptions: {
        requestMetadata: {
          path_references: [
            {
              path: "/Users/demo/Downloads",
              name: "Downloads",
              is_dir: true,
              isDir: true,
              size: null,
              mime_type: null,
              mimeType: null,
              source: "file_manager",
            },
          ],
          harness: {
            file_references: [
              {
                path: "/Users/demo/Downloads",
                name: "Downloads",
                is_dir: true,
                isDir: true,
                size: null,
                mime_type: null,
                mimeType: null,
                source: "file_manager",
              },
            ],
            fileReferences: [
              {
                path: "/Users/demo/Downloads",
                name: "Downloads",
                is_dir: true,
                isDir: true,
                size: null,
                mime_type: null,
                mimeType: null,
                source: "file_manager",
              },
            ],
          },
        },
      },
    });
  });

  it("无法读取系统文件路径时应按 en-US 资源提示", async () => {
    await changeLimeLocale("en-US");
    const onAddPathReferences = vi.fn();
    const { container } = renderInputbar({
      onAddPathReferences,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const dropEvent = dispatchInputbarDrop(container, {
      files: [
        new File(["notes"], "notes.txt", {
          type: "text/plain",
        }),
      ],
      getData: () => "",
    });

    expect(toast.error).toHaveBeenCalledWith(
      "Cannot read system file paths. Drag files in from the built-in file manager.",
    );
    expect(onAddPathReferences).not.toHaveBeenCalled();
    expect(dropEvent.defaultPrevented).toBe(true);
  });

  it("应在输入框底栏提供文件管理器打开按钮", async () => {
    const onToggleFileManager = vi.fn();
    const { container } = renderInputbar({
      onToggleFileManager,
      fileManagerOpen: false,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const toggleButton = container.querySelector(
      '[data-testid="inputbar-file-manager-toggle"]',
    ) as HTMLButtonElement | null;
    expect(toggleButton).toBeTruthy();
    expect(toggleButton?.getAttribute("aria-label")).toBe("打开左侧文件管理器");

    await act(async () => {
      toggleButton?.click();
      await Promise.resolve();
    });

    expect(onToggleFileManager).toHaveBeenCalledTimes(1);
  });

  it("通用输入区 chrome 文案应跟随 en-US 资源", async () => {
    await changeLimeLocale("en-US");
    const { container } = renderInputbar({
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
      onToggleFileManager: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="model-selector"]'),
    ).toBeTruthy();
    expect(container.textContent).not.toContain("Advanced settings");

    const plusTrigger = container.querySelector(
      '[data-testid="inputbar-plus-trigger"]',
    ) as HTMLButtonElement | null;
    expect(plusTrigger?.getAttribute("aria-label")).toBe("Open input settings");
    expect(plusTrigger?.getAttribute("title")).toBe("Open input settings");

    const fileManagerToggle = container.querySelector(
      '[data-testid="inputbar-file-manager-toggle"]',
    ) as HTMLButtonElement | null;
    expect(fileManagerToggle?.getAttribute("aria-label")).toBe(
      "Open file manager sidebar",
    );

    expandAdvancedControls(container);
    expect(document.body.textContent).toContain("Plan mode");
  });

  it("带着初始已安装技能进入时，应恢复 capability badge 并继续按 route 发送", async () => {
    const onSend = vi.fn();
    const skill = {
      key: "writer",
      name: "写作助手",
      description: "用于写作",
      directory: "writer",
      installed: true,
      sourceKind: "builtin",
    } as Skill;
    renderInputbar({
      input: "整理最近发布计划",
      onSend,
      activeTheme: "general",
      skills: [skill],
      initialInputCapability: {
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "local:writer",
          skillName: "写作助手",
        },
        requestKey: 20260418,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(document.querySelector('[data-testid="skill-badge"]')).toBeTruthy();

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectInputbarSend(onSend, {
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

  it("选择结果模板发送时，应透传 curated_task capability route", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input:
        "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
      onSend,
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "curated_task",
        task: {
          id: "social-post-starter",
          title: "内容主稿生成",
          summary: "生成一版可继续打磨的内容首稿。",
          outputHint: "内容首稿 + 结构提纲",
          resultDestination:
            "首版主稿会先进入当前内容，方便继续改写、拆成多平台版本。",
          categoryLabel: "内容起稿",
          prompt:
            "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
          requiredInputs: ["主题或产品信息", "目标受众"],
          requiredInputFields: [
            {
              key: "subject_or_product",
              label: "主题或产品信息",
              placeholder: "输入主题、产品、活动或你已经掌握的关键信息",
              type: "textarea",
            },
            {
              key: "target_audience",
              label: "目标受众",
              placeholder:
                "例如 25-35 岁新消费品牌运营，或 正在找 AI 剪辑工具的创作者",
              type: "text",
            },
          ],
          optionalReferences: ["品牌语气", "参考案例或灵感图片"],
          outputContract: ["内容首稿", "结构提纲", "可继续扩写角度"],
          followUpActions: ["改成多平台版本", "转成口播/字幕稿"],
          badge: "结果模板",
          actionLabel: "进入生成",
          statusLabel: "可直接开始",
          statusTone: "emerald",
          recentUsedAt: null,
          isRecent: false,
        },
      });
      await Promise.resolve();
    });

    expect(
      document.querySelector('[data-testid="curated-task-badge"]'),
    ).toBeTruthy();

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride:
        "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
      sendOptions: expect.objectContaining({
        capabilityRoute: {
          kind: "curated_task",
          taskId: "social-post-starter",
          taskTitle: "内容主稿生成",
          prompt:
            "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
        },
        displayContent:
          "请先帮我起草一版内容首稿：明确目标受众、标题方向、正文结构、核心观点和可继续扩写的角度，并给我一版适合继续打磨的正文。",
        requestMetadata: {
          harness: {
            curated_task: expect.objectContaining({
              task_id: "social-post-starter",
              task_title: "内容主稿生成",
            }),
          },
        },
      }),
    });
  });

  it("Generate 内发送 curated_task 后，应把最新启动事实写回 recent usage", async () => {
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();

    const launchInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const referenceEntries = [
      {
        id: "memory-1",
        sourceKind: "memory" as const,
        title: "品牌风格样本",
        summary: "轻盈但专业的品牌语气参考。",
        category: "context" as const,
        categoryLabel: "参考",
        tags: ["品牌", "语气"],
      },
    ];
    const prompt = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: launchInputValues,
      referenceEntries,
    });
    const onSend = vi.fn().mockResolvedValue(true);

    renderInputbar({
      input: prompt,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt,
          launchInputValues,
          referenceMemoryIds: ["memory-1"],
          referenceEntries,
        },
        requestKey: 2026042101,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(resolveCuratedTaskTemplateLaunchPrefill(template!)).toEqual(
      expect.objectContaining({
        inputValues: launchInputValues,
        referenceMemoryIds: ["memory-1"],
        hint: expect.stringContaining("已根据你上次启动"),
        referenceEntries: [
          expect.objectContaining({
            id: "memory-1",
            sourceKind: "memory",
            title: "品牌风格样本",
            summary: "轻盈但专业的品牌语气参考。",
          }),
        ],
      }),
    );
  });

  it("发送被上层拦截时，不应写入 curated task recent usage", async () => {
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();

    const prompt = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: {
        theme_target: "AI 内容创作",
      },
    });
    const onSend = vi.fn().mockResolvedValue(false);

    renderInputbar({
      input: prompt,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt,
          launchInputValues: {
            theme_target: "AI 内容创作",
          },
        },
        requestKey: 2026042102,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(resolveCuratedTaskTemplateLaunchPrefill(template!)).toBeNull();
  });

  it("带着缺少启动槽位的初始结果模板进入时，应先打开 launcher 而不是直接预填 prompt", async () => {
    const setInput = vi.fn();
    renderInputbar({
      input: "",
      setInput,
      onSend: vi.fn(),
      activeTheme: "general",
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
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setInput).not.toHaveBeenCalled();
    expect(
      document.querySelector(
        '[data-testid="curated-task-dialog-field-theme_target"]',
      ),
    ).toBeTruthy();
    expect(
      document.querySelector(
        '[data-testid="curated-task-dialog-field-platform_region"]',
      ),
    ).toBeTruthy();

    const confirmButton = document.querySelector(
      '[data-testid="curated-task-dialog-confirm"]',
    );
    expect(confirmButton).toBeTruthy();

    const curatedTaskBadge = document.querySelector(
      '[data-testid="curated-task-badge"]',
    ) as HTMLElement | null;
    expect(curatedTaskBadge).toBeTruthy();
    expect(curatedTaskBadge?.textContent).toContain("每日趋势摘要");
    expect(curatedTaskBadge?.textContent).toContain(
      "继续展开其中一个选题、生成首条内容主稿",
    );
  });

  it("带着已确认启动信息的初始结果模板进入时，应恢复 capability badge 并在输入为空时预填 prompt", async () => {
    const onSend = vi.fn();
    const setInput = vi.fn();
    const initialLaunchInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const initialPrompt = buildCuratedTaskLaunchPrompt({
      task: {
        prompt:
          "请先给我做一版每日趋势摘要：围绕当前主题梳理最近值得关注的趋势、热点内容方向、代表案例、用户正在关心的问题，以及最值得立即开工的 3 个选题。",
        requiredInputFields: [
          {
            key: "theme_target",
            label: "主题或赛道",
            placeholder: "",
            type: "text",
          },
          {
            key: "platform_region",
            label: "希望关注的平台/地域",
            placeholder: "",
            type: "text",
          },
        ],
        outputContract: ["趋势摘要", "3 个优先选题", "代表案例线索"],
      },
      inputValues: initialLaunchInputValues,
    });
    const rendered = renderInputbar({
      input: "",
      setInput,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 20260418,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith(initialPrompt);
    expect(
      document.querySelector(
        '[data-testid="curated-task-dialog-field-theme_target"]',
      ),
    ).toBeFalsy();

    rendered.rerender({
      input: initialPrompt,
      setInput,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 20260418,
      },
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: initialPrompt,
      sendOptions: expect.objectContaining({
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        displayContent: initialPrompt,
        requestMetadata: {
          harness: {
            curated_task: expect.objectContaining({
              task_id: "daily-trend-briefing",
              task_title: "每日趋势摘要",
              launch_input_values: initialLaunchInputValues,
            }),
          },
        },
      }),
    });
  });

  it("输入条已激活结果模板时，应把 projectId/sessionId 透传给 badge 和编辑态 launcher", async () => {
    const initialLaunchInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const initialPrompt = buildCuratedTaskLaunchPrompt({
      task: {
        prompt:
          "请先给我做一版每日趋势摘要：围绕当前主题梳理最近值得关注的趋势、热点内容方向、代表案例、用户正在关心的问题，以及最值得立即开工的 3 个选题。",
        requiredInputFields: [
          {
            key: "theme_target",
            label: "主题或赛道",
            placeholder: "",
            type: "text",
          },
          {
            key: "platform_region",
            label: "希望关注的平台/地域",
            placeholder: "",
            type: "text",
          },
        ],
        outputContract: ["趋势摘要", "3 个优先选题", "代表案例线索"],
      },
      inputValues: initialLaunchInputValues,
    });

    renderInputbar({
      input: initialPrompt,
      activeTheme: "general",
      projectId: "project-review-chain",
      sessionId: "session-review-chain",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 2026042201,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const curatedTaskBadge = document.querySelector(
      '[data-testid="curated-task-badge"]',
    ) as HTMLDivElement | null;
    expect(curatedTaskBadge?.dataset.projectId).toBe("project-review-chain");
    expect(curatedTaskBadge?.dataset.sessionId).toBe("session-review-chain");

    const editButton = document.querySelector(
      '[data-testid="curated-task-badge-edit"]',
    ) as HTMLButtonElement | null;
    expect(editButton).toBeTruthy();

    await act(async () => {
      editButton?.click();
      await Promise.resolve();
    });

    const launcherDialog = document.querySelector(
      '[data-testid="curated-task-launcher-dialog"]',
    ) as HTMLDivElement | null;
    expect(launcherDialog?.dataset.projectId).toBe("project-review-chain");
    expect(launcherDialog?.dataset.sessionId).toBe("session-review-chain");
  });

  it("输入条已激活结果模板时，应支持从 badge 直接切到最近判断推荐模板", async () => {
    const setInput = vi.fn();
    const onSend = vi.fn();
    const initialLaunchInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const initialPrompt = buildCuratedTaskLaunchPrompt({
      task: {
        prompt:
          "请先给我做一版每日趋势摘要：围绕当前主题梳理最近值得关注的趋势、热点内容方向、代表案例、用户正在关心的问题，以及最值得立即开工的 3 个选题。",
        requiredInputFields: [
          {
            key: "theme_target",
            label: "主题或赛道",
            placeholder: "",
            type: "text",
          },
          {
            key: "platform_region",
            label: "希望关注的平台/地域",
            placeholder: "",
            type: "text",
          },
        ],
        outputContract: ["趋势摘要", "3 个优先选题", "代表案例线索"],
      },
      inputValues: initialLaunchInputValues,
    });
    const suggestedTask = findCuratedTaskTemplateById("account-project-review");
    expect(suggestedTask).toBeTruthy();
    const expectedPrompt = buildCuratedTaskLaunchPrompt({
      task: suggestedTask!,
      inputValues: initialLaunchInputValues,
    });
    const rendered = renderInputbar({
      input: initialPrompt,
      setInput,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 2026042401,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const reviewAction = document.querySelector(
      '[data-testid="curated-task-badge-review-action"]',
    ) as HTMLButtonElement | null;
    expect(reviewAction).toBeTruthy();

    await act(async () => {
      reviewAction?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenLastCalledWith(expectedPrompt);

    const curatedTaskBadge = document.querySelector(
      '[data-testid="curated-task-badge"]',
    ) as HTMLDivElement | null;
    expect(curatedTaskBadge?.dataset.taskId).toBe("account-project-review");

    rendered.rerender({
      input: expectedPrompt,
      setInput,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 2026042401,
      },
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: expectedPrompt,
      sendOptions: expect.objectContaining({
        capabilityRoute: expect.objectContaining({
          kind: "curated_task",
          taskId: "account-project-review",
          prompt: expectedPrompt,
          launchInputValues: initialLaunchInputValues,
        }),
        displayContent: expectedPrompt,
      }),
    });
  });

  it("输入条编辑态 launcher 按最近判断切模板时，应保留参考选择并显示 en-US 提示", async () => {
    await changeLimeLocale("en-US");
    const initialLaunchInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const initialPrompt = buildCuratedTaskLaunchPrompt({
      task: {
        prompt:
          "请先给我做一版每日趋势摘要：围绕当前主题梳理最近值得关注的趋势、热点内容方向、代表案例、用户正在关心的问题，以及最值得立即开工的 3 个选题。",
        requiredInputFields: [
          {
            key: "theme_target",
            label: "主题或赛道",
            placeholder: "",
            type: "text",
          },
          {
            key: "platform_region",
            label: "希望关注的平台/地域",
            placeholder: "",
            type: "text",
          },
        ],
        outputContract: ["趋势摘要", "3 个优先选题", "代表案例线索"],
      },
      inputValues: initialLaunchInputValues,
    });

    renderInputbar({
      input: initialPrompt,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 2026042402,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const editButton = document.querySelector(
      '[data-testid="curated-task-badge-edit"]',
    ) as HTMLButtonElement | null;
    expect(editButton).toBeTruthy();

    await act(async () => {
      editButton?.click();
      await Promise.resolve();
    });

    const referenceToggle = document.querySelector(
      '[data-testid="curated-task-dialog-reference-toggle"]',
    ) as HTMLButtonElement | null;
    expect(referenceToggle).toBeTruthy();

    await act(async () => {
      referenceToggle?.click();
      await Promise.resolve();
    });

    const reviewAction = document.querySelector(
      '[data-testid="curated-task-dialog-review-action"]',
    ) as HTMLButtonElement | null;
    expect(reviewAction).toBeTruthy();

    await act(async () => {
      reviewAction?.click();
      await Promise.resolve();
    });

    const launcherDialog = document.querySelector(
      '[data-testid="curated-task-launcher-dialog"]',
    ) as HTMLDivElement | null;
    expect(launcherDialog?.dataset.taskId).toBe("account-project-review");
    expect(launcherDialog?.dataset.referenceMemoryCount).toBe("1");

    const prefillHint = document.querySelector(
      '[data-testid="curated-task-launcher-prefill-hint"]',
    );
    expect(prefillHint?.textContent).toBe(
      "Switched to a better-fit result template based on the latest review. You can keep editing before sending.",
    );
  });

  it("输入条已激活复盘模板时，应把 sceneapp 项目结果引用透传给 badge", async () => {
    renderInputbar({
      input: "请帮我判断这个账号或项目当前该怎么推进",
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "account-project-review",
          taskTitle: "复盘这个账号/项目",
          prompt: "请帮我判断这个账号或项目当前该怎么推进",
          launchInputValues: {
            project_goal: "AI 内容周报",
            existing_results: "当前已有一轮项目结果，可直接作为复盘基线。",
          },
          referenceEntries: [
            {
              id: "sceneapp:content-pack:run:1",
              sourceKind: "sceneapp_execution_summary",
              title: "AI 内容周报",
              summary: "当前已有一轮项目结果，可直接作为复盘基线。",
              category: "experience",
              categoryLabel: "成果",
              tags: ["复盘"],
            },
          ],
        },
        requestKey: 2026042301,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const curatedTaskBadge = document.querySelector(
      '[data-testid="curated-task-badge"]',
    ) as HTMLDivElement | null;
    expect(curatedTaskBadge?.dataset.referenceCount).toBe("1");
    expect(curatedTaskBadge?.dataset.firstSourceKind).toBe(
      "sceneapp_execution_summary",
    );
  });

  it("结果模板带着灵感引用发送时，应附带结构化 request metadata", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input: "请先给我做一版每日趋势摘要",
      onSend,
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "curated_task",
        task: {
          id: "daily-trend-briefing",
          title: "每日趋势摘要",
          summary: "先收一版趋势摘要。",
          outputHint: "趋势摘要 + 选题方向",
          resultDestination:
            "趋势摘要会先写回当前内容，方便继续展开选题和主稿。",
          categoryLabel: "趋势与选题",
          prompt: "请先给我做一版每日趋势摘要",
          requiredInputs: ["主题或赛道", "希望关注的平台/地域"],
          requiredInputFields: [],
          optionalReferences: ["已有账号方向", "过去爆款链接"],
          outputContract: ["趋势摘要", "3 个优先选题"],
          followUpActions: ["继续展开其中一个选题"],
          badge: "结果模板",
          actionLabel: "进入生成",
          statusLabel: "可直接开始",
          statusTone: "emerald",
          recentUsedAt: null,
          isRecent: false,
        },
        referenceMemoryIds: ["memory-1"],
        referenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      });
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "请先给我做一版每日趋势摘要",
      sendOptions: expect.objectContaining({
        capabilityRoute: expect.objectContaining({
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          referenceMemoryIds: ["memory-1"],
        }),
        requestMetadata: {
          harness: {
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
          },
        },
      }),
    });
  });

  it("已激活结果模板后重新编辑启动信息时，应更新输入与 curated_task route", async () => {
    const setInput = vi.fn();
    const onSend = vi.fn();
    const initialLaunchInputValues = {
      theme_target: "AI 内容创作",
      platform_region: "X 与 TikTok 北美区",
    };
    const editedLaunchInputValues = {
      theme_target: "品牌内容中台",
      platform_region: "LinkedIn 与 X（海外）",
    };
    const taskDefinition = {
      prompt:
        "请先给我做一版每日趋势摘要：围绕当前主题梳理最近值得关注的趋势、热点内容方向、代表案例、用户正在关心的问题，以及最值得立即开工的 3 个选题。",
      requiredInputFields: [
        {
          key: "theme_target",
          label: "主题或赛道",
          placeholder: "",
          type: "text" as const,
        },
        {
          key: "platform_region",
          label: "希望关注的平台/地域",
          placeholder: "",
          type: "text" as const,
        },
      ],
      outputContract: ["趋势摘要", "3 个优先选题", "代表案例线索"],
    };
    const initialPrompt = buildCuratedTaskLaunchPrompt({
      task: taskDefinition,
      inputValues: initialLaunchInputValues,
    });
    const editedPrompt = buildCuratedTaskLaunchPrompt({
      task: taskDefinition,
      inputValues: editedLaunchInputValues,
    });
    const rendered = renderInputbar({
      input: initialPrompt,
      setInput,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 20260419,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const editButton = document.querySelector(
      '[data-testid="curated-task-badge-edit"]',
    ) as HTMLButtonElement | null;
    expect(editButton).toBeTruthy();

    await act(async () => {
      editButton?.click();
      await Promise.resolve();
    });

    const themeInput = document.querySelector(
      '[data-testid="curated-task-dialog-field-theme_target"]',
    ) as HTMLInputElement | null;
    const platformInput = document.querySelector(
      '[data-testid="curated-task-dialog-field-platform_region"]',
    ) as HTMLInputElement | null;

    expect(themeInput?.value).toBe("AI 内容创作");
    expect(platformInput?.value).toBe("X 与 TikTok 北美区");

    await act(async () => {
      updateFieldValue(themeInput, editedLaunchInputValues.theme_target);
      updateFieldValue(platformInput, editedLaunchInputValues.platform_region);
      await Promise.resolve();
    });

    const confirmButton = document.querySelector(
      '[data-testid="curated-task-dialog-confirm"]',
    ) as HTMLButtonElement | null;
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenLastCalledWith(editedPrompt);

    rendered.rerender({
      input: editedPrompt,
      setInput,
      onSend,
      activeTheme: "general",
      initialInputCapability: {
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: initialPrompt,
          launchInputValues: initialLaunchInputValues,
        },
        requestKey: 20260419,
      },
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: editedPrompt,
      sendOptions: expect.objectContaining({
        capabilityRoute: {
          kind: "curated_task",
          taskId: "daily-trend-briefing",
          taskTitle: "每日趋势摘要",
          prompt: editedPrompt,
          launchInputValues: editedLaunchInputValues,
        },
        displayContent: editedPrompt,
        requestMetadata: {
          harness: {
            curated_task: expect.objectContaining({
              task_id: "daily-trend-briefing",
              launch_input_values: editedLaunchInputValues,
            }),
          },
        },
      }),
    });
  });

  it("选择内建命令发送时，应透传 capability route 与原始显示文案", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input: "整理最近发布计划",
      onSend,
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    const builtinCommand: BuiltinInputCommand = {
      key: "research",
      label: "搜索",
      mentionLabel: "搜索",
      commandPrefix: "@搜索",
      description: "搜索资料",
      aliases: [],
    };
    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "builtin_command",
        command: builtinCommand,
      });
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectInputbarSend(onSend, {
      textOverride: "整理最近发布计划",
      sendOptions: {
        capabilityRoute: {
          kind: "builtin_command",
          commandKey: "research",
          commandPrefix: "@搜索",
        },
        displayContent: "整理最近发布计划",
      },
    });
  });

  it("选择运行时场景发送时，应透传 runtime scene route 与原始显示文案", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input: "帮我做一版新品活动启动方案",
      onSend,
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    const runtimeScene: RuntimeSceneSlashCommand = {
      key: "campaign-launch",
      label: "活动启动场景",
      commandPrefix: "/campaign-launch",
      description: "围绕活动目标生成启动方案",
      aliases: [],
    };
    const latestCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      latestCall?.onSelectInputCapability?.({
        kind: "runtime_scene",
        command: runtimeScene,
      });
      await Promise.resolve();
    });

    expect(
      document.querySelector('[data-testid="runtime-scene-badge"]'),
    ).toBeTruthy();

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectInputbarSend(onSend, {
      textOverride: "帮我做一版新品活动启动方案",
      sendOptions: {
        capabilityRoute: {
          kind: "runtime_scene",
          sceneKey: "campaign-launch",
          commandPrefix: "/campaign-launch",
        },
        displayContent: "帮我做一版新品活动启动方案",
      },
    });
  });

  it("先选内建命令再切到运行时场景时，不应继续残留旧命令 route", async () => {
    const onSend = vi.fn();
    renderInputbar({
      input: "帮我做一版新品活动启动方案",
      onSend,
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    const builtinCommand: BuiltinInputCommand = {
      key: "research",
      label: "搜索",
      mentionLabel: "搜索",
      commandPrefix: "@搜索",
      description: "搜索资料",
      aliases: [],
    };
    const runtimeScene: RuntimeSceneSlashCommand = {
      key: "campaign-launch",
      label: "活动启动场景",
      commandPrefix: "/campaign-launch",
      description: "围绕活动目标生成启动方案",
      aliases: [],
    };
    const firstCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      firstCall?.onSelectInputCapability?.({
        kind: "builtin_command",
        command: builtinCommand,
      });
      await Promise.resolve();
    });

    const secondCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      secondCall?.onSelectInputCapability?.({
        kind: "runtime_scene",
        command: runtimeScene,
      });
      await Promise.resolve();
    });

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectInputbarSend(onSend, {
      textOverride: "帮我做一版新品活动启动方案",
      sendOptions: {
        capabilityRoute: {
          kind: "runtime_scene",
          sceneKey: "campaign-launch",
          commandPrefix: "/campaign-launch",
        },
        displayContent: "帮我做一版新品活动启动方案",
      },
    });
  });

  it("先选内建命令再切到服务技能入口时，不应继续残留旧命令前缀", async () => {
    const onSend = vi.fn();
    const onSelectServiceSkill = vi.fn();
    const serviceSkill = {
      id: "daily-trend-briefing",
      title: "每日趋势摘要",
    } as ServiceSkillHomeItem;
    renderInputbar({
      input: "整理最近发布计划",
      onSend,
      onSelectServiceSkill,
      serviceSkills: [serviceSkill],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const builtinCommand: BuiltinInputCommand = {
      key: "research",
      label: "搜索",
      mentionLabel: "搜索",
      commandPrefix: "@搜索",
      description: "搜索资料",
      aliases: [],
    };
    const firstCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    expect(firstCall?.onSelectInputCapability).toBeTypeOf("function");

    await act(async () => {
      firstCall?.onSelectInputCapability?.({
        kind: "builtin_command",
        command: builtinCommand,
      });
      await Promise.resolve();
    });

    const secondCall =
      mockCharacterMention.mock.calls[
        mockCharacterMention.mock.calls.length - 1
      ]?.[0];

    await act(async () => {
      secondCall?.onSelectInputCapability?.({
        kind: "service_skill",
        skill: serviceSkill,
      });
      await Promise.resolve();
    });

    expect(onSelectServiceSkill).toHaveBeenCalledWith(serviceSkill);

    const sendButton = document.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expectInputbarSend(onSend, {
      textOverride: "整理最近发布计划",
    });
  });

  it("没有任务文件和额外控件时不应渲染 overlay row", async () => {
    const { container } = renderInputbar({
      overlayAccessory: null,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-secondary-controls"]'),
    ).toBeNull();
  });

  it("主线程任务状态不应再在输入区单独渲染 task strip", async () => {
    const { container } = renderInputbar({
      input: "分析 claudecode 项目结构并继续执行",
      activeTheme: "general",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="agent-task-strip"]'),
    ).toBeNull();
  });

  it("工作区输入区默认隐藏技能选择器，通过加号菜单打开独立技能浮层", async () => {
    const { container } = renderInputbar({
      activeTheme: "general",
      skills: [
        {
          key: "writer",
          name: "写作助手",
          description: "用于写作",
          directory: "writer",
          installed: true,
          sourceKind: "builtin",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="skill-selector-trigger"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-plus-trigger"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="character-mention-stub"]'),
    ).toBeTruthy();

    openSkillsPanel(container);

    expect(
      document.body.querySelector('[data-testid="skill-selector-inline"]'),
    ).toBeTruthy();
  });

  it("加号菜单选择插件时应写回显式触发前缀并显示插件标记", async () => {
    const setInput = vi.fn();
    const { container, rerender } = renderInputbar({
      input: "整理今天的选题",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    expect(pluginsPanel.textContent).toContain("选择插件");
    const option = pluginsPanel.querySelector(
      '[data-testid="inputbar-plugin-option"]',
    ) as HTMLButtonElement | null;
    expect(option).toBeTruthy();

    await act(async () => {
      option?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith("@内容工厂 整理今天的选题");

    rerender({
      input: "@内容工厂 整理今天的选题",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-plugin-badge"]'),
    ).toBeTruthy();
  });

  it("加号菜单选择内容工厂写文章入口时应写回 @写文章", async () => {
    const setInput = vi.fn();
    const { container } = renderInputbar({
      input: "写一篇关于登山的文章",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-factory-app",
          displayName: "写文章",
          trigger: "@写文章",
          description: "@写文章 · 启动内容工厂文章工作流",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    const option = pluginsPanel.querySelector(
      '[data-testid="inputbar-plugin-option"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      option?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith("@写文章 写一篇关于登山的文章");
  });

  it("mention 面板选择插件时应保留插件标记但不重复写回触发词", async () => {
    const setInput = vi.fn();
    const plugin: InputbarPluginCapability = {
      pluginId: "content-factory-app",
      displayName: "写文章",
      trigger: "@写文章",
      description: "生成文章草稿",
    };
    const { container, rerender } = renderInputbar({
      input: "@写文章",
      setInput,
      pluginSuggestions: [plugin],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const mentionProps = mockCharacterMention.mock.calls.at(-1)?.[0];
    expect(mentionProps?.onSelectPlugin).toBeTruthy();

    await act(async () => {
      mentionProps?.onSelectPlugin?.(plugin, undefined, {
        inputOverride: "",
        preserveInputOverride: true,
      });
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith("");

    rerender({
      input: "",
      setInput,
      pluginSuggestions: [plugin],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-plugin-badge"]')
        ?.textContent,
    ).toContain("写文章");
    expect(setInput).not.toHaveBeenCalledWith("@写文章");
  });

  it("加号菜单选择插件技能时应写回 @插件:技能 前缀", async () => {
    const setInput = vi.fn();
    const { container, rerender } = renderInputbar({
      input: "整理今天的选题",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
          skills: [
            {
              skillId: "article-writer",
              title: "文章写作",
              description: "生成文章草稿",
            },
          ],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    expect(pluginsPanel.textContent).toContain("文章写作");
    const skillOption = pluginsPanel.querySelector(
      '[data-testid="inputbar-plugin-skill-option"]',
    ) as HTMLButtonElement | null;
    expect(skillOption).toBeTruthy();

    await act(async () => {
      skillOption?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith("@内容工厂:文章写作 整理今天的选题");

    rerender({
      input: "@内容工厂:文章写作 整理今天的选题",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
          skills: [
            {
              skillId: "article-writer",
              title: "文章写作",
              description: "生成文章草稿",
            },
          ],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-plugin-badge"]')
        ?.textContent,
    ).toContain("内容工厂:文章写作");
  });

  it("插件技能同一 skillId 的多个 alias 不应触发重复 key 警告", async () => {
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);

    try {
      const { container } = renderInputbar({
        pluginSuggestions: [
          {
            pluginId: "content-factory-app",
            displayName: "写文章",
            trigger: "@写文章",
            description: "@写文章 · 启动内容工厂文章工作流",
            skills: [
              {
                skillId: "content_article_generate",
                title: "@写文章",
                trigger: "@写文章",
              },
              {
                skillId: "content_article_generate",
                title: "@写作",
                trigger: "@写作",
              },
            ],
          },
        ],
      });

      await act(async () => {
        await Promise.resolve();
      });

      const pluginsPanel = openPluginsPanel(container);

      expect(
        pluginsPanel.querySelectorAll(
          '[data-testid="inputbar-plugin-skill-option"]',
        ),
      ).toHaveLength(2);
      expect(pluginsPanel.textContent).toContain("@写文章");
      expect(pluginsPanel.textContent).toContain("@写作");
      expect(
        consoleErrorSpy.mock.calls.some((call) =>
          call
            .map((item) => String(item))
            .join(" ")
            .includes("Encountered two children with the same key"),
        ),
      ).toBe(false);
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });

  it("用户手动删除插件技能前缀时不应自动恢复前缀", async () => {
    const setInput = vi.fn();
    const { container, rerender } = renderInputbar({
      input: "整理今天的选题",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
          skills: [
            {
              skillId: "article-writer",
              title: "文章写作",
              description: "生成文章草稿",
            },
          ],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    const skillOption = pluginsPanel.querySelector(
      '[data-testid="inputbar-plugin-skill-option"]',
    ) as HTMLButtonElement | null;
    expect(skillOption).toBeTruthy();

    await act(async () => {
      skillOption?.click();
      await Promise.resolve();
    });

    expect(setInput).toHaveBeenCalledWith("@内容工厂:文章写作 整理今天的选题");

    rerender({
      input: "@内容工厂:文章写作 整理今天的选题",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
          skills: [
            {
              skillId: "article-writer",
              title: "文章写作",
              description: "生成文章草稿",
            },
          ],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-plugin-badge"]'),
    ).toBeTruthy();
    setInput.mockClear();

    rerender({
      input: "@内容工厂:Ar",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: "内容工厂",
          description: "整理内容生产资料",
          skills: [
            {
              skillId: "article-writer",
              title: "文章写作",
              description: "生成文章草稿",
            },
          ],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setInput).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="inputbar-plugin-badge"]'),
    ).toBeNull();
  });

  it("没有插件候选时加号菜单仍应显示插件入口和空态", async () => {
    const { container } = renderInputbar();

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    expect(pluginsPanel.textContent).toContain("当前没有可选插件");
  });

  it("插件候选加载中时加号菜单不应提前显示空态", async () => {
    const { container } = renderInputbar({
      pluginSuggestionsLoading: true,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    expect(pluginsPanel.textContent).toContain("正在读取已安装插件");
    expect(pluginsPanel.textContent).not.toContain("当前没有可选插件");
  });

  it("插件候选名称为空时应回退显示插件 id", async () => {
    const { container } = renderInputbar({
      pluginSuggestions: [
        {
          pluginId: "content-workbench",
          displayName: " ",
          description: "整理内容生产资料",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    expect(pluginsPanel.textContent).toContain("content-workbench");
  });

  it("加号菜单中的阻断插件应禁用且不写回输入", async () => {
    const setInput = vi.fn();
    const { container } = renderInputbar({
      input: "继续生成",
      setInput,
      pluginSuggestions: [
        {
          pluginId: "blocked-workbench",
          displayName: "受限工作台",
          disabled: true,
          blockerCodes: ["PLUGIN_DISABLED"],
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const pluginsPanel = openPluginsPanel(container);
    const option = pluginsPanel.querySelector(
      '[data-testid="inputbar-plugin-option"]',
    ) as HTMLButtonElement | null;
    expect(option).toBeTruthy();
    expect(option?.disabled).toBe(true);

    await act(async () => {
      option?.click();
      await Promise.resolve();
    });

    expect(setInput).not.toHaveBeenCalled();
  });

  it("存在 executionRuntime 时底栏也应直接显示模型切换器", async () => {
    const { container } = renderInputbar({
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-5.4-mini",
      setModel: vi.fn(),
      executionRuntime: {
        session_id: "session-1",
        provider_selector: "openai",
        provider_name: "openai",
        model_name: "gpt-5.4",
        source: "turn_context",
        output_schema_runtime: {
          source: "turn",
          strategy: "native",
          providerName: "openai",
          modelName: "gpt-5.4",
        },
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("最近执行模型");
    expect(
      container.querySelector('[data-testid="model-selector"]'),
    ).toBeTruthy();
    expect(container.textContent).not.toContain("结构化输出");
    expect(container.textContent).not.toContain("Native schema");
  });

  it("高级设置只保留权限模式，不再渲染编程执行前置开关", async () => {
    const setAccessMode = vi.fn();
    const { container } = renderInputbar({
      accessMode: "current",
      setAccessMode,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-plan-toggle"]'),
    ).toBeNull();
    expect(
      container.querySelector('select[aria-label="权限模式"]'),
    ).toBeTruthy();

    const planToggle = container.querySelector(
      '[data-testid="inputbar-plan-toggle"]',
    ) as HTMLButtonElement | null;
    const accessSelect = container.querySelector(
      'select[aria-label="权限模式"]',
    ) as HTMLSelectElement | null;

    expect(planToggle).toBeNull();
    expect(accessSelect).toBeTruthy();
    expect(container.querySelector('select[aria-label="执行模式"]')).toBeNull();
    expect(container.textContent).not.toContain("编程执行");

    act(() => {
      accessSelect!.value = "full-access";
      accessSelect?.dispatchEvent(new Event("change", { bubbles: true }));
    });

    expect(setAccessMode).toHaveBeenCalledWith("full-access");
  });

  it("高级设置 en-US 也不再渲染编程执行前置开关", async () => {
    await changeLimeLocale("en-US");
    const { container } = renderInputbar({
      accessMode: "current",
      setAccessMode: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    const planToggle = container.querySelector(
      '[data-testid="inputbar-plan-toggle"]',
    ) as HTMLButtonElement | null;

    expect(planToggle).toBeNull();
    expect(container.textContent).not.toContain("Code");
    expect(container.textContent).not.toContain("coding execution");
  });

  it("受控模式下不再渲染联网搜索前置开关", async () => {
    const onToolStatesChange = vi.fn();
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    act(() => {
      root.render(
        <Inputbar
          input=""
          setInput={vi.fn()}
          onSend={vi.fn()}
          isLoading={false}
          characters={[]}
          skills={[]}
          toolStates={{ subagent: false }}
          onToolStatesChange={onToolStatesChange}
        />,
      );
    });

    mountedRoots.push({ root, container });
    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="toggle-web-search"]'),
    ).toBeNull();

    const toggleButton = container.querySelector(
      '[data-testid="toggle-web-search"]',
    ) as HTMLButtonElement | null;
    expect(toggleButton).toBeNull();
    expect(onToolStatesChange).not.toHaveBeenCalled();
    expect(toast.info).not.toHaveBeenCalledWith("联网搜索已开启");
  });

  it("en-US 下也不再渲染联网搜索前置开关", async () => {
    await changeLimeLocale("en-US");
    const onToolStatesChange = vi.fn();
    const { container } = renderInputbar({
      input: "please search the web",
      toolStates: { subagent: false },
      onToolStatesChange,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const toggleButton = container.querySelector(
      '[data-testid="toggle-web-search"]',
    ) as HTMLButtonElement | null;
    expect(toggleButton).toBeNull();
    expect(onToolStatesChange).not.toHaveBeenCalled();
    expect(toast.info).not.toHaveBeenCalledWith("Web search is on");
  });

  it("移除执行策略选择后普通发送不再提交 execution strategy", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const { container } = renderInputbar({
      input: "请联网检查这个依赖变更并修复编译错误",
      onSend,
      toolStates: { subagent: false },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "请联网检查这个依赖变更并修复编译错误",
    });
  });

  it("通用聊天态复杂任务不再主动显示 Subagents 建议", async () => {
    const onToolStatesChange = vi.fn();
    const { container } = renderInputbar({
      input:
        "请拆分这个多代理调试任务，分别分析、实现、验证，再由主线程汇总结论。",
      activeTheme: "general",
      toolStates: { subagent: false },
      onToolStatesChange,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const retiredSuggestionTestId = ["team", "suggestion", "bar"].join("-");
    const suggestionBar = container.querySelector(
      `[data-testid="${retiredSuggestionTestId}"]`,
    );
    expect(suggestionBar).toBeNull();
    expect(onToolStatesChange).not.toHaveBeenCalled();
  });

  it("启用项目资料后发送应携带 knowledge_pack metadata", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const { container } = renderInputbar({
      input: "基于项目资料写一段介绍",
      onSend,
      knowledgePackSelection: {
        enabled: true,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "品牌产品资料",
        status: "ready",
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "基于项目资料写一段介绍",
      sendOptions: expect.objectContaining({
        requestMetadata: {
          knowledge_pack: {
            pack_name: "brand-product-demo",
            working_dir: "/tmp/lime-project",
            source: "inputbar",
          },
        },
      }),
    });
  });

  it("启用运营资料后发送应携带隐式 persona 协同 pack", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const { container } = renderInputbar({
      input: "基于运营资料写一段社群预热文案",
      onSend,
      knowledgePackSelection: {
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
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "基于运营资料写一段社群预热文案",
      sendOptions: expect.objectContaining({
        requestMetadata: expect.objectContaining({
          knowledge_pack: {
            pack_name: "content-calendar",
            working_dir: "/tmp/lime-project",
            source: "inputbar",
            packs: [
              {
                name: "founder-persona",
                activation: "implicit",
              },
            ],
          },
          persona_context: expect.objectContaining({
            source: "knowledge_pack",
            scope: "style_context_only",
            packs: [
              {
                name: "founder-persona",
                activation: "implicit",
                role: "companion",
              },
            ],
          }),
        }),
      }),
    });
  });

  it("资料中枢应支持显式追加多份 data 协同资料", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const onToggleKnowledgeCompanionPack = vi.fn();
    const { container, rerender } = renderInputbar({
      input: "基于运营资料生成本周计划",
      onSend,
      knowledgePackSelection: {
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
      knowledgePackOptions: [
        {
          packName: "content-calendar",
          label: "内容运营资料",
          status: "ready",
          runtimeMode: "data",
        },
        {
          packName: "founder-persona",
          label: "创始人人设资料",
          status: "ready",
          defaultForWorkspace: true,
          runtimeMode: "persona",
        },
        {
          packName: "campaign-plan",
          label: "618 活动资料",
          status: "ready",
          runtimeMode: "data",
        },
      ],
      onToggleKnowledgeCompanionPack,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    const hub = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-hub"]',
    );
    expect(hub?.textContent).toContain("已自动搭配人设资料：创始人人设资料");
    expect(hub?.textContent).toContain("协同资料（可多选）");

    const campaignOption = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-companion-option-campaign-plan"]',
    ) as HTMLButtonElement | null;
    expect(campaignOption).toBeTruthy();

    act(() => {
      campaignOption?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onToggleKnowledgeCompanionPack).toHaveBeenCalledWith(
      "campaign-plan",
      true,
    );

    rerender({
      knowledgePackSelection: {
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
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "基于运营资料生成本周计划",
      sendOptions: expect.objectContaining({
        requestMetadata: expect.objectContaining({
          knowledge_pack: {
            pack_name: "content-calendar",
            working_dir: "/tmp/lime-project",
            source: "inputbar",
            packs: [
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
          persona_context: expect.objectContaining({
            packs: [
              {
                name: "founder-persona",
                activation: "implicit",
                role: "companion",
              },
            ],
          }),
        }),
      }),
    });
  });

  it("项目资料控件应在输入框主路径常显并使用用户语言", async () => {
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: false,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "品牌产品资料",
        status: "ready",
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    expect(knowledgePanel.textContent).toContain("品牌产品资料");
    expect(knowledgePanel.textContent).toContain("已确认，可直接用于本轮生成");
    expect(container.textContent).not.toContain("知识包");
  });

  it("启用资料应在输入区直接显示当前资料和协同数量", async () => {
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: true,
        packName: "content-calendar",
        workingDir: "/tmp/lime-project",
        label: "内容运营资料",
        status: "ready",
        companionPacks: [
          {
            name: "founder-persona",
            activation: "explicit",
          },
          {
            name: "campaign-plan",
            activation: "explicit",
          },
        ],
      },
      knowledgePackOptions: [
        {
          packName: "content-calendar",
          label: "内容运营资料",
          status: "ready",
          runtimeMode: "data",
        },
        {
          packName: "founder-persona",
          label: "写作口吻资料",
          status: "ready",
          runtimeMode: "persona",
        },
        {
          packName: "campaign-plan",
          label: "活动资料",
          status: "ready",
          runtimeMode: "data",
        },
      ],
    });

    await act(async () => {
      await Promise.resolve();
    });

    const visibleToggle = container.querySelector(
      '[data-testid="top-extra"] [data-testid="inputbar-knowledge-pack-toggle"]',
    ) as HTMLButtonElement | null;
    expect(visibleToggle).toBeTruthy();
    expect(visibleToggle?.textContent).toContain("资料：内容运营资料 +2");
    expect(visibleToggle?.getAttribute("title")).toContain("内容运营资料");
  });

  it("项目资料控件应在英文界面读取 agent namespace 文案", async () => {
    await changeLimeLocale("en-US");

    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: true,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "Brand knowledge",
        status: "ready",
        companionPacks: [
          {
            name: "founder-persona",
            activation: "implicit",
          },
        ],
      },
      knowledgePackOptions: [
        {
          packName: "brand-product-demo",
          label: "Brand knowledge",
          status: "ready",
          defaultForWorkspace: true,
          runtimeMode: "data",
        },
        {
          packName: "founder-persona",
          label: "Founder persona",
          status: "ready",
          runtimeMode: "persona",
        },
        {
          packName: "campaign-plan",
          label: "Campaign plan",
          status: "ready",
          runtimeMode: "data",
        },
      ],
      onToggleKnowledgeCompanionPack: vi.fn(),
      onStartKnowledgeOrganize: vi.fn(),
      onManageKnowledgePacks: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);

    const hub = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-hub"]',
    );
    expect(hub?.textContent).toContain("Using: Brand knowledge");
    expect(hub?.textContent).toContain("Main knowledge");
    expect(hub?.textContent).toContain("Default");
    expect(hub?.textContent).toContain("Reviewed and ready for generation");
    expect(hub?.textContent).toContain(
      "Persona knowledge added automatically: Founder persona.",
    );
    expect(hub?.textContent).toContain("Companion knowledge (multi-select)");
    expect(hub?.textContent).toContain("Available");
    expect(
      knowledgePanel
        .querySelector('[data-testid="inputbar-knowledge-hub-dismiss"]')
        ?.getAttribute("aria-label"),
    ).toBeUndefined();
    expect(hub?.textContent).toContain("Add to knowledge");
    expect(hub?.textContent).toContain("Check knowledge");
  });

  it("资料中枢应在加号菜单中作为独立二级浮层展示", async () => {
    const onToggleKnowledgePack = vi.fn();
    const onKnowledgePacksNeeded = vi.fn();
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: false,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "品牌产品资料",
        status: "ready",
      },
      onKnowledgePacksNeeded,
      onToggleKnowledgePack,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    expect(onKnowledgePacksNeeded).toHaveBeenCalledTimes(1);
    expect(knowledgePanel).toBeTruthy();
    expect(
      knowledgePanel.querySelector('[data-testid="inputbar-knowledge-hub"]'),
    ).toBeTruthy();
    expect(
      knowledgePanel.querySelector(
        '[data-testid="inputbar-knowledge-hub-dismiss"]',
      ),
    ).toBeNull();
    expect(onToggleKnowledgePack).not.toHaveBeenCalled();
  });

  it("待确认资料即使残留启用状态也不应显示为正在使用", async () => {
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: true,
        packName: "draft-without-source",
        workingDir: "/tmp/lime-project",
        label: "待确认资料",
        status: "needs-review",
      },
      onManageKnowledgePacks: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    const hub = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-hub"]',
    );
    expect(hub?.textContent).toContain("资料待确认");
    expect(hub?.textContent).toContain("去确认资料");
    expect(hub?.textContent).not.toContain("正在使用");
    expect(hub?.textContent).not.toContain("管理资料");
    expect(hub?.textContent).not.toContain("关闭资料");
  });

  it("没有项目资料时应能从输入框触发资料整理入口", async () => {
    const onStartKnowledgeOrganize = vi.fn();
    const { container } = renderInputbar({
      onStartKnowledgeOrganize,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    const organizeCard = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-hub"]',
    );
    expect(organizeCard?.textContent).toContain("添加项目资料");
    expect(organizeCard?.textContent).toContain("开始添加资料");

    const organizeAction = Array.from(
      knowledgePanel.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("开始添加资料"));
    act(() => {
      organizeAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onStartKnowledgeOrganize).toHaveBeenCalledTimes(1);
  });

  it("没有项目资料时不应把普通用户带到管理动作", async () => {
    const onManageKnowledgePacks = vi.fn();
    const { container } = renderInputbar({
      onManageKnowledgePacks,
      onStartKnowledgeOrganize: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector(
        '[data-testid="inputbar-knowledge-pack-menu-toggle"]',
      ),
    ).toBeNull();

    const knowledgePanel = openKnowledgePanel(container);
    const hub = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-hub"]',
    );
    expect(hub?.textContent).toContain("开始添加资料");
    expect(hub?.textContent).not.toContain("管理资料");
  });

  it("可从输入区菜单切换具体项目资料并用选中资料发送", async () => {
    const onSend = vi.fn().mockResolvedValue(true);
    const onSelectKnowledgePack = vi.fn();
    const onToggleKnowledgePack = vi.fn();
    const { container, rerender } = renderInputbar({
      input: "基于选中的项目资料写介绍",
      onSend,
      knowledgePackSelection: {
        enabled: false,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "品牌产品资料",
        status: "ready",
      },
      knowledgePackOptions: [
        {
          packName: "brand-product-demo",
          label: "品牌产品资料",
          status: "ready",
          defaultForWorkspace: true,
        },
        {
          packName: "org-knowhow-demo",
          label: "组织经验资料",
          status: "ready",
        },
      ],
      onSelectKnowledgePack,
      onToggleKnowledgePack,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    const secondPackOption = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-pack-option-org-knowhow-demo"]',
    ) as HTMLButtonElement | null;
    expect(secondPackOption).toBeTruthy();
    expect(secondPackOption?.textContent).toContain("组织经验资料");

    act(() => {
      secondPackOption?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(onSelectKnowledgePack).toHaveBeenCalledWith("org-knowhow-demo");
    expect(onToggleKnowledgePack).toHaveBeenCalledWith(true);

    rerender({
      knowledgePackSelection: {
        enabled: true,
        packName: "org-knowhow-demo",
        workingDir: "/tmp/lime-project",
        label: "组织经验资料",
        status: "ready",
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    await act(async () => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "基于选中的项目资料写介绍",
      sendOptions: expect.objectContaining({
        requestMetadata: {
          knowledge_pack: {
            pack_name: "org-knowhow-demo",
            working_dir: "/tmp/lime-project",
            source: "inputbar",
          },
        },
      }),
    });
  });

  it("已有可用资料时仍应提供添加新资料入口", async () => {
    const onStartKnowledgeOrganize = vi.fn();
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: false,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "品牌产品资料",
        status: "ready",
      },
      onStartKnowledgeOrganize,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);

    const addButton = Array.from(
      knowledgePanel.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("添加新资料"));
    expect(addButton).toBeTruthy();

    act(() => {
      addButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onStartKnowledgeOrganize).toHaveBeenCalledTimes(1);
  });

  it("资料中枢只把已确认资料作为可用选项，待确认资料回到管理页处理", async () => {
    const onManageKnowledgePacks = vi.fn();
    const { container } = renderInputbar({
      knowledgePackSelection: {
        enabled: true,
        packName: "brand-product-demo",
        workingDir: "/tmp/lime-project",
        label: "品牌产品资料",
        status: "ready",
      },
      knowledgePackOptions: [
        {
          packName: "brand-product-demo",
          label: "品牌产品资料",
          status: "ready",
          defaultForWorkspace: true,
        },
        {
          packName: "no-source-draft",
          label: "看起来你还没有提供原始素材",
          status: "needs-review",
        },
      ],
      onManageKnowledgePacks,
      onStartKnowledgeOrganize: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
    });

    const knowledgePanel = openKnowledgePanel(container);
    const hub = knowledgePanel.querySelector(
      '[data-testid="inputbar-knowledge-hub"]',
    );
    expect(hub?.textContent).toContain("品牌产品资料");
    expect(hub?.textContent).toContain(
      "还有 1 份资料待确认，确认后才会出现在可用列表里。",
    );
    expect(hub?.textContent).toContain("补充资料");
    expect(hub?.textContent).toContain("检查资料");
    expect(hub?.textContent).not.toContain("看起来你还没有提供原始素材");
    expect(
      knowledgePanel.querySelector(
        '[data-testid="inputbar-knowledge-pack-option-no-source-draft"]',
      ),
    ).toBeNull();

    const manageButton = Array.from(
      knowledgePanel.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("检查资料"));

    act(() => {
      manageButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onManageKnowledgePacks).toHaveBeenCalledTimes(1);
  });

  it("Subagents 开关只反映偏好状态", async () => {
    const { container } = renderInputbar({
      activeTheme: "general",
      toolStates: { subagent: false },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expandAdvancedControls(container);

    expect(
      container.querySelector('[data-testid="inputbar-plus-subagent-mode"]')
        ?.textContent,
    ).toContain("off");

    const { container: enabledContainer } = renderInputbar({
      activeTheme: "general",
      toolStates: { subagent: true },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expandAdvancedControls(enabledContainer);

    expect(
      enabledContainer.querySelector(
        '[data-testid="inputbar-plus-subagent-mode"]',
      )?.textContent,
    ).toContain("on");
  });

  it("未开启 Subagents 时默认不暴露多代理开关，展开高级设置后可启用", async () => {
    const onToolStatesChange = vi.fn();
    const { container } = renderInputbar({
      activeTheme: "general",
      toolStates: { subagent: false },
      onToolStatesChange,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="team-mode-enable-button"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="toggle-subagent-mode"]'),
    ).toBeNull();

    expandAdvancedControls(container);

    const enableButton = container.querySelector(
      '[data-testid="inputbar-plus-subagent-mode"]',
    ) as HTMLButtonElement | null;

    expect(enableButton).toBeTruthy();

    act(() => {
      enableButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onToolStatesChange).toHaveBeenCalledWith({
      subagent: true,
    });
  });

  it("点击多代理图标后只切换 Subagents 偏好", async () => {
    const onToolStatesChange = vi.fn();
    const { container } = renderInputbar({
      activeTheme: "general",
      onToolStatesChange,
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expandAdvancedControls(container);

    const enableButton = container.querySelector(
      '[data-testid="inputbar-plus-subagent-mode"]',
    ) as HTMLButtonElement | null;

    expect(enableButton).toBeTruthy();

    act(() => {
      enableButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onToolStatesChange).toHaveBeenCalledWith({
      subagent: true,
    });
    expect(
      container.querySelector('[data-testid="inputbar-plus-subagent-mode"]'),
    ).toBeTruthy();
  });

  it("复杂任务但未开启 Subagents 时，不再渲染主动推荐入口", async () => {
    const { container } = renderInputbar({
      activeTheme: "general",
      input: "请把这个跨模块问题拆分成分析、实现、验证三个并行子任务再汇总",
      toolStates: { subagent: false },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const recommendationButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("打开 Subagents"));

    expect(
      container.querySelector('[data-testid="team-mode-enable-button"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="toggle-subagent-mode"]'),
    ).toBeNull();
    expect(recommendationButton).toBeUndefined();
    const retiredSuggestionTestId = ["team", "suggestion", "bar"].join("-");
    const suggestionBar = container.querySelector(
      `[data-testid="${retiredSuggestionTestId}"]`,
    );
    expect(suggestionBar).toBeNull();
  });

  it("内容主题默认发送时不应再注入旧 skill 前缀", async () => {
    const onSend = vi.fn();
    const { container } = renderInputbar({
      activeTheme: "general",
      input: "写一篇春季上新种草文案",
      onSend,
    });

    await act(async () => {
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "写一篇春季上新种草文案",
    });
  });

  it("社媒主题输入 slash 命令时不应重复注入默认 skill", async () => {
    const onSend = vi.fn();
    const { container } = renderInputbar({
      activeTheme: "general",
      input: "/custom_skill 写一篇品牌故事",
      onSend,
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const sendButton = container.querySelector(
      '[data-testid="send-btn"]',
    ) as HTMLButtonElement | null;
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton?.click();
      await Promise.resolve();
    });

    expectInputbarSend(onSend, {
      textOverride: "/custom_skill 写一篇品牌故事",
    });
  });

  it("工作区工作流模式应使用 attach-only 浮动输入配置", async () => {
    renderInputbar({
      variant: "workspace",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const latestCall =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCall).toBeTruthy();
    expect(latestCall.toolMode).toBe("attach-only");
    expect(
      Object.prototype.hasOwnProperty.call(latestCall, "showTranslate"),
    ).toBe(false);
    expect(latestCall.placeholder).toContain("试着输入任何指令");
    expect(latestCall.leftExtra).toBeUndefined();
  });

  it("已选择 Provider 时不应再将 prompt cache 提示组件常驻挂到输入区顶部", async () => {
    const { container } = renderInputbar({
      providerType: "custom-provider-id",
      setProviderType: vi.fn(),
      model: "glm-5.1",
      setModel: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-prompt-cache-warning"]'),
    ).toBeNull();
  });

  it("生成工作区应使用继续推进型输入提示", async () => {
    renderInputbar({
      variant: "workspace",
      contextVariant: "task-center",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const latestCall =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCall).toBeTruthy();
    expect(latestCall.placeholder).toContain(
      "继续补充这轮生成，或回到左侧继续旧历史",
    );
  });

  it("工作区输入提示应跟随 en-US 资源", async () => {
    await changeLimeLocale("en-US");
    const { rerender } = renderInputbar({
      variant: "workspace",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    let latestCall =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCall?.placeholder).toBe(
      "Try entering any instruction and I will handle the rest.",
    );

    rerender({
      variant: "workspace",
      contextVariant: "task-center",
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    latestCall =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCall?.placeholder).toBe(
      "Keep adding to this generation, or use the left side to return to earlier history.",
    );

    rerender({
      variant: "workspace",
      workflowGate: {
        key: "review_choice",
        title: "Review choice",
        status: "waiting",
        description: "Choose how to continue.",
      },
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    latestCall =
      mockInputbarCore.mock.calls[mockInputbarCore.mock.calls.length - 1]?.[0];
    expect(latestCall?.placeholder).toBe(
      "Tell me your choice and I will handle the rest.",
    );
  });

  it("工作区工作流在待启动状态下不应显示闸门条", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      workflowGate: {
        key: "draft_start",
        title: "编排待启动",
        status: "idle",
        description: "输入目标后将自动进入编排执行。",
      },
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("编排待启动");
    expect(container.textContent).not.toContain("待启动");
  });

  it("工作区工作流闸门快捷操作应能快速填充输入", async () => {
    const setInput = vi.fn();
    const { container } = renderInputbar({
      variant: "workspace",
      setInput,
      workflowGate: {
        key: "topic_select",
        title: "选题闸门",
        status: "waiting",
        description: "请选择优先推进的选题方向。",
      },
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("选题闸门");
    expect(container.textContent).not.toContain("当前闸门");
    expect(container.textContent).not.toContain("请选择优先推进的选题方向。");

    const quickActionButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("生成 3 个选题"));
    expect(quickActionButton).toBeTruthy();

    act(() => {
      quickActionButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(setInput).toHaveBeenCalledWith(
      "请给我 3 个可执行选题方向，并说明目标读者与传播价值。",
    );
  });

  it("工作区工作流生成中应展示任务面板并支持停止", async () => {
    const onStop = vi.fn();
    const { container } = renderInputbar({
      variant: "workspace",
      isLoading: true,
      onStop,
      workflowSteps: [
        { id: "research", title: "检索项目资料", status: "active" },
        { id: "write", title: "编写正文草稿", status: "pending" },
      ],
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("当前进展");
    expect(container.textContent).toContain("检索项目资料");
    expect(container.textContent).toContain("任务队列");
    expect(container.querySelector('[data-testid="inputbar-core"]')).toBeNull();

    const stopButton = container.querySelector(
      '[data-testid="workflow-stop"]',
    ) as HTMLButtonElement | null;
    expect(stopButton).toBeTruthy();
    act(() => {
      stopButton?.click();
    });
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("工作区工作流面板 chrome 文案应跟随 en-US 资源", async () => {
    await changeLimeLocale("en-US");
    const onStop = vi.fn();
    const { container } = renderInputbar({
      variant: "workspace",
      isLoading: true,
      onStop,
      workflowSteps: [
        { id: "research", title: "Review project material", status: "active" },
        { id: "write", title: "Write body draft", status: "pending" },
      ],
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("Current progress");
    expect(container.textContent).toContain("Task queue");
    expect(container.textContent).toContain("In progress · 0/2");
    expect(container.textContent).toContain("Moving forward; 1 item(s) remain");
    expect(container.textContent).not.toContain("当前进展");
    expect(container.textContent).not.toContain("任务队列");

    const stopButton = container.querySelector(
      '[data-testid="workflow-stop"]',
    ) as HTMLButtonElement | null;
    expect(stopButton?.getAttribute("aria-label")).toBe("Stop generation");

    const collapseButton = Array.from(
      container.querySelectorAll("button"),
    ).find(
      (button) => button.getAttribute("aria-label") === "Collapse task queue",
    );
    expect(collapseButton).toBeTruthy();
  });

  it("工作区工作流生成中应支持折叠与展开待办列表", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      isLoading: true,
      workflowSteps: [
        { id: "research", title: "检索项目资料", status: "active" },
        { id: "write", title: "编写正文草稿", status: "pending" },
      ],
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("检索项目资料");

    const collapseButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.getAttribute("aria-label") === "折叠任务队列");
    expect(collapseButton).toBeTruthy();

    act(() => {
      collapseButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelectorAll('[data-testid="workflow-queue-item"]'),
    ).toHaveLength(0);

    const expandButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.getAttribute("aria-label") === "展开任务队列",
    );
    expect(expandButton).toBeTruthy();

    act(() => {
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelectorAll('[data-testid="workflow-queue-item"]'),
    ).toHaveLength(2);
  });

  it("工作区任务队列应展示统一进度计数并按优先级排序", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      isLoading: true,
      workflowSteps: [
        { id: "done", title: "完成选题", status: "completed" },
        { id: "pending", title: "等待补充案例", status: "pending" },
        { id: "active", title: "撰写主稿", status: "active" },
        { id: "error", title: "封面生成失败", status: "error" },
      ],
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("已完成 1/4");

    const queueItems = Array.from(
      container.querySelectorAll('[data-testid="workflow-queue-item"]'),
    );
    expect(queueItems.map((item) => item.getAttribute("data-status"))).toEqual([
      "active",
      "error",
      "pending",
    ]);
  });

  it("工作区工作流在 auto_running 状态下应展示生成面板（不依赖 isLoading）", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      isLoading: false,
      workflowRunState: "auto_running",
      workflowSteps: [
        { id: "research", title: "检索项目资料", status: "active" },
        { id: "write", title: "编写正文草稿", status: "pending" },
      ],
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("当前进展");
    expect(container.textContent).toContain("检索项目资料");
    expect(container.querySelector('[data-testid="inputbar-core"]')).toBeNull();
  });

  it("工作区工作流在 await_user_decision 状态下应显示输入框", async () => {
    const { container } = renderInputbar({
      variant: "workspace",
      isLoading: true,
      workflowRunState: "await_user_decision",
      workflowGate: {
        key: "topic_select",
        title: "等待用户确认选题",
        status: "waiting",
        description: "等待你确认后继续推进下一步。",
      },
      workflowSteps: [
        { id: "topic", title: "等待用户确认选题", status: "pending" },
      ],
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-core"]'),
    ).toBeTruthy();
    expect(container.textContent).toContain("当前进展");
    expect(container.textContent).toContain("等待用户确认选题");
    expect(container.textContent).not.toContain("正在生成中");
  });

  it("对话页输入框底部不应再显示项目上下文条", async () => {
    const { container } = renderInputbar({
      projectId: "project-default",
      openedProjects: [
        {
          id: "project-default",
          name: "默认项目",
          rootPath: "/workspace/default",
        },
      ],
      projectContextModeLabel: "本地模式",
      projectContextBranchLabel: "main",
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-context-bar-slot"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-project-context-bar"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("默认项目");
    expect(container.textContent).not.toContain("本地模式");
    expect(container.textContent).not.toContain("main");
  });
});
