import {
  MessageSquareText,
  PanelRightClose,
  PanelRightOpen,
  Pencil,
  Plus,
  Loader2,
  X,
} from "lucide-react";
import type { CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import type { TaskStatus } from "../hooks/agentChatShared";
import { cn } from "@/lib/utils";
import {
  TASK_CENTER_CHROME_STAGE_BLEND,
  TASK_CENTER_CHROME_STAGE_SEAM,
} from "../workspace/taskCenterChromeTokens";
import {
  buildFileNameTabTooltip,
  resolveFileNameTabLabel,
} from "../utils/tabFileDisplay";

const TASK_CENTER_TAB_STATUS_META: Record<
  TaskStatus,
  { label: string; iconClassName: string }
> = {
  draft: {
    label: "待补充",
    iconClassName: "text-[color:var(--lime-text-muted)]",
  },
  running: {
    label: "进行中",
    iconClassName: "text-[color:var(--lime-info)]",
  },
  queued: {
    label: "排队中",
    iconClassName: "text-sky-600 dark:text-sky-300",
  },
  waiting: {
    label: "待继续",
    iconClassName: "text-[color:var(--lime-warning)]",
  },
  done: {
    label: "已完成",
    iconClassName: "text-[color:var(--lime-brand-strong)]",
  },
  failed: {
    label: "有异常",
    iconClassName: "text-[color:var(--lime-danger)]",
  },
};

export interface TaskCenterTabItem {
  id: string;
  title: string;
  status: TaskStatus;
  updatedAt: Date;
  isActive: boolean;
  hasUnread: boolean;
  renamable?: boolean;
  closable?: boolean;
}

interface TaskCenterTabStripProps {
  items: TaskCenterTabItem[];
  onSelectTask: (taskId: string) => void | Promise<void>;
  onRenameTask?: (taskId: string) => void | Promise<void>;
  onCloseTask: (taskId: string) => void | Promise<void>;
  onCreateTask: () => void;
  showWorkbenchToggle?: boolean;
  workbenchVisible?: boolean;
  onWorkbenchToggle?: () => void;
  embedded?: boolean;
}

function formatTaskTabTitle(
  item: TaskCenterTabItem,
  displayTitle: string,
): string {
  const statusMeta =
    TASK_CENTER_TAB_STATUS_META[item.status] ??
    TASK_CENTER_TAB_STATUS_META.done;
  return `${buildFileNameTabTooltip({
    label: displayTitle,
    source: item.title,
  })} · ${statusMeta.label} · 更新于 ${item.updatedAt.toLocaleString("zh-CN")}`;
}

const conversationTabShellClassName =
  "group flex h-[28px] items-center gap-0 rounded-[14px] border border-transparent px-1 transition-[background-color,border-color,box-shadow,color] duration-150 ease-out";

const activeConversationTabClassName =
  "border-[color:var(--lime-chrome-divider)] bg-[color:var(--lime-chrome-tab-hover)] text-[color:var(--lime-text-strong)] shadow-[inset_0_1px_0_rgba(255,255,255,0.5)] dark:bg-slate-700 dark:text-slate-100";

const inactiveConversationTabClassName =
  "bg-transparent text-[color:var(--lime-chrome-muted)] hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text)] dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200";

const conversationTabButtonClassName =
  "flex h-full min-w-0 items-center gap-1 rounded-[13px] px-1.5 text-left";

const tabUtilityButtonClassName =
  "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[14px] bg-transparent text-[color:var(--lime-chrome-muted)] transition-colors hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text)] dark:text-slate-400 dark:hover:bg-white/10 dark:hover:text-slate-200";

const tabWorkbenchButtonClassName =
  "inline-flex h-7 shrink-0 items-center gap-1 rounded-[14px] border border-transparent px-2 text-[11px] font-medium text-[color:var(--lime-text-muted)] transition-[background-color,border-color,box-shadow,color] hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text)] dark:text-slate-400 dark:hover:bg-white/10 dark:hover:text-slate-100";

const taskCenterTabStripStyle = {
  "--task-center-tab-strip-background": TASK_CENTER_CHROME_STAGE_BLEND,
  "--task-center-tab-strip-seam": TASK_CENTER_CHROME_STAGE_SEAM,
  background: "var(--task-center-tab-strip-background)",
  borderBottomColor: "var(--task-center-tab-strip-seam)",
  boxShadow:
    "inset 0 1px 0 rgba(255, 255, 255, 0.34), 0 12px 24px -34px var(--lime-shadow-color)",
} as CSSProperties;

export function TaskCenterTabStrip({
  items,
  onSelectTask,
  onRenameTask,
  onCloseTask,
  onCreateTask,
  showWorkbenchToggle = false,
  workbenchVisible = false,
  onWorkbenchToggle,
  embedded = false,
}: TaskCenterTabStripProps) {
  const { t: tNavigation } = useTranslation("navigation");
  const { t: tAgent } = useTranslation("agent");
  const showToolbarActions = showWorkbenchToggle;
  const renameActionLabel = tNavigation(
    "navigation.sidebar.conversations.menu.rename",
  );
  const createConversationLabel = tNavigation(
    "navigation.sidebar.conversations.newConversation",
  );
  const closeTabLabel = (label: string) =>
    tNavigation("navigation.sidebar.conversations.closeTab", {
      label,
      defaultValue: "关闭 {{label}}",
    });
  const workbenchLabel = tNavigation(
    "navigation.sidebar.conversations.workbench.label",
  );
  const workbenchToggleLabel = workbenchVisible
    ? tNavigation("navigation.sidebar.conversations.workbench.collapse")
    : tNavigation("navigation.sidebar.conversations.workbench.expand");
  const openChatLabel = tAgent("agentChat.navbar.openChat");

  return (
    <section
      className={cn(
        "relative z-10 min-h-[42px] shrink-0 bg-[color:var(--lime-chrome-tab-active-surface)] px-4 pb-2 pt-1.5",
        !embedded && "border-b border-[color:var(--lime-chrome-divider)]",
      )}
      data-testid="task-center-tab-strip"
      role="tablist"
      style={embedded ? undefined : taskCenterTabStripStyle}
    >
      <div className="flex items-center gap-1">
        <div className="min-w-0 flex-1 overflow-x-auto [scrollbar-width:none]">
          <div className="flex min-w-max items-center gap-1">
            {items.map((item) => {
              const statusMeta =
                TASK_CENTER_TAB_STATUS_META[item.status] ??
                TASK_CENTER_TAB_STATUS_META.done;
              const displayTitle =
                resolveFileNameTabLabel(item.title) || openChatLabel;
              const tabTitle = formatTaskTabTitle(item, displayTitle);
              const closeLabel = closeTabLabel(displayTitle);

              return (
                <div
                  key={item.id}
                  className={cn(
                    conversationTabShellClassName,
                    item.isActive
                      ? activeConversationTabClassName
                      : inactiveConversationTabClassName,
                  )}
                  data-testid={`task-center-tab-${item.id}`}
                  data-active={item.isActive ? "true" : "false"}
                >
                  <button
                    type="button"
                    className={conversationTabButtonClassName}
                    role="tab"
                    aria-selected={item.isActive ? "true" : "false"}
                    title={tabTitle}
                    onClick={() => {
                      void onSelectTask(item.id);
                    }}
                  >
                    {item.status === "running" ? (
                      <Loader2
                        className={cn(
                          "h-3.5 w-3.5 shrink-0 animate-spin",
                          statusMeta.iconClassName,
                        )}
                        data-testid={`task-center-tab-loading-${item.id}`}
                        aria-label={statusMeta.label}
                      />
                    ) : (
                      <MessageSquareText
                        className={cn(
                          "h-3.5 w-3.5 shrink-0",
                          statusMeta.iconClassName,
                        )}
                        aria-hidden="true"
                      />
                    )}
                    <span className="truncate text-[11px] font-semibold">
                      {displayTitle}
                    </span>
                    {item.hasUnread ? (
                      <span
                        className="h-2 w-2 shrink-0 rounded-full bg-[color:var(--lime-brand)]"
                        data-testid={`task-center-tab-unread-${item.id}`}
                        aria-hidden="true"
                      />
                    ) : null}
                  </button>
                  {item.renamable !== false && onRenameTask ? (
                    <button
                      type="button"
                      className={cn(
                        "rounded-full p-1 text-[color:var(--lime-text-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text)] focus-visible:text-[color:var(--lime-text)] dark:hover:bg-white/10 dark:hover:text-slate-200",
                        item.isActive
                          ? "opacity-100"
                          : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                      )}
                      aria-label={renameActionLabel}
                      title={renameActionLabel}
                      data-testid={`task-center-tab-rename-${item.id}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        void onRenameTask(item.id);
                      }}
                    >
                      <Pencil className="h-2.5 w-2.5" />
                    </button>
                  ) : null}
                  {item.closable === false ? null : (
                    <button
                      type="button"
                      className={cn(
                        "mr-1 rounded-full p-1 text-[color:var(--lime-text-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text)] focus-visible:text-[color:var(--lime-text)] dark:hover:bg-white/10 dark:hover:text-slate-200",
                        item.isActive
                          ? "opacity-100"
                          : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                      )}
                      aria-label={closeLabel}
                      title={closeLabel}
                      data-testid={`task-center-tab-close-${item.id}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        void onCloseTask(item.id);
                      }}
                    >
                      <X className="h-2.5 w-2.5" />
                    </button>
                  )}
                </div>
              );
            })}

            <button
              type="button"
              className={cn(tabUtilityButtonClassName, "ml-0.5")}
              data-testid="task-center-tab-create-button"
              aria-label={createConversationLabel}
              title={createConversationLabel}
              onClick={onCreateTask}
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {showToolbarActions ? (
          <div
            className="flex shrink-0 items-center gap-1 border-l border-[color:var(--lime-chrome-divider)] pl-1.5 dark:border-slate-700/80"
            data-testid="task-center-tab-toolbar"
          >
            {showWorkbenchToggle ? (
              <button
                type="button"
                className={cn(
                  tabWorkbenchButtonClassName,
                  workbenchVisible &&
                    "border-[color:var(--lime-chrome-border)] bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)] shadow-[inset_0_1px_0_rgba(255,255,255,0.72)] dark:bg-slate-700 dark:text-slate-100",
                )}
                data-testid="task-center-tab-workbench"
                aria-label={workbenchToggleLabel}
                title={workbenchToggleLabel}
                onClick={onWorkbenchToggle}
              >
                {workbenchVisible ? (
                  <PanelRightClose className="h-3.5 w-3.5" />
                ) : (
                  <PanelRightOpen className="h-3.5 w-3.5" />
                )}
                <span>{workbenchLabel}</span>
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
