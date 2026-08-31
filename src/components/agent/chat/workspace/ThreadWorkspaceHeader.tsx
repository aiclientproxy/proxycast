import type { ReactNode } from "react";
import {
  ArrowLeft,
  CheckCircle2,
  CircleAlert,
  CircleDot,
  Clock3,
  FolderOpen,
  GitFork,
  LoaderCircle,
  MoreHorizontal,
  PanelRightClose,
  PanelRightOpen,
  Pencil,
  Archive,
  Home,
  Settings,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import type { TaskStatus } from "../hooks/agentChatShared";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

interface ThreadWorkspaceHeaderProps {
  sessionId: string;
  title: string;
  status: TaskStatus | null;
  workingDirectory: string | null;
  canAcceptDirectInput?: boolean | null;
  onRename?: (title: string) => Promise<void>;
  onArchive?: () => Promise<void>;
  onFork?: () => Promise<void>;
  onBackHome?: () => void;
  onBackToResources?: () => void;
  onBackToProjectManagement?: () => void;
  showCanvasToggle?: boolean;
  isCanvasOpen?: boolean;
  onToggleCanvas?: () => void;
  onOpenSettings?: () => void;
  actions?: ReactNode;
}

const statusMeta: Record<
  TaskStatus,
  {
    key: string;
    defaultValue: string;
    className: string;
    Icon: typeof CircleDot;
    animated?: boolean;
  }
> = {
  draft: {
    key: "agentChat.threadTimeline.status.pending",
    defaultValue: "待处理",
    className: "text-[color:var(--lime-text-muted)]",
    Icon: CircleDot,
  },
  running: {
    key: "agentChat.inputbar.runtimeStatus.status.running",
    defaultValue: "处理中",
    className: "text-[color:var(--lime-info)]",
    Icon: LoaderCircle,
    animated: true,
  },
  queued: {
    key: "agentChat.inputbar.runtimeStatus.status.queued",
    defaultValue: "排队中",
    className: "text-sky-700 dark:text-sky-300",
    Icon: Clock3,
  },
  waiting: {
    key: "agentChat.inputbar.runtimeStatus.status.waitingInput",
    defaultValue: "等待补充",
    className: "text-[color:var(--lime-warning)]",
    Icon: CircleAlert,
  },
  done: {
    key: "agentChat.inputbar.runtimeStatus.status.completed",
    defaultValue: "已完成",
    className: "text-[color:var(--lime-text-muted)]",
    Icon: CheckCircle2,
  },
  failed: {
    key: "agentChat.inputbar.runtimeStatus.status.failed",
    defaultValue: "失败",
    className: "text-[color:var(--lime-danger)]",
    Icon: CircleAlert,
  },
};

export function ThreadWorkspaceHeader({
  sessionId,
  title,
  status,
  workingDirectory,
  canAcceptDirectInput = null,
  onRename,
  onArchive,
  onFork,
  onBackHome,
  onBackToResources,
  onBackToProjectManagement,
  showCanvasToggle = false,
  isCanvasOpen = false,
  onToggleCanvas,
  onOpenSettings,
  actions,
}: ThreadWorkspaceHeaderProps) {
  const { t } = useTranslation("agent");
  const { t: tNavigation } = useTranslation("navigation");
  const currentStatus = status ? statusMeta[status] : null;
  const statusLabel = currentStatus
    ? String(
        t(
          currentStatus.key as never,
          {
            defaultValue: currentStatus.defaultValue,
          } as never,
        ),
      )
    : null;
  const StatusIcon = currentStatus?.Icon;
  const hasNavigationActions = Boolean(
    onBackHome ||
    onBackToResources ||
    onBackToProjectManagement ||
    (showCanvasToggle && onToggleCanvas) ||
    onOpenSettings,
  );
  const handleRename = () => {
    if (!onRename || typeof window === "undefined") {
      return;
    }
    const nextTitle = window.prompt(
      String(tNavigation("sidebar.conversations.rename.prompt")),
      title,
    );
    if (nextTitle?.trim()) {
      void onRename(nextTitle);
    }
  };

  return (
    <header
      className="flex h-[52px] min-w-0 items-center gap-3 border-b border-[color:var(--lime-surface-border)] bg-[color:var(--lime-stage-surface,var(--lime-app-bg,#f4f7f1))] px-4"
      data-testid="thread-workspace-header"
      data-session-id={sessionId}
      data-status={status ?? undefined}
      data-can-accept-direct-input={
        canAcceptDirectInput === null ? undefined : String(canAcceptDirectInput)
      }
    >
      {hasNavigationActions ? (
        <div
          className="flex shrink-0 items-center gap-0.5"
          data-testid="thread-workspace-header-navigation"
        >
          {onBackHome ? (
            <button
              type="button"
              className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]"
              onClick={onBackHome}
              aria-label={String(t("agentChat.navbar.backHome"))}
              title={String(t("agentChat.navbar.backHome"))}
            >
              <Home className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          ) : null}
          {onBackToResources ? (
            <button
              type="button"
              className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]"
              onClick={onBackToResources}
              aria-label={String(t("agentChat.navbar.backResources"))}
              title={String(t("agentChat.navbar.backResources"))}
            >
              <FolderOpen className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          ) : null}
          {onBackToProjectManagement ? (
            <button
              type="button"
              className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]"
              onClick={onBackToProjectManagement}
              aria-label={String(t("agentChat.navbar.projectManagement"))}
              title={String(t("agentChat.navbar.projectManagement"))}
            >
              <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          ) : null}
          {showCanvasToggle && onToggleCanvas ? (
            <button
              type="button"
              className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]"
              onClick={onToggleCanvas}
              aria-label={String(
                t(
                  isCanvasOpen
                    ? "agentChat.navbar.collapseCanvas"
                    : "agentChat.navbar.expandCanvas",
                ),
              )}
              title={String(
                t(
                  isCanvasOpen
                    ? "agentChat.navbar.collapseCanvas"
                    : "agentChat.navbar.expandCanvas",
                ),
              )}
            >
              {isCanvasOpen ? (
                <PanelRightClose className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <PanelRightOpen className="h-3.5 w-3.5" aria-hidden="true" />
              )}
            </button>
          ) : null}
          {onOpenSettings ? (
            <button
              type="button"
              className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]"
              onClick={onOpenSettings}
              aria-label={String(t("agentChat.navbar.openSettings"))}
              title={String(t("agentChat.navbar.openSettings"))}
            >
              <Settings className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          ) : null}
        </div>
      ) : null}
      <div className="flex min-w-0 flex-1 items-center gap-2.5">
        <h1
          className="min-w-0 truncate text-[14px] font-semibold leading-5 text-[color:var(--lime-text-strong)]"
          data-testid="thread-workspace-header-title"
          title={title}
        >
          {title}
        </h1>
        {currentStatus && StatusIcon && statusLabel ? (
          <span
            className={cn(
              "inline-flex shrink-0 items-center gap-1 text-[11px] font-medium",
              currentStatus.className,
            )}
            data-testid="thread-workspace-header-status"
          >
            <StatusIcon
              className={cn(
                "h-3.5 w-3.5",
                currentStatus.animated && "animate-spin",
              )}
              aria-hidden="true"
            />
            {statusLabel}
          </span>
        ) : null}
        {workingDirectory ? (
          <span
            className="hidden min-w-0 items-center gap-1 text-[11px] text-[color:var(--lime-text-muted)] min-[900px]:inline-flex"
            data-testid="thread-workspace-header-directory"
            title={workingDirectory}
          >
            <FolderOpen className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span className="max-w-[min(34vw,420px)] truncate">
              {workingDirectory}
            </span>
          </span>
        ) : null}
      </div>
      {actions || onRename || onArchive || onFork ? (
        <div
          className="flex min-w-0 shrink-0 items-center justify-end gap-1"
          data-testid="thread-workspace-header-actions"
        >
          {actions}
          {onRename || onArchive || onFork ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[10px] text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]"
                  aria-label={String(
                    tNavigation("sidebar.conversations.openActionMenu", {
                      title,
                    }),
                  )}
                  title={String(
                    tNavigation("sidebar.conversations.moreActions"),
                  )}
                  data-testid="thread-workspace-header-action-menu"
                >
                  <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-36 p-1">
                {onRename ? (
                  <DropdownMenuItem onClick={handleRename}>
                    <Pencil className="h-3.5 w-3.5" aria-hidden="true" />
                    {String(tNavigation("sidebar.conversations.menu.rename"))}
                  </DropdownMenuItem>
                ) : null}
                {onFork ? (
                  <DropdownMenuItem
                    onClick={() => {
                      void onFork();
                    }}
                  >
                    <GitFork className="h-3.5 w-3.5" aria-hidden="true" />
                    {String(
                      tNavigation("navigation.sidebar.conversations.menu.fork"),
                    )}
                  </DropdownMenuItem>
                ) : null}
                {onArchive ? (
                  <DropdownMenuItem
                    onClick={() => {
                      void onArchive();
                    }}
                  >
                    <Archive className="h-3.5 w-3.5" aria-hidden="true" />
                    {String(tNavigation("sidebar.conversations.menu.archive"))}
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
        </div>
      ) : null}
    </header>
  );
}
