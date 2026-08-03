import {
  ArrowLeftRight,
  Check,
  CloudCog,
  CloudOff,
  ExternalLink,
  Monitor,
} from "lucide-react";
import { PopoverContent } from "@/components/ui/popover";

export interface TaskCenterLocationPanelProps {
  canOpenLocal: boolean;
  onOpenLocal: () => void;
  onOpenCodexWeb: () => void;
  canCreateWorktree: boolean;
  onCreateWorktree: () => void;
  translate: (key: string, options?: Record<string, unknown>) => unknown;
}

function text(
  translate: TaskCenterLocationPanelProps["translate"],
  key: string,
  defaultValue: string,
): string {
  return String(translate(key, { defaultValue }));
}

export function TaskCenterLocationPanel({
  canOpenLocal,
  onOpenLocal,
  onOpenCodexWeb,
  canCreateWorktree,
  onCreateWorktree,
  translate,
}: TaskCenterLocationPanelProps) {
  const localUnavailable = text(
    translate,
    "agentChat.navbar.appSwitcher.localUnavailable",
    "当前项目缺少本地目录",
  );

  return (
    <PopoverContent
      align="end"
      sideOffset={8}
      className="w-[min(216px,calc(100vw-1rem))] rounded-xl border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-2 text-[color:var(--lime-text)] shadow-xl shadow-slate-950/10"
      data-testid="task-center-app-switcher-popover"
    >
      <p className="px-1.5 pb-1 text-[11px] font-medium text-[color:var(--lime-text-muted)]">
        {text(
          translate,
          "agentChat.navbar.appSwitcher.continueUsing",
          "继续使用",
        )}
      </p>

      <button
        type="button"
        className="flex w-full min-w-0 items-center gap-2 rounded-lg px-1.5 py-1.5 text-left text-[13px] leading-4 transition hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-transparent"
        disabled={!canOpenLocal}
        title={canOpenLocal ? undefined : localUnavailable}
        onClick={onOpenLocal}
        data-testid="task-center-location-local"
      >
        <Monitor className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
        <span className="min-w-0 flex-1 truncate">
          {text(translate, "agentChat.navbar.appSwitcher.local", "在本地处理")}
        </span>
        <Check className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
      </button>

      <button
        type="button"
        className="flex w-full min-w-0 items-center gap-2 rounded-lg px-1.5 py-1.5 text-left text-[13px] leading-4 transition hover:bg-[color:var(--lime-surface-hover)]"
        onClick={onOpenCodexWeb}
        data-testid="task-center-location-codex-web"
      >
        <CloudCog className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
        <span className="min-w-0 flex-1 truncate">
          {text(
            translate,
            "agentChat.navbar.appSwitcher.codexWeb",
            "关联 Codex web",
          )}
        </span>
        <ExternalLink className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
      </button>

      <button
        type="button"
        disabled
        className="flex w-full min-w-0 cursor-not-allowed items-center gap-2 rounded-lg px-1.5 py-1.5 text-left text-[13px] leading-4 text-[color:var(--lime-text-muted)] opacity-45"
        title={text(
          translate,
          "agentChat.navbar.appSwitcher.cloudUnavailable",
          "云端同步尚未接入",
        )}
        data-testid="task-center-location-cloud"
      >
        <CloudOff className="h-3.5 w-3.5 shrink-0" />
        <span className="min-w-0 flex-1 truncate">
          {text(translate, "agentChat.navbar.appSwitcher.cloud", "发送至云端")}
        </span>
      </button>

      <div className="my-1 border-t border-[color:var(--lime-surface-border)]" />

      <button
        type="button"
        disabled={!canCreateWorktree}
        className="flex w-full min-w-0 items-center gap-2 rounded-lg px-1.5 py-1.5 text-left text-[13px] leading-4 transition hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:text-[color:var(--lime-text-muted)] disabled:opacity-55 disabled:hover:bg-transparent"
        title={
          canCreateWorktree
            ? undefined
            : text(
                translate,
                "agentChat.navbar.appSwitcher.worktreeUnavailable",
                "请先选择本地项目目录",
              )
        }
        onClick={onCreateWorktree}
        data-testid="task-center-location-worktree"
      >
        <ArrowLeftRight className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
        <span className="min-w-0 flex-1 truncate">
          {text(translate, "agentChat.navbar.appSwitcher.worktree", "工作树")}
        </span>
      </button>
    </PopoverContent>
  );
}
