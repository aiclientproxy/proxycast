import {
  Check,
  ChevronDown,
  CircleAlert,
  CircleCheck,
  CircleDot,
  GitBranch,
  GitCommitHorizontal,
  GitCompare,
  ExternalLink,
  Loader2,
  Monitor,
  Plus,
  Search,
  Server,
} from "lucide-react";
import React from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { ProjectGitStatus } from "@/lib/api/projectGit";
import type { ThreadEnvironmentLifecycleState } from "../hooks/useThreadEnvironmentLifecycleStatus";

export interface TaskCenterEnvironmentPanelProps {
  normalizedProjectRootPath: string | null;
  status: ProjectGitStatus | null;
  environmentStatusLabel: string;
  lifecycleStatuses?: readonly ThreadEnvironmentLifecycleState[];
  changeSummary?: { additions: number; deletions: number } | null;
  branchLabel: string;
  changeCount: number;
  onCheckoutBranch?: (branch: string) => void | Promise<void>;
  onCreateBranch?: (branch: string) => void | Promise<void>;
  translate: (key: string, options?: Record<string, unknown>) => unknown;
}

function text(
  translate: TaskCenterEnvironmentPanelProps["translate"],
  key: string,
  defaultValue: string,
  options?: Record<string, unknown>,
): string {
  return String(translate(key, { defaultValue, ...options }));
}

export function TaskCenterEnvironmentPanel({
  normalizedProjectRootPath,
  status,
  environmentStatusLabel,
  lifecycleStatuses = [],
  changeSummary,
  branchLabel,
  changeCount,
  onCheckoutBranch,
  onCreateBranch,
  translate,
}: TaskCenterEnvironmentPanelProps) {
  const hasGitRepository = Boolean(status?.hasGitRepository);
  const [branchMenuOpen, setBranchMenuOpen] = React.useState(false);
  const [branchSearchQuery, setBranchSearchQuery] = React.useState("");
  const branches = React.useMemo(() => {
    if (!hasGitRepository) {
      return [];
    }
    const options = (status?.branches ?? []).filter(
      (branch) => branch !== branchLabel,
    );
    return branchLabel ? [branchLabel, ...options] : options;
  }, [branchLabel, hasGitRepository, status?.branches]);
  const filteredBranches = branchSearchQuery.trim()
    ? branches.filter((branch) =>
        branch.toLowerCase().includes(branchSearchQuery.trim().toLowerCase()),
      )
    : branches;
  const canCreateBranch =
    branchSearchQuery.trim().length > 0 &&
    !branches.includes(branchSearchQuery.trim());
  const visibleLifecycleStatuses =
    lifecycleStatuses.length > 0
      ? lifecycleStatuses
      : ([
          { environmentId: "local", status: "connected" },
        ] satisfies ThreadEnvironmentLifecycleState[]);

  return (
    <PopoverContent
      align="end"
      sideOffset={8}
      className="w-[min(300px,calc(100vw-1rem))] rounded-2xl border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-3 py-2.5 text-[color:var(--lime-text)] shadow-xl shadow-slate-950/10"
      data-testid="task-center-environment-popover"
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span className="min-w-0 truncate text-xs font-medium text-[color:var(--lime-text-muted)]">
          {text(translate, "agentChat.navbar.environment.title", "环境信息")}
        </span>
        <button
          type="button"
          disabled
          className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-[color:var(--lime-text-faint)] disabled:cursor-not-allowed disabled:opacity-75"
          aria-label={text(
            translate,
            "agentChat.navbar.environment.addUnavailable",
            "添加环境",
          )}
          title={text(
            translate,
            "agentChat.navbar.environment.addUnavailable",
            "添加环境",
          )}
          data-testid="task-center-environment-add"
        >
          <Plus className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      </div>

      <div className="mt-2.5 space-y-0.5 text-[13px]">
        <div
          className="flex min-w-0 items-center justify-between gap-2 rounded-lg px-1.5 py-1.5 leading-4"
          data-testid="task-center-environment-changes"
        >
          <span className="flex min-w-0 items-center gap-2">
            <GitCommitHorizontal className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
            <span className="truncate">
              {text(translate, "agentChat.navbar.environment.changes", "变更")}
            </span>
          </span>
          <span className="shrink-0 text-xs text-[color:var(--lime-text-muted)]">
            {hasGitRepository && changeSummary ? (
              <span className="inline-flex items-center gap-1.5 font-medium">
                <span className="text-emerald-600">
                  +{changeSummary.additions.toLocaleString()}
                </span>{" "}
                <span className="text-rose-600">
                  -{changeSummary.deletions.toLocaleString()}
                </span>
              </span>
            ) : hasGitRepository ? (
              text(
                translate,
                "agentChat.navbar.environment.uncommittedFiles",
                "{{count}} 个文件",
                { count: changeCount },
              )
            ) : (
              environmentStatusLabel
            )}
          </span>
        </div>

        <div className="border-y border-[color:var(--lime-surface-border)] py-1">
          <div className="px-1.5 pb-1 pt-0.5 text-[11px] font-medium text-[color:var(--lime-text-faint)]">
            {text(
              translate,
              "agentChat.navbar.environment.runtimeTitle",
              "运行环境",
            )}
          </div>
          {visibleLifecycleStatuses.map((environment) => {
            const isLocal = environment.environmentId === "local";
            const statusLabel =
              environment.status === "connected"
                ? text(
                    translate,
                    "agentChat.navbar.environment.status.connected",
                    "已连接",
                  )
                : environment.status === "disconnected"
                  ? text(
                      translate,
                      "agentChat.navbar.environment.status.disconnected",
                      "连接已断开",
                    )
                  : text(
                      translate,
                      "agentChat.navbar.environment.status.pending",
                      "连接中",
                    );
            const StatusIcon =
              environment.status === "connected"
                ? CircleCheck
                : environment.status === "disconnected"
                  ? CircleAlert
                  : Loader2;
            return (
              <div
                key={environment.environmentId}
                className="flex min-h-8 w-full min-w-0 items-center gap-2 rounded-lg px-1.5 py-1.5 leading-4"
                data-testid={
                  isLocal
                    ? "task-center-environment-local"
                    : "task-center-environment-runtime"
                }
                data-environment-id={environment.environmentId}
                data-environment-status={environment.status}
                data-protocol-method={
                  lifecycleStatuses.length === 0
                    ? undefined
                    : environment.status === "disconnected"
                      ? "thread/environment/disconnected"
                      : environment.status === "connected"
                        ? "thread/environment/connected"
                        : undefined
                }
              >
                {isLocal ? (
                  <Monitor className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
                ) : (
                  <Server className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
                )}
                <span
                  className="min-w-0 flex-1 truncate"
                  title={environment.environmentId}
                >
                  {isLocal
                    ? text(
                        translate,
                        "agentChat.navbar.environment.local",
                        "本地",
                      )
                    : environment.environmentId}
                </span>
                <span
                  className={`inline-flex shrink-0 items-center gap-1 text-xs ${
                    environment.status === "connected"
                      ? "text-emerald-600 dark:text-emerald-400"
                      : environment.status === "disconnected"
                        ? "text-rose-600 dark:text-rose-400"
                        : "text-amber-600 dark:text-amber-400"
                  }`}
                >
                  <StatusIcon
                    className={`h-3.5 w-3.5 ${
                      environment.status === "pending" ? "animate-spin" : ""
                    }`}
                    aria-hidden="true"
                  />
                  {statusLabel}
                </span>
              </div>
            );
          })}
        </div>

        {hasGitRepository ? (
          <Popover open={branchMenuOpen} onOpenChange={setBranchMenuOpen}>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="flex w-full min-w-0 items-center gap-2 rounded-lg px-1.5 py-1.5 text-left leading-4 transition hover:bg-[color:var(--lime-surface-hover)]"
                data-testid="task-center-environment-branch"
              >
                <GitBranch className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
                <span className="min-w-0 flex-1 truncate">{branchLabel}</span>
                <ChevronDown className="h-3 w-3 shrink-0 text-[color:var(--lime-text-faint)]" />
              </button>
            </PopoverTrigger>
            <PopoverContent
              align="start"
              side="right"
              sideOffset={8}
              className="z-[80] w-[min(19rem,calc(100vw-1rem))] rounded-xl border border-slate-200 bg-white p-1.5 text-slate-900 shadow-xl shadow-slate-950/10"
              data-testid="task-center-environment-branch-menu"
            >
              <div className="flex h-8 items-center gap-2 rounded-lg border border-slate-200 bg-white px-2">
                <Search className="h-3.5 w-3.5 text-slate-400" />
                <input
                  autoFocus
                  className="min-w-0 flex-1 bg-transparent text-[13px] font-medium text-slate-900 outline-none placeholder:text-slate-400"
                  value={branchSearchQuery}
                  placeholder={text(
                    translate,
                    "agentChat.navbar.environment.branchSearchPlaceholder",
                    "搜索分支",
                  )}
                  onChange={(event) => setBranchSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && canCreateBranch) {
                      event.preventDefault();
                      void onCreateBranch?.(branchSearchQuery.trim());
                    }
                  }}
                />
              </div>
              <div className="px-2 pb-1 pt-2 text-[11px] font-semibold text-slate-500">
                {text(
                  translate,
                  "agentChat.navbar.environment.branchMenuTitle",
                  "分支",
                )}
              </div>
              <div className="max-h-[220px] overflow-auto pb-1">
                {filteredBranches.map((branch) => (
                  <button
                    key={branch}
                    type="button"
                    className="flex min-h-10 w-full min-w-0 items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] font-medium text-slate-800 transition hover:bg-slate-50"
                    onClick={() => {
                      setBranchMenuOpen(false);
                      void onCheckoutBranch?.(branch);
                    }}
                  >
                    <GitBranch className="h-4 w-4 shrink-0 text-slate-500" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate">{branch}</span>
                      {branch === branchLabel && changeCount > 0 ? (
                        <span className="block text-[11px] font-normal text-slate-500">
                          {text(
                            translate,
                            "agentChat.navbar.environment.branchUncommittedFiles",
                            "未提交：{{count}} 个文件",
                            { count: changeCount },
                          )}
                        </span>
                      ) : null}
                    </span>
                    {branch === branchLabel ? (
                      <Check className="h-3.5 w-3.5 shrink-0 text-slate-500" />
                    ) : null}
                  </button>
                ))}
              </div>
              <div className="my-1 h-px bg-slate-100" />
              <button
                type="button"
                className={`flex min-h-9 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium transition ${canCreateBranch ? "text-slate-700 hover:bg-slate-50 hover:text-slate-950" : "text-slate-400"}`}
                disabled={!canCreateBranch}
                onClick={() => {
                  setBranchMenuOpen(false);
                  void onCreateBranch?.(branchSearchQuery.trim());
                }}
                data-testid="task-center-environment-create-branch"
              >
                <Plus className="h-4 w-4 text-slate-400" />
                <span className="min-w-0 flex-1 truncate">
                  {canCreateBranch
                    ? text(
                        translate,
                        "agentChat.navbar.environment.branchCreateNamedAction",
                        "创建并检出 {{branch}}",
                        { branch: branchSearchQuery.trim() },
                      )
                    : text(
                        translate,
                        "agentChat.navbar.environment.branchCreateAction",
                        "创建并检出新分支...",
                      )}
                </span>
              </button>
            </PopoverContent>
          </Popover>
        ) : (
          <div
            className="flex w-full min-w-0 items-center gap-2 rounded-lg px-1.5 py-1.5 leading-4"
            data-testid="task-center-environment-branch"
          >
            <GitBranch className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
            <span className="min-w-0 flex-1 truncate">{branchLabel}</span>
            <ChevronDown className="h-3 w-3 shrink-0 text-[color:var(--lime-text-faint)]" />
          </div>
        )}

        <button
          type="button"
          className="flex w-full min-w-0 cursor-not-allowed items-center gap-2 rounded-lg px-1.5 py-1.5 text-left leading-4 text-[color:var(--lime-text-muted)] opacity-70"
          disabled
          title={text(
            translate,
            "agentChat.navbar.environment.submitUnavailable",
            "提交和推送需要后续接入 Git 写操作",
          )}
          data-testid="task-center-environment-submit"
        >
          <CircleDot className="h-3.5 w-3.5 shrink-0" />
          <span>
            {text(
              translate,
              "agentChat.navbar.environment.submit",
              "提交或推送",
            )}
          </span>
        </button>

        <button
          type="button"
          className="flex w-full min-w-0 cursor-not-allowed items-center gap-2 rounded-lg px-1.5 py-1.5 text-left leading-4 text-[color:var(--lime-text-muted)] opacity-70"
          disabled
          title={text(
            translate,
            "agentChat.navbar.environment.compareUnavailable",
            "比较分支需要先选择对比目标",
          )}
          data-testid="task-center-environment-compare"
        >
          <GitCompare className="h-3.5 w-3.5 shrink-0" />
          <span>
            {text(
              translate,
              "agentChat.navbar.environment.compare",
              "比较分支",
            )}
          </span>
          <ExternalLink className="ml-auto h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-faint)]" />
        </button>
      </div>

      {!normalizedProjectRootPath ? (
        <span className="sr-only">{environmentStatusLabel}</span>
      ) : null}
    </PopoverContent>
  );
}
