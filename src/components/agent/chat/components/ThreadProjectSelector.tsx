import { useMemo, useState } from "react";
import { Check, ChevronDown, FolderTree, Plus, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useThreadProjectDirectory } from "../hooks/useThreadProjectDirectory";

interface ThreadProjectSelectorProps {
  threadId?: string | null;
  workspaceName?: string | null;
  workspaceRootPath?: string | null;
  compact?: boolean;
  className?: string;
}

export function ThreadProjectSelector({
  threadId,
  workspaceName,
  workspaceRootPath,
  compact = false,
  className,
}: ThreadProjectSelectorProps) {
  const { t } = useTranslation("agent");
  const [open, setOpen] = useState(false);
  const directory = useThreadProjectDirectory({ threadId });
  const selectedProject = useMemo(
    () =>
      directory.projects.find(
        (project) => project.id === directory.projectId,
      ) ?? null,
    [directory.projectId, directory.projects],
  );
  const hasWorkspaceRoot = Boolean(workspaceRootPath?.trim());
  const currentLabel = selectedProject?.name
    ? selectedProject.name
    : directory.projectId
      ? t("agentChat.threadProjectSelector.assigned", "已归入项目")
      : t("agentChat.threadProjectSelector.unassigned", "未归入项目");
  const buttonLabel = t(
    "agentChat.threadProjectSelector.open",
    "对话项目：{{project}}",
    { project: currentLabel },
  );
  const errorLabel = t(
    "agentChat.threadProjectSelector.loadFailed",
    "项目目录暂时不可用",
  );

  if (!threadId?.trim()) {
    return null;
  }

  const runAction = (action: () => Promise<void>) => {
    void action().catch(() => undefined);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className={cn(
            "h-8 min-w-0 max-w-[220px] shrink-0 gap-1.5 rounded-[14px] border border-transparent px-2.5 text-xs font-medium text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)] hover:text-[color:var(--lime-text)]",
            compact && "max-w-[180px]",
            className,
          )}
          aria-label={buttonLabel}
          title={buttonLabel}
          data-testid="thread-project-selector"
          data-thread-project-id={directory.projectId ?? ""}
          disabled={directory.mutating}
        >
          <FolderTree className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span className="min-w-0 truncate">{currentLabel}</span>
          <ChevronDown className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        side="bottom"
        sideOffset={8}
        className="w-[min(320px,calc(100vw-24px))] p-2"
        data-testid="thread-project-directory"
      >
        <div className="flex items-center justify-between gap-2 px-2 pb-2">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-[color:var(--lime-text-strong)]">
              {t("agentChat.threadProjectSelector.title", "对话项目")}
            </p>
            <p className="truncate text-[11px] text-[color:var(--lime-text-muted)]">
              {t(
                "agentChat.threadProjectSelector.subtitle",
                "当前对话的项目归属",
              )}
            </p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0 rounded-[10px] text-[color:var(--lime-text-muted)]"
            onClick={() => runAction(directory.refresh)}
            aria-label={t(
              "agentChat.threadProjectSelector.refresh",
              "刷新项目目录",
            )}
            title={t("agentChat.threadProjectSelector.refresh", "刷新项目目录")}
            disabled={directory.loading || directory.mutating}
          >
            <RefreshCw
              className={cn("h-3.5 w-3.5", directory.loading && "animate-spin")}
            />
          </Button>
        </div>

        {directory.error ? (
          <div className="mx-1 mb-2 flex items-center justify-between gap-2 rounded-[10px] border border-[color:var(--lime-danger-border)] bg-[color:var(--lime-danger-surface)] px-2.5 py-2 text-xs text-[color:var(--lime-danger)]">
            <span>{errorLabel}</span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-6 shrink-0 rounded-[8px] px-2 text-xs"
              onClick={() => runAction(directory.refresh)}
              disabled={directory.loading || directory.mutating}
            >
              {t("agentChat.threadProjectSelector.retry", "重试")}
            </Button>
          </div>
        ) : null}

        <div className="max-h-[280px] space-y-1 overflow-y-auto">
          <ProjectOption
            active={!directory.projectId}
            disabled={directory.mutating}
            projectId=""
            label={t("agentChat.threadProjectSelector.noProject", "不归入项目")}
            onClick={() =>
              runAction(async () => {
                await directory.assign(null);
                setOpen(false);
              })
            }
          />
          {directory.projects.map((project) => (
            <ProjectOption
              key={project.id}
              active={project.id === directory.projectId}
              disabled={directory.mutating}
              projectId={project.id}
              label={project.name}
              path={project.roots[0]?.path}
              onClick={() =>
                runAction(async () => {
                  await directory.assign(project.id);
                  setOpen(false);
                })
              }
            />
          ))}
        </div>

        {!directory.loading && directory.projects.length === 0 ? (
          <p className="px-2 py-3 text-xs text-[color:var(--lime-text-muted)]">
            {t("agentChat.threadProjectSelector.empty", "项目目录为空")}
          </p>
        ) : null}

        {hasWorkspaceRoot ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-2 h-8 w-full justify-start gap-1.5 rounded-[10px] text-xs"
            onClick={() =>
              runAction(async () => {
                await directory.createAndAssign({
                  name:
                    workspaceName?.trim() ||
                    workspaceRootPath
                      ?.split(/[\\/]+/)
                      .filter(Boolean)
                      .at(-1) ||
                    "当前文件夹",
                  rootPath: workspaceRootPath!.trim(),
                });
                setOpen(false);
              })
            }
            disabled={directory.mutating}
            data-testid="thread-project-create-from-workspace"
          >
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            {t(
              "agentChat.threadProjectSelector.createFromWorkspace",
              "将当前文件夹加入项目",
            )}
          </Button>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

function ProjectOption({
  active,
  disabled,
  label,
  path,
  projectId,
  onClick,
}: {
  active: boolean;
  disabled: boolean;
  label: string;
  path?: string;
  projectId: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="flex min-h-10 w-full items-center gap-2 rounded-[10px] px-2.5 py-2 text-left text-sm text-[color:var(--lime-text)] hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:opacity-60"
      onClick={onClick}
      disabled={disabled}
      data-project-id={projectId}
    >
      <Check
        className={cn(
          "h-3.5 w-3.5 shrink-0",
          active ? "opacity-100" : "opacity-0",
        )}
        aria-hidden="true"
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate">{label}</span>
        {path ? (
          <span className="block truncate text-[11px] text-[color:var(--lime-text-muted)]">
            {path}
          </span>
        ) : null}
      </span>
    </button>
  );
}
