/**
 * 创建项目对话框
 *
 * 用于创建新项目
 */

import { useState, useEffect, useMemo } from "react";
import { open as openDialog } from "@/lib/desktop-host/dialog";
import { useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import {
  ProjectType,
  USER_PROJECT_TYPES,
  extractErrorMessage,
  getCreateProjectErrorMessage,
  getProjectTypeIcon,
  getProjectByRootPath,
  getWorkspaceProjectsRoot,
  resolveProjectRootPath,
} from "@/lib/api/project";
import { toast } from "sonner";

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (
    name: string,
    type: ProjectType,
    rootPath: string,
  ) => Promise<void>;
  defaultType?: ProjectType;
  defaultName?: string;
  allowedTypes?: ProjectType[];
}

type UserProjectType = (typeof USER_PROJECT_TYPES)[number];

export function CreateProjectDialog({
  open,
  onOpenChange,
  onSubmit,
  defaultType,
  defaultName,
  allowedTypes,
}: CreateProjectDialogProps) {
  const { t } = useTranslation("common");
  const [name, setName] = useState("");
  const [type, setType] = useState<UserProjectType>("general");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [parentRootPath, setParentRootPath] = useState("");
  const [resolvedProjectPath, setResolvedProjectPath] = useState("");
  const [pathChecking, setPathChecking] = useState(false);
  const [pathConflictMessage, setPathConflictMessage] = useState("");
  const [pathErrorMessage, setPathErrorMessage] = useState("");
  const [isChoosingParentPath, setIsChoosingParentPath] = useState(false);

  const visibleTypes = useMemo(() => {
    const candidates =
      allowedTypes && allowedTypes.length > 0
        ? allowedTypes
        : USER_PROJECT_TYPES;

    return candidates.filter((candidate): candidate is UserProjectType =>
      USER_PROJECT_TYPES.includes(
        candidate as (typeof USER_PROJECT_TYPES)[number],
      ),
    );
  }, [allowedTypes]);

  const fallbackType = visibleTypes[0] || "general";
  const showTypeSelector = visibleTypes.length > 1;
  const projectTypeLabels: Record<UserProjectType, string> = {
    general: t("common.createProjectDialog.projectType.general"),
  };
  const getProjectTypeDisplayLabel = (projectType: UserProjectType) =>
    projectTypeLabels[projectType];
  const createProjectErrorMessageCopy = useMemo(
    () => ({
      invalidPath: t("common.createProjectDialog.error.invalidPath"),
      objectError: t("common.createProjectDialog.error.object"),
      pathExists: t("common.createProjectDialog.error.pathExists"),
      staleSchema: t("common.createProjectDialog.error.staleSchema"),
      unknown: t("common.createProjectDialog.error.unknown"),
    }),
    [t],
  );

  // 当对话框打开且 defaultType 变化时，更新类型选择
  useEffect(() => {
    if (!open) {
      return;
    }

    if (
      defaultType &&
      USER_PROJECT_TYPES.includes(defaultType as UserProjectType) &&
      visibleTypes.includes(defaultType as UserProjectType)
    ) {
      setType(defaultType as UserProjectType);
      return;
    }

    if (!visibleTypes.includes(type)) {
      setType(fallbackType);
    }
  }, [defaultType, fallbackType, open, type, visibleTypes]);

  // 当对话框打开且 defaultName 变化时，更新项目名称
  useEffect(() => {
    if (open) {
      setName(defaultName || "");
    }
  }, [open, defaultName]);

  useEffect(() => {
    if (!open) {
      return;
    }

    let mounted = true;
    setResolvedProjectPath("");
    setPathChecking(false);
    setPathConflictMessage("");
    setPathErrorMessage("");

    const loadWorkspaceRoot = async () => {
      try {
        const root = await getWorkspaceProjectsRoot();
        if (mounted) {
          setParentRootPath(root);
        }
      } catch (error) {
        console.error("加载 workspace 目录失败:", error);
        if (mounted) {
          setParentRootPath("");
        }
      }
    };

    void loadWorkspaceRoot();

    return () => {
      mounted = false;
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const projectName = name.trim();
    const rootPath = parentRootPath.trim();
    if (!projectName) {
      setResolvedProjectPath("");
      setPathChecking(false);
      setPathConflictMessage("");
      setPathErrorMessage("");
      return;
    }

    if (!rootPath) {
      setResolvedProjectPath("");
      setPathChecking(false);
      setPathConflictMessage("");
      setPathErrorMessage(t("common.createProjectDialog.path.parentRequired"));
      return;
    }

    let mounted = true;
    setPathChecking(true);
    setPathConflictMessage("");
    setPathErrorMessage("");

    const resolveAndCheckPath = async () => {
      try {
        const path = await resolveProjectRootPath(projectName, rootPath);
        if (!mounted) {
          return;
        }

        setResolvedProjectPath(path);
        const existingProject = await getProjectByRootPath(path);
        if (!mounted) {
          return;
        }

        if (existingProject) {
          setPathConflictMessage(
            t("common.createProjectDialog.path.conflict", {
              name: existingProject.name,
            }),
          );
        } else {
          setPathConflictMessage("");
        }
      } catch (error) {
        console.error("解析或检查项目目录失败:", error);
        if (mounted) {
          setResolvedProjectPath("");
          setPathConflictMessage("");
          setPathErrorMessage(t("common.createProjectDialog.error.invalidPath"));
        }
      } finally {
        if (mounted) {
          setPathChecking(false);
        }
      }
    };

    void resolveAndCheckPath();

    return () => {
      mounted = false;
    };
  }, [open, name, parentRootPath, t]);

  const resetType = () => {
    setType(
      defaultType &&
        USER_PROJECT_TYPES.includes(defaultType as UserProjectType) &&
        visibleTypes.includes(defaultType as UserProjectType)
        ? (defaultType as UserProjectType)
        : fallbackType,
    );
  };

  const handleChooseParentPath = async () => {
    setIsChoosingParentPath(true);
    try {
      const selectedPath = await openDialog({
        directory: true,
        multiple: false,
        defaultPath: parentRootPath || undefined,
      });
      if (typeof selectedPath === "string" && selectedPath.trim()) {
        setParentRootPath(selectedPath);
      }
    } catch (error) {
      toast.error(
        t("common.createProjectDialog.path.selectFailed", {
          message: extractErrorMessage(error),
        }),
      );
    } finally {
      setIsChoosingParentPath(false);
    }
  };

  const canSubmit =
    Boolean(name.trim()) &&
    Boolean(parentRootPath.trim()) &&
    Boolean(resolvedProjectPath) &&
    !isSubmitting &&
    !pathChecking &&
    !pathConflictMessage &&
    !pathErrorMessage;

  const handleSubmit = async () => {
    if (!canSubmit) return;

    setIsSubmitting(true);
    try {
      await onSubmit(name.trim(), type, resolvedProjectPath);
      setName("");
      setResolvedProjectPath("");
      setPathConflictMessage("");
      setPathErrorMessage("");
      resetType();
      onOpenChange(false);
    } catch (error) {
      console.error("创建项目失败:", error);
      const message = extractErrorMessage(error);
      const friendlyMessage = getCreateProjectErrorMessage(
        message,
        createProjectErrorMessageCopy,
      );
      toast.error(
        t("common.createProjectDialog.toast.createFailed", {
          message: friendlyMessage,
        }),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const pathFeedback =
    pathErrorMessage ||
    pathConflictMessage ||
    (pathChecking ? t("common.createProjectDialog.path.checking") : "");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden border-slate-200 bg-white p-0 sm:max-w-[480px]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void handleSubmit();
          }}
        >
          <DialogHeader className="border-b border-slate-100 px-6 py-5">
            <DialogTitle>{t("common.createProjectDialog.title")}</DialogTitle>
            <DialogDescription className="sr-only">
              {t("common.createProjectDialog.description")}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-5 px-6 py-5">
            <div className="grid gap-2">
              <Label htmlFor="name">
                {t("common.createProjectDialog.name.label")}
              </Label>
              <Input
                id="name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder={t("common.createProjectDialog.name.placeholder")}
                autoFocus
              />
            </div>

            {showTypeSelector ? (
              <div className="grid gap-2">
                <Label>{t("common.createProjectDialog.type.title")}</Label>
                <div className="grid grid-cols-2 gap-2">
                  {visibleTypes.map((projectType) => (
                    <button
                      key={projectType}
                      type="button"
                      aria-pressed={type === projectType}
                      className={cn(
                        "flex h-10 items-center justify-center gap-2 rounded-md border px-3 text-sm transition",
                        type === projectType
                          ? "border-slate-900 bg-slate-900 text-white"
                          : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
                      )}
                      onClick={() => setType(projectType)}
                    >
                      <span>{getProjectTypeIcon(projectType)}</span>
                      <span>{getProjectTypeDisplayLabel(projectType)}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="grid gap-2">
              <Label htmlFor="parent-root">
                {t("common.createProjectDialog.path.workspaceRootLabel")}
              </Label>
              <div className="flex gap-2">
                <Input
                  id="parent-root"
                  value={parentRootPath}
                  placeholder={t("common.loading")}
                  readOnly
                  title={parentRootPath}
                  className="min-w-0 flex-1 bg-slate-50"
                />
                <Button
                  type="button"
                  variant="outline"
                  className="shrink-0"
                  disabled={isChoosingParentPath || isSubmitting}
                  onClick={() => {
                    void handleChooseParentPath();
                  }}
                >
                  {t("common.createProjectDialog.path.select")}
                </Button>
              </div>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="project-path-preview">
                {t("common.createProjectDialog.path.previewLabel")}
              </Label>
              <div
                id="project-path-preview"
                title={resolvedProjectPath}
                className="min-h-10 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm leading-5 text-slate-700"
              >
                <span className="block truncate">
                  {resolvedProjectPath ||
                    t("common.createProjectDialog.path.nameRequired")}
                </span>
              </div>
              {pathFeedback ? (
                <p
                  className={cn(
                    "text-xs leading-5",
                    pathErrorMessage || pathConflictMessage
                      ? "text-destructive"
                      : "text-slate-500",
                  )}
                >
                  {pathFeedback}
                </p>
              ) : null}
            </div>
          </div>

          <DialogFooter className="border-t border-slate-100 px-6 py-4">
            <Button
              type="button"
              variant="outline"
              className="border-slate-200 bg-white"
              onClick={() => onOpenChange(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button type="submit" disabled={!canSubmit}>
              {isSubmitting
                ? t("common.createProjectDialog.action.creating")
                : t("common.createProjectDialog.action.create")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
