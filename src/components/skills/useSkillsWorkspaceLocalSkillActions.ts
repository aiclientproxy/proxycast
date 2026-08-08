import {
  useCallback,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { open as openDialog, save as saveDialog } from "@/lib/desktop-host/dialog";
import { toast } from "sonner";
import { skillsApi, type Skill } from "@/lib/api/skills";
import { ensureSkillPackageExtension } from "./SkillsWorkspacePageContent";
import type { SkillsWorkspaceTranslate } from "./SkillsWorkspacePageTypes";
import {
  getSkillAutoLoadPreferenceKey,
  readSkillAutoLoadPreferences,
  writeSkillAutoLoadPreferences,
} from "./skillAutoLoadPreferences";

interface UseSkillsWorkspaceLocalSkillActionsParams {
  detailInstalledSkillDirectory: string | null;
  highlightedInstalledSkillDirectory: string | null;
  optimisticInstalledSkillDirectory: string | null;
  refreshLocalSkills: () => Promise<void>;
  setDetailInstalledSkillDirectory: Dispatch<SetStateAction<string | null>>;
  setHighlightedInstalledSkillDirectory: Dispatch<SetStateAction<string | null>>;
  setOptimisticInstalledSkill: Dispatch<SetStateAction<Skill | null>>;
  setOptimisticallyHiddenSkillDirectories: Dispatch<
    SetStateAction<Set<string>>
  >;
  t: SkillsWorkspaceTranslate;
  uninstallLocalSkill: (directory: string) => Promise<void>;
}

export function useSkillsWorkspaceLocalSkillActions({
  detailInstalledSkillDirectory,
  highlightedInstalledSkillDirectory,
  optimisticInstalledSkillDirectory,
  refreshLocalSkills,
  setDetailInstalledSkillDirectory,
  setHighlightedInstalledSkillDirectory,
  setOptimisticInstalledSkill,
  setOptimisticallyHiddenSkillDirectories,
  t,
  uninstallLocalSkill,
}: UseSkillsWorkspaceLocalSkillActionsParams) {
  const [uninstallingSkillDirectory, setUninstallingSkillDirectory] = useState<
    string | null
  >(null);
  const [exportingSkillDirectory, setExportingSkillDirectory] = useState<
    string | null
  >(null);
  const [renamingSkillDirectory, setRenamingSkillDirectory] = useState<
    string | null
  >(null);
  const [replacingSkillDirectory, setReplacingSkillDirectory] = useState<
    string | null
  >(null);
  const [revealingSkillDirectory, setRevealingSkillDirectory] = useState<
    string | null
  >(null);
  const [skillAutoLoadPreferences, setSkillAutoLoadPreferences] = useState(() =>
    readSkillAutoLoadPreferences(),
  );

  const handleUninstallLocalSkill = useCallback(
    async (skill: Skill) => {
      if (skill.sourceKind === "builtin" || skill.catalogSource === "project") {
        return;
      }

      setUninstallingSkillDirectory(skill.directory);
      setOptimisticallyHiddenSkillDirectories((previous) => {
        const next = new Set(previous);
        next.add(skill.directory);
        return next;
      });
      try {
        await uninstallLocalSkill(skill.directory);
        if (optimisticInstalledSkillDirectory === skill.directory) {
          setOptimisticInstalledSkill(null);
        }
        if (highlightedInstalledSkillDirectory === skill.directory) {
          setHighlightedInstalledSkillDirectory(null);
        }
        const preferenceKey = getSkillAutoLoadPreferenceKey(skill);
        setSkillAutoLoadPreferences((previous) => {
          if (!(preferenceKey in previous)) {
            return previous;
          }
          const next = { ...previous };
          delete next[preferenceKey];
          writeSkillAutoLoadPreferences(next);
          return next;
        });
        toast.success(
          t("skills.workspace.installedSkill.uninstallSuccess", {
            name: skill.name,
          }),
        );
      } catch (error) {
        setOptimisticallyHiddenSkillDirectories((previous) => {
          if (!previous.has(skill.directory)) {
            return previous;
          }
          const next = new Set(previous);
          next.delete(skill.directory);
          return next;
        });
        toast.error(
          t("skills.workspace.installedSkill.uninstallFailed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setUninstallingSkillDirectory(null);
      }
    },
    [
      highlightedInstalledSkillDirectory,
      optimisticInstalledSkillDirectory,
      setHighlightedInstalledSkillDirectory,
      setOptimisticInstalledSkill,
      setOptimisticallyHiddenSkillDirectories,
      t,
      uninstallLocalSkill,
    ],
  );

  const handleExportLocalSkillPackage = useCallback(
    async (skill: Skill) => {
      if (!skill.directory) {
        return;
      }

      let selectedPath: string | null;
      try {
        selectedPath = await saveDialog({
          title: t("skills.workspace.export.dialogTitle"),
          defaultPath: `${skill.directory}.skills`,
          filters: [
            {
              name: t("skills.workspace.export.filterName"),
              extensions: ["skills", "skill"],
            },
          ],
        });
      } catch (error) {
        toast.error(
          t("skills.workspace.export.failed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
        return;
      }

      if (!selectedPath) {
        return;
      }

      setExportingSkillDirectory(skill.directory);
      try {
        await skillsApi.exportLocalSkillPackage(
          skill.directory,
          ensureSkillPackageExtension(selectedPath),
          "lime",
        );
        toast.success(
          t("skills.workspace.export.success", {
            name: skill.name,
          }),
        );
      } catch (error) {
        toast.error(
          t("skills.workspace.export.failed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setExportingSkillDirectory(null);
      }
    },
    [t],
  );

  const handleRenameLocalSkill = useCallback(
    async (skill: Skill) => {
      if (
        !skill.directory ||
        skill.sourceKind === "builtin" ||
        skill.catalogSource === "project"
      ) {
        return;
      }

      const nextDirectory = window
        .prompt(
          t("skills.workspace.installedSkill.rename.prompt", {
            name: skill.name,
          }),
          skill.directory,
        )
        ?.trim();

      if (!nextDirectory || nextDirectory === skill.directory) {
        return;
      }

      setRenamingSkillDirectory(skill.directory);
      try {
        const result = await skillsApi.renameLocalSkill(
          skill.directory,
          nextDirectory,
          "lime",
        );
        await refreshLocalSkills();
        setHighlightedInstalledSkillDirectory(result.directory);
        if (detailInstalledSkillDirectory === skill.directory) {
          setDetailInstalledSkillDirectory(result.directory);
        }
        const oldPreferenceKey = getSkillAutoLoadPreferenceKey(skill);
        setSkillAutoLoadPreferences((previous) => {
          if (!(oldPreferenceKey in previous)) {
            return previous;
          }
          const next = { ...previous };
          next[result.directory] = previous[oldPreferenceKey];
          delete next[oldPreferenceKey];
          writeSkillAutoLoadPreferences(next);
          return next;
        });
        toast.success(
          t("skills.workspace.installedSkill.rename.success", {
            name: skill.name,
            directory: result.directory,
          }),
        );
      } catch (error) {
        toast.error(
          t("skills.workspace.installedSkill.rename.failed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setRenamingSkillDirectory(null);
      }
    },
    [
      detailInstalledSkillDirectory,
      refreshLocalSkills,
      setDetailInstalledSkillDirectory,
      setHighlightedInstalledSkillDirectory,
      t,
    ],
  );

  const handleReplaceLocalSkillPackage = useCallback(
    async (skill: Skill) => {
      if (
        !skill.directory ||
        skill.sourceKind === "builtin" ||
        skill.catalogSource === "project"
      ) {
        return;
      }

      let selected: string | string[] | null;
      setReplacingSkillDirectory(skill.directory);
      try {
        selected = await openDialog({
          directory: false,
          multiple: false,
          title: t("skills.workspace.installedSkill.replace.dialogTitle", {
            name: skill.name,
          }),
          filters: [
            {
              name: t("skills.workspace.export.filterName"),
              extensions: ["skills", "skill"],
            },
          ],
        });
      } catch (error) {
        toast.error(
          t("skills.workspace.installedSkill.replace.failed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
        setReplacingSkillDirectory(null);
        return;
      }

      if (!selected || Array.isArray(selected)) {
        setReplacingSkillDirectory(null);
        return;
      }

      try {
        const result = await skillsApi.replaceLocalSkillPackage(
          skill.directory,
          selected,
          "lime",
        );
        await refreshLocalSkills();
        setHighlightedInstalledSkillDirectory(result.directory);
        toast.success(
          t("skills.workspace.installedSkill.replace.success", {
            name: skill.name,
          }),
        );
      } catch (error) {
        toast.error(
          t("skills.workspace.installedSkill.replace.failed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setReplacingSkillDirectory(null);
      }
    },
    [refreshLocalSkills, setHighlightedInstalledSkillDirectory, t],
  );

  const handleRevealLocalSkill = useCallback(
    async (skill: Skill) => {
      if (!skill.directory) {
        return;
      }

      setRevealingSkillDirectory(skill.directory);
      try {
        await skillsApi.revealLocalSkill(skill.directory, "lime");
      } catch (error) {
        toast.error(
          t("skills.workspace.installedSkill.showInFolder.failed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
      } finally {
        setRevealingSkillDirectory(null);
      }
    },
    [t],
  );

  const handleSkillAutoLoadChange = useCallback(
    (skill: Skill, enabled: boolean) => {
      setSkillAutoLoadPreferences((previous) => {
        const key = getSkillAutoLoadPreferenceKey(skill);
        const next = { ...previous, [key]: enabled };
        writeSkillAutoLoadPreferences(next);
        return next;
      });
      toast.success(
        t(
          enabled
            ? "skills.workspace.autoLoad.enabledToast"
            : "skills.workspace.autoLoad.disabledToast",
          {
            name: skill.name,
          },
        ),
      );
    },
    [t],
  );

  return {
    exportingSkillDirectory,
    handleExportLocalSkillPackage,
    handleRenameLocalSkill,
    handleReplaceLocalSkillPackage,
    handleRevealLocalSkill,
    handleSkillAutoLoadChange,
    handleUninstallLocalSkill,
    renamingSkillDirectory,
    replacingSkillDirectory,
    revealingSkillDirectory,
    skillAutoLoadPreferences,
    uninstallingSkillDirectory,
  };
}
