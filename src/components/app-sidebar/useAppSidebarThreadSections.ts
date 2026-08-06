import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import type { AppServerThreadSection } from "@/lib/api/appServer";
import {
  createThreadSection,
  deleteThreadSection,
  listThreadSections,
  updateThreadSection,
} from "@/lib/api/threadSections";

interface UseAppSidebarThreadSectionsParams {
  onSectionsChanged?: () => Promise<void> | void;
}

export function useAppSidebarThreadSections({
  onSectionsChanged,
}: UseAppSidebarThreadSectionsParams) {
  const { t } = useTranslation("navigation");
  const [sections, setSections] = useState<AppServerThreadSection[] | null>(
    null,
  );
  const [pendingSectionId, setPendingSectionId] = useState<string | null>(null);

  const refreshSections = useCallback(async () => {
    try {
      setSections(await listThreadSections());
    } catch (error) {
      console.warn("加载会话分组失败:", error);
    }
  }, []);

  useEffect(() => {
    void refreshSections();
  }, [refreshSections]);

  const createSection = useCallback(async () => {
    const name = window
      .prompt(
        t("navigation.sidebar.conversations.section.create.prompt", "新建分组"),
      )
      ?.trim();
    if (!name) {
      return;
    }

    setPendingSectionId("create");
    try {
      const section = await createThreadSection({ name });
      setSections((current) => [...(current ?? []), section]);
      toast.success(
        t(
          "navigation.sidebar.conversations.section.create.success",
          "已新建分组",
        ),
      );
      await onSectionsChanged?.();
    } catch (error) {
      console.error("新建会话分组失败:", error);
      toast.error(
        t(
          "navigation.sidebar.conversations.section.create.error",
          "新建分组失败",
        ),
      );
    } finally {
      setPendingSectionId(null);
    }
  }, [onSectionsChanged, t]);

  const renameSection = useCallback(
    async (section: AppServerThreadSection) => {
      const name = window
        .prompt(
          t(
            "navigation.sidebar.conversations.section.rename.prompt",
            "重命名分组",
          ),
          section.name,
        )
        ?.trim();
      if (!name || name === section.name) {
        return;
      }

      setPendingSectionId(section.id);
      try {
        const updated = await updateThreadSection({
          sectionId: section.id,
          name,
        });
        setSections(
          (current) =>
            current?.map((item) =>
              item.id === updated.id ? updated : item,
            ) ?? [updated],
        );
        toast.success(
          t(
            "navigation.sidebar.conversations.section.rename.success",
            "已重命名分组",
          ),
        );
        await onSectionsChanged?.();
      } catch (error) {
        console.error("重命名会话分组失败:", error);
        toast.error(
          t(
            "navigation.sidebar.conversations.section.rename.error",
            "重命名分组失败",
          ),
        );
      } finally {
        setPendingSectionId(null);
      }
    },
    [onSectionsChanged, t],
  );

  const removeSection = useCallback(
    async (section: AppServerThreadSection) => {
      const confirmed = window.confirm(
        t("navigation.sidebar.conversations.section.delete.confirm", {
          name: section.name,
          defaultValue:
            "确定要删除分组“{{name}}”吗？其中的对话会回到未分组列表。",
        }),
      );
      if (!confirmed) {
        return;
      }

      setPendingSectionId(section.id);
      try {
        await deleteThreadSection({ sectionId: section.id });
        setSections(
          (current) => current?.filter((item) => item.id !== section.id) ?? [],
        );
        toast.success(
          t(
            "navigation.sidebar.conversations.section.delete.success",
            "已删除分组",
          ),
        );
        await onSectionsChanged?.();
      } catch (error) {
        console.error("删除会话分组失败:", error);
        toast.error(
          t(
            "navigation.sidebar.conversations.section.delete.error",
            "删除分组失败",
          ),
        );
      } finally {
        setPendingSectionId(null);
      }
    },
    [onSectionsChanged, t],
  );

  return {
    createSection,
    pendingSectionId,
    removeSection,
    renameSection,
    sections,
  };
}
