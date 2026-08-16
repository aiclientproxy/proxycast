import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import type { Artifact } from "@/lib/artifact/types";
import {
  openPathWithDefaultApp,
  revealPathInFinder,
} from "@/lib/api/fileSystem";
import { getFileName } from "./harnessStatusPanelViewModel";

export interface HarnessFilePreviewResult {
  artifactId?: string;
  artifactRef?: string;
  path?: string;
  content?: string | null;
  error?: string | null;
  isBinary?: boolean;
  metadata?: unknown;
  size?: number;
  title?: string;
}

export interface HarnessPreviewDialogState {
  open: boolean;
  title: string;
  description?: string;
  path?: string;
  displayName: string;
  content?: string;
  preview?: string;
  artifact?: Artifact;
  error?: string;
  isBinary: boolean;
  size?: number;
  loading: boolean;
}

export interface OpenHarnessPreviewOptions {
  title: string;
  description?: string;
  path?: string;
  content?: string;
  preview?: string;
}

interface UseHarnessPreviewDialogOptions {
  onLoadFilePreview?: (path: string) => Promise<HarnessFilePreviewResult>;
  onOpenFile?: (fileName: string, content: string) => void;
  onRevealPath?: (path: string) => Promise<void>;
  onOpenPath?: (path: string) => Promise<void>;
}

export function useHarnessPreviewDialog({
  onLoadFilePreview,
  onOpenFile,
  onRevealPath,
  onOpenPath,
}: UseHarnessPreviewDialogOptions) {
  const [previewDialog, setPreviewDialog] = useState<HarnessPreviewDialogState>(
    {
      open: false,
      title: "",
      displayName: "",
      isBinary: false,
      loading: false,
    },
  );
  const previewRequestIdRef = useRef(0);

  const handlePreviewOpenChange = useCallback((open: boolean) => {
    setPreviewDialog((current) => ({
      ...current,
      open,
      loading: open ? current.loading : false,
    }));
  }, []);

  const openPreview = useCallback(
    async ({
      title,
      description,
      path,
      content,
      preview,
    }: OpenHarnessPreviewOptions) => {
      const requestId = previewRequestIdRef.current + 1;
      previewRequestIdRef.current = requestId;

      const shouldLoad =
        !content?.trim() && !!path && typeof onLoadFilePreview === "function";

      setPreviewDialog({
        open: true,
        title,
        description,
        path,
        displayName: path ? getFileName(path) : title,
        content: content?.trim() || preview?.trim(),
        preview,
        artifact: undefined,
        error:
          content?.trim() || preview?.trim()
            ? undefined
            : shouldLoad
              ? undefined
              : "暂无可预览内容",
        isBinary: false,
        loading: shouldLoad,
      });

      if (!shouldLoad || !path) {
        return;
      }

      try {
        const result = await onLoadFilePreview(path);
        if (previewRequestIdRef.current !== requestId) {
          return;
        }

        const nextPath = result.path || path;
        const normalizedContent = result.content ?? undefined;

        setPreviewDialog((current) => ({
          ...current,
          path: nextPath,
          displayName: getFileName(nextPath),
          content: normalizedContent?.trim()
            ? normalizedContent
            : current.content,
          isBinary: result.isBinary === true,
          size: result.size,
          error:
            result.isBinary === true
              ? undefined
              : result.error || (normalizedContent ? undefined : current.error),
          loading: false,
        }));
      } catch (error) {
        if (previewRequestIdRef.current !== requestId) {
          return;
        }

        setPreviewDialog((current) => ({
          ...current,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        }));
      }
    },
    [onLoadFilePreview],
  );

  const handleOpenFile = useCallback(() => {
    if (!onOpenFile || !previewDialog.content?.trim()) {
      return;
    }

    onOpenFile(
      previewDialog.path || previewDialog.displayName,
      previewDialog.content,
    );
  }, [
    onOpenFile,
    previewDialog.content,
    previewDialog.displayName,
    previewDialog.path,
  ]);

  const handleCopyPath = useCallback(async () => {
    const path = previewDialog.path?.trim();
    if (!path) {
      toast.error("当前没有可复制的文件路径");
      return;
    }

    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      toast.error("当前环境不支持剪贴板复制");
      return;
    }

    try {
      await navigator.clipboard.writeText(path);
      toast.success("文件路径已复制");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "复制路径失败");
    }
  }, [previewDialog.path]);

  const handleCopyContent = useCallback(async () => {
    const content = previewDialog.content?.trim();
    if (!content) {
      toast.error("当前没有可复制的内容");
      return;
    }

    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      toast.error("当前环境不支持剪贴板复制");
      return;
    }

    try {
      await navigator.clipboard.writeText(previewDialog.content || "");
      toast.success("内容已复制");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "复制内容失败");
    }
  }, [previewDialog.content]);

  const handleOpenPathValue = useCallback(
    async (path: string) => {
      const normalizedPath = path.trim();
      if (!normalizedPath) {
        toast.error("当前没有可打开的文件路径");
        return;
      }

      try {
        await (onOpenPath ?? openPathWithDefaultApp)(normalizedPath);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "打开文件失败");
      }
    },
    [onOpenPath],
  );

  const handleRevealPath = useCallback(async () => {
    const path = previewDialog.path?.trim();
    if (!path) {
      toast.error("当前没有可定位的文件路径");
      return;
    }

    try {
      await (onRevealPath ?? revealPathInFinder)(path);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "定位文件失败");
    }
  }, [onRevealPath, previewDialog.path]);

  const handleOpenPath = useCallback(async () => {
    const path = previewDialog.path?.trim();
    if (!path) {
      toast.error("当前没有可打开的文件路径");
      return;
    }

    await handleOpenPathValue(path);
  }, [handleOpenPathValue, previewDialog.path]);

  return {
    previewDialog,
    handlePreviewOpenChange,
    openPreview,
    handleOpenFile,
    handleCopyPath,
    handleCopyContent,
    handleOpenPathValue,
    handleRevealPath,
    handleOpenPath,
  };
}
