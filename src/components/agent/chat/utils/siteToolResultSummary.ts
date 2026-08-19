import type { SiteSavedContentTarget } from "../types";
import { normalizeManagedWorkspacePathForDisplay } from "../workspace/workspacePath";

export interface SiteToolResultSummary {
  savedContent?: {
    contentId?: string;
    projectId?: string;
    title?: string;
    projectRootPath?: string;
    bundleRelativeDir?: string;
    markdownRelativePath?: string;
    imagesRelativeDir?: string;
    metaRelativePath?: string;
    imageCount?: number;
  };
  savedProjectId?: string;
  savedBy?: string;
  saveSkippedProjectId?: string;
  saveSkippedBy?: string;
  saveErrorMessage?: string;
  adapterSourceKind?: string;
  adapterSourceVersion?: string;
}

type SiteAdapterRunResult = {
  saved_content?: {
    content_id?: string;
    project_id?: string;
    title?: string;
    markdown_relative_path?: string;
  } | null;
  saved_project_id?: string | null;
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function readString(
  candidates: Array<Record<string, unknown> | undefined>,
  keys: string[],
): string | undefined {
  for (const candidate of candidates) {
    for (const key of keys) {
      const value = candidate?.[key];
      if (typeof value === "string" && value.trim()) {
        return value.trim();
      }
    }
  }
  return undefined;
}

function readNumber(
  candidate: Record<string, unknown> | undefined,
  keys: string[],
): number | undefined {
  for (const key of keys) {
    const value = candidate?.[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value.trim());
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return undefined;
}

export function isPreloadSiteToolResultMetadata(rawMetadata: unknown): boolean {
  const metadata = asRecord(rawMetadata);
  const result = asRecord(metadata?.result);
  return (
    readString([metadata, result], ["execution_origin", "executionOrigin"]) ===
      "preload" ||
    metadata?.preload === true ||
    result?.preload === true
  );
}

export function normalizeSiteToolResultSummary(
  rawMetadata: unknown,
): SiteToolResultSummary | null {
  const metadata = asRecord(rawMetadata);
  if (!metadata) {
    return null;
  }
  const result = asRecord(metadata.result);
  const savedContent =
    asRecord(metadata.saved_content) || asRecord(result?.saved_content);
  const candidates = [metadata, result, savedContent];
  const toolFamily = readString(candidates, ["tool_family", "toolFamily"]);
  const savedProjectId = readString(candidates, [
    "saved_project_id",
    "savedProjectId",
  ]);
  const saveSkippedProjectId = readString(candidates, [
    "save_skipped_project_id",
    "saveSkippedProjectId",
  ]);
  const saveErrorMessage = readString(candidates, [
    "save_error_message",
    "saveErrorMessage",
  ]);
  const adapterSourceKind = readString(candidates, [
    "adapter_source_kind",
    "adapterSourceKind",
  ]);
  const adapterSourceVersion = readString(candidates, [
    "adapter_source_version",
    "adapterSourceVersion",
  ]);
  const hasSavedContent = Boolean(
    savedContent &&
      [
        "content_id",
        "contentId",
        "project_id",
        "projectId",
        "title",
      ].some((key) => typeof savedContent[key] === "string" && savedContent[key]),
  );
  if (
    toolFamily !== "site" &&
    !hasSavedContent &&
    !savedProjectId &&
    !saveSkippedProjectId &&
    !saveErrorMessage &&
    !adapterSourceKind
  ) {
    return null;
  }

  const projectRootPath = readString([savedContent], [
    "project_root_path",
    "projectRootPath",
  ]);
  return {
    savedContent: hasSavedContent
      ? {
          contentId: readString([savedContent], ["content_id", "contentId"]),
          projectId: readString([savedContent], ["project_id", "projectId"]),
          title: readString([savedContent], ["title"]),
          projectRootPath:
            normalizeManagedWorkspacePathForDisplay(projectRootPath) ||
            undefined,
          bundleRelativeDir: readString([savedContent], [
            "bundle_relative_dir",
            "bundleRelativeDir",
          ]),
          markdownRelativePath: readString([savedContent], [
            "markdown_relative_path",
            "markdownRelativePath",
          ]),
          imagesRelativeDir: readString([savedContent], [
            "images_relative_dir",
            "imagesRelativeDir",
          ]),
          metaRelativePath: readString([savedContent], [
            "meta_relative_path",
            "metaRelativePath",
          ]),
          imageCount: readNumber(savedContent, ["image_count", "imageCount"]),
        }
      : undefined,
    savedProjectId,
    savedBy: readString(candidates, ["saved_by", "savedBy"]),
    saveSkippedProjectId,
    saveSkippedBy: readString(candidates, ["save_skipped_by", "saveSkippedBy"]),
    saveErrorMessage,
    adapterSourceKind,
    adapterSourceVersion,
  };
}

export function resolveSiteSavedContentTarget(
  summary: SiteToolResultSummary | null,
): SiteSavedContentTarget | null {
  const content = summary?.savedContent;
  const contentId = content?.contentId?.trim();
  const projectId = content?.projectId?.trim() || summary?.savedProjectId?.trim();
  if (!content || !contentId || !projectId) {
    return null;
  }
  const relativePath = content.markdownRelativePath?.trim();
  return {
    projectId,
    contentId,
    ...(content.title?.trim() ? { title: content.title.trim() } : {}),
    ...(relativePath
      ? { preferredTarget: "project_file", projectFile: { relativePath } }
      : {}),
  };
}

export function resolveSiteSavedContentTargetFromMetadata(
  rawMetadata: unknown,
): SiteSavedContentTarget | null {
  if (isPreloadSiteToolResultMetadata(rawMetadata)) {
    return null;
  }
  return resolveSiteSavedContentTarget(normalizeSiteToolResultSummary(rawMetadata));
}

export function resolveSiteSavedContentTargetFromRunResult(
  result: Pick<SiteAdapterRunResult, "saved_content" | "saved_project_id"> | null,
): SiteSavedContentTarget | null {
  const savedContent = result?.saved_content;
  if (!savedContent?.content_id?.trim()) {
    return null;
  }
  const projectId = savedContent.project_id?.trim() || result?.saved_project_id?.trim();
  if (!projectId) {
    return null;
  }
  const relativePath = savedContent.markdown_relative_path?.trim();
  return {
    projectId,
    contentId: savedContent.content_id.trim(),
    ...(savedContent.title?.trim() ? { title: savedContent.title.trim() } : {}),
    ...(relativePath
      ? { preferredTarget: "project_file", projectFile: { relativePath } }
      : {}),
  };
}

export function hasMeaningfulSiteToolResultSignal(rawMetadata: unknown): boolean {
  const summary = normalizeSiteToolResultSummary(rawMetadata);
  return Boolean(
    summary?.savedContent ||
      summary?.savedProjectId ||
      summary?.saveSkippedProjectId ||
      summary?.saveErrorMessage,
  );
}

export function resolveSiteSavedContentTargetRelativePath(
  target: SiteSavedContentTarget | null | undefined,
): string | null {
  return target?.projectFile?.relativePath?.trim() || null;
}

export function resolveSiteSavedContentTargetDisplayName(
  target: SiteSavedContentTarget | null | undefined,
): string | null {
  const relativePath = resolveSiteSavedContentTargetRelativePath(target);
  if (relativePath) {
    const segments = relativePath.replace(/\\/g, "/").split("/").filter(Boolean);
    return segments.at(-1) || relativePath;
  }
  return target?.title?.trim() || null;
}

export function resolveSiteProjectSourceLabel(source?: string): string | null {
  if (source === "context_project") return "来自当前项目上下文";
  if (source === "explicit_project") return "来自显式项目参数";
  return null;
}

export function resolveSiteProjectTargetLabel(params: {
  source?: string;
  projectId?: string;
}): string {
  if (params.source === "context_project") return "当前项目";
  if (params.source === "explicit_project") return "所选项目";
  return params.projectId?.trim() ? `项目 ${params.projectId.trim()}` : "项目";
}

export function resolveSiteAdapterSourceLabel(
  summary: SiteToolResultSummary,
): string | null {
  if (summary.adapterSourceKind === "server_synced") {
    return summary.adapterSourceVersion
      ? `服务端脚本 · ${summary.adapterSourceVersion}`
      : "服务端脚本";
  }
  if (summary.adapterSourceKind === "bundled") {
    return summary.adapterSourceVersion
      ? `内置脚本 · ${summary.adapterSourceVersion}`
      : "内置脚本";
  }
  return null;
}
