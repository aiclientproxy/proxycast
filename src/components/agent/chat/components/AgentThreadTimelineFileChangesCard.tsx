import { useCallback, useMemo } from "react";

import {
  readAgentRuntimeTimelineArtifactContent,
  type AgentRuntimeTimelineArtifactContent,
} from "@/lib/api/agentRuntime/appServerArtifactClient";
import {
  aggregateFileChangeSummaries,
  type FileChangeDiffLine,
  type FileChangeKind,
  type FileChangesAggregate,
  type FileChangeSummary,
} from "../utils/fileChangeSummary";
import type { AgentThreadItem } from "../types";
import { buildTimelinePatchContentPart } from "./messageListTimelineContentPartBuilders";
import {
  resolveTimelineArtifactNavigation,
  type ArtifactTimelineOpenTarget,
} from "../utils/artifactTimelineNavigation";
import { isTimelineReadOnlyFileArtifact } from "../utils/timelineFileArtifactKind";
import { FileChangesSummaryCard } from "./FileChangesSummaryCard";

type FileArtifactItem = Extract<AgentThreadItem, { type: "file_artifact" }>;
type PatchItem = Extract<AgentThreadItem, { type: "patch" }>;
type FileChangeEvidenceItem = FileArtifactItem | PatchItem;

interface AgentThreadTimelineFileChangesCardProps {
  items: FileChangeEvidenceItem[];
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  readTimelineArtifactContent?: (
    item: FileArtifactItem,
  ) => Promise<AgentRuntimeTimelineArtifactContent | null>;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(
  record: Record<string, unknown> | null,
  keys: string[],
): string | null {
  for (const key of keys) {
    const value = record?.[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function readNumber(
  record: Record<string, unknown> | null,
  keys: string[],
): number | null {
  for (const key of keys) {
    const value = record?.[key];
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
  return null;
}

function resolveFileChangeRecord(
  item: FileArtifactItem,
): Record<string, unknown> | null {
  const metadata = asRecord(item.metadata);
  return asRecord(metadata?.file_change) || asRecord(metadata?.fileChange);
}

// eslint-disable-next-line react-refresh/only-export-components
export function hasTimelineFileChangeEvidence(
  item: AgentThreadItem,
): item is FileChangeEvidenceItem {
  if (item.type === "patch") {
    const part = buildTimelinePatchContentPart(item);
    return part?.type === "file_changes_batch" && part.aggregate.fileCount > 0;
  }
  if (item.type !== "file_artifact") {
    return false;
  }
  if (isTimelineReadOnlyFileArtifact(item)) {
    return false;
  }
  return Boolean(resolveFileChangeRecord(item));
}

function normalizeFileChangeKind(value: unknown): FileChangeKind {
  if (value === "add" || value === "delete" || value === "update") {
    return value;
  }
  if (value === "added") {
    return "add";
  }
  if (value === "deleted") {
    return "delete";
  }
  return "update";
}

function normalizeDiffLine(value: unknown): FileChangeDiffLine | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const kind = record.kind;
  if (kind !== "context" && kind !== "add" && kind !== "remove") {
    return null;
  }
  const text =
    typeof record.value === "string"
      ? record.value
      : typeof record.text === "string"
        ? record.text
        : "";
  const oldLine = readNumber(record, ["old_line", "oldLine"]);
  const newLine = readNumber(record, ["new_line", "newLine"]);
  return {
    kind,
    value: text,
    ...(oldLine !== null ? { oldLine } : {}),
    ...(newLine !== null ? { newLine } : {}),
  };
}

function resolveDiffLines(
  fileChange: Record<string, unknown> | null,
): FileChangeDiffLine[] {
  const diff = fileChange?.diff;
  if (!Array.isArray(diff)) {
    return [];
  }
  return diff
    .map(normalizeDiffLine)
    .filter((line): line is FileChangeDiffLine => Boolean(line));
}

function countDiffLines(diff: FileChangeDiffLine[]): {
  added: number;
  removed: number;
} {
  return diff.reduce(
    (stats, line) => {
      if (line.kind === "add") {
        stats.added += 1;
      } else if (line.kind === "remove") {
        stats.removed += 1;
      }
      return stats;
    },
    { added: 0, removed: 0 },
  );
}

function buildFileChangeSummary(item: FileArtifactItem): FileChangeSummary {
  const fileChange = resolveFileChangeRecord(item);
  const diff = resolveDiffLines(fileChange);
  const counted = countDiffLines(diff);
  const path =
    readString(fileChange, ["path", "filePath", "file_path"]) || item.path;
  const linesAdded =
    readNumber(fileChange, [
      "lines_added",
      "linesAdded",
      "additions",
      "addedCount",
    ]) ?? counted.added;
  const linesRemoved =
    readNumber(fileChange, [
      "lines_removed",
      "linesRemoved",
      "deletions",
      "removedCount",
    ]) ?? counted.removed;

  return {
    path,
    kind: normalizeFileChangeKind(fileChange?.kind),
    linesAdded,
    linesRemoved,
    diff,
    truncated: fileChange?.truncated === true,
    source: fileChange ? "backend" : "approx",
    status: item.status === "failed" ? "failed" : "completed",
  };
}

function buildFileChangeSummaries(
  item: FileChangeEvidenceItem,
): FileChangeSummary[] {
  if (item.type === "patch") {
    const part = buildTimelinePatchContentPart(item);
    return part?.type === "file_changes_batch" ? part.aggregate.files : [];
  }
  return [buildFileChangeSummary(item)];
}

function buildAggregate(items: FileChangeEvidenceItem[]): FileChangesAggregate {
  return aggregateFileChangeSummaries(items.flatMap(buildFileChangeSummaries));
}

function resolvePatchFilePath(item: PatchItem): string {
  return (
    item.paths?.find((path) => path.trim())?.trim() ||
    item.summary?.find((path) => path.trim())?.trim() ||
    ""
  );
}

export function AgentThreadTimelineFileChangesCard({
  items,
  onFileClick,
  onOpenArtifactFromTimeline,
  readTimelineArtifactContent = readAgentRuntimeTimelineArtifactContent,
}: AgentThreadTimelineFileChangesCardProps) {
  const aggregate = useMemo(() => buildAggregate(items), [items]);
  const itemByPath = useMemo(() => {
    const entries = new Map<string, FileChangeEvidenceItem>();
    for (const item of items) {
      for (const summary of buildFileChangeSummaries(item)) {
        entries.set(summary.path, item);
      }
      if (item.type === "file_artifact") {
        entries.set(item.path, item);
      }
    }
    return entries;
  }, [items]);

  const openTimelineItem = useCallback(
    (item: FileChangeEvidenceItem) => {
      void (async () => {
        if (item.type === "patch") {
          const target = {
            filePath: resolvePatchFilePath(item),
            content: [item.text, item.stdout, item.stderr]
              .filter((value): value is string => Boolean(value?.trim()))
              .join("\n"),
            timelineItemId: item.id,
            openMode: "file_preview" as const,
          };
          if (onOpenArtifactFromTimeline) {
            onOpenArtifactFromTimeline(target);
            return;
          }
          onFileClick?.(target.filePath, target.content);
          return;
        }
        const navigation = resolveTimelineArtifactNavigation(item);
        const baseTarget = navigation?.rootTarget ?? {
          filePath: item.path,
          content: item.content || "",
          timelineItemId: item.id,
          openMode: "file_preview" as const,
        };
        let target = baseTarget;

        if (!target.content.trim()) {
          const artifactContent = await readTimelineArtifactContent(item).catch(
            () => null,
          );
          if (artifactContent?.content.trim()) {
            target = {
              ...target,
              artifactId: artifactContent.artifactId || target.artifactId,
              content: artifactContent.content,
              filePath: artifactContent.filePath || target.filePath,
            };
          }
        }

        if (onOpenArtifactFromTimeline) {
          onOpenArtifactFromTimeline(target);
          return;
        }

        onFileClick?.(target.filePath, target.content);
      })();
    },
    [onFileClick, onOpenArtifactFromTimeline, readTimelineArtifactContent],
  );

  return (
    <div className="py-1.5" data-testid="timeline-file-artifact-group">
      <FileChangesSummaryCard
        aggregate={aggregate}
        variant="timeline"
        onOpenFile={(file) => {
          const item = itemByPath.get(file.path);
          if (item) {
            openTimelineItem(item);
          }
        }}
      />
    </div>
  );
}
