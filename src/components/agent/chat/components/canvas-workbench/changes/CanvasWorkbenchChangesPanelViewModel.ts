import {
  normalizeCanvasWorkbenchPath,
  resolvePreviewPath,
  type CanvasWorkbenchSelectionContextLike,
} from "../../CanvasWorkbenchLayoutViewModel";
import { extractFileNameFromPath } from "../../../workspace/workspacePath";
import {
  buildCanvasWorkbenchDiff,
  type CanvasWorkbenchDiffLine,
} from "../../../utils/canvasWorkbenchDiff";
import type { AgentThreadItem } from "@/lib/api/agentProtocolCoreTypes";

export type CanvasWorkbenchChangeKind =
  | "added"
  | "modified"
  | "deleted"
  | "renamed"
  | "copied"
  | "unknown";

export interface CanvasWorkbenchChangeItem {
  id: string;
  path: string;
  absolutePath?: string | null;
  displayName?: string;
  source?: string;
  status?: "in_progress" | "completed" | "failed";
  changeKind?: CanvasWorkbenchChangeKind | string | null;
  preview?: string;
  diffLines?: CanvasWorkbenchDiffLine[];
  currentContent?: string | null;
  previousContent?: string | null;
  checkpointPath?: string | null;
  checkpointLabel?: string | null;
  hunkStartLine?: number;
  hunkEndLine?: number;
  hunkHeader?: string;
}

export interface CanvasWorkbenchChangeView {
  items: CanvasWorkbenchChangeItem[];
  checkpointCount?: number;
  latestCheckpointPath?: string | null;
  onOpenFile?: (path: string) => void | Promise<void>;
  threadItems?: readonly AgentThreadItem[];
  reviewThreadId?: string | null;
  currentTurnId?: string | null;
  turnDiff?: string;
  canInterrupt?: boolean;
  onStartReview?: () => Promise<unknown>;
}

export interface CanvasWorkbenchChangeDiffStats {
  additions: number;
  removals: number;
}

export interface CanvasWorkbenchChangeDisplayMeta {
  kind: CanvasWorkbenchChangeKind;
  shortLabelKey: string;
  labelKey: string;
  className: string;
}

export interface CanvasWorkbenchChangeTreeFileNode {
  type: "file";
  id: string;
  name: string;
  path: string;
  depth: number;
  item: CanvasWorkbenchChangeItem;
}

export interface CanvasWorkbenchChangeTreeFolderNode {
  type: "folder";
  id: string;
  name: string;
  path: string;
  depth: number;
  children: CanvasWorkbenchChangeTreeNode[];
}

export type CanvasWorkbenchChangeTreeNode =
  | CanvasWorkbenchChangeTreeFileNode
  | CanvasWorkbenchChangeTreeFolderNode;

export function isPendingChangeItem(item: CanvasWorkbenchChangeItem): boolean {
  return item.status === "in_progress";
}

export function resolveChangeItemDisplayName(
  item: CanvasWorkbenchChangeItem,
): string {
  return item.displayName?.trim() || extractFileNameFromPath(item.path);
}

function normalizeChangeItemPathForMatch(value: string | null | undefined) {
  return normalizeCanvasWorkbenchPath(value || "")
    .trim()
    .toLowerCase();
}

export function findChangeItemForSelection(
  items: CanvasWorkbenchChangeItem[],
  context: CanvasWorkbenchSelectionContextLike | null,
): CanvasWorkbenchChangeItem | null {
  if (!context) {
    return null;
  }

  const selectionCandidates = [
    context.selectionPath,
    resolvePreviewPath(context.target),
    context.subtitle,
    context.title,
  ]
    .map(normalizeChangeItemPathForMatch)
    .filter(Boolean);

  return (
    items.find((item) => {
      const itemCandidates = [
        item.path,
        item.absolutePath,
        resolveChangeItemDisplayName(item),
      ]
        .map(normalizeChangeItemPathForMatch)
        .filter(Boolean);

      return itemCandidates.some((candidate) =>
        selectionCandidates.includes(candidate),
      );
    }) || null
  );
}

function hasChangeItemReviewEvidence(item: CanvasWorkbenchChangeItem): boolean {
  return Boolean(
    item.preview?.trim() ||
    item.diffLines?.length ||
    item.currentContent != null ||
    item.previousContent != null ||
    item.checkpointPath?.trim(),
  );
}

function findLatestRuntimeChangeItemWithEvidence(
  items: readonly CanvasWorkbenchChangeItem[],
): CanvasWorkbenchChangeItem | undefined {
  const itemsByLatest = [...items].reverse();
  return (
    itemsByLatest.find(
      (item) => item.source === "runtime" && hasChangeItemReviewEvidence(item),
    ) || itemsByLatest.find((item) => item.source === "runtime")
  );
}

export function resolveCanvasWorkbenchSelectedChangeItem({
  items,
  selectedChangeId,
  activeSelectionChangeItem,
  preferRuntimeEvidenceDefault = false,
}: {
  items: readonly CanvasWorkbenchChangeItem[];
  selectedChangeId: string | null;
  activeSelectionChangeItem: CanvasWorkbenchChangeItem | null;
  preferRuntimeEvidenceDefault?: boolean;
}): CanvasWorkbenchChangeItem | undefined {
  const explicitSelection = selectedChangeId
    ? items.find((item) => item.id === selectedChangeId)
    : undefined;
  if (explicitSelection) {
    return explicitSelection;
  }

  const latestRuntimeEvidenceItem =
    findLatestRuntimeChangeItemWithEvidence(items);
  if (preferRuntimeEvidenceDefault && latestRuntimeEvidenceItem) {
    if (
      activeSelectionChangeItem?.source === "runtime" &&
      hasChangeItemReviewEvidence(activeSelectionChangeItem) &&
      activeSelectionChangeItem.previousContent != null
    ) {
      return activeSelectionChangeItem;
    }
    return latestRuntimeEvidenceItem;
  }

  return activeSelectionChangeItem || latestRuntimeEvidenceItem || items[0];
}

export function resolveChangeStatusCopyKey(
  item: CanvasWorkbenchChangeItem,
): string {
  if (item.status === "failed") {
    return "agentChat.canvasWorkbench.coding.changes.status.failed";
  }
  if (item.status === "in_progress") {
    return "agentChat.canvasWorkbench.coding.changes.status.inProgress";
  }
  return "agentChat.canvasWorkbench.coding.changes.status.completed";
}

export function resolveChangeStatusClassName(
  item: CanvasWorkbenchChangeItem,
): string {
  if (item.status === "failed") {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }
  if (item.status === "in_progress") {
    return "border-sky-200 bg-sky-50 text-sky-700";
  }
  if (item.status === "completed") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  return "border-slate-200 bg-slate-50 text-slate-600";
}

export function normalizeCanvasWorkbenchChangeKind(
  value: CanvasWorkbenchChangeItem["changeKind"],
): CanvasWorkbenchChangeKind {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_\s-]+/g, "");
  if (
    normalized === "add" ||
    normalized === "added" ||
    normalized === "new" ||
    normalized === "create" ||
    normalized === "created"
  ) {
    return "added";
  }
  if (
    normalized === "delete" ||
    normalized === "deleted" ||
    normalized === "remove" ||
    normalized === "removed"
  ) {
    return "deleted";
  }
  if (
    normalized === "rename" ||
    normalized === "renamed" ||
    normalized === "move"
  ) {
    return "renamed";
  }
  if (normalized === "copy" || normalized === "copied") {
    return "copied";
  }
  if (
    normalized === "modify" ||
    normalized === "modified" ||
    normalized === "change" ||
    normalized === "changed" ||
    normalized === "edit" ||
    normalized === "edited"
  ) {
    return "modified";
  }
  return "unknown";
}

export function inferCanvasWorkbenchChangeKind(
  item: CanvasWorkbenchChangeItem,
): CanvasWorkbenchChangeKind {
  const explicitKind = normalizeCanvasWorkbenchChangeKind(item.changeKind);
  if (explicitKind !== "unknown") {
    return explicitKind;
  }
  if (item.previousContent === null && item.currentContent !== null) {
    return "added";
  }
  if (item.previousContent !== null && item.currentContent === null) {
    return "deleted";
  }
  if (item.previousContent != null || item.currentContent != null) {
    return "modified";
  }
  return "unknown";
}

export function resolveChangeDisplayMeta(
  item: CanvasWorkbenchChangeItem,
): CanvasWorkbenchChangeDisplayMeta {
  const kind = inferCanvasWorkbenchChangeKind(item);
  if (kind === "added") {
    return {
      kind,
      shortLabelKey: "agentChat.canvasWorkbench.coding.changes.kindShort.added",
      labelKey: "agentChat.canvasWorkbench.coding.changes.kind.added",
      className: "border-emerald-200 bg-emerald-50 text-emerald-700",
    };
  }
  if (kind === "deleted") {
    return {
      kind,
      shortLabelKey:
        "agentChat.canvasWorkbench.coding.changes.kindShort.deleted",
      labelKey: "agentChat.canvasWorkbench.coding.changes.kind.deleted",
      className: "border-rose-200 bg-rose-50 text-rose-700",
    };
  }
  if (kind === "renamed") {
    return {
      kind,
      shortLabelKey:
        "agentChat.canvasWorkbench.coding.changes.kindShort.renamed",
      labelKey: "agentChat.canvasWorkbench.coding.changes.kind.renamed",
      className: "border-sky-200 bg-sky-50 text-sky-700",
    };
  }
  if (kind === "copied") {
    return {
      kind,
      shortLabelKey:
        "agentChat.canvasWorkbench.coding.changes.kindShort.copied",
      labelKey: "agentChat.canvasWorkbench.coding.changes.kind.copied",
      className: "border-amber-200 bg-amber-50 text-amber-700",
    };
  }
  if (kind === "modified") {
    return {
      kind,
      shortLabelKey:
        "agentChat.canvasWorkbench.coding.changes.kindShort.modified",
      labelKey: "agentChat.canvasWorkbench.coding.changes.kind.modified",
      className: "border-slate-200 bg-slate-50 text-slate-700",
    };
  }
  return {
    kind,
    shortLabelKey: "agentChat.canvasWorkbench.coding.changes.kindShort.unknown",
    labelKey: "agentChat.canvasWorkbench.coding.changes.kind.unknown",
    className: "border-slate-200 bg-white text-slate-500",
  };
}

export function countCanvasWorkbenchDiffStats(
  diffLines: CanvasWorkbenchDiffLine[],
): CanvasWorkbenchChangeDiffStats {
  return diffLines.reduce<CanvasWorkbenchChangeDiffStats>(
    (stats, line) => {
      if (line.type === "add") {
        stats.additions += 1;
      }
      if (line.type === "remove") {
        stats.removals += 1;
      }
      return stats;
    },
    { additions: 0, removals: 0 },
  );
}

export function countCanvasWorkbenchChangeItemStats(
  item: CanvasWorkbenchChangeItem,
): CanvasWorkbenchChangeDiffStats {
  if (item.diffLines?.length) {
    return countCanvasWorkbenchDiffStats(item.diffLines);
  }
  if (item.previousContent != null && item.currentContent != null) {
    return countCanvasWorkbenchDiffStats(
      item.previousContent === item.currentContent
        ? []
        : buildCanvasWorkbenchDiff(item.previousContent, item.currentContent),
    );
  }
  if (item.previousContent === null && item.currentContent != null) {
    return {
      additions: splitChangeContentLines(item.currentContent).length,
      removals: 0,
    };
  }
  if (item.previousContent != null && item.currentContent === null) {
    return {
      additions: 0,
      removals: splitChangeContentLines(item.previousContent).length,
    };
  }
  return { additions: 0, removals: 0 };
}

export function countCanvasWorkbenchChangeItemsStats(
  items: readonly CanvasWorkbenchChangeItem[],
): CanvasWorkbenchChangeDiffStats {
  return items.reduce<CanvasWorkbenchChangeDiffStats>(
    (stats, item) => {
      const itemStats = countCanvasWorkbenchChangeItemStats(item);
      stats.additions += itemStats.additions;
      stats.removals += itemStats.removals;
      return stats;
    },
    { additions: 0, removals: 0 },
  );
}

export function buildCanvasWorkbenchChangeDiffLines(
  item: CanvasWorkbenchChangeItem | undefined,
  loadedContent?: string,
): CanvasWorkbenchDiffLine[] {
  if (!item) {
    return [];
  }
  if (item.diffLines?.length) {
    return item.diffLines;
  }
  const currentContent = loadedContent ?? item.currentContent;
  if (item.previousContent != null && currentContent != null) {
    return buildCanvasWorkbenchDiff(item.previousContent, currentContent);
  }
  if (item.previousContent === null && currentContent != null) {
    return buildCanvasWorkbenchDiff("", currentContent);
  }
  if (item.previousContent != null && currentContent === null) {
    return buildCanvasWorkbenchDiff(item.previousContent, "");
  }
  return [];
}

export function parseCanvasWorkbenchGitPatchToChangeItems(
  patch: string,
): CanvasWorkbenchChangeItem[] {
  const sections = splitGitPatchSections(patch);
  return sections.flatMap((section, index) => {
    const item = parseGitPatchSection(section, index);
    return item ? [item] : [];
  });
}

function splitGitPatchSections(patch: string): string[][] {
  const sections: string[][] = [];
  let current: string[] = [];

  patch
    .replace(/\r\n/g, "\n")
    .split("\n")
    .forEach((line) => {
      if (line.startsWith("diff --git ")) {
        if (current.length > 0) {
          sections.push(current);
        }
        current = [line];
        return;
      }
      if (current.length > 0) {
        current.push(line);
      }
    });

  if (current.length > 0) {
    sections.push(current);
  }
  return sections;
}

function parseGitPatchSection(
  lines: string[],
  index: number,
): CanvasWorkbenchChangeItem | null {
  const diffHeader = lines.find((line) => line.startsWith("diff --git "));
  if (!diffHeader) {
    return null;
  }

  const oldPath = parseDiffHeaderPath(diffHeader, "a/");
  const newPath = parseDiffHeaderPath(diffHeader, "b/");
  const minusPath = lines
    .find((line) => line.startsWith("--- "))
    ?.slice(4)
    .trim();
  const plusPath = lines
    .find((line) => line.startsWith("+++ "))
    ?.slice(4)
    .trim();
  const path =
    normalizeGitPatchPath(plusPath) ||
    normalizeGitPatchPath(newPath) ||
    normalizeGitPatchPath(minusPath) ||
    normalizeGitPatchPath(oldPath);
  if (!path) {
    return null;
  }

  const diffLines = parseGitPatchDiffLines(lines);
  const preview = diffLines.find((line) => line.type !== "context")?.value;
  const changeKind = inferGitPatchChangeKind(lines);
  const hunk = parseGitPatchHunk(lines);

  return {
    id: `git:${index}:${path}`,
    path,
    displayName: extractFileNameFromPath(path),
    source: "git",
    status: "completed",
    changeKind,
    preview,
    diffLines,
    hunkStartLine: hunk?.newStart ?? undefined,
    hunkEndLine:
      hunk && hunk.newLength > 0
        ? hunk.newStart + hunk.newLength - 1
        : (hunk?.newStart ?? undefined),
    hunkHeader: hunk?.header,
  };
}

interface ParsedGitPatchHunk {
  header: string;
  oldStart: number;
  oldLength: number;
  newStart: number;
  newLength: number;
}

function parseGitPatchHunk(lines: string[]): ParsedGitPatchHunk | null {
  const hunkLine = lines.find((line) => line.startsWith("@@"));
  if (!hunkLine) {
    return null;
  }

  const match = hunkLine.match(
    /^@@ -(?<oldStart>\d+)(?:,(?<oldLength>\d+))? \+(?<newStart>\d+)(?:,(?<newLength>\d+))? @@/,
  );
  if (!match?.groups) {
    return null;
  }

  const oldStart = Number.parseInt(match.groups.oldStart || "0", 10);
  const oldLength = Number.parseInt(match.groups.oldLength || "1", 10);
  const newStart = Number.parseInt(match.groups.newStart || "0", 10);
  const newLength = Number.parseInt(match.groups.newLength || "1", 10);

  return {
    header: hunkLine,
    oldStart: Number.isFinite(oldStart) ? oldStart : 0,
    oldLength: Number.isFinite(oldLength) ? oldLength : 0,
    newStart: Number.isFinite(newStart) ? newStart : 0,
    newLength: Number.isFinite(newLength) ? newLength : 0,
  };
}

function parseDiffHeaderPath(header: string, prefix: "a/" | "b/"): string {
  const marker = ` ${prefix}`;
  const index = header.indexOf(marker);
  if (index < 0) {
    return "";
  }
  return header.slice(index + 1).split(/\s+/)[0] || "";
}

function normalizeGitPatchPath(value: string | null | undefined): string {
  const trimmed = (value || "").trim();
  if (!trimmed || trimmed === "/dev/null" || trimmed === "NUL") {
    return "";
  }
  return normalizeCanvasWorkbenchPath(
    trimmed.replace(/^"|"$/g, "").replace(/^[ab]\//, ""),
  ).replace(/^\/+/, "");
}

function parseGitPatchDiffLines(lines: string[]): CanvasWorkbenchDiffLine[] {
  const diffLines: CanvasWorkbenchDiffLine[] = [];
  let inHunk = false;
  let oldLineNumber = 0;
  let newLineNumber = 0;

  lines.forEach((line) => {
    if (line.startsWith("@@")) {
      inHunk = true;
      const hunk = parseGitPatchHunk([line]);
      oldLineNumber = hunk?.oldStart ?? 0;
      newLineNumber = hunk?.newStart ?? 0;
      return;
    }
    if (!inHunk || line.startsWith("\\ No newline")) {
      return;
    }
    if (line.startsWith("+") && !line.startsWith("+++ ")) {
      diffLines.push({
        type: "add",
        value: line.slice(1),
        oldLineNumber: null,
        newLineNumber,
      });
      newLineNumber += 1;
      return;
    }
    if (line.startsWith("-") && !line.startsWith("--- ")) {
      diffLines.push({
        type: "remove",
        value: line.slice(1),
        oldLineNumber,
        newLineNumber: null,
      });
      oldLineNumber += 1;
      return;
    }
    if (line.startsWith(" ")) {
      diffLines.push({
        type: "context",
        value: line.slice(1),
        oldLineNumber,
        newLineNumber,
      });
      oldLineNumber += 1;
      newLineNumber += 1;
    }
  });

  return diffLines;
}

function inferGitPatchChangeKind(lines: string[]): CanvasWorkbenchChangeKind {
  if (lines.some((line) => line.startsWith("new file mode "))) {
    return "added";
  }
  if (lines.some((line) => line.startsWith("deleted file mode "))) {
    return "deleted";
  }
  if (lines.some((line) => line.startsWith("rename from "))) {
    return "renamed";
  }
  if (lines.some((line) => line.startsWith("copy from "))) {
    return "copied";
  }
  return "modified";
}

export function buildCanvasWorkbenchGitApplyPatch(
  items: CanvasWorkbenchChangeItem[],
): string {
  return items
    .map(buildCanvasWorkbenchGitApplyPatchForItem)
    .filter(Boolean)
    .join("\n");
}

function splitChangeContentLines(content: string): string[] {
  if (!content) {
    return [];
  }
  return content.replace(/\r\n/g, "\n").split("\n");
}

function buildCanvasWorkbenchGitApplyPatchForItem(
  item: CanvasWorkbenchChangeItem,
): string {
  const kind = inferCanvasWorkbenchChangeKind(item);
  const oldContent = item.previousContent ?? "";
  const newContent = item.currentContent ?? "";
  const diffLines = buildCanvasWorkbenchDiff(oldContent, newContent);
  if (diffLines.length === 0) {
    return "";
  }

  const normalizedPath = normalizeCanvasWorkbenchPath(item.path).replace(
    /^\/+/,
    "",
  );
  if (!normalizedPath) {
    return "";
  }

  const oldLineCount = splitChangeContentLines(oldContent).length;
  const newLineCount = splitChangeContentLines(newContent).length;
  const lines = [`diff --git a/${normalizedPath} b/${normalizedPath}`];
  if (kind === "added") {
    lines.push(
      "new file mode 100644",
      "--- /dev/null",
      `+++ b/${normalizedPath}`,
    );
  } else if (kind === "deleted") {
    lines.push(
      "deleted file mode 100644",
      `--- a/${normalizedPath}`,
      "+++ /dev/null",
    );
  } else {
    lines.push(`--- a/${normalizedPath}`, `+++ b/${normalizedPath}`);
  }
  lines.push(
    `@@ -${oldLineCount > 0 ? 1 : 0},${oldLineCount} +${newLineCount > 0 ? 1 : 0},${newLineCount} @@`,
  );
  lines.push(...diffLines.map(formatCanvasWorkbenchPatchLine));
  return lines.join("\n");
}

function formatCanvasWorkbenchPatchLine(line: CanvasWorkbenchDiffLine): string {
  if (line.type === "add") {
    return `+${line.value}`;
  }
  if (line.type === "remove") {
    return `-${line.value}`;
  }
  return ` ${line.value}`;
}

function splitChangeItemPath(path: string): string[] {
  return normalizeCanvasWorkbenchPath(path)
    .split("/")
    .map((segment) => segment.trim())
    .filter(Boolean);
}

function normalizeChangeSearchQuery(value: string): string {
  return value.trim().toLowerCase();
}

function changeItemMatchesSearch(
  item: CanvasWorkbenchChangeItem,
  query: string,
): boolean {
  if (!query) {
    return true;
  }

  return [
    item.path,
    item.absolutePath,
    item.displayName,
    item.source,
    item.preview,
    resolveChangeItemDisplayName(item),
  ]
    .filter(Boolean)
    .some((value) => String(value).toLowerCase().includes(query));
}

function sortChangeTreeNodes(
  nodes: CanvasWorkbenchChangeTreeNode[],
): CanvasWorkbenchChangeTreeNode[] {
  return [...nodes].sort((left, right) => {
    if (left.type !== right.type) {
      return left.type === "folder" ? -1 : 1;
    }
    return left.name.localeCompare(right.name, undefined, {
      sensitivity: "base",
    });
  });
}

export function buildCanvasWorkbenchChangeFileTree(
  items: CanvasWorkbenchChangeItem[],
  searchQuery = "",
): CanvasWorkbenchChangeTreeNode[] {
  const query = normalizeChangeSearchQuery(searchQuery);
  const roots: CanvasWorkbenchChangeTreeNode[] = [];
  const folders = new Map<string, CanvasWorkbenchChangeTreeFolderNode>();

  items
    .filter((item) => changeItemMatchesSearch(item, query))
    .forEach((item) => {
      const segments = splitChangeItemPath(item.path);
      const fileName =
        resolveChangeItemDisplayName(item) ||
        (segments.length > 0 ? segments[segments.length - 1] : item.path);
      let currentChildren = roots;
      let currentPath = "";

      segments.slice(0, -1).forEach((segment, depth) => {
        currentPath = currentPath ? `${currentPath}/${segment}` : segment;
        const existingFolder = folders.get(currentPath);
        if (existingFolder) {
          currentChildren = existingFolder.children;
          return;
        }

        const folderNode: CanvasWorkbenchChangeTreeFolderNode = {
          type: "folder",
          id: `folder:${currentPath}`,
          name: segment,
          path: currentPath,
          depth,
          children: [],
        };
        folders.set(currentPath, folderNode);
        currentChildren.push(folderNode);
        currentChildren = folderNode.children;
      });

      currentChildren.push({
        type: "file",
        id: `file:${item.id}`,
        name: fileName,
        path: item.path,
        depth: Math.max(segments.length - 1, 0),
        item,
      });
    });

  const sortNested = (
    nodes: CanvasWorkbenchChangeTreeNode[],
  ): CanvasWorkbenchChangeTreeNode[] =>
    sortChangeTreeNodes(nodes).map((node) => {
      if (node.type === "file") {
        return node;
      }
      return {
        ...node,
        children: sortNested(node.children),
      };
    });

  return sortNested(roots);
}

export function flattenCanvasWorkbenchChangeFileTree(
  nodes: CanvasWorkbenchChangeTreeNode[],
): CanvasWorkbenchChangeItem[] {
  return nodes.flatMap((node) =>
    node.type === "file"
      ? [node.item]
      : flattenCanvasWorkbenchChangeFileTree(node.children),
  );
}
