import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  listProjectGitCommits,
  readProjectGitDiff,
  type ProjectGitCommitList,
} from "@/lib/api/projectGit";
import { resolveAbsoluteWorkspacePath } from "../../../workspace/workspacePath";
import type { HarnessFilePreviewResult } from "../../HarnessStatusPanel";
import type {
  CanvasWorkbenchChangeItem,
  CanvasWorkbenchChangeView,
} from "./CanvasWorkbenchChangesPanelViewModel";
import {
  buildCanvasWorkbenchGitApplyPatch,
  parseCanvasWorkbenchGitPatchToChangeItems,
} from "./CanvasWorkbenchChangesPanelViewModel";
import type {
  CanvasWorkbenchReviewBase,
  CanvasWorkbenchReviewCommitOption,
  CanvasWorkbenchTranslation,
} from "./CanvasWorkbenchChangesTypes";

interface CanvasWorkbenchReviewStateInput {
  changeView: CanvasWorkbenchChangeView | null | undefined;
  workspaceRoot?: string | null;
  loadFilePreview?: (path: string) => Promise<HarnessFilePreviewResult>;
  translateWorkbench: CanvasWorkbenchTranslation;
}

function buildGitApplyCommand(patch: string): string {
  return `git apply <<'PATCH'\n${patch.trimEnd()}\nPATCH\n`;
}

function resolveBackendPatchForCopy(
  selectedBase: CanvasWorkbenchReviewBase,
  backendPatch: string | null,
  fallbackPatch: string,
): string {
  return selectedBase === "previousConversation"
    ? fallbackPatch
    : backendPatch || "";
}

function projectGitCommitsToReviewOptions(
  commits: ProjectGitCommitList["commits"],
): CanvasWorkbenchReviewCommitOption[] {
  return commits.map((commit) => ({
    sha: commit.sha,
    shortSha: commit.shortSha,
    subject: commit.subject,
    committedAt: commit.committedAt,
  }));
}

export function useCanvasWorkbenchReviewState({
  changeView,
  workspaceRoot,
  loadFilePreview,
  translateWorkbench,
}: CanvasWorkbenchReviewStateInput) {
  const [selectedBase, setSelectedBase] = useState<CanvasWorkbenchReviewBase>(
    "previousConversation",
  );
  const [reviewActionBusy, setReviewActionBusy] = useState(false);
  const [backendPatch, setBackendPatch] = useState<string | null>(null);
  const [backendChangeItems, setBackendChangeItems] = useState<
    CanvasWorkbenchChangeItem[]
  >([]);
  const [backendHasGitRepository, setBackendHasGitRepository] = useState<
    boolean | null
  >(null);
  const [backendRepositoryRoot, setBackendRepositoryRoot] = useState<
    string | null
  >(null);
  const [branchCompareLabel, setBranchCompareLabel] = useState<string | null>(
    null,
  );
  const [commitOptions, setCommitOptions] = useState<
    CanvasWorkbenchReviewCommitOption[]
  >([]);
  const [commitsLoading, setCommitsLoading] = useState(false);
  const [selectedCommitSha, setSelectedCommitSha] = useState<string | null>(
    null,
  );
  const [fullFileContentById, setFullFileContentById] = useState<
    Record<string, string>
  >({});
  const [loadFullFile, setLoadFullFile] = useState(false);

  const selectedBaseUsesGit = selectedBase !== "previousConversation";
  const changeItems = useMemo(
    () =>
      selectedBaseUsesGit ? backendChangeItems : (changeView?.items ?? []),
    [backendChangeItems, changeView, selectedBaseUsesGit],
  );
  const fallbackPatch = useMemo(
    () =>
      changeView?.turnDiff !== undefined
        ? changeView.turnDiff
        : buildCanvasWorkbenchGitApplyPatch(changeItems),
    [changeItems, changeView?.turnDiff],
  );

  const loadGitDiffBase = useCallback(
    async (base: CanvasWorkbenchReviewBase, commitSha?: string) => {
      if (base === "previousConversation") {
        setBackendPatch(null);
        setBackendChangeItems([]);
        setBackendHasGitRepository(null);
        setBackendRepositoryRoot(null);
        setBranchCompareLabel(null);
        setSelectedBase(base);
        setSelectedCommitSha(null);
        return true;
      }
      if (!workspaceRoot) {
        toast.error(
          translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.toast.missingWorkspaceRoot",
          ),
        );
        return false;
      }

      setReviewActionBusy(true);
      try {
        const diff = await readProjectGitDiff(
          workspaceRoot,
          3,
          base,
          commitSha,
        );
        setBackendPatch(diff.patch);
        setBackendHasGitRepository(diff.hasGitRepository);
        setBackendRepositoryRoot(diff.repositoryRoot ?? null);
        setBranchCompareLabel(
          base === "branch" && diff.currentRef && diff.comparisonBaseRef
            ? `${diff.currentRef} -> ${diff.comparisonBaseRef}`
            : null,
        );
        setBackendChangeItems(
          parseCanvasWorkbenchGitPatchToChangeItems(diff.patch),
        );
        setSelectedBase(base);
        if (base !== "commit") {
          setSelectedCommitSha(null);
        }
        toast.success(
          translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.toast.refreshed",
          ),
        );
        return true;
      } catch (error) {
        setBackendPatch(null);
        setBackendChangeItems([]);
        setBackendHasGitRepository(null);
        setBackendRepositoryRoot(null);
        setBranchCompareLabel(null);
        toast.error(
          error instanceof Error
            ? error.message
            : translateWorkbench(
                "agentChat.canvasWorkbench.coding.changes.toast.refreshFailed",
              ),
        );
        return false;
      } finally {
        setReviewActionBusy(false);
      }
    },
    [translateWorkbench, workspaceRoot],
  );

  const loadGitCommits = useCallback(async () => {
    if (!workspaceRoot) {
      toast.error(
        translateWorkbench(
          "agentChat.canvasWorkbench.coding.changes.toast.missingWorkspaceRoot",
        ),
      );
      return false;
    }
    setCommitsLoading(true);
    try {
      const result = await listProjectGitCommits(workspaceRoot, 30);
      setCommitOptions(projectGitCommitsToReviewOptions(result.commits));
      return true;
    } catch (error) {
      setCommitOptions([]);
      toast.error(
        error instanceof Error
          ? error.message
          : translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.toast.refreshFailed",
            ),
      );
      return false;
    } finally {
      setCommitsLoading(false);
    }
  }, [translateWorkbench, workspaceRoot]);

  const selectBase = useCallback(
    async (base: CanvasWorkbenchReviewBase) => {
      if (base === "commit") {
        return false;
      }
      return loadGitDiffBase(base);
    },
    [loadGitDiffBase],
  );

  const openCommitMenu = useCallback(() => {
    if (commitOptions.length === 0 && !commitsLoading) {
      void loadGitCommits();
    }
  }, [commitOptions.length, commitsLoading, loadGitCommits]);

  const selectCommit = useCallback(
    (commit: CanvasWorkbenchReviewCommitOption) => {
      setSelectedCommitSha(commit.sha);
      void loadGitDiffBase("commit", commit.sha);
    },
    [loadGitDiffBase],
  );

  const refreshChanges = useCallback(async () => {
    if (selectedBase === "commit") {
      if (selectedCommitSha) {
        await loadGitDiffBase("commit", selectedCommitSha);
      } else {
        await loadGitCommits();
      }
      return;
    }
    if (selectedBaseUsesGit) {
      await loadGitDiffBase(selectedBase);
      return;
    }
    if (!workspaceRoot) {
      toast.error(
        translateWorkbench(
          "agentChat.canvasWorkbench.coding.changes.toast.missingWorkspaceRoot",
        ),
      );
      return;
    }

    setReviewActionBusy(true);
    try {
      const diff = await readProjectGitDiff(workspaceRoot, 3);
      setBackendPatch(diff.patch);
      setBackendHasGitRepository(diff.hasGitRepository);
      setBackendRepositoryRoot(diff.repositoryRoot ?? null);
      toast.success(
        translateWorkbench(
          "agentChat.canvasWorkbench.coding.changes.toast.refreshed",
        ),
      );
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.toast.refreshFailed",
            ),
      );
    } finally {
      setReviewActionBusy(false);
    }
  }, [
    loadGitDiffBase,
    loadGitCommits,
    selectedBase,
    selectedBaseUsesGit,
    selectedCommitSha,
    translateWorkbench,
    workspaceRoot,
  ]);

  const copyGitApply = useCallback(async () => {
    const patch = resolveBackendPatchForCopy(
      selectedBase,
      backendPatch,
      fallbackPatch,
    );
    if (!patch.trim()) {
      toast.error(
        translateWorkbench(
          "agentChat.canvasWorkbench.coding.changes.toast.noPatch",
        ),
      );
      return;
    }
    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      toast.error(
        translateWorkbench("agentChat.canvasWorkbench.clipboard.unsupported"),
      );
      return;
    }

    setReviewActionBusy(true);
    try {
      await navigator.clipboard.writeText(buildGitApplyCommand(patch));
      toast.success(
        translateWorkbench(
          "agentChat.canvasWorkbench.coding.changes.toast.gitApplyCopied",
        ),
      );
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.toast.gitApplyCopyFailed",
            ),
      );
    } finally {
      setReviewActionBusy(false);
    }
  }, [backendPatch, fallbackPatch, selectedBase, translateWorkbench]);

  const toggleLoadFullFile = useCallback(
    async (selectedChangeItem: CanvasWorkbenchChangeItem | undefined) => {
      if (
        selectedChangeItem &&
        loadFullFile &&
        fullFileContentById[selectedChangeItem.id] != null
      ) {
        setLoadFullFile(false);
        return;
      }
      if (!selectedChangeItem || !loadFilePreview) {
        toast.error(
          translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.toast.fullFileLoadFailed",
          ),
        );
        return;
      }
      if (fullFileContentById[selectedChangeItem.id] != null) {
        setLoadFullFile(true);
        return;
      }

      const previewPath = resolveAbsoluteWorkspacePath(
        workspaceRoot,
        selectedChangeItem.absolutePath || selectedChangeItem.path,
      );
      if (!previewPath) {
        toast.error(
          translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.toast.fullFileLoadFailed",
          ),
        );
        setLoadFullFile(false);
        return;
      }

      setReviewActionBusy(true);
      try {
        const preview = await loadFilePreview(previewPath);
        if (preview.error || preview.isBinary || preview.content == null) {
          toast.error(
            preview.error ||
              translateWorkbench(
                "agentChat.canvasWorkbench.coding.changes.toast.fullFileLoadFailed",
              ),
          );
          setLoadFullFile(false);
          return;
        }
        setFullFileContentById((current) => ({
          ...current,
          [selectedChangeItem.id]: preview.content || "",
        }));
        setLoadFullFile(true);
      } catch (error) {
        toast.error(
          error instanceof Error
            ? error.message
            : translateWorkbench(
                "agentChat.canvasWorkbench.coding.changes.toast.fullFileLoadFailed",
              ),
        );
        setLoadFullFile(false);
      } finally {
        setReviewActionBusy(false);
      }
    },
    [
      fullFileContentById,
      loadFilePreview,
      loadFullFile,
      translateWorkbench,
      workspaceRoot,
    ],
  );

  return {
    backendHasGitRepository,
    backendPatch,
    backendRepositoryRoot,
    branchCompareLabel,
    changeItems,
    commitOptions,
    commitsLoading,
    copyGitApply,
    fallbackPatch,
    fullFileContentById,
    loadFullFile,
    openCommitMenu,
    refreshChanges,
    reviewActionBusy,
    selectBase,
    selectCommit,
    selectedBase,
    selectedBaseUsesGit,
    selectedCommitSha,
    toggleLoadFullFile,
  };
}
