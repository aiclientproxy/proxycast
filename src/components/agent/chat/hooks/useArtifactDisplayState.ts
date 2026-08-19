import { useEffect, useMemo, useRef, useState } from "react";
import type { Artifact } from "@/lib/artifact/types";
import { resolveArtifactProtocolFilePath } from "@/lib/artifact-protocol";
import { resolveArtifactWritePhase } from "../utils/messageArtifacts";

export type ArtifactDisplayMode =
  | "content"
  | "overlay-on-previous"
  | "typed-skeleton"
  | "empty-finished"
  | "error";

export type ArtifactOverlayPhase =
  | "creating"
  | "streaming_content"
  | "updating_file"
  | "finalized_empty"
  | "failed";

export interface ArtifactDisplayOverlayState {
  phase: ArtifactOverlayPhase;
  phaseLabel: string;
  title: string;
  detail: string;
  displayName: string;
  filePath: string;
  showProgress: boolean;
}

export interface ArtifactDisplayState {
  liveArtifact: Artifact | null;
  displayArtifact: Artifact | null;
  mode: ArtifactDisplayMode;
  overlay: ArtifactDisplayOverlayState | null;
  showPreviousVersionBadge: boolean;
}

export interface ResolveArtifactDisplayStateOptions {
  liveArtifact: Artifact | null;
  artifacts: Artifact[];
  previousRenderableArtifact?: Artifact | null;
  isSlowTransition?: boolean;
}

const SLOW_TRANSITION_THRESHOLD_MS = 900;

export function settleLiveArtifactAfterStreamStops(
  artifact: Artifact | null,
  options: {
    streamActive: boolean;
  },
): Artifact | null {
  if (!artifact || options.streamActive) {
    return artifact;
  }

  const writePhase = resolveArtifactWritePhase(artifact);
  const isPendingLike =
    artifact.status === "pending" ||
    artifact.status === "streaming" ||
    writePhase === "preparing" ||
    writePhase === "streaming" ||
    writePhase === "persisted";

  if (!isPendingLike || !artifact.content.trim()) {
    return artifact;
  }

  return {
    ...artifact,
    status: "complete",
    meta: {
      ...artifact.meta,
      writePhase: "completed",
      isPartial: false,
    },
  };
}

function hasRenderableArtifactContent(
  artifact: Artifact | null | undefined,
): boolean {
  return Boolean(artifact?.content.trim());
}

function resolveArtifactDisplayName(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  const segments = normalized.split("/");
  return segments[segments.length - 1] || path;
}

function artifactStillExists(
  artifact: Artifact | null | undefined,
  artifacts: Artifact[],
): artifact is Artifact {
  if (!artifact) {
    return false;
  }
  return artifacts.some((candidate) => candidate.id === artifact.id);
}

function findPreviousRenderableArtifact(
  liveArtifact: Artifact,
  artifacts: Artifact[],
  preferred: Artifact | null | undefined,
): Artifact | null {
  if (
    preferred &&
    preferred.id !== liveArtifact.id &&
    artifactStillExists(preferred, artifacts) &&
    hasRenderableArtifactContent(preferred)
  ) {
    return preferred;
  }

  for (let index = artifacts.length - 1; index >= 0; index -= 1) {
    const candidate = artifacts[index];
    if (candidate.id === liveArtifact.id) {
      continue;
    }
    if (hasRenderableArtifactContent(candidate)) {
      return candidate;
    }
  }

  return null;
}

function buildOverlayState(
  artifact: Artifact,
  phase: ArtifactOverlayPhase,
  options: { isSlowTransition: boolean },
): ArtifactDisplayOverlayState {
  const filePath = resolveArtifactProtocolFilePath(artifact);
  const displayName = resolveArtifactDisplayName(filePath);

  switch (phase) {
    case "creating":
      return {
        phase,
        phaseLabel: "准备写入",
        title: "正在创建文件",
        detail: options.isSlowTransition
          ? "文件已创建，正在生成首段内容。"
          : "正在准备首段内容，画布会在内容到达后立即切换。",
        displayName,
        filePath,
        showProgress: true,
      };
    case "streaming_content":
      return {
        phase,
        phaseLabel: "正在写入",
        title: "正在生成新版本",
        detail: options.isSlowTransition
          ? "内容还在流式生成中，当前先保留上一份可见版本。"
          : "新的内容片段正在写入，首段到达后会切换到最新版本。",
        displayName,
        filePath,
        showProgress: true,
      };
    case "updating_file":
      return {
        phase,
        phaseLabel: "已落盘",
        title: "正在同步最新内容",
        detail: "文件已经落盘，正在等待可渲染内容同步到画布。",
        displayName,
        filePath,
        showProgress: true,
      };
    case "finalized_empty":
      return {
        phase,
        phaseLabel: "已完成",
        title: "写入已结束",
        detail:
          "文件已经完成，但当前还没有可直接渲染的内容，暂时保留上一版本。",
        displayName,
        filePath,
        showProgress: false,
      };
    case "failed":
      return {
        phase,
        phaseLabel: "失败",
        title: "写入未完成",
        detail:
          artifact.error?.trim() ||
          "文件写入过程中出现异常，当前先保留上一份可见内容。",
        displayName,
        filePath,
        showProgress: false,
      };
  }
}

export function resolveArtifactDisplayState({
  liveArtifact,
  artifacts,
  previousRenderableArtifact,
  isSlowTransition = false,
}: ResolveArtifactDisplayStateOptions): ArtifactDisplayState {
  if (!liveArtifact) {
    return {
      liveArtifact: null,
      displayArtifact: null,
      mode: "content",
      overlay: null,
      showPreviousVersionBadge: false,
    };
  }

  if (hasRenderableArtifactContent(liveArtifact)) {
    return {
      liveArtifact,
      displayArtifact: liveArtifact,
      mode: "content",
      overlay: null,
      showPreviousVersionBadge: false,
    };
  }

  const writePhase = resolveArtifactWritePhase(liveArtifact);
  const previousArtifact = findPreviousRenderableArtifact(
    liveArtifact,
    artifacts,
    previousRenderableArtifact,
  );

  if (liveArtifact.status === "error" || writePhase === "failed") {
    return {
      liveArtifact,
      displayArtifact: previousArtifact || liveArtifact,
      mode: "error",
      overlay: previousArtifact
        ? buildOverlayState(liveArtifact, "failed", { isSlowTransition })
        : null,
      showPreviousVersionBadge: Boolean(previousArtifact),
    };
  }

  if (
    liveArtifact.status === "complete" ||
    writePhase === "completed" ||
    writePhase === "persisted"
  ) {
    return {
      liveArtifact,
      displayArtifact: previousArtifact || liveArtifact,
      mode: previousArtifact ? "overlay-on-previous" : "empty-finished",
      overlay: previousArtifact
        ? buildOverlayState(liveArtifact, "finalized_empty", {
            isSlowTransition,
          })
        : null,
      showPreviousVersionBadge: Boolean(previousArtifact),
    };
  }

  if (liveArtifact.status === "pending" || writePhase === "preparing") {
    return {
      liveArtifact,
      displayArtifact: previousArtifact || liveArtifact,
      mode: previousArtifact ? "overlay-on-previous" : "typed-skeleton",
      overlay: previousArtifact
        ? buildOverlayState(liveArtifact, "creating", { isSlowTransition })
        : null,
      showPreviousVersionBadge: Boolean(previousArtifact),
    };
  }

  if (liveArtifact.status === "streaming" || writePhase === "streaming") {
    return {
      liveArtifact,
      displayArtifact: previousArtifact || liveArtifact,
      mode: previousArtifact ? "overlay-on-previous" : "typed-skeleton",
      overlay: previousArtifact
        ? buildOverlayState(liveArtifact, "streaming_content", {
            isSlowTransition,
          })
        : null,
      showPreviousVersionBadge: Boolean(previousArtifact),
    };
  }

  return {
    liveArtifact,
    displayArtifact: previousArtifact || liveArtifact,
    mode: previousArtifact ? "overlay-on-previous" : "typed-skeleton",
    overlay: previousArtifact
      ? buildOverlayState(liveArtifact, "updating_file", { isSlowTransition })
      : null,
    showPreviousVersionBadge: Boolean(previousArtifact),
  };
}

export function useArtifactDisplayState(
  liveArtifact: Artifact | null,
  artifacts: Artifact[],
): ArtifactDisplayState {
  const lastRenderableArtifactRef = useRef<Artifact | null>(null);
  const [isSlowTransition, setIsSlowTransition] = useState(false);

  useEffect(() => {
    if (!liveArtifact && artifacts.length === 0) {
      lastRenderableArtifactRef.current = null;
      setIsSlowTransition(false);
      return;
    }

    if (hasRenderableArtifactContent(liveArtifact)) {
      lastRenderableArtifactRef.current = liveArtifact;
      setIsSlowTransition(false);
      return;
    }

    const writePhase = liveArtifact
      ? resolveArtifactWritePhase(liveArtifact)
      : null;
    const isPendingTransition =
      liveArtifact &&
      !hasRenderableArtifactContent(liveArtifact) &&
      (liveArtifact.status === "pending" ||
        liveArtifact.status === "streaming" ||
        writePhase === "preparing" ||
        writePhase === "streaming");

    if (!isPendingTransition) {
      setIsSlowTransition(false);
      return;
    }

    setIsSlowTransition(false);
    const timer = window.setTimeout(() => {
      setIsSlowTransition(true);
    }, SLOW_TRANSITION_THRESHOLD_MS);

    return () => {
      window.clearTimeout(timer);
    };
  }, [artifacts.length, liveArtifact]);

  return useMemo(
    () =>
      resolveArtifactDisplayState({
        liveArtifact,
        artifacts,
        previousRenderableArtifact: lastRenderableArtifactRef.current,
        isSlowTransition,
      }),
    [artifacts, isSlowTransition, liveArtifact],
  );
}
