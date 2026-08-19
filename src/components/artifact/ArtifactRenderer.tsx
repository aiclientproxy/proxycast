/**
 * @file Artifact 统一渲染入口组件
 * @description 根据 Artifact 类型分发到对应的渲染器，支持 Canvas 类型委托、错误边界和流式状态处理
 * @module components/artifact/ArtifactRenderer
 * @requirements 3.4, 11.1, 11.2, 11.3, 12.1, 14.2
 */

import React, {
  Suspense,
  memo,
  useCallback,
  useState,
  useEffect,
  useRef,
} from "react";
import { useTranslation } from "react-i18next";
import {
  AlertCircle,
  AlertTriangle,
  FileCode2,
  FileText,
  LayoutTemplate,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { artifactRegistry } from "@/lib/artifact/registry";
import { useDebouncedValue } from "@/lib/artifact/hooks";
import {
  resolveArtifactProtocolDocumentPayload,
  resolveArtifactProtocolFilePath,
} from "@/lib/artifact-protocol";
import {
  resolveDefaultArtifactViewMode,
  resolveArtifactWritePhase,
} from "@/components/agent/chat/utils/messageArtifacts";
import { ErrorFallbackRenderer } from "./ErrorFallbackRenderer";
import {
  PreviewArtifactFallbackSurface,
  type PreviewArtifactFallbackMode,
} from "./PreviewArtifactFallbackSurface";
import {
  PreviewMediaRenderer,
  type PreviewMediaContentKind,
} from "./PreviewMediaRenderer";
import {
  PreviewSourceSummaryRenderer,
  type PreviewSourceSummaryKind,
} from "./PreviewSourceSummaryRenderer";
import {
  CanvasAdapter,
  type CanvasAdapterCanvasFactoryProps,
} from "./CanvasAdapter";
import type { Artifact, ArtifactRendererProps } from "@/lib/artifact/types";

// ============================================================================
// 错误边界组件
// ============================================================================

/**
 * 错误边界状态
 */
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * 错误边界 Props
 */
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

/**
 * Artifact 错误边界组件
 * 捕获渲染器中的错误，显示友好的错误信息
 *
 * @requirements 14.2
 */
class ArtifactErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error("[ArtifactRenderer] 渲染错误:", error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// ============================================================================
// 辅助组件
// ============================================================================

type ArtifactRendererWritePhase = NonNullable<
  ReturnType<typeof resolveArtifactWritePhase>
>;

const ARTIFACT_RENDERER_WRITE_PHASE_LABEL_KEYS = {
  preparing: "workspace.artifactRenderer.writePhase.preparing",
  streaming: "workspace.artifactRenderer.writePhase.streaming",
  persisted: "workspace.artifactRenderer.writePhase.persisted",
  completed: "workspace.artifactRenderer.writePhase.completed",
  failed: "workspace.artifactRenderer.writePhase.failed",
} as const satisfies Record<ArtifactRendererWritePhase, string>;

const ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS = {
  failed: "workspace.artifactRenderer.empty.failed.detail",
  finished: "workspace.artifactRenderer.empty.finished.detail",
  preparing: "workspace.artifactRenderer.empty.writing.preparingDetail",
  streaming: "workspace.artifactRenderer.empty.writing.streamingDetail",
} as const;

type ArtifactRendererTranslationKey =
  | (typeof ARTIFACT_RENDERER_WRITE_PHASE_LABEL_KEYS)[ArtifactRendererWritePhase]
  | (typeof ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS)[keyof typeof ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS]
  | "workspace.artifactRenderer.empty.failed.title"
  | "workspace.artifactRenderer.empty.finished.title";

/**
 * 渲染器加载骨架屏
 */
const RendererSkeleton: React.FC<{ tone?: "dark" | "light" }> = memo(
  ({ tone = "dark" }) => {
    const { t } = useTranslation("workspace");

    return (
      <div
        className={cn(
          "flex items-center justify-center h-full min-h-[200px]",
          tone === "light" ? "bg-background" : "bg-[#1e2227]",
        )}
      >
        <div
          className={cn(
            "flex flex-col items-center gap-3",
            tone === "light" ? "text-muted-foreground" : "text-gray-400",
          )}
        >
          <Loader2 className="w-8 h-8 animate-spin" />
          <span className="text-sm">
            {t("workspace.artifactRenderer.loadingRenderer")}
          </span>
        </div>
      </div>
    );
  },
);
RendererSkeleton.displayName = "RendererSkeleton";

type EmptyArtifactSurfaceMode = "writing" | "finished" | "failed";
type EmptyArtifactSkeletonKind = "document" | "code" | "preview" | "generic";

interface EmptyArtifactSurfaceState {
  mode: EmptyArtifactSurfaceMode;
  titleKey: ArtifactRendererTranslationKey;
  detailKey: ArtifactRendererTranslationKey;
  detail?: string;
  skeletonKind: EmptyArtifactSkeletonKind;
}

function resolveEmptyArtifactSkeletonKind(
  artifact: Pick<Artifact, "type">,
): EmptyArtifactSkeletonKind {
  if (artifact.type === "document" || artifact.type === "canvas:document") {
    return "document";
  }
  if (artifact.type === "code") {
    return "code";
  }
  if (
    artifact.type === "html" ||
    artifact.type === "svg" ||
    artifact.type === "react" ||
    artifact.type === "mermaid"
  ) {
    return "preview";
  }
  return "generic";
}

function resolveEmptyArtifactSurfaceState(
  artifact: Pick<Artifact, "content" | "error" | "meta" | "status" | "type">,
): EmptyArtifactSurfaceState | null {
  if (
    artifact.meta.previewArtifact === true &&
    artifact.meta.renderMode === "media"
  ) {
    return null;
  }

  const artifactDocument = resolveArtifactProtocolDocumentPayload({
    content: artifact.content,
    metadata: artifact.meta,
  });

  if (artifactDocument) {
    return null;
  }

  if (artifact.content.trim()) {
    return null;
  }

  const writePhase = resolveArtifactWritePhase(artifact);
  const skeletonKind = resolveEmptyArtifactSkeletonKind(artifact);

  if (artifact.status === "error" || writePhase === "failed") {
    return {
      mode: "failed",
      titleKey: "workspace.artifactRenderer.empty.failed.title",
      detailKey: ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS.failed,
      detail: artifact.error?.trim() || undefined,
      skeletonKind,
    };
  }

  if (
    artifact.status === "complete" ||
    writePhase === "completed" ||
    writePhase === "persisted"
  ) {
    return {
      mode: "finished",
      titleKey: "workspace.artifactRenderer.empty.finished.title",
      detailKey: ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS.finished,
      skeletonKind,
    };
  }

  if (
    artifact.status === "pending" ||
    artifact.status === "streaming" ||
    writePhase === "preparing" ||
    writePhase === "streaming"
  ) {
    const titleKey =
      writePhase && ARTIFACT_RENDERER_WRITE_PHASE_LABEL_KEYS[writePhase]
        ? ARTIFACT_RENDERER_WRITE_PHASE_LABEL_KEYS[writePhase]
        : ARTIFACT_RENDERER_WRITE_PHASE_LABEL_KEYS.preparing;
    return {
      mode: "writing",
      titleKey,
      detailKey:
        writePhase === "preparing"
          ? ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS.preparing
          : ARTIFACT_RENDERER_EMPTY_DETAIL_KEYS.streaming,
      skeletonKind,
    };
  }

  return null;
}

const SkeletonBar: React.FC<{
  className?: string;
  tone?: "dark" | "light";
}> = memo(({ className, tone = "dark" }) => (
  <div
    className={cn(
      "h-3 animate-pulse rounded-full",
      tone === "light" ? "bg-muted" : "bg-white/10",
      className,
    )}
  />
));
SkeletonBar.displayName = "SkeletonBar";

const DocumentSkeleton: React.FC<{ tone?: "dark" | "light" }> = memo(
  ({ tone = "dark" }) => (
    <div className="space-y-4">
      <SkeletonBar tone={tone} className="h-6 w-1/3" />
      <SkeletonBar tone={tone} className="w-full" />
      <SkeletonBar tone={tone} className="w-5/6" />
      <SkeletonBar tone={tone} className="w-[92%]" />
      <div className="pt-4">
        <SkeletonBar tone={tone} className="h-4 w-1/4" />
      </div>
      <SkeletonBar tone={tone} className="w-full" />
      <SkeletonBar tone={tone} className="w-4/5" />
      <SkeletonBar tone={tone} className="w-[88%]" />
    </div>
  ),
);
DocumentSkeleton.displayName = "DocumentSkeleton";

const CodeSkeleton: React.FC<{ tone?: "dark" | "light" }> = memo(
  ({ tone = "dark" }) => (
    <div className="space-y-3">
      {Array.from({ length: 8 }).map((_, index) => (
        <div key={index} className="flex items-center gap-4">
          <div
            className={cn(
              "w-6 text-right text-xs",
              tone === "light" ? "text-muted-foreground/70" : "text-gray-500",
            )}
          >
            {index + 1}
          </div>
          <SkeletonBar
            tone={tone}
            className={cn(
              "h-3",
              index % 3 === 0 ? "w-3/5" : index % 3 === 1 ? "w-4/5" : "w-2/3",
            )}
          />
        </div>
      ))}
    </div>
  ),
);
CodeSkeleton.displayName = "CodeSkeleton";

const PreviewSkeleton: React.FC<{ tone?: "dark" | "light" }> = memo(
  ({ tone = "dark" }) => (
    <div className="space-y-4">
      <div
        className={cn(
          "h-56 animate-pulse rounded-3xl border",
          tone === "light"
            ? "border-border bg-muted/70"
            : "border-white/10 bg-white/5",
        )}
      />
      <SkeletonBar tone={tone} className="w-1/3" />
      <SkeletonBar tone={tone} className="w-2/3" />
    </div>
  ),
);
PreviewSkeleton.displayName = "PreviewSkeleton";

const GenericSkeleton: React.FC<{ tone?: "dark" | "light" }> = memo(
  ({ tone = "dark" }) => (
    <div className="space-y-4">
      <div
        className={cn(
          "flex h-28 w-full animate-pulse items-center justify-center rounded-3xl border",
          tone === "light"
            ? "border-border bg-muted/60"
            : "border-white/10 bg-white/5",
        )}
      >
        <FileText
          className={cn(
            "h-8 w-8",
            tone === "light" ? "text-muted-foreground/70" : "text-gray-500",
          )}
        />
      </div>
      <SkeletonBar tone={tone} className="w-1/2" />
      <SkeletonBar tone={tone} className="w-5/6" />
      <SkeletonBar tone={tone} className="w-2/3" />
    </div>
  ),
);
GenericSkeleton.displayName = "GenericSkeleton";

const EmptyArtifactSurface: React.FC<{
  artifact: Artifact;
  state: EmptyArtifactSurfaceState;
  tone?: "dark" | "light";
}> = memo(({ artifact, state, tone = "dark" }) => {
  const { t } = useTranslation("workspace");
  const filePath = resolveArtifactProtocolFilePath(artifact);
  const isWriting = state.mode === "writing";
  const isFailed = state.mode === "failed";
  const title = t(state.titleKey);
  const detail = state.detail ?? t(state.detailKey);
  const Icon = isFailed
    ? AlertCircle
    : state.skeletonKind === "code"
      ? FileCode2
      : state.skeletonKind === "preview"
        ? LayoutTemplate
        : FileText;

  return (
    <div
      data-testid="artifact-empty-surface"
      data-empty-mode={state.mode}
      className="flex h-full min-h-[320px] flex-col"
    >
      <div
        className={cn(
          "border-b px-5 py-4",
          tone === "light"
            ? "border-border bg-background"
            : "border-white/10 bg-[#1e2227]",
        )}
      >
        <div className="flex items-start gap-3">
          <div
            className={cn(
              "mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl",
              isFailed
                ? "bg-destructive/10 text-destructive"
                : tone === "light"
                  ? "bg-primary/10 text-primary"
                  : "bg-white/10 text-white",
            )}
          >
            <Icon className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div
              className={cn(
                "text-sm font-semibold",
                tone === "light" ? "text-foreground" : "text-white",
              )}
            >
              {title}
            </div>
            <div
              className={cn(
                "mt-1 text-sm leading-6",
                tone === "light" ? "text-muted-foreground" : "text-gray-400",
              )}
            >
              {detail}
            </div>
            <div
              className={cn(
                "mt-2 truncate text-xs",
                tone === "light" ? "text-muted-foreground/90" : "text-gray-500",
              )}
            >
              {filePath}
            </div>
          </div>
        </div>
        {isWriting ? (
          <div
            className={cn(
              "mt-4 h-1.5 overflow-hidden rounded-full",
              tone === "light" ? "bg-muted" : "bg-white/10",
            )}
          >
            <div
              className={cn(
                "h-full w-2/5 rounded-r-full animate-pulse",
                tone === "light" ? "bg-primary/60" : "bg-white/40",
              )}
            />
          </div>
        ) : null}
      </div>
      <div
        className={cn(
          "flex-1 px-6 py-6",
          tone === "light" ? "bg-background" : "bg-[#1e2227]",
        )}
      >
        {state.skeletonKind === "document" ? (
          <DocumentSkeleton tone={tone} />
        ) : state.skeletonKind === "code" ? (
          <CodeSkeleton tone={tone} />
        ) : state.skeletonKind === "preview" ? (
          <PreviewSkeleton tone={tone} />
        ) : (
          <GenericSkeleton tone={tone} />
        )}
      </div>
    </div>
  );
});
EmptyArtifactSurface.displayName = "EmptyArtifactSurface";

/**
 * 流式状态指示器 Props
 * @requirements 11.1
 */
interface StreamingIndicatorProps {
  /** 是否正在完成（用于平滑过渡） */
  isCompleting?: boolean;
}

/**
 * 流式状态指示器
 * 显示正在生成的视觉指示，支持平滑过渡到完成状态
 *
 * @requirements 11.1, 11.3
 */
const StreamingIndicator: React.FC<StreamingIndicatorProps> = memo(
  ({ isCompleting = false }) => {
    const { t } = useTranslation("workspace");

    return (
      <div
        className={cn(
          "absolute bottom-3 right-3 flex items-center gap-2 px-3 py-1.5 rounded-full transition-all duration-300",
          isCompleting
            ? "bg-green-500/20 scale-95 opacity-80"
            : "bg-blue-500/20",
        )}
      >
        {isCompleting ? (
          <>
            <div className="w-4 h-4 text-green-400">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <span className="text-xs text-green-400">
              {t("workspace.artifactRenderer.streaming.completed")}
            </span>
          </>
        ) : (
          <>
            <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
            <span className="text-xs text-blue-400">
              {t("workspace.artifactRenderer.streaming.generating")}
            </span>
            {/* 进度动画条 */}
            <div className="w-12 h-1 bg-blue-500/30 rounded-full overflow-hidden">
              <div className="h-full bg-blue-400 rounded-full animate-streaming-progress" />
            </div>
          </>
        )}
      </div>
    );
  },
);
StreamingIndicator.displayName = "StreamingIndicator";

/**
 * 未知类型回退渲染器
 * 当 Artifact 类型没有对应的渲染器时显示
 */
const FallbackRenderer: React.FC<{
  artifact: Artifact;
  tone?: "dark" | "light";
}> = memo(({ artifact, tone = "dark" }) => {
  const { t } = useTranslation("workspace");

  return (
    <div
      className={cn(
        "flex flex-col h-full",
        tone === "light" ? "bg-background" : "bg-[#1e2227]",
      )}
    >
      {/* 提示区域 */}
      <div className="p-4 border-b border-yellow-500/20 bg-yellow-500/10">
        <div className="flex items-center gap-2 text-yellow-400 font-medium mb-2">
          <AlertTriangle className="w-5 h-5" />
          <span>{t("workspace.artifactRenderer.fallback.unknownType")}</span>
        </div>
        <div
          className={cn(
            "text-sm",
            tone === "light" ? "text-muted-foreground" : "text-gray-400",
          )}
        >
          {t("workspace.artifactRenderer.fallback.detail", {
            type: artifact.type,
          })}
        </div>
      </div>

      {/* 内容区域 */}
      <div className="flex-1 overflow-auto p-4">
        <pre
          className={cn(
            "overflow-auto whitespace-pre-wrap rounded p-3 text-sm",
            tone === "light"
              ? "border border-border bg-muted/30 text-foreground"
              : "bg-black/30 text-gray-300",
          )}
        >
          {artifact.content}
        </pre>
      </div>
    </div>
  );
});
FallbackRenderer.displayName = "FallbackRenderer";

function readStringMeta(meta: Artifact["meta"], key: string): string | null {
  const value = meta[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function resolvePreviewArtifactContentKind(artifact: Artifact): string | null {
  if (artifact.meta.previewArtifact !== true) {
    return null;
  }
  return readStringMeta(artifact.meta, "contentKind");
}

function resolvePreviewArtifactFallbackMode(
  artifact: Artifact,
): PreviewArtifactFallbackMode | null {
  if (artifact.meta.previewArtifact !== true) {
    return null;
  }

  const renderMode = readStringMeta(artifact.meta, "renderMode");
  return renderMode === "system_open" || renderMode === "unsupported"
    ? renderMode
    : null;
}

function resolvePreviewSourceSummaryKind(
  artifact: Artifact,
): PreviewSourceSummaryKind | null {
  if (artifact.meta.previewArtifact !== true) {
    return null;
  }

  const source = readStringMeta(artifact.meta, "source");
  const contentKind = readStringMeta(artifact.meta, "contentKind");

  if (source === "url") {
    return "url";
  }
  if (source === "database_record") {
    return "database_record";
  }
  if (source === "app" || contentKind === "app_shell") {
    return "app";
  }
  return null;
}

// ============================================================================
// 主组件
// ============================================================================

/**
 * ArtifactRenderer Props
 */
interface ArtifactRendererComponentProps extends ArtifactRendererProps {
  /** 自定义类名 */
  className?: string;
  /** 防抖延迟（毫秒），默认 100ms */
  debounceDelay?: number;
  /** Canvas Artifact 运行上下文，用于 current 工作台 project 保存与图层生成 */
  canvasFactoryProps?: CanvasAdapterCanvasFactoryProps;
  /** 是否隐藏内部工具栏 */
  hideToolbar?: boolean;
  /** 视图模式（用于代码预览） */
  viewMode?: "source" | "preview";
  /** 预览尺寸 */
  previewSize?: "mobile" | "tablet" | "desktop";
}

/**
 * Artifact 统一渲染入口组件
 *
 * 功能特性：
 * - 根据 Artifact 类型分发到对应的渲染器 (Requirement 3.4)
 * - Canvas 类型委托给 Canvas 系统处理 (Requirement 12.1)
 * - 错误边界捕获渲染错误，显示友好的错误信息 (Requirement 14.2)
 * - 流式指示器显示正在生成的视觉指示 (Requirement 11.1)
 * - 防抖更新避免频繁重渲染 (Requirement 11.2)
 * - 流式完成时平滑过渡到完成状态 (Requirement 11.3)
 * - 使用 React.Suspense 支持懒加载渲染器
 *
 * @param artifact - 要渲染的 Artifact 对象
 * @param isStreaming - 是否处于流式生成状态
 * @param onContentChange - 内容变更回调
 * @param className - 自定义类名
 * @param debounceDelay - 防抖延迟（毫秒）
 *
 * @requirements 3.4, 11.1, 11.2, 11.3, 12.1, 14.2
 */
export const ArtifactRenderer: React.FC<ArtifactRendererComponentProps> = memo(
  ({
    artifact,
    isStreaming = false,
    onContentChange,
    className,
    debounceDelay = 100,
    hideToolbar = false,
    viewMode,
    previewSize = "desktop",
    tone = "dark",
    canvasFactoryProps,
  }) => {
    const { t } = useTranslation("workspace");
    // 错误状态管理
    const [renderError, setRenderError] = useState<Error | null>(null);
    const [showSourceOnError, setShowSourceOnError] = useState(false);
    const [retryKey, setRetryKey] = useState(0);

    // 流式完成过渡状态
    // @requirements 11.3
    const [isCompleting, setIsCompleting] = useState(false);
    const prevStreamingRef = useRef(isStreaming);

    // 防抖处理内容更新，避免频繁重渲染
    // @requirements 11.2
    const debouncedContent = useDebouncedValue(artifact.content, debounceDelay);

    // 创建带防抖内容的 artifact 副本
    const debouncedArtifact: Artifact = {
      ...artifact,
      content: isStreaming ? debouncedContent : artifact.content,
    };

    // 监听流式状态变化，实现平滑过渡
    // @requirements 11.3
    useEffect(() => {
      // 从流式状态变为非流式状态时，显示完成动画
      if (prevStreamingRef.current && !isStreaming) {
        setIsCompleting(true);
        // 500ms 后隐藏完成指示器
        const timer = setTimeout(() => {
          setIsCompleting(false);
        }, 500);
        return () => clearTimeout(timer);
      }
      prevStreamingRef.current = isStreaming;
    }, [isStreaming]);

    /**
     * 处理渲染错误
     */
    const handleError = useCallback((error: Error) => {
      setRenderError(error);
    }, []);

    /**
     * 重试渲染
     */
    const handleRetry = useCallback(() => {
      setRenderError(null);
      setShowSourceOnError(false);
      setRetryKey((k) => k + 1);
    }, []);

    /**
     * 显示源码
     */
    const handleShowSource = useCallback(() => {
      setShowSourceOnError(true);
    }, []);

    // 获取渲染器注册项
    const entry = artifactRegistry.get(artifact.type);
    const previewArtifactContentKind =
      resolvePreviewArtifactContentKind(artifact);
    const previewArtifactFallbackMode =
      resolvePreviewArtifactFallbackMode(artifact);
    const previewSourceSummaryKind =
      resolvePreviewSourceSummaryKind(artifact);
    const emptySurfaceState = resolveEmptyArtifactSurfaceState(artifact);
    const effectiveViewMode =
      viewMode ?? resolveDefaultArtifactViewMode(artifact);

    if (previewArtifactFallbackMode) {
      return (
        <div className={cn("relative h-full", className)}>
          <PreviewArtifactFallbackSurface
            artifact={debouncedArtifact}
            mode={previewArtifactFallbackMode}
            tone={tone}
          />
          {(isStreaming || isCompleting) && (
            <StreamingIndicator isCompleting={isCompleting} />
          )}
        </div>
      );
    }

    if (previewSourceSummaryKind) {
      return (
        <div className={cn("relative h-full", className)}>
          <PreviewSourceSummaryRenderer
            artifact={debouncedArtifact}
            sourceKind={previewSourceSummaryKind}
            tone={tone}
          />
          {(isStreaming || isCompleting) && (
            <StreamingIndicator isCompleting={isCompleting} />
          )}
        </div>
      );
    }

    if (emptySurfaceState) {
      return (
        <div className={cn("relative h-full", className)}>
          <EmptyArtifactSurface
            artifact={artifact}
            state={emptySurfaceState}
            tone={tone}
          />
          {(isStreaming || isCompleting) && (
            <StreamingIndicator isCompleting={isCompleting} />
          )}
        </div>
      );
    }

    // 如果有错误且选择显示源码，直接显示源码
    if (renderError && showSourceOnError) {
      return (
        <div className={cn("relative h-full", className)}>
          <div
            className={cn(
              "flex flex-col h-full",
              tone === "light" ? "bg-background" : "bg-[#1e2227]",
            )}
          >
            <div className="flex items-center justify-between border-b border-yellow-500/20 bg-yellow-500/10 p-2">
              <span className="text-sm text-yellow-400">
                {t("workspace.artifactRenderer.sourceOnError.title")}
              </span>
              <button
                type="button"
                onClick={handleRetry}
                className={cn(
                  "flex items-center gap-1 rounded px-2 py-1 text-xs",
                  tone === "light"
                    ? "bg-black/5 text-foreground hover:bg-black/10"
                    : "bg-white/10 hover:bg-white/15",
                )}
              >
                <RefreshCw className="w-3 h-3" />
                {t("workspace.artifactRenderer.sourceOnError.retry")}
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              <pre
                className={cn(
                  "whitespace-pre-wrap text-sm",
                  tone === "light" ? "text-foreground" : "text-gray-300",
                )}
              >
                {artifact.content}
              </pre>
            </div>
          </div>
        </div>
      );
    }

    // Canvas 类型委托给 Canvas 系统
    // @requirements 12.1
    if (artifactRegistry.isCanvasType(artifact.type)) {
      return (
        <div className={cn("relative h-full", className)}>
          <CanvasAdapter
            artifact={debouncedArtifact}
            isStreaming={isStreaming}
            onContentChange={onContentChange}
            canvasFactoryProps={canvasFactoryProps}
          />
          {(isStreaming || isCompleting) && (
            <StreamingIndicator isCompleting={isCompleting} />
          )}
        </div>
      );
    }

    if (
      previewArtifactContentKind === "image" ||
      previewArtifactContentKind === "audio" ||
      previewArtifactContentKind === "video"
    ) {
      return (
        <div className={cn("relative h-full", className)}>
          <PreviewMediaRenderer
            artifact={debouncedArtifact}
            contentKind={previewArtifactContentKind as PreviewMediaContentKind}
            tone={tone}
          />
          {(isStreaming || isCompleting) && (
            <StreamingIndicator isCompleting={isCompleting} />
          )}
        </div>
      );
    }

    // 未注册的类型，使用回退渲染器
    if (!entry) {
      return (
        <div className={cn("relative h-full", className)}>
          <FallbackRenderer artifact={debouncedArtifact} tone={tone} />
          {(isStreaming || isCompleting) && (
            <StreamingIndicator isCompleting={isCompleting} />
          )}
        </div>
      );
    }

    // 获取渲染器组件
    const RendererComponent = entry.component;

    // 错误回退组件
    const errorFallback = (
      <ErrorFallbackRenderer
        artifact={debouncedArtifact}
        error={renderError}
        onRetry={handleRetry}
        onShowSource={handleShowSource}
      />
    );

    return (
      <div className={cn("relative h-full", className)} key={retryKey}>
        <Suspense fallback={<RendererSkeleton tone={tone} />}>
          <ArtifactErrorBoundary fallback={errorFallback} onError={handleError}>
            <RendererComponent
              artifact={debouncedArtifact}
              isStreaming={isStreaming}
              tone={tone}
              onContentChange={onContentChange}
              hideToolbar={hideToolbar}
              viewMode={effectiveViewMode}
              previewSize={previewSize}
            />
          </ArtifactErrorBoundary>
        </Suspense>
        {(isStreaming || isCompleting) && (
          <StreamingIndicator isCompleting={isCompleting} />
        )}
      </div>
    );
  },
);

ArtifactRenderer.displayName = "ArtifactRenderer";
