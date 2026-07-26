/**
 * @file 布局过渡组件
 * @description 处理对话和画布之间的布局切换动画
 * @module components/workspace/layout/LayoutTransition
 */

import React, {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import styled from "styled-components";
import { MessageSquareText, PanelsTopLeft } from "lucide-react";
import { LayoutMode } from "@/lib/workspace/workflowTypes";
import { useLayoutTransition, TransitionConfig } from "./useLayoutTransition";

const COMPACT_CHAT_CANVAS_CONTAINER_WIDTH = 900;
const COMPACT_CHAT_CANVAS_BREAKPOINT_HEIGHT = 620;
const CHAT_CANVAS_RESIZE_MIN_CHAT_WIDTH = 360;
const CHAT_CANVAS_RESIZE_MIN_CANVAS_WIDTH = 420;

function shouldUseCompactChatCanvasLayout(
  mode: LayoutMode,
  containerWidth?: number,
): boolean {
  if (mode !== "chat-canvas" || typeof window === "undefined") {
    return false;
  }

  const availableWidth =
    typeof containerWidth === "number" && containerWidth > 0
      ? containerWidth
      : window.innerWidth;

  return (
    availableWidth <= COMPACT_CHAT_CANVAS_CONTAINER_WIDTH ||
    window.innerHeight <= COMPACT_CHAT_CANVAS_BREAKPOINT_HEIGHT
  );
}

const Container = styled.div`
  display: flex;
  position: relative;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
`;

const CompactModeBar = styled.div<{ $visible: boolean; $topInset: string }>`
  display: ${({ $visible }) => ($visible ? "flex" : "none")};
  min-width: 0;
  height: 42px;
  flex: 0 0 42px;
  margin-top: ${({ $topInset }) => $topInset};
  align-items: center;
  justify-content: flex-start;
  border-bottom: 1px solid var(--lime-surface-border, rgba(226, 232, 240, 0.9));
  background: var(--lime-home-card-surface, #fff);
  padding: 5px 12px;
`;

const CompactModeControl = styled.div`
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 2px;
  border: 1px solid var(--lime-surface-border, rgba(226, 232, 240, 0.9));
  border-radius: 7px;
  background: var(--lime-surface-soft, #f8fafc);
  padding: 2px;
`;

const CompactModeButton = styled.button<{ $active: boolean }>`
  display: inline-flex;
  height: 28px;
  min-width: 0;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 0;
  border-radius: 5px;
  background: ${({ $active }) =>
    $active ? "var(--lime-home-card-surface, #fff)" : "transparent"};
  color: ${({ $active }) =>
    $active
      ? "var(--lime-text-strong, #0f172a)"
      : "var(--lime-text-muted, #64748b)"};
  box-shadow: ${({ $active }) =>
    $active ? "0 1px 2px rgba(15, 23, 42, 0.08)" : "none"};
  padding: 0 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;

  &:hover {
    color: var(--lime-text-strong, #0f172a);
  }

  &:focus-visible {
    outline: 2px solid var(--lime-focus-ring, #38bdf8);
    outline-offset: 1px;
  }
`;

const PanelViewport = styled.div<{ $compactSinglePanel: boolean }>`
  position: relative;
  display: flex;
  flex-direction: row;
  flex: 1;
  width: 100%;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  gap: ${({ $compactSinglePanel }) => ($compactSinglePanel ? "0" : "12px")};
`;

const SplitResizeHandle = styled.div<{ $dragging: boolean }>`
  position: relative;
  z-index: 12;
  flex: 0 0 10px;
  align-self: stretch;
  margin: 0 -6px;
  cursor: col-resize;
  touch-action: none;
  user-select: none;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
    transform: translateX(-50%);
    background: ${({ $dragging }) =>
      $dragging
        ? "var(--lime-surface-border-strong, #bbf7d0)"
        : "var(--lime-surface-border, rgba(226, 240, 226, 0.94))"};
  }

  &::after {
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 4px;
    height: 52px;
    border-radius: 999px;
    transform: translate(-50%, -50%);
    background: ${({ $dragging }) =>
      $dragging ? "var(--lime-brand-soft, #ecfdf5)" : "transparent"};
  }

  &:hover::before {
    background: var(--lime-surface-border-strong, #bbf7d0);
  }

  &:hover::after {
    background: var(--lime-brand-soft, #ecfdf5);
  }
`;

const ChatPanel = styled.div<{
  $width: string;
  $duration: number;
  $minWidth: string;
  $compactSinglePanel: boolean;
  $hidden: boolean;
  $chrome: "panel" | "plain";
}>`
  position: relative;
  height: ${({ $hidden }) => ($hidden ? "0" : "100%")};
  max-height: ${({ $hidden }) => ($hidden ? "0" : "100%")};
  overflow: hidden;
  transition:
    width ${({ $duration }) => $duration}ms ease-out,
    height ${({ $duration }) => $duration}ms ease-out;
  width: ${({ $compactSinglePanel, $width, $hidden }) =>
    $hidden ? "0" : $compactSinglePanel ? "100%" : $width};
  min-width: ${({ $compactSinglePanel, $minWidth }) =>
    $compactSinglePanel ? "0" : $minWidth};
  min-height: ${({ $hidden }) => ($hidden ? "0" : "100%")};
  flex: ${({ $compactSinglePanel, $hidden }) =>
    $hidden ? "0 0 0" : $compactSinglePanel ? "1 1 100%" : "0 0 auto"};
  will-change: width, height;
  display: ${({ $hidden }) => ($hidden ? "none" : "flex")};
  flex-direction: column;
  padding: ${({ $chrome }) => ($chrome === "plain" ? "0" : "16px 16px 16px 0")};
`;

const ChatPanelInner = styled.div<{ $topInset: string }>`
  height: 100%;
  display: flex;
  flex-direction: column;
  background: hsl(var(--background));
  border-radius: 12px;
  border: 1px solid hsl(var(--border));
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  box-sizing: border-box;
  padding-top: ${({ $topInset }) => $topInset};
`;

const PlainChatPanelInner = styled.div<{ $topInset: string }>`
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  padding-top: ${({ $topInset }) => $topInset};
`;

const CanvasPanel = styled.div<{
  $visible: boolean;
  $transform: string;
  $opacity: number;
  $duration: number;
  $topInset: string;
}>`
  position: relative;
  height: 100%;
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  transition:
    transform ${({ $duration }) => $duration}ms ease-out,
    opacity ${({ $duration }) => $duration}ms ease-out;
  transform: ${({ $transform }) => $transform};
  opacity: ${({ $opacity }) => $opacity};
  display: ${({ $visible }) => ($visible ? "block" : "none")};
  will-change: transform, opacity;
  box-sizing: border-box;
  padding-top: ${({ $topInset }) => $topInset};
`;

interface LayoutTransitionProps {
  /** 当前布局模式 */
  mode: LayoutMode;
  /** 对话区域内容 */
  chatContent: React.ReactNode;
  /** 画布区域内容 */
  canvasContent: React.ReactNode;
  /** 过渡配置 */
  transitionConfig?: TransitionConfig;
  /** 聊天区域是否使用额外面板壳 */
  chatPanelChrome?: "panel" | "plain";
  /** chat-canvas 模式下聊天面板宽度 */
  chatPanelWidth?: string;
  /** chat-canvas 模式下聊天面板最小宽度 */
  chatPanelMinWidth?: string;
  /** chat-canvas 模式下聊天面板顶部预留 */
  chatPanelTopInset?: string;
  /** chat-canvas 模式下画布面板顶部预留 */
  canvasPanelTopInset?: string;
  /** 紧凑单面板态下强制切回聊天区 */
  forceOpenChatPanel?: boolean;
}

/**
 * 布局过渡组件
 *
 * 处理纯对话和对话+画布两种布局之间的平滑切换
 */
export const LayoutTransition: React.FC<LayoutTransitionProps> = memo(
  ({
    mode,
    chatContent,
    canvasContent,
    transitionConfig,
    chatPanelChrome = "panel",
    chatPanelWidth,
    chatPanelMinWidth,
    chatPanelTopInset = "0px",
    canvasPanelTopInset = "0px",
    forceOpenChatPanel = false,
  }) => {
    const { t } = useTranslation("workspace");
    const containerRef = useRef<HTMLDivElement | null>(null);
    const resizingChatPanelRef = useRef(false);
    const [resizedChatPanelWidth, setResizedChatPanelWidth] = useState<
      number | null
    >(null);
    const [resizingChatPanel, setResizingChatPanel] = useState(false);
    const hasCanvasContent = React.Children.count(canvasContent) > 0;
    const effectiveMode: LayoutMode = hasCanvasContent ? mode : "chat";
    const effectiveChatPanelWidth = useMemo(
      () =>
        resizedChatPanelWidth !== null && effectiveMode === "chat-canvas"
          ? `${resizedChatPanelWidth}px`
          : chatPanelWidth,
      [chatPanelWidth, effectiveMode, resizedChatPanelWidth],
    );
    const { isCanvasVisible, getTransitionStyles } = useLayoutTransition(
      effectiveMode,
      transitionConfig,
      {
        chatCanvasPanelWidth: effectiveChatPanelWidth,
      },
    );
    const [compactChatCanvasLayout, setCompactChatCanvasLayout] = useState(() =>
      shouldUseCompactChatCanvasLayout(effectiveMode),
    );
    const [compactPrimaryPanel, setCompactPrimaryPanel] = useState<
      "chat" | "canvas"
    >("chat");

    const chatStyles = getTransitionStyles("chat");
    const canvasStyles = getTransitionStyles("canvas");
    const resolvedChatPanelMinWidth =
      effectiveMode === "chat-canvas" ? chatPanelMinWidth || "360px" : "0px";
    const shouldRenderCanvas = hasCanvasContent && isCanvasVisible;
    const isCompactChatCanvas =
      compactChatCanvasLayout && effectiveMode === "chat-canvas";
    const compactModeTopInset = isCompactChatCanvas ? chatPanelTopInset : "0px";
    const resolvedChatPanelTopInset = isCompactChatCanvas
      ? "0px"
      : chatPanelTopInset;
    const resolvedCanvasPanelTopInset = isCompactChatCanvas
      ? "0px"
      : canvasPanelTopInset;
    const shouldRenderResizeHandle =
      effectiveMode === "chat-canvas" &&
      shouldRenderCanvas &&
      !isCompactChatCanvas;

    const resolveConstrainedChatWidth = useCallback((clientX: number) => {
      const containerRect = containerRef.current?.getBoundingClientRect();
      if (!containerRect) {
        return null;
      }
      const maxWidth = Math.max(
        CHAT_CANVAS_RESIZE_MIN_CHAT_WIDTH,
        containerRect.width - CHAT_CANVAS_RESIZE_MIN_CANVAS_WIDTH,
      );
      return Math.min(
        Math.max(
          clientX - containerRect.left,
          CHAT_CANVAS_RESIZE_MIN_CHAT_WIDTH,
        ),
        maxWidth,
      );
    }, []);

    const handleResizePointerDown = useCallback(
      (event: React.PointerEvent<HTMLDivElement>) => {
        const nextWidth = resolveConstrainedChatWidth(event.clientX);
        if (nextWidth === null) {
          return;
        }
        event.currentTarget.setPointerCapture(event.pointerId);
        event.preventDefault();
        resizingChatPanelRef.current = true;
        setResizingChatPanel(true);
        setResizedChatPanelWidth(nextWidth);
      },
      [resolveConstrainedChatWidth],
    );

    const handleResizePointerMove = useCallback(
      (event: React.PointerEvent<HTMLDivElement>) => {
        if (!resizingChatPanelRef.current) {
          return;
        }
        const nextWidth = resolveConstrainedChatWidth(event.clientX);
        if (nextWidth !== null) {
          setResizedChatPanelWidth(nextWidth);
        }
      },
      [resolveConstrainedChatWidth],
    );

    const handleResizePointerEnd = useCallback(
      (event: React.PointerEvent<HTMLDivElement>) => {
        if (!resizingChatPanelRef.current) {
          return;
        }
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
        resizingChatPanelRef.current = false;
        setResizingChatPanel(false);
      },
      [],
    );

    useEffect(() => {
      if (effectiveMode !== "chat-canvas" || compactChatCanvasLayout) {
        resizingChatPanelRef.current = false;
        setResizingChatPanel(false);
      }
    }, [compactChatCanvasLayout, effectiveMode]);

    useEffect(() => {
      const updateLayout = (containerWidth?: number) => {
        setCompactChatCanvasLayout(
          shouldUseCompactChatCanvasLayout(effectiveMode, containerWidth),
        );
      };

      if (typeof window === "undefined") {
        return;
      }

      const updateFromWindow = () => {
        const containerWidth =
          containerRef.current?.getBoundingClientRect().width;
        updateLayout(containerWidth);
      };
      updateFromWindow();
      window.addEventListener("resize", updateFromWindow);

      const container = containerRef.current;
      const resizeObserver =
        container && typeof ResizeObserver !== "undefined"
          ? new ResizeObserver((entries) => {
              const observedWidth = entries[0]?.contentRect.width;
              updateLayout(observedWidth);
            })
          : null;
      if (container && resizeObserver) {
        resizeObserver.observe(container);
      }

      return () => {
        window.removeEventListener("resize", updateFromWindow);
        resizeObserver?.disconnect();
      };
    }, [effectiveMode]);

    useEffect(() => {
      if (isCompactChatCanvas) {
        setCompactPrimaryPanel("chat");
      }
    }, [isCompactChatCanvas]);

    useEffect(() => {
      if (!forceOpenChatPanel || !isCompactChatCanvas) {
        return;
      }

      setCompactPrimaryPanel("chat");
    }, [forceOpenChatPanel, isCompactChatCanvas]);

    const canvasPanelNode = (
      <CanvasPanel
        $visible={
          shouldRenderCanvas &&
          (!isCompactChatCanvas || compactPrimaryPanel === "canvas")
        }
        $transform={canvasStyles.transform as string}
        $opacity={canvasStyles.opacity as number}
        $topInset={resolvedCanvasPanelTopInset}
        $duration={parseInt(
          canvasStyles.transition?.match(/\d+/)?.[0] || "300",
        )}
        data-top-inset={resolvedCanvasPanelTopInset}
        data-testid="layout-canvas-panel"
      >
        {canvasContent}
      </CanvasPanel>
    );

    const chatPanelNode = (
      <ChatPanel
        $width={chatStyles.width as string}
        $duration={parseInt(chatStyles.transition?.match(/\d+/)?.[0] || "300")}
        $minWidth={resolvedChatPanelMinWidth}
        $compactSinglePanel={isCompactChatCanvas}
        $hidden={
          effectiveMode === "canvas" ||
          (isCompactChatCanvas && compactPrimaryPanel !== "chat")
        }
        $chrome={chatPanelChrome}
        data-testid="layout-chat-panel"
        data-chat-panel-width={chatStyles.width as string}
        data-chat-panel-min-width={resolvedChatPanelMinWidth}
        data-overlay-state={
          isCompactChatCanvas
            ? compactPrimaryPanel === "chat"
              ? "single-active"
              : "single-hidden"
            : "inline"
        }
      >
        {chatPanelChrome === "plain" ? (
          <PlainChatPanelInner
            $topInset={resolvedChatPanelTopInset}
            data-top-inset={resolvedChatPanelTopInset}
            data-testid="layout-chat-panel-plain"
          >
            {chatContent}
          </PlainChatPanelInner>
        ) : (
          <ChatPanelInner
            $topInset={resolvedChatPanelTopInset}
            data-top-inset={resolvedChatPanelTopInset}
            data-testid="layout-chat-panel-inner"
          >
            {chatContent}
          </ChatPanelInner>
        )}
      </ChatPanel>
    );

    return (
      <Container
        ref={containerRef}
        data-testid="layout-transition-root"
        data-effective-mode={effectiveMode}
        data-has-canvas={shouldRenderCanvas ? "true" : "false"}
        data-layout-axis={isCompactChatCanvas ? "single" : "horizontal"}
        data-chat-panel-placement={
          isCompactChatCanvas ? "single-panel" : "inline"
        }
        data-compact-primary-panel={
          isCompactChatCanvas ? compactPrimaryPanel : undefined
        }
      >
        <CompactModeBar
          $visible={isCompactChatCanvas}
          $topInset={compactModeTopInset}
          data-testid="layout-compact-mode-bar"
          data-visible={isCompactChatCanvas ? "true" : "false"}
          data-top-inset={compactModeTopInset}
        >
          {isCompactChatCanvas ? (
            <CompactModeControl
              role="tablist"
              aria-label={t("workspace.layout.compactMode.ariaLabel")}
            >
              <CompactModeButton
                type="button"
                role="tab"
                aria-selected={compactPrimaryPanel === "chat"}
                $active={compactPrimaryPanel === "chat"}
                data-testid="layout-compact-chat-tab"
                onClick={() => setCompactPrimaryPanel("chat")}
              >
                <MessageSquareText size={15} aria-hidden="true" />
                <span>{t("workspace.layout.compactMode.chat")}</span>
              </CompactModeButton>
              <CompactModeButton
                type="button"
                role="tab"
                aria-selected={compactPrimaryPanel === "canvas"}
                $active={compactPrimaryPanel === "canvas"}
                data-testid="layout-compact-canvas-tab"
                onClick={() => setCompactPrimaryPanel("canvas")}
              >
                <PanelsTopLeft size={15} aria-hidden="true" />
                <span>{t("workspace.layout.compactMode.workbench")}</span>
              </CompactModeButton>
            </CompactModeControl>
          ) : null}
        </CompactModeBar>
        <PanelViewport
          $compactSinglePanel={isCompactChatCanvas}
          data-testid="layout-panel-viewport"
        >
          {chatPanelNode}
          {shouldRenderResizeHandle ? (
            <SplitResizeHandle
              role="separator"
              aria-orientation="vertical"
              aria-label={t("workspace.layout.splitter.resizeAria")}
              $dragging={resizingChatPanel}
              data-testid="layout-chat-canvas-resize-handle"
              data-dragging={resizingChatPanel ? "true" : "false"}
              onPointerDown={handleResizePointerDown}
              onPointerMove={handleResizePointerMove}
              onPointerUp={handleResizePointerEnd}
              onPointerCancel={handleResizePointerEnd}
            />
          ) : null}
          {canvasPanelNode}
        </PanelViewport>
      </Container>
    );
  },
);

LayoutTransition.displayName = "LayoutTransition";
