import styled from "styled-components";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LIME_STAGE_SURFACE } from "../workspace/taskCenterChromeTokens";
import { CONVERSATION_CONTENT_MAX_WIDTH } from "./conversationLayoutTokens";

export const Navbar = styled.div<{
  $compact?: boolean;
  $collapsed?: boolean;
  $taskCenter?: boolean;
}>`
  display: flex;
  align-items: center;
  justify-content: ${({ $collapsed }) =>
    $collapsed ? "flex-end" : "space-between"};
  gap: ${({ $compact, $collapsed, $taskCenter }) =>
    $collapsed ? "10px" : $taskCenter ? "8px" : $compact ? "8px" : "12px"};
  padding: ${({ $compact, $collapsed, $taskCenter }) =>
    $collapsed
      ? $compact
        ? "6px 8px 2px"
        : "6px 14px 2px"
      : $taskCenter
        ? $compact
          ? "1px 8px 0"
          : "1px 8px 0"
        : $compact
          ? "7px 10px 7px"
          : "12px 16px 10px"};
  min-height: ${({ $compact, $collapsed, $taskCenter }) =>
    $collapsed ? "auto" : $taskCenter ? "auto" : $compact ? "50px" : "64px"};
  border-bottom: ${({ $collapsed }) =>
    $collapsed
      ? "none"
      : "1px solid var(--lime-surface-border, rgba(226, 240, 226, 0.72))"};
  border-bottom-color: ${({ $collapsed, $taskCenter }) =>
    $collapsed
      ? "transparent"
      : $taskCenter
        ? "transparent"
        : "var(--lime-surface-border, rgba(226, 240, 226, 0.72))"};
  background: ${({ $collapsed, $taskCenter }) =>
    $collapsed
      ? "transparent"
      : $taskCenter
        ? "transparent"
        : "var(--lime-composer-surface-floating)"};
  box-shadow: ${({ $collapsed, $taskCenter }) =>
    $collapsed
      ? "none"
      : $taskCenter
        ? "none"
        : `inset 0 -1px 0 rgba(255, 255, 255, 0.74),
    0 10px 28px rgba(15, 23, 42, 0.04)`};
  backdrop-filter: ${({ $collapsed, $taskCenter }) =>
    $collapsed || $taskCenter ? "none" : "blur(18px)"};
  align-self: ${({ $collapsed }) => ($collapsed ? "flex-end" : "stretch")};
  width: ${({ $collapsed }) => ($collapsed ? "fit-content" : "auto")};
  max-width: ${({ $collapsed }) => ($collapsed ? "calc(100% - 24px)" : "100%")};
  margin-left: ${({ $collapsed }) => ($collapsed ? "auto" : "0")};
  flex-shrink: 0;
  position: relative;
  z-index: 10;
`;

export const MessageListContainer = styled(ScrollArea)<{
  $taskCenterSurface?: boolean;
}>`
  flex: 1;
  min-height: 0;
  height: 100%;
  padding: ${({ $taskCenterSurface }) =>
    $taskCenterSurface ? "6px 0 16px" : "22px 0 16px"};
  overscroll-behavior: contain;
  background: ${({ $taskCenterSurface }) =>
    $taskCenterSurface ? LIME_STAGE_SURFACE : "#ffffff"};
`;

export const MessageListFrame = styled.div`
  position: relative;
  display: flex;
  flex: 1;
  min-height: 0;
  height: 100%;
`;

export const MessageTurnGroup = styled.section<{
  $deferOffscreenWork: boolean;
}>`
  content-visibility: ${({ $deferOffscreenWork }) =>
    $deferOffscreenWork ? "auto" : "visible"};
  contain-intrinsic-size: ${({ $deferOffscreenWork }) =>
    $deferOffscreenWork ? "auto 320px" : "none"};
`;

// Linear Layout Wrapper: Always Row, Left Aligned
export const MessageWrapper = styled.div<{
  $isUser: boolean;
  $compactLeadingSpacing?: boolean;
}>`
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  justify-content: ${({ $isUser }) => ($isUser ? "flex-end" : "flex-start")};
  padding: ${({ $compactLeadingSpacing }) =>
    $compactLeadingSpacing ? "6px 2px" : "6px 4px"};
  gap: 0;
  width: 100%;
  max-width: none;
  margin: 0;

  &:hover .message-actions,
  &:focus-within .message-actions {
    opacity: 1;
    max-height: 48px;
    margin-top: 8px;
    transform: translateY(0);
    pointer-events: auto;
    overflow: visible;
  }

  .user-message-footer {
    max-height: 0;
    margin-top: 0;
    overflow: hidden;
    opacity: 0;
    transform: translateY(-2px);
    pointer-events: none;
    transition:
      opacity 0.18s ease,
      max-height 0.18s ease,
      margin-top 0.18s ease,
      transform 0.18s ease;
  }

  &:hover .user-message-footer,
  &:focus-within .user-message-footer {
    max-height: 28px;
    margin-top: 6px;
    overflow: visible;
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
  }
`;

export const ContentColumn = styled.div<{ $isUser: boolean }>`
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0;
  align-items: ${({ $isUser }) => ($isUser ? "flex-end" : "stretch")};
`;

export const MessageBubble = styled.div<{
  $isUser: boolean;
  $bareMedia?: boolean;
}>`
  width: ${({ $isUser, $bareMedia }) =>
    $bareMedia || $isUser ? "fit-content" : "100%"};
  max-width: ${({ $isUser, $bareMedia }) =>
    $bareMedia
      ? "min(100%, 800px)"
      : $isUser
        ? "min(72%, 560px)"
        : CONVERSATION_CONTENT_MAX_WIDTH};
  padding: ${({ $isUser, $bareMedia }) =>
    $isUser ? "6px 12px" : $bareMedia ? "0" : "0 4px"};
  display: flex;
  flex-direction: column;
  gap: ${({ $isUser, $bareMedia }) =>
    $isUser ? "8px" : $bareMedia ? "0" : "10px"};
  border-radius: ${({ $isUser, $bareMedia }) =>
    $bareMedia || !$isUser ? "0" : "14px"};
  border: ${({ $isUser, $bareMedia }) =>
    $isUser && !$bareMedia ? "1px solid rgba(229, 231, 235, 0.92)" : "0"};
  background: ${({ $isUser, $bareMedia }) =>
    $isUser && !$bareMedia ? "#f4f4f3" : "transparent"};
  box-shadow: ${({ $isUser, $bareMedia }) =>
    $isUser && !$bareMedia ? "none" : "none"};
  color: ${({ $isUser }) => ($isUser ? "rgb(31, 41, 55)" : "rgb(31, 41, 55)")};
  font-size: ${({ $isUser }) => ($isUser ? "14px" : "15px")};
  line-height: ${({ $isUser }) => ($isUser ? "1.58" : "1.72")};
  font-weight: 400;
  position: relative;

  .markdown-renderer,
  .markdown-renderer * {
    color: inherit;
  }

  &.message-bubble-user-command {
    max-width: min(72%, 560px);
    border: 1px solid rgba(226, 232, 240, 0.76);
    border-radius: 12px;
    background: #f7f7f6;
    box-shadow: none;
  }
`;

export const MessageActions = styled.div`
  display: flex;
  gap: 4px;
  align-self: flex-end;
  position: relative;
  z-index: 30;
  isolation: isolate;
  max-height: 0;
  overflow: hidden;
  opacity: 0;
  pointer-events: none;
  margin-top: 0;
  transform: translateY(-2px);
  transition:
    opacity 0.18s ease,
    max-height 0.18s ease,
    margin-top 0.18s ease,
    transform 0.18s ease;
  background-color: transparent;

  &.message-actions-persistent {
    max-height: 48px;
    margin-top: 8px;
    overflow: visible;
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
  }

  &.image-workbench-message-actions {
    align-self: flex-start;
    margin-top: 8px;
  }
`;
