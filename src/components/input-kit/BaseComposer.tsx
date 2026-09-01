import React, { useCallback, useEffect, useId, useMemo, useRef } from "react";
import type { ComposerController } from "./ComposerController";

interface BaseComposerRenderContext {
  textareaRef: React.RefObject<HTMLTextAreaElement>;
  textareaProps: React.TextareaHTMLAttributes<HTMLTextAreaElement>;
  hasContent: boolean;
  canSend: boolean;
  isPrimaryDisabled: boolean;
  onPrimaryActionStart: () => void;
  onPrimaryAction: () => void;
}

export type BaseComposerSendTriggerSource = "button" | "enter" | "ime";

export interface BaseComposerSendMetadata {
  triggeredAt: number;
  triggerSource: BaseComposerSendTriggerSource;
  submitTarget?: "start" | "queue" | "steer" | "interrupt" | "command";
}

export interface BaseComposerProps {
  text: string;
  setText: (value: string) => void;
  onSend: (metadata?: BaseComposerSendMetadata) => void;
  onStop?: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  onPaste?: (event: React.ClipboardEvent<HTMLTextAreaElement>) => void;
  onKeyDown?: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onEscape?: () => void;
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
  isFullscreen?: boolean;
  fillHeightWhenFullscreen?: boolean;
  sendOnEnter?: boolean;
  deferSendOnEnter?: boolean;
  maxAutoHeight?: number;
  hasAdditionalContent?: boolean;
  rows?: number;
  autoFocus?: boolean;
  allowSendWhileLoading?: boolean;
  allowEmptySend?: boolean;
  sendOnPointerDown?: boolean;
  controller?: ComposerController;
  children: (context: BaseComposerRenderContext) => React.ReactNode;
}

export const BaseComposer: React.FC<BaseComposerProps> = ({
  text,
  setText,
  onSend,
  onStop,
  isLoading = false,
  disabled = false,
  placeholder,
  onPaste,
  onKeyDown,
  onEscape,
  textareaRef: externalTextareaRef,
  isFullscreen = false,
  fillHeightWhenFullscreen = false,
  sendOnEnter = true,
  deferSendOnEnter = false,
  maxAutoHeight = 300,
  hasAdditionalContent = false,
  rows = 1,
  autoFocus = false,
  allowSendWhileLoading = false,
  allowEmptySend = false,
  sendOnPointerDown = false,
  controller,
  children,
}) => {
  const internalTextareaRef = useRef<HTMLTextAreaElement>(null);
  const textareaRef = externalTextareaRef || internalTextareaRef;
  const textareaId = useId();
  const pendingImeSendRef = useRef(false);
  const pendingPointerSendClickRef = useRef(false);
  const canSendRef = useRef(false);
  const onSendRef = useRef(onSend);

  const hasContent = useMemo(() => {
    return allowEmptySend || text.trim().length > 0 || hasAdditionalContent;
  }, [allowEmptySend, hasAdditionalContent, text]);

  const canSend =
    hasContent && !disabled && (!isLoading || allowSendWhileLoading);
  const isPrimaryDisabled =
    isLoading && !allowSendWhileLoading ? false : !canSend;

  useEffect(() => {
    canSendRef.current = canSend;
  }, [canSend]);

  useEffect(() => {
    onSendRef.current = onSend;
  }, [onSend]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    if (isFullscreen && fillHeightWhenFullscreen) {
      textarea.style.height = "100%";
      return;
    }

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxAutoHeight)}px`;
  }, [
    fillHeightWhenFullscreen,
    isFullscreen,
    maxAutoHeight,
    text,
    textareaRef,
  ]);

  useEffect(() => {
    if (!autoFocus || disabled) return;
    textareaRef.current?.focus();
  }, [autoFocus, disabled, textareaRef]);

  const isImeComposing = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      const nativeEvent = event.nativeEvent as KeyboardEvent & {
        isComposing?: boolean;
      };
      return Boolean(
        nativeEvent.isComposing ||
        nativeEvent.key === "Process" ||
        nativeEvent.keyCode === 229,
      );
    },
    [],
  );

  const dispatchPrimaryAction = useCallback(
    (triggeredAt: number) => {
      if (isLoading && !allowSendWhileLoading) {
        onStop?.();
        return;
      }

      if (!canSend) {
        return;
      }

      onSend({ triggeredAt, triggerSource: "button" });
    },
    [allowSendWhileLoading, canSend, isLoading, onSend, onStop],
  );

  const onPrimaryActionStart = useCallback(() => {
    if (!sendOnPointerDown || pendingPointerSendClickRef.current) {
      return;
    }

    pendingPointerSendClickRef.current = true;
    if (typeof window !== "undefined") {
      window.setTimeout(() => {
        pendingPointerSendClickRef.current = false;
      }, 1_000);
    }

    dispatchPrimaryAction(Date.now());
  }, [dispatchPrimaryAction, sendOnPointerDown]);

  const onPrimaryAction = useCallback(() => {
    if (sendOnPointerDown && pendingPointerSendClickRef.current) {
      pendingPointerSendClickRef.current = false;
      return;
    }

    const triggeredAt = Date.now();
    if (isLoading && !allowSendWhileLoading) {
      onStop?.();
      return;
    }

    if (!canSend) {
      return;
    }

    onSend({ triggeredAt, triggerSource: "button" });
  }, [
    allowSendWhileLoading,
    canSend,
    isLoading,
    onSend,
    onStop,
    sendOnPointerDown,
  ]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      onKeyDown?.(event);
      if (event.defaultPrevented) {
        return;
      }

      const composing = isImeComposing(event);
      if (composing) {
        if (event.key === "Enter" && sendOnEnter && !event.shiftKey) {
          pendingImeSendRef.current = true;
        }
        return;
      }

      if (
        controller &&
        !event.shiftKey &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        (event.key === "ArrowUp" || event.key === "ArrowDown")
      ) {
        const atHistoryBoundary =
          event.key === "ArrowUp"
            ? event.currentTarget.selectionStart === 0 &&
              event.currentTarget.selectionEnd === 0
            : event.currentTarget.selectionStart === event.currentTarget.value.length &&
              event.currentTarget.selectionEnd === event.currentTarget.value.length;
        if (atHistoryBoundary) {
          const recalled = controller.recallHistory(
            event.key === "ArrowUp" ? "previous" : "next",
          );
          if (recalled) {
            event.preventDefault();
            setText(recalled.text);
            window.requestAnimationFrame(() => {
              const textarea = textareaRef.current;
              if (!textarea) {
                return;
              }
              textarea.setSelectionRange(
                recalled.selectionStart,
                recalled.selectionEnd,
              );
            });
          }
          return;
        }
      }

      if (event.key === "Enter" && sendOnEnter && !event.shiftKey) {
        const triggeredAt = Date.now();
        event.preventDefault();
        if (canSend) {
          if (deferSendOnEnter && typeof window !== "undefined") {
            window.requestAnimationFrame(() => {
              if (canSendRef.current) {
                onSendRef.current({ triggeredAt, triggerSource: "enter" });
              }
            });
          } else {
            onSend({ triggeredAt, triggerSource: "enter" });
          }
        }
        return;
      }

      if (event.key === "Escape" && isFullscreen) {
        onEscape?.();
      }
    },
    [
      canSend,
      controller,
      deferSendOnEnter,
      isFullscreen,
      isImeComposing,
      onEscape,
      onKeyDown,
      onSend,
      sendOnEnter,
      setText,
      textareaRef,
    ],
  );

  const handleCompositionEnd = useCallback(() => {
    if (!pendingImeSendRef.current) {
      return;
    }

    pendingImeSendRef.current = false;
    if (!sendOnEnter) {
      return;
    }

    const triggeredAt = Date.now();
    window.requestAnimationFrame(() => {
      if (canSendRef.current) {
        onSendRef.current({ triggeredAt, triggerSource: "ime" });
      }
    });
  }, [sendOnEnter]);

  const handlePaste = useCallback(
    (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
      onPaste?.(event);
      if (event.defaultPrevented || !controller) {
        return;
      }

      const pasted = event.clipboardData.getData("text");
      if (!pasted) {
        return;
      }

      event.preventDefault();
      controller.ingestPaste(pasted, {
        platform:
          typeof navigator !== "undefined" && /Win/i.test(navigator.platform)
            ? "windows"
            : typeof navigator !== "undefined" &&
                /Mac/i.test(navigator.platform)
              ? "macos"
              : "unknown",
      });
      setText(controller.getSnapshot().document.text);
    },
    [controller, onPaste, setText],
  );

  const textareaProps: React.TextareaHTMLAttributes<HTMLTextAreaElement> = {
    id: textareaId,
    name: "agent-chat-message",
    value: text,
    onChange: (event) => {
      controller?.setText(event.target.value, {
        start: event.target.selectionStart,
        end: event.target.selectionEnd,
      });
      setText(event.target.value);
    },
    onKeyDown: handleKeyDown,
    onCompositionEnd: handleCompositionEnd,
    onSelect: (event) => {
      controller?.setSelection(
        event.currentTarget.selectionStart,
        event.currentTarget.selectionEnd,
      );
    },
    onPaste: handlePaste,
    placeholder,
    disabled,
    rows,
    autoFocus,
  };

  return (
    <>
      {children({
        textareaRef,
        textareaProps,
        hasContent,
        canSend,
        isPrimaryDisabled,
        onPrimaryActionStart,
        onPrimaryAction,
      })}
    </>
  );
};
