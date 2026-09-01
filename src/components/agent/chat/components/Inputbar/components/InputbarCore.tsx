import React, { useCallback, useRef, useState } from "react";
import {
  ActionButtonGroup,
  Container,
  DictationRecordingDot,
  DictationRecordingDuration,
  DictationRecordingWaveform,
  DictationLiveTranscript,
  InputBarContainer,
  InputColumn,
  InputIconButton,
  MainRow,
  MetaSlot,
  StyledTextarea,
  BottomBar,
  LeftSection,
  TrailingSection,
  SendButton,
  SecondaryActionButton,
  DragHandle,
  ImagePreviewContainer,
  ImagePreviewItem,
  ImagePreviewImg,
  ImageRemoveButton,
  PathReferenceChip,
  PathReferenceContainer,
  PathReferenceIcon,
  PathReferenceKnowledgeButton,
  PathReferenceName,
  PathReferencePath,
  PathReferenceRemoveButton,
  PathReferenceText,
} from "../styles";
import { InputbarTools } from "./InputbarTools";
import {
  ArrowUp,
  ChevronDown,
  ChevronUp,
  FileText,
  Folder,
  ImagePlus,
  Loader2,
  Mic,
  ListPlus,
  Plus,
  Square,
  X,
} from "lucide-react";
import {
  BaseComposer,
  type BaseComposerSendMetadata,
} from "@/components/input-kit";
import type { ComposerController } from "@/components/input-kit";
import { isKnowledgeTextSourceCandidate } from "@/features/knowledge/import/knowledgeSourceSupport";
import type { MessageImage, MessagePathReference } from "../../../types";
import type { InputbarCoreCopy } from "./inputbarCoreCopy";
import {
  InputbarPlusMenu,
  type InputbarPlusMenuConfig,
} from "./InputbarPlusMenu";
import { useInputbarDictation } from "../hooks/useInputbarDictation";

const INTERACTIVE_TARGET_SELECTOR =
  "button, a, input, textarea, select, option, [role='button'], [contenteditable=''], [contenteditable='true'], [contenteditable='plaintext-only']";

function shouldFocusComposerTextarea(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) {
    return true;
  }
  return !target.closest(INTERACTIVE_TARGET_SELECTOR);
}

function resolvePendingImagePreviewSrc(image: MessageImage): string {
  const previewUrl = image.previewUrl?.trim();
  if (previewUrl) {
    return previewUrl;
  }
  if (image.data.trim()) {
    return `data:${image.mediaType};base64,${image.data}`;
  }
  return image.sourceUri?.trim() || image.sourcePath?.trim() || "";
}

function formatRecordingDuration(durationSecs: number): string {
  const totalSeconds = Math.max(0, Math.floor(durationSecs));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = `${totalSeconds % 60}`.padStart(2, "0");
  return `${minutes}:${seconds}`;
}

interface InputbarCoreProps {
  uiCopy: InputbarCoreCopy;
  controller?: ComposerController;
  text: string;
  setText: (text: string) => void;
  onSend: (metadata?: BaseComposerSendMetadata) => void;
  /** 停止生成回调 */
  onStop?: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  activeTools: Record<string, boolean>;
  onToolClick: (tool: string) => void;
  pendingImages?: MessageImage[];
  onRemoveImage?: (index: number) => void;
  pathReferences?: MessagePathReference[];
  onImportPathReferenceAsKnowledge?: (reference: MessagePathReference) => void;
  onRemovePathReference?: (id: string) => void;
  onPaste?: (e: React.ClipboardEvent) => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDrop?: (e: React.DragEvent) => void;
  isFullscreen?: boolean;
  /** Textarea ref（用于 CharacterMention） */
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
  /** 输入框底栏左侧扩展区域 */
  leftExtra?: React.ReactNode;
  /** 输入框底栏尾部元信息区域 */
  trailingMeta?: React.ReactNode;
  /** 输入框内部顶部扩展区域（textarea 上方） */
  topExtra?: React.ReactNode;
  /** 输入框提示文案 */
  placeholder?: string;
  /** 工具栏模式 */
  toolMode?: "default" | "attach-only";
  /** 是否显示顶部拖拽条 */
  showDragHandle?: boolean;
  /** 视觉风格 */
  visualVariant?: "default" | "floating";
  /** 下方是否连接外部上下文托盘 */
  connectedContextBar?: boolean;
  /** Enter 发送延后一帧，优先释放首页首帧渲染。 */
  deferSendOnEnter?: boolean;
  /** 首页首发按钮在 pointerdown 阶段预提交，避免 click 前后出现空首页帧。 */
  sendOnPointerDown?: boolean;
  /** 当前输入所属会话，仅用于运行态可观测性与稳定回归定位。 */
  sessionId?: string | null;
  activeTheme?: string;
  showMetaTools?: boolean;
  showTextareaExpandButton?: boolean;
  plusMenu?: InputbarPlusMenuConfig;
}

export const InputbarCore: React.FC<InputbarCoreProps> = ({
  uiCopy,
  controller,
  text,
  setText,
  onSend,
  onStop,
  isLoading = false,
  disabled = false,
  activeTools,
  onToolClick,
  pendingImages = [],
  onRemoveImage,
  pathReferences = [],
  onImportPathReferenceAsKnowledge,
  onRemovePathReference,
  onPaste,
  onDragOver,
  onDrop,
  isFullscreen = false,
  textareaRef: externalTextareaRef,
  leftExtra,
  trailingMeta,
  topExtra,
  placeholder,
  toolMode = "default",
  showDragHandle = true,
  visualVariant = "default",
  connectedContextBar = false,
  deferSendOnEnter = false,
  sendOnPointerDown = false,
  sessionId = null,
  activeTheme,
  showMetaTools = true,
  showTextareaExpandButton = true,
  plusMenu,
}) => {
  const [isTextareaExpanded, setIsTextareaExpanded] = useState(false);
  const inputBarContainerRef = useRef<HTMLDivElement | null>(null);
  const fallbackTextareaRef = useRef<HTMLTextAreaElement>(null);
  const resolvedTextareaRef = externalTextareaRef ?? fallbackTextareaRef;
  const {
    dictationEnabled,
    dictationState,
    liveTranscript,
    recordingStatus,
    isDictationBusy,
    isDictationProcessing,
    handleDictationToggle,
  } = useInputbarDictation({
    text,
    setText,
    textareaRef: resolvedTextareaRef,
    disabled: disabled || isLoading,
  });
  const composerDisabled = disabled || isDictationBusy;
  const isFloatingVariant = visualVariant === "floating";
  const hasInlineComposerContent =
    text.trim().length > 0 ||
    pendingImages.length > 0 ||
    pathReferences.length > 0;
  const shouldCollapseFloatingTools =
    isFloatingVariant &&
    toolMode === "attach-only" &&
    !hasInlineComposerContent;
  const shouldUseCompactFloatingComposer =
    shouldCollapseFloatingTools && !topExtra && !isTextareaExpanded;
  const containerClassName = [
    isFullscreen ? "flex-1 flex flex-col" : "",
    isFloatingVariant ? "floating-composer" : "",
    connectedContextBar ? "context-connected" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const inputBarClassName = [
    isFullscreen ? "flex-1 flex flex-col" : "",
    isFloatingVariant ? "floating-composer" : "",
    shouldUseCompactFloatingComposer ? "floating-collapsed" : "",
    connectedContextBar ? "context-connected" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const textareaClassName = [
    isFullscreen ? "flex-1 resize-none" : "",
    isFloatingVariant ? "floating-composer" : "",
    shouldUseCompactFloatingComposer ? "floating-collapsed" : "",
    isTextareaExpanded ? "composer-expanded" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const bottomBarClassName = [
    isFloatingVariant ? "floating-composer" : "",
    shouldUseCompactFloatingComposer ? "floating-collapsed" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const mainRowClassName = [
    isFloatingVariant ? "floating-composer" : "",
    shouldUseCompactFloatingComposer ? "floating-collapsed" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const leftSectionClassName = shouldCollapseFloatingTools
    ? "floating-collapsed"
    : "";
  const shouldRenderMetaBar =
    !shouldUseCompactFloatingComposer &&
    (Boolean(plusMenu) ||
      Boolean(leftExtra) ||
      Boolean(trailingMeta) ||
      (showMetaTools &&
        toolMode === "default" &&
        !shouldCollapseFloatingTools));
  const handleRemoveImageMouseDown = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
    },
    [],
  );

  const handleRemoveImageClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>, index: number) => {
      event.preventDefault();
      event.stopPropagation();
      onRemoveImage?.(index);
    },
    [onRemoveImage],
  );

  const handleRemovePathReferenceMouseDown = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
    },
    [],
  );

  const handleRemovePathReferenceClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>, id: string) => {
      event.preventDefault();
      event.stopPropagation();
      onRemovePathReference?.(id);
    },
    [onRemovePathReference],
  );
  const handleImportPathReferenceMouseDown = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
    },
    [],
  );
  const handleImportPathReferenceClick = useCallback(
    (
      event: React.MouseEvent<HTMLButtonElement>,
      reference: MessagePathReference,
    ) => {
      event.preventDefault();
      event.stopPropagation();
      onImportPathReferenceAsKnowledge?.(reference);
    },
    [onImportPathReferenceAsKnowledge],
  );

  const handleToggleTextareaExpanded = useCallback(() => {
    setIsTextareaExpanded((previous) => !previous);
  }, []);

  return (
    <BaseComposer
      controller={controller}
      text={text}
      setText={setText}
      onSend={onSend}
      onStop={onStop}
      isLoading={isLoading}
      disabled={composerDisabled}
      onPaste={onPaste}
      isFullscreen={isFullscreen}
      fillHeightWhenFullscreen
      hasAdditionalContent={
        pendingImages.length > 0 || pathReferences.length > 0
      }
      maxAutoHeight={isTextareaExpanded ? 360 : isFloatingVariant ? 240 : 120}
      textareaRef={resolvedTextareaRef}
      onEscape={() => onToolClick("fullscreen")}
      allowSendWhileLoading
      deferSendOnEnter={deferSendOnEnter}
      sendOnPointerDown={sendOnPointerDown}
      rows={isTextareaExpanded ? 7 : isFloatingVariant ? 3 : 1}
      placeholder={
        placeholder ||
        (isFullscreen
          ? uiCopy.placeholder.fullscreen
          : uiCopy.placeholder.default)
      }
    >
      {({
        textareaProps,
        textareaRef,
        isPrimaryDisabled,
        onPrimaryAction,
        onPrimaryActionStart,
      }) => {
        const loadingSecondaryActionLabel = isPrimaryDisabled
          ? uiCopy.action.running
          : uiCopy.action.defer;
        const recordingLabel = uiCopy.dictation.recording(
          formatRecordingDuration(recordingStatus?.duration ?? 0),
        );
        const dictationButtonLabel =
          dictationState === "listening"
            ? uiCopy.dictation.stopRecording(recordingLabel)
            : isDictationProcessing
              ? uiCopy.dictation.transcribing
              : uiCopy.dictation.start;
        const dictationButtonClassName = [
          "is-dictation",
          dictationState === "listening" ? "is-recording" : "",
          isDictationProcessing ? "is-processing" : "",
        ]
          .filter(Boolean)
          .join(" ");
        const handleContainerMouseDownCapture = (
          event: React.MouseEvent<HTMLDivElement>,
        ) => {
          if (!isFloatingVariant || toolMode !== "attach-only") {
            return;
          }
          if (!shouldFocusComposerTextarea(event.target)) {
            return;
          }
          window.requestAnimationFrame(() => {
            textareaRef.current?.focus();
          });
        };

        return (
          <Container className={containerClassName}>
            <InputBarContainer
              ref={inputBarContainerRef}
              data-testid="inputbar-core-container"
              className={inputBarClassName}
              onMouseDownCapture={handleContainerMouseDownCapture}
              onDragEnterCapture={onDragOver}
              onDragOverCapture={onDragOver}
              onDropCapture={onDrop}
            >
              {!isFullscreen && showDragHandle && <DragHandle />}

              {pendingImages.length > 0 && (
                <ImagePreviewContainer>
                  {pendingImages.map((img, index) => (
                    <ImagePreviewItem key={index}>
                      <ImagePreviewImg
                        src={resolvePendingImagePreviewSrc(img)}
                        alt={uiCopy.image.previewAlt(index + 1)}
                      />
                      <ImageRemoveButton
                        type="button"
                        aria-label={uiCopy.image.remove(index + 1)}
                        onMouseDown={handleRemoveImageMouseDown}
                        onClick={(event) =>
                          handleRemoveImageClick(event, index)
                        }
                      >
                        <X size={12} />
                      </ImageRemoveButton>
                    </ImagePreviewItem>
                  ))}
                </ImagePreviewContainer>
              )}

              {pathReferences.length > 0 ? (
                <PathReferenceContainer aria-label={uiCopy.path.containerLabel}>
                  {pathReferences.map((reference) => {
                    const ReferenceIcon = reference.isDir ? Folder : FileText;
                    return (
                      <PathReferenceChip
                        key={reference.id}
                        title={reference.name}
                        data-testid="inputbar-path-reference-chip"
                      >
                        <PathReferenceIcon $isDir={reference.isDir}>
                          <ReferenceIcon size={14} aria-hidden />
                        </PathReferenceIcon>
                        <PathReferenceText>
                          <PathReferenceName>
                            {reference.name}
                          </PathReferenceName>
                          <PathReferencePath>
                            {reference.isDir
                              ? uiCopy.path.localFolder
                              : uiCopy.path.localFile}
                          </PathReferencePath>
                        </PathReferenceText>
                        {onImportPathReferenceAsKnowledge &&
                        isKnowledgeTextSourceCandidate(reference) ? (
                          <PathReferenceKnowledgeButton
                            type="button"
                            aria-label={uiCopy.path.importAsKnowledge(
                              reference.name,
                            )}
                            onMouseDown={handleImportPathReferenceMouseDown}
                            onClick={(event) =>
                              handleImportPathReferenceClick(event, reference)
                            }
                          >
                            <FileText size={12} aria-hidden />
                            {uiCopy.path.importAction}
                          </PathReferenceKnowledgeButton>
                        ) : null}
                        <PathReferenceRemoveButton
                          type="button"
                          aria-label={uiCopy.path.remove(reference.name)}
                          onMouseDown={handleRemovePathReferenceMouseDown}
                          onClick={(event) =>
                            handleRemovePathReferenceClick(event, reference.id)
                          }
                        >
                          <X size={12} />
                        </PathReferenceRemoveButton>
                      </PathReferenceChip>
                    );
                  })}
                </PathReferenceContainer>
              ) : null}

              {topExtra}
              <MainRow className={mainRowClassName}>
                <InputIconButton
                  type="button"
                  onClick={() => onToolClick("attach")}
                  aria-label={uiCopy.image.add}
                  title={uiCopy.image.add}
                >
                  <ImagePlus size={14} />
                </InputIconButton>
                <InputColumn>
                  <StyledTextarea
                    ref={textareaRef}
                    {...textareaProps}
                    data-session-id={sessionId || undefined}
                    className={textareaClassName}
                  />
                  {liveTranscript.trim() ? (
                    <DictationLiveTranscript
                      role="status"
                      aria-label={uiCopy.dictation.liveTranscript}
                      data-testid="inputbar-dictation-live-transcript"
                    >
                      {liveTranscript}
                    </DictationLiveTranscript>
                  ) : null}
                </InputColumn>
                <ActionButtonGroup data-testid="inputbar-primary-actions">
                  {showTextareaExpandButton ? (
                    <InputIconButton
                      type="button"
                      onClick={handleToggleTextareaExpanded}
                      disabled={disabled}
                      data-testid="inputbar-expand-toggle"
                      className={isTextareaExpanded ? "is-active" : ""}
                      aria-label={
                        isTextareaExpanded
                          ? uiCopy.textarea.collapse
                          : uiCopy.textarea.expand
                      }
                      title={
                        isTextareaExpanded
                          ? uiCopy.textarea.collapse
                          : uiCopy.textarea.expand
                      }
                    >
                      {isTextareaExpanded ? (
                        <ChevronDown size={14} />
                      ) : (
                        <ChevronUp size={14} />
                      )}
                    </InputIconButton>
                  ) : null}
                  {dictationEnabled ? (
                    <InputIconButton
                      type="button"
                      data-testid="inputbar-dictation-toggle"
                      onClick={() => {
                        void handleDictationToggle();
                      }}
                      disabled={
                        disabled ||
                        isLoading ||
                        (isDictationBusy && dictationState !== "listening")
                      }
                      className={dictationButtonClassName}
                      aria-label={dictationButtonLabel}
                      aria-pressed={dictationState === "listening"}
                      title={dictationButtonLabel}
                    >
                      {dictationState === "listening" ? (
                        <>
                          <DictationRecordingDot aria-hidden />
                          <DictationRecordingDuration>
                            {formatRecordingDuration(
                              recordingStatus?.duration ?? 0,
                            )}
                          </DictationRecordingDuration>
                          <DictationRecordingWaveform aria-hidden />
                          <Mic size={14} aria-hidden />
                        </>
                      ) : isDictationProcessing ? (
                        <Loader2 size={14} aria-hidden />
                      ) : (
                        <Mic size={14} aria-hidden />
                      )}
                    </InputIconButton>
                  ) : null}
                  {isLoading ? (
                    <SecondaryActionButton
                      type="button"
                      onClick={onPrimaryAction}
                      disabled={isPrimaryDisabled}
                      aria-label={loadingSecondaryActionLabel}
                      title={loadingSecondaryActionLabel}
                    >
                      <span>{loadingSecondaryActionLabel}</span>
                    </SecondaryActionButton>
                  ) : null}
                  {isLoading ? (
                    <InputIconButton
                      type="button"
                      onClick={() =>
                        onSend({
                          triggeredAt: Date.now(),
                          triggerSource: "button",
                          submitTarget: "queue",
                        })
                      }
                      disabled={isPrimaryDisabled}
                      aria-label={uiCopy.action.queue}
                      title={uiCopy.action.queue}
                    >
                      <ListPlus size={14} />
                    </InputIconButton>
                  ) : null}
                  {isLoading ? (
                    <InputIconButton
                      type="button"
                      onClick={onStop}
                      disabled={!onStop}
                      $destructive
                      aria-label={uiCopy.action.stop}
                      title={uiCopy.action.stop}
                    >
                      <Square size={14} fill="currentColor" />
                    </InputIconButton>
                  ) : null}
                  {!isLoading && !isDictationBusy ? (
                    <SendButton
                      type="button"
                      data-testid="send-btn"
                      onPointerDown={
                        sendOnPointerDown ? onPrimaryActionStart : undefined
                      }
                      onClick={onPrimaryAction}
                      disabled={isPrimaryDisabled}
                      aria-label={uiCopy.action.send}
                      title={uiCopy.action.send}
                    >
                      <ArrowUp size={16} strokeWidth={2.4} />
                    </SendButton>
                  ) : null}
                </ActionButtonGroup>
              </MainRow>

              {shouldRenderMetaBar ? (
                <BottomBar className={bottomBarClassName}>
                  <LeftSection
                    className={leftSectionClassName}
                    data-testid="inputbar-meta-left"
                  >
                    {plusMenu ? (
                      <InputbarPlusMenu config={plusMenu} disabled={disabled}>
                        <InputIconButton
                          type="button"
                          aria-label={plusMenu.labels.open}
                          title={plusMenu.labels.open}
                          data-testid="inputbar-plus-trigger"
                        >
                          <Plus size={15} />
                        </InputIconButton>
                      </InputbarPlusMenu>
                    ) : null}
                    {leftExtra ? <MetaSlot>{leftExtra}</MetaSlot> : null}
                    {!shouldCollapseFloatingTools && showMetaTools ? (
                      <InputbarTools
                        onToolClick={onToolClick}
                        activeTools={activeTools}
                        toolMode={toolMode}
                        activeTheme={activeTheme}
                      />
                    ) : null}
                  </LeftSection>
                  {trailingMeta ? (
                    <TrailingSection data-testid="inputbar-meta-trailing">
                      <MetaSlot>{trailingMeta}</MetaSlot>
                    </TrailingSection>
                  ) : null}
                </BottomBar>
              ) : null}
            </InputBarContainer>
          </Container>
        );
      }}
    </BaseComposer>
  );
};
