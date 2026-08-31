import {
  useCallback,
  useMemo,
  type ComponentProps,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import { useTranslation } from "react-i18next";
import { Info, Palette } from "lucide-react";
import styled from "styled-components";
import { Switch } from "@/components/ui/switch";
import type { Character } from "@/lib/api/projectMemory";
import { Inputbar } from "../components/Inputbar";
import type { TaskFile } from "../components/TaskFiles";
import { CONVERSATION_CONTENT_MAX_WIDTH } from "../styles/conversationLayoutTokens";
import { type RuntimeToolAvailability } from "../utils/runtimeToolAvailability";
import { resolveCanvasTaskFileTarget } from "../utils/taskFileCanvasSync";
import { GeneralWorkbenchDialogSection } from "./WorkspaceHarnessDialogs";
import { isRenderableTaskFile } from "./generalWorkbenchHelpers";
import type { GeneralWorkbenchEntryPromptState } from "./workspaceSendHelpers";

interface GeneralWorkbenchEntryPromptAccessoryProps {
  prompt: GeneralWorkbenchEntryPromptState;
  restartLabel: string;
  onRestart: () => void;
  onContinue: () => Promise<void> | void;
}

interface SoulArtifactVoiceAccessoryProps {
  enabled: boolean;
  title: string;
  enabledLabel: string;
  disabledLabel: string;
  toggleAria: string;
  onEnabledChange: (enabled: boolean) => void;
}

const InputbarOverlayAccessoryStack = styled.div`
  display: flex;
  width: 100%;
  max-width: 100%;
  flex-direction: column;
  align-items: stretch;
  gap: 8px;
`;

const InputbarControlReplacement = styled.div`
  width: min(100%, ${CONVERSATION_CONTENT_MAX_WIDTH});
  max-width: 100%;
`;

const SoulArtifactVoiceCard = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 10px;
  max-width: min(320px, calc(100vw - 48px));
  min-height: 42px;
  padding: 8px 10px 8px 12px;
  border-radius: 12px;
  border: 1px solid rgba(167, 243, 208, 0.9);
  background: rgba(255, 255, 255, 0.98);
  color: #0f172a;
  box-shadow: 0 14px 28px -24px rgba(15, 23, 42, 0.24);
`;

const SoulArtifactVoiceIcon = styled.span`
  display: inline-flex;
  width: 24px;
  height: 24px;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: rgba(236, 253, 245, 0.98);
  color: #047857;
`;

const SoulArtifactVoiceText = styled.span`
  display: inline-flex;
  min-width: 0;
  flex-direction: column;
  gap: 1px;
`;

const SoulArtifactVoiceTitle = styled.span`
  overflow: hidden;
  color: #0f172a;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const SoulArtifactVoiceStatus = styled.span<{ $enabled: boolean }>`
  color: ${({ $enabled }) => ($enabled ? "#047857" : "#64748b")};
  font-size: 11px;
  font-weight: 600;
  line-height: 1.25;
  white-space: nowrap;
`;

const GeneralWorkbenchEntryPromptCard = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: min(360px, calc(100vw - 48px));
  max-width: min(420px, calc(100vw - 48px));
  padding: 12px 14px;
  border-radius: 18px;
  border: 1px solid rgba(191, 219, 254, 0.92);
  background: linear-gradient(
    180deg,
    rgba(255, 255, 255, 0.98) 0%,
    rgba(239, 246, 255, 0.96) 100%
  );
  color: #0f172a;
  box-shadow: 0 18px 34px -28px rgba(15, 23, 42, 0.26);
`;

const GeneralWorkbenchEntryPromptHeader = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
`;

const GeneralWorkbenchEntryPromptTitleWrap = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
`;

const GeneralWorkbenchEntryPromptTitle = styled.span`
  font-size: 13px;
  font-weight: 700;
  line-height: 1.4;
`;

const GeneralWorkbenchEntryPromptDescription = styled.span`
  font-size: 12px;
  line-height: 1.5;
  color: #475569;
`;

const GeneralWorkbenchEntryPromptActions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 8px;
`;

const GeneralWorkbenchEntryPromptButton = styled.button<{
  $variant?: "primary" | "ghost";
}>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 88px;
  height: 32px;
  padding: 0 12px;
  border-radius: 999px;
  border: 1px solid
    ${({ $variant }) =>
      $variant === "ghost"
        ? "rgba(191, 219, 254, 0.92)"
        : "rgba(59, 130, 246, 0.94)"};
  background: ${({ $variant }) =>
    $variant === "ghost"
      ? "rgba(255, 255, 255, 0.92)"
      : "linear-gradient(180deg, rgba(59,130,246,0.96) 0%, rgba(37,99,235,0.96) 100%)"};
  color: ${({ $variant }) => ($variant === "ghost" ? "#1e293b" : "#eff6ff")};
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition:
    transform 0.16s ease,
    box-shadow 0.2s ease,
    background 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px -18px rgba(37, 99, 235, 0.46);
    background: ${({ $variant }) =>
      $variant === "ghost"
        ? "rgba(239, 246, 255, 0.98)"
        : "linear-gradient(180deg, rgba(37,99,235,0.98) 0%, rgba(29,78,216,0.98) 100%)"};
  }
`;

function renderSoulArtifactVoiceAccessory({
  enabled,
  title,
  enabledLabel,
  disabledLabel,
  toggleAria,
  onEnabledChange,
}: SoulArtifactVoiceAccessoryProps): ReactNode {
  return (
    <SoulArtifactVoiceCard data-testid="soul-artifact-voice-turn-toggle">
      <SoulArtifactVoiceIcon aria-hidden="true">
        <Palette className="h-3.5 w-3.5" />
      </SoulArtifactVoiceIcon>
      <SoulArtifactVoiceText>
        <SoulArtifactVoiceTitle>{title}</SoulArtifactVoiceTitle>
        <SoulArtifactVoiceStatus $enabled={enabled}>
          {enabled ? enabledLabel : disabledLabel}
        </SoulArtifactVoiceStatus>
      </SoulArtifactVoiceText>
      <Switch
        checked={enabled}
        aria-label={toggleAria}
        data-testid="soul-artifact-voice-turn-switch"
        onCheckedChange={onEnabledChange}
      />
    </SoulArtifactVoiceCard>
  );
}

function renderGeneralWorkbenchEntryPromptAccessory({
  prompt,
  restartLabel,
  onRestart,
  onContinue,
}: GeneralWorkbenchEntryPromptAccessoryProps): ReactNode {
  return (
    <GeneralWorkbenchEntryPromptCard data-testid="theme-workbench-entry-prompt">
      <GeneralWorkbenchEntryPromptHeader>
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
        <GeneralWorkbenchEntryPromptTitleWrap>
          <GeneralWorkbenchEntryPromptTitle>
            {prompt.title}
          </GeneralWorkbenchEntryPromptTitle>
          <GeneralWorkbenchEntryPromptDescription>
            {prompt.description}
          </GeneralWorkbenchEntryPromptDescription>
        </GeneralWorkbenchEntryPromptTitleWrap>
      </GeneralWorkbenchEntryPromptHeader>
      <GeneralWorkbenchEntryPromptActions>
        <GeneralWorkbenchEntryPromptButton
          type="button"
          $variant="ghost"
          data-testid="theme-workbench-entry-restart"
          onClick={onRestart}
        >
          {restartLabel}
        </GeneralWorkbenchEntryPromptButton>
        <GeneralWorkbenchEntryPromptButton
          type="button"
          data-testid="theme-workbench-entry-continue"
          onClick={() => {
            void onContinue();
          }}
        >
          {prompt.actionLabel}
        </GeneralWorkbenchEntryPromptButton>
      </GeneralWorkbenchEntryPromptActions>
    </GeneralWorkbenchEntryPromptCard>
  );
}

export type WorkspaceInputbarBuilderParams = Omit<
  ComponentProps<typeof Inputbar>,
  "overlayAccessory"
>;

export interface UseWorkspaceInputbarScenePresentationRuntimeParams {
  setMentionedCharacters: Dispatch<SetStateAction<Character[]>>;
  taskFiles?: TaskFile[];
  selectedFileId?: string;
  isThemeWorkbench: boolean;
  inputbarPresentation: {
    inputbar: Omit<WorkspaceInputbarBuilderParams, "onSelectCharacter">;
    generalWorkbenchEntryPrompt: GeneralWorkbenchEntryPromptState | null;
    onRestartGeneralWorkbenchEntryPrompt: () => void;
    onContinueGeneralWorkbenchEntryPrompt: () => Promise<void> | void;
    planDecisionAccessory?: ReactNode;
    approvalAccessory?: ReactNode;
    soulArtifactVoiceGenerationBrief?: Record<string, unknown> | null;
    soulArtifactVoiceEnabledForTurn: boolean;
    onSoulArtifactVoiceEnabledForTurnChange: (enabled: boolean) => void;
    generalWorkbenchDialog: ComponentProps<
      typeof GeneralWorkbenchDialogSection
    >;
  };
}

export interface WorkspaceInputbarScenePresentationRuntimeResult {
  activeCanvasTaskFile: TaskFile | null;
  inputbarNode: ReactNode;
  generalWorkbenchDialog: ReactNode;
  runtimeToolAvailability: RuntimeToolAvailability | null | undefined;
}

export type InputbarScenePresentationParams =
  UseWorkspaceInputbarScenePresentationRuntimeParams;
export type InputbarPresentationParams =
  InputbarScenePresentationParams["inputbarPresentation"];
export type InputbarParams = InputbarPresentationParams["inputbar"];
export type GeneralWorkbenchDialogParams =
  InputbarPresentationParams["generalWorkbenchDialog"];
export type WorkspaceInputbarToolStates = NonNullable<
  ComponentProps<typeof Inputbar>["toolStates"]
>;

export function useWorkspaceInputbarScenePresentationRuntime({
  setMentionedCharacters,
  taskFiles = [],
  selectedFileId,
  isThemeWorkbench,
  inputbarPresentation,
}: UseWorkspaceInputbarScenePresentationRuntimeParams): WorkspaceInputbarScenePresentationRuntimeResult {
  const { t } = useTranslation("agent");
  const handleSelectCharacter = useCallback(
    (character: Character) => {
      setMentionedCharacters((previous) => {
        if (previous.find((item) => item.id === character.id)) {
          return previous;
        }
        return [...previous, character];
      });
    },
    [setMentionedCharacters],
  );

  const visibleTaskFiles = useMemo(
    () =>
      taskFiles.filter((file) => isRenderableTaskFile(file, isThemeWorkbench)),
    [isThemeWorkbench, taskFiles],
  );

  const visibleSelectedFileId = useMemo(() => {
    if (!selectedFileId) {
      return undefined;
    }

    return visibleTaskFiles.some((file) => file.id === selectedFileId)
      ? selectedFileId
      : undefined;
  }, [selectedFileId, visibleTaskFiles]);

  const activeCanvasTaskFile = useMemo(
    () =>
      resolveCanvasTaskFileTarget(visibleTaskFiles, visibleSelectedFileId)
        .targetFile,
    [visibleSelectedFileId, visibleTaskFiles],
  );

  const generalWorkbenchEntryPromptAccessory = useMemo(
    () =>
      inputbarPresentation.generalWorkbenchEntryPrompt
        ? renderGeneralWorkbenchEntryPromptAccessory({
            prompt: inputbarPresentation.generalWorkbenchEntryPrompt,
            restartLabel: t(
              "agentChat.workspace.generalWorkbenchEntryPrompt.restart",
            ),
            onRestart:
              inputbarPresentation.onRestartGeneralWorkbenchEntryPrompt,
            onContinue:
              inputbarPresentation.onContinueGeneralWorkbenchEntryPrompt,
          })
        : null,
    [
      inputbarPresentation.generalWorkbenchEntryPrompt,
      inputbarPresentation.onContinueGeneralWorkbenchEntryPrompt,
      inputbarPresentation.onRestartGeneralWorkbenchEntryPrompt,
      t,
    ],
  );

  const soulArtifactVoiceAccessory = useMemo(
    () =>
      inputbarPresentation.soulArtifactVoiceGenerationBrief
        ? renderSoulArtifactVoiceAccessory({
            enabled: inputbarPresentation.soulArtifactVoiceEnabledForTurn,
            title: t("agentChat.workspace.soulArtifactVoice.title"),
            enabledLabel: t("agentChat.workspace.soulArtifactVoice.enabled"),
            disabledLabel: t("agentChat.workspace.soulArtifactVoice.disabled"),
            toggleAria: t("agentChat.workspace.soulArtifactVoice.toggleAria"),
            onEnabledChange:
              inputbarPresentation.onSoulArtifactVoiceEnabledForTurnChange,
          })
        : null,
    [
      inputbarPresentation.onSoulArtifactVoiceEnabledForTurnChange,
      inputbarPresentation.soulArtifactVoiceEnabledForTurn,
      inputbarPresentation.soulArtifactVoiceGenerationBrief,
      t,
    ],
  );

  const workspaceInputbarProps = useMemo<WorkspaceInputbarBuilderParams>(
    () => ({
      ...inputbarPresentation.inputbar,
      onSelectCharacter: handleSelectCharacter,
    }),
    [handleSelectCharacter, inputbarPresentation.inputbar],
  );

  const overlayAccessory =
    generalWorkbenchEntryPromptAccessory || soulArtifactVoiceAccessory ? (
      <InputbarOverlayAccessoryStack>
        {generalWorkbenchEntryPromptAccessory}
        {soulArtifactVoiceAccessory}
      </InputbarOverlayAccessoryStack>
    ) : undefined;

  // Runtime permission gates are stricter than plan confirmation: if both
  // exist, approval owns the input area until the current request is submitted.
  const inputbarNode = inputbarPresentation.approvalAccessory ? (
    <InputbarControlReplacement
      data-testid="inputbar-approval-replacement"
      data-action-owner="composer"
    >
      {inputbarPresentation.approvalAccessory}
    </InputbarControlReplacement>
  ) : inputbarPresentation.planDecisionAccessory ? (
    <InputbarControlReplacement
      data-testid="plan-decision-inputbar-replacement"
      data-action-owner="composer"
    >
      {inputbarPresentation.planDecisionAccessory}
    </InputbarControlReplacement>
  ) : (
    <Inputbar {...workspaceInputbarProps} overlayAccessory={overlayAccessory} />
  );
  const generalWorkbenchDialog = (
    <GeneralWorkbenchDialogSection
      {...inputbarPresentation.generalWorkbenchDialog}
    />
  );

  return {
    activeCanvasTaskFile,
    inputbarNode,
    generalWorkbenchDialog,
    runtimeToolAvailability:
      inputbarPresentation.generalWorkbenchDialog.runtimeToolAvailability,
  };
}
