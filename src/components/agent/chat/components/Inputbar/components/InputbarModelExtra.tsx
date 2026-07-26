import React from "react";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import { ChatModelSelector } from "../../ChatModelSelector";
import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";

interface InputbarModelExtraProps {
  isFullscreen?: boolean;
  disabled?: boolean;
  providerType?: string;
  setProviderType: (type: string) => void;
  model?: string;
  setModel: (model: string) => void;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
  activeTheme?: string;
  onManageProviders?: () => void;
  executionRuntime?: AgentSessionExecutionRuntime | null;
}

export const InputbarModelExtra: React.FC<InputbarModelExtraProps> = ({
  isFullscreen = false,
  disabled = false,
  providerType,
  setProviderType,
  model,
  setModel,
  reasoningEffort,
  setReasoningEffort,
  activeTheme,
  onManageProviders,
  executionRuntime: _executionRuntime = null,
}) => {
  if (isFullscreen) {
    return null;
  }

  return (
    <div className="flex items-center flex-wrap gap-2">
      <ChatModelSelector
        disabled={disabled}
        providerType={providerType ?? ""}
        setProviderType={setProviderType}
        model={model ?? ""}
        setModel={setModel}
        reasoningEffort={reasoningEffort ?? ""}
        setReasoningEffort={setReasoningEffort}
        activeTheme={activeTheme}
        compactTrigger
        popoverSide="top"
        onManageProviders={onManageProviders}
        backgroundPreload="disabled"
        preserveUnknownModelSelection
      />
    </div>
  );
};
