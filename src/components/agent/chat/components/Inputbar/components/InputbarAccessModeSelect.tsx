import React from "react";
import { ShieldCheck } from "lucide-react";
import { MetaSelect, MetaSelectIcon, MetaSelectWrap } from "../styles";
import {
  DEFAULT_AGENT_ACCESS_MODE,
  type AgentAccessMode,
} from "../../../hooks/agentChatStorage";

interface InputbarAccessModeSelectProps {
  isFullscreen?: boolean;
  disabled?: boolean;
  accessMode?: AgentAccessMode;
  setAccessMode?: (mode: AgentAccessMode) => void;
}

const ACCESS_MODE_OPTIONS: Array<{
  value: AgentAccessMode;
  label: string;
}> = [
  { value: "read-only", label: "只读" },
  { value: "current", label: "按需确认" },
  { value: "full-access", label: "完全访问" },
];

export const InputbarAccessModeSelect: React.FC<
  InputbarAccessModeSelectProps
> = ({
  isFullscreen = false,
  disabled = false,
  accessMode = DEFAULT_AGENT_ACCESS_MODE,
  setAccessMode,
}) => {
  if (isFullscreen || !setAccessMode) {
    return null;
  }

  return (
    <MetaSelectWrap>
      <MetaSelectIcon aria-hidden>
        <ShieldCheck strokeWidth={1.8} />
      </MetaSelectIcon>
      <MetaSelect
        disabled={disabled}
        aria-label="权限模式"
        data-testid="inputbar-access-mode-select"
        value={accessMode}
        onChange={(event) =>
          setAccessMode(event.target.value as AgentAccessMode)
        }
        $width="92px"
      >
        {ACCESS_MODE_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </MetaSelect>
    </MetaSelectWrap>
  );
};
