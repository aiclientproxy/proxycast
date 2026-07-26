import React from "react";
import { X } from "lucide-react";
import {
  ModeStatusChip as StyledModeStatusChip,
  ModeStatusLabel,
  ModeStatusRemoveMark,
} from "../styles";

interface InputbarModeStatusChipProps {
  label: string;
  testId: string;
  disabled?: boolean;
  onRemove: () => void;
}

export function InputbarModeStatusChip({
  label,
  testId,
  disabled = false,
  onRemove,
}: InputbarModeStatusChipProps) {
  return (
    <StyledModeStatusChip
      type="button"
      aria-label={label}
      title={label}
      data-testid={testId}
      disabled={disabled}
      onMouseDown={(event) => event.preventDefault()}
      onClick={onRemove}
    >
      <ModeStatusRemoveMark aria-hidden data-testid={`${testId}-remove-mark`}>
        <X />
      </ModeStatusRemoveMark>
      <ModeStatusLabel>{label}</ModeStatusLabel>
    </StyledModeStatusChip>
  );
}
