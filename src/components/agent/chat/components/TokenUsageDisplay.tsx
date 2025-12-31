/**
 * Token 使用量显示组件
 *
 * 在响应完成后显示 token 使用量
 * Requirements: 9.5 - THE Frontend SHALL display token usage statistics after each Agent response
 */

import React from "react";
import styled from "styled-components";
import { Coins } from "lucide-react";
import type { TokenUsage } from "@/lib/api/agent";

const UsageContainer = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 6px;
  background-color: hsl(var(--muted) / 0.5);
  font-size: 11px;
  color: hsl(var(--muted-foreground));
  margin-top: 8px;
`;

const UsageIcon = styled(Coins)`
  width: 12px;
  height: 12px;
  opacity: 0.7;
`;

const UsageText = styled.span`
  font-family:
    ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
    "Courier New", monospace;
`;

const Separator = styled.span`
  opacity: 0.5;
`;

interface TokenUsageDisplayProps {
  usage: TokenUsage;
  className?: string;
}

/**
 * Token 使用量显示组件
 *
 * 显示输入/输出 token 数量
 */
export const TokenUsageDisplay: React.FC<TokenUsageDisplayProps> = ({
  usage,
  className,
}) => {
  const total = usage.input_tokens + usage.output_tokens;

  return (
    <UsageContainer className={className}>
      <UsageIcon />
      <UsageText>{usage.input_tokens.toLocaleString()} in</UsageText>
      <Separator>/</Separator>
      <UsageText>{usage.output_tokens.toLocaleString()} out</UsageText>
      <Separator>·</Separator>
      <UsageText>{total.toLocaleString()} total</UsageText>
    </UsageContainer>
  );
};

export default TokenUsageDisplay;
