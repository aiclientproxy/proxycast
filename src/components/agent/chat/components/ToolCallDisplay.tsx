/**
 * 工具调用显示组件
 *
 * 显示工具执行状态和结果
 * Requirements: 9.1, 9.2 - 工具执行指示器和结果折叠面板
 */

import React, { useState } from "react";
import styled, { keyframes } from "styled-components";
import {
  Terminal,
  FileText,
  Edit3,
  FolderOpen,
  ChevronDown,
  ChevronRight,
  Check,
  X,
  Loader2,
} from "lucide-react";
import type { ToolCallState } from "@/lib/api/agent";

// 动画
const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
`;

// 样式组件
const ToolCallContainer = styled.div`
  margin: 8px 0;
  border: 1px solid hsl(var(--border));
  border-radius: 8px;
  overflow: hidden;
  background-color: hsl(var(--muted) / 0.3);
`;

const ToolCallHeader = styled.div<{ $status: string }>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  cursor: pointer;
  transition: background-color 0.2s;
  background-color: ${(props) =>
    props.$status === "running"
      ? "hsl(var(--primary) / 0.1)"
      : props.$status === "failed"
        ? "hsl(var(--destructive) / 0.1)"
        : "transparent"};

  &:hover {
    background-color: hsl(var(--muted) / 0.5);
  }
`;

const ToolIcon = styled.div<{ $status: string }>`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 4px;
  background-color: ${(props) =>
    props.$status === "running"
      ? "hsl(var(--primary))"
      : props.$status === "failed"
        ? "hsl(var(--destructive))"
        : "hsl(var(--primary) / 0.8)"};
  color: white;
`;

const SpinningLoader = styled(Loader2)`
  animation: ${spin} 1s linear infinite;
`;

const ToolName = styled.span`
  font-size: 13px;
  font-weight: 500;
  color: hsl(var(--foreground));
  flex: 1;
`;

const ToolStatus = styled.span<{ $status: string }>`
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 4px;
  background-color: ${(props) =>
    props.$status === "running"
      ? "hsl(var(--primary) / 0.2)"
      : props.$status === "failed"
        ? "hsl(var(--destructive) / 0.2)"
        : "hsl(var(--primary) / 0.1)"};
  color: ${(props) =>
    props.$status === "running"
      ? "hsl(var(--primary))"
      : props.$status === "failed"
        ? "hsl(var(--destructive))"
        : "hsl(var(--primary))"};
  animation: ${(props) => (props.$status === "running" ? pulse : "none")} 1.5s
    ease-in-out infinite;
`;

const ExpandIcon = styled.div`
  color: hsl(var(--muted-foreground));
  transition: transform 0.2s;
`;

const ToolResultPanel = styled.div<{ $expanded: boolean }>`
  display: ${(props) => (props.$expanded ? "block" : "none")};
  border-top: 1px solid hsl(var(--border));
  background-color: hsl(var(--background));
`;

const ResultContent = styled.pre`
  margin: 0;
  padding: 12px;
  font-size: 12px;
  font-family:
    ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
    "Courier New", monospace;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow-y: auto;
  color: hsl(var(--foreground));
  line-height: 1.5;
`;

const ErrorContent = styled(ResultContent)`
  color: hsl(var(--destructive));
  background-color: hsl(var(--destructive) / 0.05);
`;

const ExecutionTime = styled.span`
  font-size: 11px;
  color: hsl(var(--muted-foreground));
  margin-left: auto;
  margin-right: 8px;
`;

// 工具图标映射
const getToolIcon = (toolName: string) => {
  const name = toolName.toLowerCase();
  if (
    name.includes("bash") ||
    name.includes("shell") ||
    name.includes("exec")
  ) {
    return Terminal;
  }
  if (name.includes("read") || name.includes("file")) {
    return FileText;
  }
  if (name.includes("edit") || name.includes("write")) {
    return Edit3;
  }
  if (name.includes("list") || name.includes("dir")) {
    return FolderOpen;
  }
  return Terminal;
};

// 工具名称显示映射
const getToolDisplayName = (toolName: string): string => {
  const nameMap: Record<string, string> = {
    bash: "执行命令",
    read_file: "读取文件",
    write_file: "写入文件",
    edit_file: "编辑文件",
    list_directory: "列出目录",
  };
  return nameMap[toolName] || toolName;
};

// 状态显示文本
const getStatusText = (status: string): string => {
  switch (status) {
    case "running":
      return "执行中...";
    case "completed":
      return "完成";
    case "failed":
      return "失败";
    default:
      return status;
  }
};

interface ToolCallDisplayProps {
  toolCall: ToolCallState;
  defaultExpanded?: boolean;
}

/**
 * 单个工具调用显示组件
 */
export const ToolCallDisplay: React.FC<ToolCallDisplayProps> = ({
  toolCall,
  defaultExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const IconComponent = getToolIcon(toolCall.name);

  // 计算执行时间
  const executionTime =
    toolCall.endTime && toolCall.startTime
      ? Math.round(
          (toolCall.endTime.getTime() - toolCall.startTime.getTime()) / 1000,
        )
      : null;

  const hasResult = toolCall.status !== "running" && toolCall.result;

  return (
    <ToolCallContainer>
      <ToolCallHeader
        $status={toolCall.status}
        onClick={() => hasResult && setExpanded(!expanded)}
      >
        <ToolIcon $status={toolCall.status}>
          {toolCall.status === "running" ? (
            <SpinningLoader size={14} />
          ) : toolCall.status === "failed" ? (
            <X size={14} />
          ) : (
            <IconComponent size={14} />
          )}
        </ToolIcon>

        <ToolName>{getToolDisplayName(toolCall.name)}</ToolName>

        {executionTime !== null && (
          <ExecutionTime>{executionTime}s</ExecutionTime>
        )}

        <ToolStatus $status={toolCall.status}>
          {toolCall.status === "completed" && <Check size={12} />}
          {getStatusText(toolCall.status)}
        </ToolStatus>

        {hasResult && (
          <ExpandIcon>
            {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </ExpandIcon>
        )}
      </ToolCallHeader>

      {hasResult && (
        <ToolResultPanel $expanded={expanded}>
          {toolCall.result?.error ? (
            <ErrorContent>{toolCall.result.error}</ErrorContent>
          ) : (
            <ResultContent>
              {toolCall.result?.output || "(无输出)"}
            </ResultContent>
          )}
        </ToolResultPanel>
      )}
    </ToolCallContainer>
  );
};

interface ToolCallListProps {
  toolCalls: ToolCallState[];
}

/**
 * 工具调用列表组件
 */
export const ToolCallList: React.FC<ToolCallListProps> = ({ toolCalls }) => {
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <div>
      {toolCalls.map((tc) => (
        <ToolCallDisplay key={tc.id} toolCall={tc} />
      ))}
    </div>
  );
};

export default ToolCallDisplay;
