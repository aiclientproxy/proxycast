/**
 * 流式消息渲染组件
 *
 * 实现实时 Markdown 渲染，区分文本响应和工具调用响应
 * Requirements: 9.3, 9.4
 */

import React, { memo, useMemo } from "react";
import styled, { keyframes } from "styled-components";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { ToolCallList } from "./ToolCallDisplay";
import type { ToolCallState } from "@/lib/api/agent";

// 光标闪烁动画
const blink = keyframes`
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
`;

const StreamingContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const TextSection = styled.div`
  position: relative;
`;

const StreamingCursor = styled.span`
  display: inline-block;
  width: 2px;
  height: 1em;
  background-color: hsl(var(--primary));
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: ${blink} 1s step-end infinite;
`;

const ToolSection = styled.div`
  margin-top: 8px;
`;

interface StreamingRendererProps {
  /** 文本内容 */
  content: string;
  /** 是否正在流式输出 */
  isStreaming?: boolean;
  /** 工具调用列表 */
  toolCalls?: ToolCallState[];
  /** 是否显示光标 */
  showCursor?: boolean;
}

/**
 * 流式消息渲染组件
 *
 * 支持实时 Markdown 渲染和工具调用显示
 * Requirements: 9.3 - THE Frontend SHALL distinguish between text responses and tool call responses visually
 * Requirements: 9.4 - WHEN streaming text, THE Frontend SHALL render markdown formatting in real-time
 */
export const StreamingRenderer: React.FC<StreamingRendererProps> = memo(
  ({ content, isStreaming = false, toolCalls, showCursor = true }) => {
    // 判断是否有正在执行的工具
    const hasRunningTools = useMemo(
      () => toolCalls?.some((tc) => tc.status === "running") ?? false,
      [toolCalls],
    );

    // 判断是否显示光标
    const shouldShowCursor = isStreaming && showCursor && !hasRunningTools;

    // 判断是否有工具调用
    const hasToolCalls = toolCalls && toolCalls.length > 0;

    return (
      <StreamingContainer>
        {/* 工具调用区域 - 显示在文本之前 */}
        {hasToolCalls && (
          <ToolSection>
            <ToolCallList toolCalls={toolCalls} />
          </ToolSection>
        )}

        {/* 文本内容区域 */}
        {content && (
          <TextSection>
            <MarkdownRenderer content={content} />
            {shouldShowCursor && <StreamingCursor />}
          </TextSection>
        )}

        {/* 如果没有内容但正在流式输出，显示光标 */}
        {!content && isStreaming && showCursor && !hasRunningTools && (
          <TextSection>
            <StreamingCursor />
          </TextSection>
        )}
      </StreamingContainer>
    );
  },
);

StreamingRenderer.displayName = "StreamingRenderer";

export default StreamingRenderer;
