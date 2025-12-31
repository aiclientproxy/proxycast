/**
 * AI Agent 聊天页面
 *
 * 包含聊天区域和侧边栏（话题/技能列表）
 */

import React, { useState, useCallback } from "react";
import styled from "styled-components";
import { useAgentChat } from "./hooks/useAgentChat";
import { ChatNavbar } from "./components/ChatNavbar";
import { ChatSidebar } from "./components/ChatSidebar";
import { ChatSettings } from "./components/ChatSettings";
import { MessageList } from "./components/MessageList";
import { Inputbar } from "./components/Inputbar";
import { EmptyState } from "./components/EmptyState";
import type { MessageImage } from "./types";

const PageContainer = styled.div`
  display: flex;
  height: 100%;
  width: 100%;
  background-color: hsl(var(--background));
`;

const MainArea = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
`;

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
`;

const ChatContent = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  padding: 0 16px;
`;

export function AgentChatPage({
  onNavigate: _onNavigate,
}: {
  onNavigate?: (page: string) => void;
}) {
  const {
    processStatus,
    providerType,
    setProviderType,
    model,
    setModel,
    messages,
    isSending,
    sendMessage,
    clearMessages,
    deleteMessage,
    editMessage,
    topics,
    sessionId,
    switchTopic,
    deleteTopic,
  } = useAgentChat();

  const [showSidebar, setShowSidebar] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [input, setInput] = useState("");

  const handleSend = useCallback(
    async (
      images?: MessageImage[],
      webSearch?: boolean,
      thinking?: boolean,
    ) => {
      if (!input.trim() && (!images || images.length === 0)) return;
      const text = input;
      setInput("");
      await sendMessage(text, images || [], webSearch, thinking);
    },
    [input, sendMessage],
  );

  const handleClearMessages = useCallback(() => {
    clearMessages();
    setInput("");
  }, [clearMessages]);

  const handleToggleSidebar = () => {
    setShowSidebar(!showSidebar);
  };

  const hasMessages = messages.length > 0;

  return (
    <PageContainer>
      {showSidebar && (
        <ChatSidebar
          onNewChat={handleClearMessages}
          topics={topics}
          currentTopicId={sessionId}
          onSwitchTopic={switchTopic}
          onDeleteTopic={deleteTopic}
        />
      )}

      <MainArea>
        <ChatNavbar
          providerType={providerType}
          setProviderType={setProviderType}
          model={model}
          setModel={setModel}
          isRunning={processStatus.running}
          onToggleHistory={handleToggleSidebar}
          onToggleFullscreen={() => {}}
          onToggleSettings={() => setShowSettings(!showSettings)}
        />

        <ChatContainer>
          {hasMessages ? (
            <ChatContent>
              <MessageList
                messages={messages}
                onDeleteMessage={deleteMessage}
                onEditMessage={editMessage}
              />
            </ChatContent>
          ) : (
            <EmptyState
              input={input}
              setInput={setInput}
              onSend={(text) => {
                setInput(text);
                // 使用 setTimeout 确保 state 更新后再发送
                setTimeout(() => handleSend([], false, false), 0);
              }}
            />
          )}

          {hasMessages && (
            <Inputbar
              input={input}
              setInput={setInput}
              onSend={handleSend}
              isLoading={isSending}
              disabled={!processStatus.running && false}
              onClearMessages={handleClearMessages}
            />
          )}
        </ChatContainer>
      </MainArea>

      {showSettings && <ChatSettings onClose={() => setShowSettings(false)} />}
    </PageContainer>
  );
}
