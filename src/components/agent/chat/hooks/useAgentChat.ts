import { useState, useEffect } from "react";
import { toast } from "sonner";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  startAgentProcess,
  stopAgentProcess,
  getAgentProcessStatus,
  createAgentSession,
  sendAgentMessageStream,
  listAgentSessions,
  deleteAgentSession,
  parseStreamEvent,
  type AgentProcessStatus,
  type SessionInfo,
  type StreamEvent,
} from "@/lib/api/agent";
import { Message, MessageImage, ContentPart, PROVIDER_CONFIG } from "../types";

/** 话题（会话）信息 */
export interface Topic {
  id: string;
  title: string;
  createdAt: Date;
  messagesCount: number;
}

// Helper for localStorage (Persistent across reloads)
const loadPersisted = <T>(key: string, defaultValue: T): T => {
  try {
    const stored = localStorage.getItem(key);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (e) {
    console.error(e);
  }
  return defaultValue;
};

const savePersisted = (key: string, value: unknown) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    console.error(e);
  }
};

// Helper for session storage (Transient data like messages)
const loadTransient = <T>(key: string, defaultValue: T): T => {
  try {
    const stored = sessionStorage.getItem(key);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (key === "agent_messages" && Array.isArray(parsed)) {
        return parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        })) as unknown as T;
      }
      return parsed;
    }
  } catch (e) {
    console.error(e);
  }
  return defaultValue;
};

const saveTransient = (key: string, value: unknown) => {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    console.error(e);
  }
};

export function useAgentChat() {
  const [processStatus, setProcessStatus] = useState<AgentProcessStatus>({
    running: false,
  });

  // Configuration State (Persistent)
  const defaultProvider = "claude";
  const defaultModel = PROVIDER_CONFIG["claude"]?.models[0] || "";

  const [providerType, setProviderType] = useState(() =>
    loadPersisted("agent_pref_provider", defaultProvider),
  );
  const [model, setModel] = useState(() =>
    loadPersisted("agent_pref_model", defaultModel),
  );

  // Session State
  const [sessionId, setSessionId] = useState<string | null>(() =>
    loadTransient("agent_curr_sessionId", null),
  );
  const [messages, setMessages] = useState<Message[]>(() =>
    loadTransient("agent_messages", []),
  );

  // 话题列表
  const [topics, setTopics] = useState<Topic[]>([]);

  const [isSending, setIsSending] = useState(false);

  // Persistence Effects
  useEffect(() => {
    savePersisted("agent_pref_provider", providerType);
  }, [providerType]);
  useEffect(() => {
    savePersisted("agent_pref_model", model);
  }, [model]);

  useEffect(() => {
    saveTransient("agent_curr_sessionId", sessionId);
  }, [sessionId]);
  useEffect(() => {
    saveTransient("agent_messages", messages);
  }, [messages]);

  // 加载话题列表
  const loadTopics = async () => {
    try {
      const sessions = await listAgentSessions();
      const topicList: Topic[] = sessions.map((s: SessionInfo) => ({
        id: s.session_id,
        title: generateTopicTitle(s),
        createdAt: new Date(s.created_at),
        messagesCount: s.messages_count,
      }));
      setTopics(topicList);
    } catch (error) {
      console.error("加载话题列表失败:", error);
    }
  };

  // 根据会话信息生成话题标题
  const generateTopicTitle = (session: SessionInfo): string => {
    if (session.messages_count === 0) {
      return "新话题";
    }
    // 使用创建时间作为默认标题
    const date = new Date(session.created_at);
    return `话题 ${date.toLocaleDateString("zh-CN")} ${date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
  };

  // Initial Load
  useEffect(() => {
    getAgentProcessStatus().then(setProcessStatus).catch(console.error);
    loadTopics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 当 sessionId 变化时刷新话题列表
  useEffect(() => {
    if (sessionId) {
      loadTopics();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Ensure an active session exists (internal helper)
  const _ensureSession = async (): Promise<string | null> => {
    // If we already have a session, we might want to continue using it.
    // However, check if we need to "re-initialize" if critical params changed?
    // User said: "选择模型后，不用和会话绑定". So we keep the session ID if it exists.
    if (sessionId) return sessionId;

    try {
      // TEMPORARY FIX: Disable skills integration due to API type mismatch (Backend expects []SystemMessage, Client sends String)
      // const [claudeSkills, proxyCastSkills] = await Promise.all([
      //     skillsApi.getAll("claude").catch(() => []),
      //     skillsApi.getInstalledProxyCastSkills().catch(() => []),
      // ]);

      // const details: SkillInfo[] = claudeSkills.filter(s => s.installed).map(s => ({
      //     name: s.name,
      //     description: s.description,
      //     path: s.directory ? `~/.claude/skills/${s.directory}/SKILL.md` : undefined,
      // }));

      // proxyCastSkills.forEach(name => {
      //     if (!details.find(d => d.name === name)) {
      //         details.push({ name, path: `~/.proxycast/skills/${name}/SKILL.md` });
      //     }
      // });

      // Create new session with CURRENT provider/model as baseline
      const response = await createAgentSession(
        providerType,
        model || undefined,
        undefined,
        undefined, // details.length > 0 ? details : undefined
      );

      setSessionId(response.session_id);
      return response.session_id;
    } catch (error) {
      console.error("Auto-creation failed", error);
      toast.error("Failed to initialize session");
      return null;
    }
  };

  const sendMessage = async (
    content: string,
    images: MessageImage[],
    webSearch?: boolean,
    thinking?: boolean,
  ) => {
    // 1. Optimistic UI Update
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      images: images.length > 0 ? images : undefined,
      timestamp: new Date(),
    };

    // Placeholder for assistant
    const assistantMsgId = crypto.randomUUID();
    let thinkingText = "思考中...";
    if (thinking && webSearch) {
      thinkingText = "深度思考 + 联网搜索中...";
    } else if (thinking) {
      thinkingText = "深度思考中...";
    } else if (webSearch) {
      thinkingText = "正在搜索网络...";
    }

    const assistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isThinking: true,
      thinkingContent: thinkingText,
      contentParts: [], // 初始化交错内容列表
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsSending(true);

    // 用于累积流式内容
    let accumulatedContent = "";
    let unlisten: UnlistenFn | null = null;

    /**
     * 辅助函数：更新 contentParts，支持交错显示
     * - text_delta: 追加到最后一个 text 类型，或创建新的 text 类型
     * - tool_start: 添加新的 tool_use 类型
     * - tool_end: 更新对应的 tool_use 状态
     */
    const appendTextToParts = (
      parts: ContentPart[],
      text: string,
    ): ContentPart[] => {
      const newParts = [...parts];
      const lastPart = newParts[newParts.length - 1];

      if (lastPart && lastPart.type === "text") {
        // 追加到最后一个 text 类型
        newParts[newParts.length - 1] = {
          type: "text",
          text: lastPart.text + text,
        };
      } else {
        // 创建新的 text 类型
        newParts.push({ type: "text", text });
      }
      return newParts;
    };

    try {
      // 2. 确保有一个活跃的 session（用于保持上下文）
      const activeSessionId = await _ensureSession();
      if (!activeSessionId) {
        throw new Error("无法创建或获取会话");
      }

      // 3. 创建唯一事件名称
      const eventName = `agent_stream_${assistantMsgId}`;

      // 4. 设置事件监听器（流式接收）
      console.log(
        `[AgentChat] 设置事件监听器: ${eventName}, sessionId: ${activeSessionId}`,
      );
      unlisten = await listen<StreamEvent>(eventName, (event) => {
        console.log("[AgentChat] 收到事件:", eventName, event.payload);
        const data = parseStreamEvent(event.payload);
        if (!data) {
          console.warn("[AgentChat] 解析事件失败:", event.payload);
          return;
        }
        console.log("[AgentChat] 解析后数据:", data);

        switch (data.type) {
          case "text_delta":
            // 累积文本并实时更新 UI（同时更新 content 和 contentParts）
            accumulatedContent += data.text;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      content: accumulatedContent,
                      thinkingContent: undefined,
                      // 更新 contentParts，支持交错显示
                      contentParts: appendTextToParts(
                        msg.contentParts || [],
                        data.text,
                      ),
                    }
                  : msg,
              ),
            );
            break;

          case "done":
            // 完成一次 API 响应，但工具循环可能还在继续
            // 不要取消监听，继续等待更多事件
            console.log("[AgentChat] 收到 done 事件，工具循环可能还在继续...");
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      // 保持 isThinking 为 true，直到收到 final_done 或 error
                      content: accumulatedContent || msg.content,
                    }
                  : msg,
              ),
            );
            // 注意：不要在这里 setIsSending(false) 或 unlisten()
            // 工具循环会继续发送事件
            break;

          case "final_done":
            // 整个对话完成（包括所有工具调用）
            console.log("[AgentChat] 收到 final_done 事件，对话完成");
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      isThinking: false,
                      content: accumulatedContent || "(No response)",
                    }
                  : msg,
              ),
            );
            setIsSending(false);
            if (unlisten) {
              unlisten();
              unlisten = null;
            }
            break;

          case "error":
            // 错误处理
            toast.error(`响应错误: ${data.message}`);
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      isThinking: false,
                      content: accumulatedContent || `错误: ${data.message}`,
                    }
                  : msg,
              ),
            );
            setIsSending(false);
            if (unlisten) {
              unlisten();
              unlisten = null;
            }
            break;

          case "tool_start": {
            // 工具开始执行 - 添加到工具调用列表和 contentParts
            console.log(`[Tool Start] ${data.tool_name} (${data.tool_id})`);
            const newToolCall = {
              id: data.tool_id,
              name: data.tool_name,
              arguments: data.arguments,
              status: "running" as const,
              startTime: new Date(),
            };
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? {
                      ...msg,
                      toolCalls: [...(msg.toolCalls || []), newToolCall],
                      // 添加到 contentParts，支持交错显示
                      contentParts: [
                        ...(msg.contentParts || []),
                        { type: "tool_use" as const, toolCall: newToolCall },
                      ],
                    }
                  : msg,
              ),
            );
            break;
          }

          case "tool_end": {
            // 工具执行完成 - 更新工具调用状态和 contentParts
            console.log(`[Tool End] ${data.tool_id}`);
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;

                // 更新 toolCalls
                const updatedToolCalls = (msg.toolCalls || []).map((tc) =>
                  tc.id === data.tool_id
                    ? {
                        ...tc,
                        status: data.result.success
                          ? ("completed" as const)
                          : ("failed" as const),
                        result: data.result,
                        endTime: new Date(),
                      }
                    : tc,
                );

                // 更新 contentParts 中对应的 tool_use
                const updatedContentParts = (msg.contentParts || []).map(
                  (part) => {
                    if (
                      part.type === "tool_use" &&
                      part.toolCall.id === data.tool_id
                    ) {
                      return {
                        ...part,
                        toolCall: {
                          ...part.toolCall,
                          status: data.result.success
                            ? ("completed" as const)
                            : ("failed" as const),
                          result: data.result,
                          endTime: new Date(),
                        },
                      };
                    }
                    return part;
                  },
                );

                return {
                  ...msg,
                  toolCalls: updatedToolCalls,
                  contentParts: updatedContentParts,
                };
              }),
            );
            break;
          }
        }
      });

      // 5. 发送流式请求（传递 sessionId 以保持上下文）
      const imagesToSend =
        images.length > 0
          ? images.map((img) => ({ data: img.data, media_type: img.mediaType }))
          : undefined;

      await sendAgentMessageStream(
        content,
        eventName,
        activeSessionId, // 传递 sessionId 以保持上下文
        model || undefined,
        imagesToSend,
        providerType, // 传递用户选择的 provider
      );
    } catch (error) {
      toast.error(`发送失败: ${error}`);
      // Remove the optimistic assistant message on failure
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantMsgId));
      setIsSending(false);
      if (unlisten) {
        unlisten();
      }
    }
  };

  // 删除单条消息
  const deleteMessage = (id: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== id));
  };

  // 编辑消息
  const editMessage = (id: string, newContent: string) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id ? { ...msg, content: newContent } : msg,
      ),
    );
  };

  const clearMessages = () => {
    setMessages([]);
    setSessionId(null);
    toast.success("新话题已创建");
  };

  // 切换话题
  const switchTopic = async (topicId: string) => {
    if (topicId === sessionId) return;

    // 清空当前消息，切换到新话题
    // 注意：后端目前没有存储消息历史，所以切换话题后消息会丢失
    // 未来可以实现消息持久化
    setMessages([]);
    setSessionId(topicId);
    toast.info("已切换话题");
  };

  // 删除话题
  const deleteTopic = async (topicId: string) => {
    try {
      await deleteAgentSession(topicId);
      setTopics((prev) => prev.filter((t) => t.id !== topicId));

      // 如果删除的是当前话题，清空状态
      if (topicId === sessionId) {
        setSessionId(null);
        setMessages([]);
      }
      toast.success("话题已删除");
    } catch (_error) {
      toast.error("删除话题失败");
    }
  };

  // Status management wrappers
  const handleStartProcess = async () => {
    try {
      await startAgentProcess();
      setProcessStatus({ running: true });
    } catch (_e) {
      toast.error("Start failed");
    }
  };

  const handleStopProcess = async () => {
    try {
      await stopAgentProcess();
      setProcessStatus({ running: false });
      setSessionId(null); // Reset session on stop
    } catch (_e) {
      toast.error("Stop failed");
    }
  };

  return {
    processStatus,
    handleStartProcess,
    handleStopProcess,

    // Config
    providerType,
    setProviderType,
    model,
    setModel,

    // Chat
    messages,
    isSending,
    sendMessage,
    clearMessages,
    deleteMessage,
    editMessage,

    // 话题管理
    topics,
    sessionId,
    switchTopic,
    deleteTopic,
    loadTopics,
  };
}
