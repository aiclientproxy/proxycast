import React, { useState, useRef, useEffect } from "react";
import {
  User,
  Bot,
  Copy,
  Edit2,
  Trash2,
  Lightbulb,
  ChevronDown,
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import {
  MessageListContainer,
  MessageWrapper,
  AvatarColumn,
  ContentColumn,
  MessageHeader,
  AvatarCircle,
  SenderName,
  TimeStamp,
  MessageBubble,
  MessageActions,
  ThinkingBox,
  ThinkingHeader,
  ThinkingContent,
} from "../styles";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { StreamingRenderer } from "./StreamingRenderer";
import { TokenUsageDisplay } from "./TokenUsageDisplay";
import { Message } from "../types";

interface MessageListProps {
  messages: Message[];
  onDeleteMessage?: (id: string) => void;
  onEditMessage?: (id: string, content: string) => void;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  onDeleteMessage,
  onEditMessage,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expandedThinking, setExpandedThinking] = useState<
    Record<string, boolean>
  >({});
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const toggleThinking = (id: string) => {
    setExpandedThinking((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const handleCopy = async (content: string, id: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(id);
      toast.success("已复制到剪贴板");
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      toast.error("复制失败");
    }
  };

  const handleEdit = (msg: Message) => {
    setEditingId(msg.id);
    setEditContent(msg.content);
  };

  const handleSaveEdit = (id: string) => {
    if (onEditMessage && editContent.trim()) {
      onEditMessage(id, editContent);
    }
    setEditingId(null);
    setEditContent("");
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditContent("");
  };

  const handleDelete = (id: string) => {
    if (onDeleteMessage) {
      onDeleteMessage(id);
      toast.success("消息已删除");
    }
  };

  return (
    <MessageListContainer>
      <div className="py-8 flex flex-col">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-64 text-muted-foreground opacity-50">
            <Bot size={48} className="mb-4 text-primary/20" />
            <p className="text-lg font-medium">开始一段新的对话吧</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageWrapper key={msg.id} $isUser={msg.role === "user"}>
            <AvatarColumn>
              <AvatarCircle $isUser={msg.role === "user"}>
                {msg.role === "user" ? <User size={18} /> : <Bot size={18} />}
              </AvatarCircle>
            </AvatarColumn>

            <ContentColumn>
              <MessageHeader>
                <SenderName>
                  {msg.role === "user" ? "用户" : "Assistant"}
                </SenderName>
                <TimeStamp>{formatTime(msg.timestamp)}</TimeStamp>
              </MessageHeader>

              <MessageBubble $isUser={msg.role === "user"}>
                {msg.isThinking && (
                  <ThinkingBox $expanded={!!expandedThinking[msg.id]}>
                    <ThinkingHeader onClick={() => toggleThinking(msg.id)}>
                      <Lightbulb size={14} className="text-yellow-500" />
                      <span>{msg.thinkingContent}</span>
                      <ChevronDown
                        size={14}
                        className={cn(
                          "ml-1 transition-transform duration-200",
                          expandedThinking[msg.id] && "rotate-180",
                        )}
                      />
                    </ThinkingHeader>
                    {expandedThinking[msg.id] && (
                      <ThinkingContent>正在深度思考...</ThinkingContent>
                    )}
                  </ThinkingBox>
                )}

                {editingId === msg.id ? (
                  <div className="flex flex-col gap-2">
                    <textarea
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                      className="w-full min-h-[100px] p-2 rounded border border-border bg-background resize-none focus:outline-none focus:ring-1 focus:ring-primary"
                      autoFocus
                    />
                    <div className="flex gap-2 justify-end">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleCancelEdit}
                      >
                        取消
                      </Button>
                      <Button size="sm" onClick={() => handleSaveEdit(msg.id)}>
                        保存
                      </Button>
                    </div>
                  </div>
                ) : msg.role === "assistant" ? (
                  /* 使用 StreamingRenderer 渲染 assistant 消息 - Requirements: 9.3, 9.4 */
                  <StreamingRenderer
                    content={msg.content}
                    isStreaming={msg.isThinking}
                    toolCalls={msg.toolCalls}
                    showCursor={msg.isThinking && !msg.content}
                  />
                ) : (
                  <MarkdownRenderer content={msg.content} />
                )}

                {msg.images && msg.images.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-3">
                    {msg.images.map((img, i) => (
                      <img
                        key={i}
                        src={`data:${img.mediaType};base64,${img.data}`}
                        className="max-w-xs rounded-lg border border-border"
                        alt="attachment"
                      />
                    ))}
                  </div>
                )}

                {/* Token 使用量显示 - Requirements: 9.5 */}
                {msg.role === "assistant" && !msg.isThinking && msg.usage && (
                  <TokenUsageDisplay usage={msg.usage} />
                )}

                {editingId !== msg.id && (
                  <MessageActions className="message-actions">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-muted-foreground hover:text-foreground"
                      onClick={() => handleCopy(msg.content, msg.id)}
                    >
                      {copiedId === msg.id ? (
                        <Check size={12} className="text-green-500" />
                      ) : (
                        <Copy size={12} />
                      )}
                    </Button>
                    {msg.role === "user" && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-muted-foreground hover:text-foreground"
                        onClick={() => handleEdit(msg)}
                      >
                        <Edit2 size={12} />
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-muted-foreground hover:text-destructive"
                      onClick={() => handleDelete(msg.id)}
                    >
                      <Trash2 size={12} />
                    </Button>
                  </MessageActions>
                )}
              </MessageBubble>
            </ContentColumn>
          </MessageWrapper>
        ))}
        <div ref={scrollRef} />
      </div>
    </MessageListContainer>
  );
};
