import type { ToolCallState, TokenUsage } from "@/lib/api/agent";

export interface MessageImage {
  data: string;
  mediaType: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: MessageImage[];
  timestamp: Date;
  isThinking?: boolean;
  thinkingContent?: string;
  search_results?: any[]; // For potential future use
  /** 工具调用列表（assistant 消息可能包含） */
  toolCalls?: ToolCallState[];
  /** Token 使用量（响应完成后） */
  usage?: TokenUsage;
}

export interface ChatSession {
  id: string;
  title: string;
  providerType: string;
  model: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export const PROVIDER_CONFIG: Record<
  string,
  { label: string; models: string[] }
> = {
  claude: {
    label: "Claude",
    models: [
      "claude-opus-4-5-20251101",
      "claude-sonnet-4-5-20250929",
      "claude-sonnet-4-20250514",
    ],
  },
  kiro: {
    label: "Kiro",
    models: ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514"],
  },
  openai: {
    label: "OpenAI",
    models: [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4-turbo",
      "o1",
      "o1-mini",
      "o3",
      "o3-mini",
    ],
  },
  gemini: {
    label: "Gemini",
    models: ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
  },
  qwen: {
    label: "通义千问",
    models: ["qwen-max", "qwen-plus", "qwen-turbo"],
  },
  codex: {
    label: "Codex",
    models: ["codex-mini-latest"],
  },
  claude_oauth: {
    label: "Claude OAuth",
    models: ["claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"],
  },
  iflow: {
    label: "iFlow",
    models: [],
  },
  antigravity: {
    label: "Antigravity",
    models: [
      "gemini-claude-sonnet-4-5",
      "gemini-claude-sonnet-4-5-thinking",
      "gemini-claude-opus-4-5-thinking",
    ],
  },
};
