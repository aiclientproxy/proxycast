import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import type { AutoContinueRequestPayload } from "@/lib/api/agentRuntime/sessionTypes";
import type { MessageImage } from "../types";
import type { SendMessageOptions } from "./agentChatShared";
import { dispatchPreparedAgentStreamSend } from "./agentStreamPreparedSendDispatch";
import type { AgentStreamPreparedSendEnv } from "./agentStreamPreparedSendEnv";
import { prepareAgentStreamUserInputSend } from "./agentStreamUserInputSendPreparation";

interface SendAgentStreamMessageOptions {
  content: string;
  images: MessageImage[];
  webSearch?: boolean;
  thinking?: boolean;
  skipUserMessage?: boolean;
  executionStrategyOverride?: AgentExecutionStrategy;
  modelOverride?: string;
  autoContinue?: AutoContinueRequestPayload;
  systemPrompt?: string;
  options?: SendMessageOptions;
  env: AgentStreamPreparedSendEnv;
}

export async function sendAgentStreamMessage(
  options: SendAgentStreamMessageOptions,
) {
  const {
    content,
    images,
    webSearch,
    thinking,
    skipUserMessage = false,
    executionStrategyOverride,
    modelOverride,
    autoContinue,
    systemPrompt,
    options: sendOptions,
    env,
  } = options;

  const currentSessionId = env.sessionIdRef.current?.trim() || "";
  const hasCurrentSession = Boolean(currentSessionId);
  const targetSessionId = sendOptions?.targetSessionId?.trim();
  const shouldBindTargetSession =
    Boolean(targetSessionId) && targetSessionId !== currentSessionId;
  if (!skipUserMessage && (!hasCurrentSession || shouldBindTargetSession)) {
    await env.ensureSession({
      targetSessionId: targetSessionId || undefined,
      skipSessionRestore: sendOptions?.skipSessionRestore === true,
      skipSessionStartHooks: sendOptions?.skipSessionStartHooks === true,
    });
  }

  const activeSessionId = env.sessionIdRef.current?.trim() || targetSessionId;
  if (activeSessionId) {
    await env.waitForSessionProviderSelectionSync?.(activeSessionId);
  }

  const preparedSend = prepareAgentStreamUserInputSend({
    content,
    images,
    webSearch,
    thinking,
    skipUserMessage,
    executionStrategyOverride,
    modelOverride,
    autoContinue,
    systemPrompt,
    options: sendOptions,
    env,
  });

  await dispatchPreparedAgentStreamSend({
    preparedSend,
    env,
  });
}
