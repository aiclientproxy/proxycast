import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const CONSUMER_PATHS = [
  "src/components/agent/chat/commands/types.ts",
  "src/components/agent/chat/skill-selection/runtimeInputCapabilityCatalog.ts",
  "src/components/agent/chat/utils/agentRuntimeStatus.ts",
  "src/components/agent/chat/utils/chatToolPreferences.ts",
  "src/components/agent/chat/hooks/agentChatStorage.ts",
  "src/components/agent/chat/hooks/agentStreamEventProcessorAuxiliary.ts",
  "src/components/agent/chat/hooks/agentStreamRuntimeStatusController.ts",
  "src/components/agent/chat/hooks/agentStreamSubmitContext.ts",
  "src/components/agent/chat/hooks/agentStreamSubmitDraft.ts",
  "src/components/agent/chat/hooks/sessionFinalizeController.ts",
  "src/components/agent/chat/hooks/sessionMetadataSyncController.ts",
  "src/components/agent/chat/hooks/useAgentContext.ts",
  "src/components/agent/chat/hooks/useAgentChat.ts",
  "src/components/agent/chat/components/Inputbar/components/InputbarModelExtra.tsx",
  "src/components/agent/chat/hooks/agentStreamRuntimeContextController.ts",
  "src/components/agent/chat/utils/accessModeRuntime.ts",
  "src/components/agent/chat/workspace/useWorkspaceChatToolPreferencesRuntime.ts",
  "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
  "src/components/agent/chat/components/Inputbar/index.tsx",
  "src/components/agent/chat/hooks/agentStreamRequestStartController.ts",
  "src/components/agent/chat/hooks/agentStreamSend.ts",
  "src/components/agent/chat/workspace/WorkspaceConversationScene.tsx",
  "src/components/agent/chat/hooks/agentSessionTopicViewModel.ts",
  "src/components/agent/chat/hooks/agentStreamPreparedSendEnv.ts",
  "src/components/agent/chat/hooks/agentStreamRuntimeHandlerTypes.ts",
  "src/components/agent/chat/hooks/agentStreamUserInputSendPreparation.ts",
  "src/components/agent/chat/utils/submitOpRuntimeCompaction.ts",
  "src/components/agent/chat/hooks/agentChatShared.ts",
  "src/components/agent/chat/hooks/agentStreamResumeBinding.ts",
  "src/components/agent/chat/hooks/agentStreamTurnEventBinding.ts",
  "src/components/agent/chat/hooks/useAgentStream.ts",
  "src/components/agent/chat/utils/buildUserInputSubmitOp.ts",
];

describe("execution runtime current owner boundary", () => {
  it("declared consumers 只从 agentExecutionRuntime 读取 execution runtime 类型", () => {
    for (const relativePath of CONSUMER_PATHS) {
      const source = readFileSync(join(cwd(), relativePath), "utf8");

      expect(source, relativePath).toContain(
        'from "@/lib/api/agentExecutionRuntime"',
      );
      expect(source, relativePath).not.toContain('"@/lib/api/agentRuntime"');
    }
  });
});
