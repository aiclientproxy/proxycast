import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const RETIRED_SNAPSHOT_CONSUMER_PATHS = [
  "src/components/agent/chat/components/MessageList.types.ts",
  "src/components/agent/chat/components/useMessageListTimelineState.ts",
  "src/components/agent/chat/utils/agentTaskRuntime.ts",
  "src/components/agent/chat/utils/inputbarRuntimeStatusLine.ts",
  "src/components/agent/chat/workspace/useSessionRuntimeProjectionDeferral.ts",
  "src/components/agent/chat/hooks/agentStreamInputRestorePlan.ts",
  "src/components/agent/chat/hooks/agentStreamInputRestoreTypes.ts",
  "src/components/agent/chat/hooks/agentStreamSubmissionLifecycle.ts",
  "src/components/agent/chat/hooks/agentStreamPreparedSendEnv.ts",
  "src/components/agent/chat/hooks/agentStreamRuntimeHandlerTypes.ts",
  "src/components/agent/chat/workspace/workspaceConversationCodingViews.tsx",
  "src/components/agent/chat/hooks/agentStreamResumeBinding.ts",
  "src/components/agent/chat/hooks/agentStreamTurnEventBinding.ts",
  "src/components/agent/chat/hooks/useAgentStream.ts",
];
const RETIRED_INPUTBAR_QUEUE_PATHS = [
  "src/components/agent/chat/components/Inputbar/components/QueuedTurnsPanel.tsx",
  "src/components/agent/chat/components/Inputbar/components/QueuedTurnsPanel.test.tsx",
  "src/components/agent/chat/components/Inputbar/components/inputbarQueuedTurnsCopy.ts",
];
const CURRENT_STEER_OWNER_PATH =
  "src/components/agent/chat/hooks/agentStreamSubmitExecution.ts";
const CURRENT_INPUTBAR_PATHS = [
  "src/components/agent/chat/components/Inputbar/index.tsx",
  "src/components/agent/chat/components/Inputbar/components/InputbarCore.tsx",
  "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
];
const RETIRED_QUEUED_TURN_SNAPSHOT_PATHS = [
  "src/lib/api/queuedTurn.ts",
  "src/lib/api/queuedTurn.test.ts",
  "src/lib/api/queuedTurn.d.ts",
];
const CURRENT_QUEUE_READER_PATHS = [
  "src/lib/api/agentRuntime/threadQueueClient.ts",
  "src/components/agent/chat/hooks/useAgentSessionThreadQueue.ts",
  "src/components/agent/chat/components/ThreadQueueStatus.tsx",
];

describe("queued turn current owner boundary", () => {
  it("已迁出的 Renderer/UI/send surface 不得重新读取 queued snapshot", () => {
    for (const relativePath of RETIRED_SNAPSHOT_CONSUMER_PATHS) {
      const source = readFileSync(join(cwd(), relativePath), "utf8");

      expect(source, relativePath).not.toContain('from "@/lib/api/queuedTurn"');
      expect(source, relativePath).not.toContain("QueuedTurnSnapshot");
      expect(source, relativePath).not.toContain("setQueuedTurns");
    }
  });

  it("Inputbar queued-turn 控制保持删除且 composer 不再读取 queuedTurns", () => {
    for (const relativePath of RETIRED_INPUTBAR_QUEUE_PATHS) {
      expect(existsSync(join(cwd(), relativePath)), relativePath).toBe(false);
    }

    const inputbarCore = readFileSync(
      join(
        cwd(),
        "src/components/agent/chat/components/Inputbar/components/InputbarCore.tsx",
      ),
      "utf8",
    );
    expect(inputbarCore).not.toContain("queuedTurns");
  });

  it("Renderer queued-turn 详细快照保持物理删除", () => {
    for (const relativePath of RETIRED_QUEUED_TURN_SNAPSHOT_PATHS) {
      expect(existsSync(join(cwd(), relativePath)), relativePath).toBe(false);
    }
  });

  it("active turn 输入只走 typed turn/steer，不恢复 public queue 写平面", () => {
    const submitExecution = readFileSync(
      join(cwd(), CURRENT_STEER_OWNER_PATH),
      "utf8",
    );

    expect(submitExecution).toContain("expectedTurnId");
    expect(submitExecution).toContain("runtime.steerTurn(params)");
    expect(submitExecution).not.toContain("queueIfBusy");
    expect(submitExecution).not.toContain("promoteQueuedTurn");
    expect(submitExecution).not.toContain("removeQueuedTurn");

    for (const relativePath of CURRENT_INPUTBAR_PATHS) {
      const source = readFileSync(join(cwd(), relativePath), "utf8");
      expect(source, relativePath).not.toContain("inputbar-queued-turn");
      expect(source, relativePath).not.toContain("onPromoteQueuedTurn");
      expect(source, relativePath).not.toContain("onRemoveQueuedTurn");
    }
  });

  it("GUI Queue consumer 只读取 typed list/changed，不恢复 renderer snapshot owner", () => {
    for (const relativePath of CURRENT_QUEUE_READER_PATHS) {
      expect(existsSync(join(cwd(), relativePath)), relativePath).toBe(true);
      const source = readFileSync(join(cwd(), relativePath), "utf8");
      expect(source, relativePath).not.toContain('from "@/lib/api/queuedTurn"');
      expect(source, relativePath).not.toContain("QueuedTurnSnapshot");
      expect(source, relativePath).not.toContain("safeInvoke(");
    }

    const gateway = readFileSync(
      join(cwd(), "src/lib/api/agentRuntime/threadQueueClient.ts"),
      "utf8",
    );
    expect(gateway).toContain("listThreadQueue");
    expect(gateway).not.toContain("addThreadQueue");
    expect(gateway).not.toContain("updateThreadQueue");
    expect(gateway).not.toContain("deleteThreadQueue");
    expect(gateway).not.toContain("reorderThreadQueue");
    expect(gateway).not.toContain("startThreadQueue");

    const hook = readFileSync(
      join(
        cwd(),
        "src/components/agent/chat/hooks/useAgentSessionThreadQueue.ts",
      ),
      "utf8",
    );
    expect(hook).toContain('notification.method !== "thread/queue/changed"');
  });
});
