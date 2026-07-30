import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const hooksDir = dirname(fileURLToPath(import.meta.url));
const canonicalThreadProjectionPath = join(
  hooksDir,
  "../../../../lib/api/agentRuntime/appServerCanonicalThreadProjection.ts",
);

const removedModules = [
  "agentStreamAgentMessageContentSync.ts",
  "agentStreamReasoningContentSync.ts",
] as const;

const currentOwners = [
  "agentChatHistory.ts",
  "agentChatHistoryReasoning.ts",
  "agentChatHistoryThreadItems.ts",
  "agentSessionState.ts",
  "agentStreamRuntimeHandler.ts",
  "agentStreamRuntimeHandlerActions.ts",
  "agentStreamRuntimeHandlerTypes.ts",
  "agentStreamRuntimeLifecycleEvents.ts",
] as const;

const removedSymbols = [
  "mergeAssistantAgentMessageContentPartsFromThreadItems",
  "mergeThreadItemReasoningIntoMessages",
  "syncAssistantAgentMessageContentPartFromThreadItem",
  "syncAssistantReasoningContentPartFromThreadItem",
  "streamedAgentMessageItemsByItemId",
] as const;

describe("canonical Item to Message synthesis boundary", () => {
  it("live AgentMessage/Reasoning Message sync 模块必须保持删除", () => {
    for (const moduleName of removedModules) {
      expect(existsSync(join(hooksDir, moduleName)), moduleName).toBe(false);
    }
  });

  it("current hooks 不得恢复 canonical Item 到 Message 的二次写回", () => {
    for (const fileName of currentOwners) {
      const source = readFileSync(join(hooksDir, fileName), "utf8");
      for (const symbol of removedSymbols) {
        expect(source, `${fileName}: ${symbol}`).not.toContain(symbol);
      }
    }
  });

  it("cold thread/read 不得恢复 canonical Item 到 Message 合成", () => {
    const source = readFileSync(canonicalThreadProjectionPath, "utf8");
    expect(source).not.toContain("canonicalItemsToMessages");
    expect(source).not.toContain("const messages =");
  });
});
