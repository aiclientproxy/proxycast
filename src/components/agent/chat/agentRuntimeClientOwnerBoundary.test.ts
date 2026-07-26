import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const CURRENT_OWNER_CONSUMERS = [
  {
    path: "src/components/agent/chat/workspace/useSessionRecentMetadataSyncRuntime.ts",
    owner: "sessionClient",
  },
  {
    path: "src/components/agent/chat/hooks/agentChatActionState.ts",
    owner: "requestTypes",
  },
  {
    path: "src/components/agent/chat/workspace/useWorkspaceImageWorkbenchCommandActionRuntime.ts",
    owner: "agentClient",
  },
  {
    path: "src/components/agent/chat/workspace/useWorkspaceArticleEditorRightSurfaceRuntime.ts",
    owner: "appServerArtifactClient",
  },
] as const;

const CURRENT_OWNER_MOCK_CONSUMERS = [
  "src/components/agent/chat/hooks/agentRuntimeAdapter.test.ts",
  "src/components/agent/chat/hooks/useAgentChat.testUtils.tsx",
] as const;

describe("agent runtime client current owner boundary", () => {
  it("declared consumers 直连各自 current owner", () => {
    for (const { owner, path } of CURRENT_OWNER_CONSUMERS) {
      const source = readFileSync(join(cwd(), path), "utf8");

      expect(source, path).toContain(`from "@/lib/api/agentRuntime/${owner}"`);
      expect(source, path).not.toContain('from "@/lib/api/agentRuntime"');
    }
  });

  it("runtime client 测试只 mock clientFactory current owner", () => {
    for (const path of CURRENT_OWNER_MOCK_CONSUMERS) {
      const source = readFileSync(join(cwd(), path), "utf8");

      expect(source, path).toContain(
        'vi.mock("@/lib/api/agentRuntime/clientFactory"',
      );
      expect(source, path).not.toContain('vi.mock("@/lib/api/agentRuntime"');
    }
  });
});
