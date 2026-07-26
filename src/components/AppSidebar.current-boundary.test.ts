import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), "utf8");
}

function sourceBetween(
  source: string,
  startNeedle: string,
  endNeedle: string,
): string {
  const start = source.indexOf(startNeedle);
  expect(start, `missing ${startNeedle}`).toBeGreaterThanOrEqual(0);
  const end = source.indexOf(endNeedle, start + startNeedle.length);
  expect(end, `missing ${endNeedle}`).toBeGreaterThan(start);
  return source.slice(start, end);
}

const SIDEBAR_SESSION_DOMAIN_SOURCES = [
  "src/components/AppSidebar.tsx",
  "src/components/AppSidebar.conversations.test.tsx",
  "src/components/AppSidebar.testFixtures.tsx",
  "src/components/app-sidebar/AppSidebarConversationMenus.tsx",
  "src/components/app-sidebar/AppSidebarConversationRow.test.tsx",
  "src/components/app-sidebar/AppSidebarConversationRow.tsx",
  "src/components/app-sidebar/AppSidebarConversationShelf.tsx",
  "src/components/app-sidebar/AppSidebarProjectConversationGroups.tsx",
  "src/components/app-sidebar/AppSidebarSearchDialog.tsx",
  "src/components/app-sidebar/sidebarConversationGroups.test.ts",
  "src/components/app-sidebar/sidebarConversationGroups.ts",
  "src/components/app-sidebar/sidebarSessionFormatting.test.ts",
  "src/components/app-sidebar/sidebarSessionFormatting.ts",
  "src/components/app-sidebar/sidebarSessions.test.ts",
  "src/components/app-sidebar/sidebarSessions.ts",
  "src/components/app-sidebar/useAppSidebarConversationActions.ts",
  "src/components/app-sidebar/useAppSidebarSessions.ts",
  "src/components/settings-v2/general/archived-conversations/index.test.tsx",
  "src/components/settings-v2/general/archived-conversations/index.tsx",
] as const;

describe("AppSidebar current App Server session boundary", () => {
  it("侧栏会话域不得回绕 agentRuntime compat 根 barrel", () => {
    for (const relativePath of SIDEBAR_SESSION_DOMAIN_SOURCES) {
      const source = readSource(relativePath);

      expect(source, relativePath).not.toContain(
        'from "@/lib/api/agentRuntime"',
      );
      expect(source, relativePath).not.toContain(
        'vi.mock("@/lib/api/agentRuntime"',
      );
    }
  });

  it("侧栏归档 / 恢复入口只能经 v2 thread lifecycle gateway", () => {
    const source = readSource(
      "src/components/app-sidebar/useAppSidebarConversationActions.ts",
    );
    const archiveHandler = sourceBetween(
      source,
      "const toggleSessionArchive = useCallback(",
      "const deleteConversation = useCallback(",
    );

    expect(archiveHandler).toContain(
      "await archiveAgentRuntimeSession(session.id)",
    );
    expect(archiveHandler).toContain(
      "await unarchiveAgentRuntimeSession(session.id)",
    );
    expect(archiveHandler).toContain(
      "moveSidebarSessionArchiveStateOptimistically(nextSession)",
    );
    expect(archiveHandler).toContain("refreshSidebarSessions");
    expect(archiveHandler).not.toContain("deleteAgentRuntimeSession");
    expect(archiveHandler).not.toContain("safeInvoke");
    expect(archiveHandler).not.toContain("invokeCommand");
    expect(archiveHandler).not.toContain("agent_runtime_");
  });

  it("agentRuntime session gateway 不应把 delete 回流到 legacy command", () => {
    const source = readSource("src/lib/api/agentRuntime/sessionClient.ts");
    const deleteFunction = sourceBetween(
      source,
      "async function deleteAgentRuntimeSession(",
      "return {",
    );

    expect(deleteFunction).toContain(
      "appServerSessionClient.deleteAgentRuntimeSession(sessionId)",
    );
    expect(deleteFunction).toContain('reason: "deleted"');
    expect(source).not.toContain('"agent_runtime_update_session"');
    expect(source).not.toContain("updateAgentRuntimeSession");
    expect(source).not.toContain('"agent_runtime_delete_session"');
    expect(source).not.toContain("invokeCommand(");
  });

  it("App Server session gateway 必须通过 thread lifecycle 与 metadata 事实源", () => {
    const source = readSource(
      "src/lib/api/agentRuntime/appServerSessionClient.ts",
    );
    const listFunction = sourceBetween(
      source,
      "async function listAgentRuntimeSessions(",
      "async function getAgentRuntimeSession(",
    );
    const getFunction = sourceBetween(
      source,
      "async function getAgentRuntimeSession(",
      "async function archiveAgentRuntimeSession(",
    );
    const lifecycleFunctions = sourceBetween(
      source,
      "async function archiveAgentRuntimeSession(",
      "return {",
    );

    expect(listFunction).toContain("listCanonicalSessionOverviews(");
    expect(source).toContain("METHOD_THREAD_LIST");
    expect(getFunction).toContain("appServerClient.readThread(");
    expect(source).not.toContain("appServerClient.updateSession(");
    expect(lifecycleFunctions).toContain("appServerClient.archiveThread(");
    expect(lifecycleFunctions).toContain("appServerClient.unarchiveThread(");
    expect(lifecycleFunctions).toContain("resolveCanonicalThreadId(");
    expect(lifecycleFunctions).toContain('"thread/delete"');
    expect(lifecycleFunctions).toContain("appServerClient.deleteThread(");
    expect(source).not.toContain("appServerClient.deleteSession(");
    expect(source).not.toContain('"agentSession/delete"');
    expect(source).not.toContain("archiveManySessions");
    expect(source).not.toContain('"agent_runtime_list_sessions"');
    expect(source).not.toContain('"agent_runtime_get_session"');
    expect(source).not.toContain('"agent_runtime_update_session"');
    expect(source).not.toContain('"agent_runtime_delete_session"');
    expect(source).not.toContain("safeInvoke");
    expect(source).not.toContain("invokeCommand");
  });
});
