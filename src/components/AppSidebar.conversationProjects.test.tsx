import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  cleanupAppSidebarTest,
  flushEffects,
  mockGetProject,
  mockListAgentRuntimeSessions,
  mountSidebarContainer,
  resetAppSidebarTest,
} from "./AppSidebar.testFixtures";
import type { AgentPageParams } from "./AppSidebar.testFixtures";

describe("AppSidebar conversation projects", () => {
  async function waitForProjectGroups(container: HTMLElement) {
    let projectGroups: HTMLElement[] = [];
    for (let attempt = 0; attempt < 10; attempt += 1) {
      await flushEffects(1);
      projectGroups = [
        ...container.querySelectorAll<HTMLElement>(
          '[data-testid="app-sidebar-project-conversation-group"]',
        ),
      ];
      if (projectGroups.length >= 2) {
        break;
      }
    }
    return projectGroups;
  }

  beforeEach(async () => {
    await resetAppSidebarTest();
    localStorage.setItem(
      "agent_opened_project_ids",
      JSON.stringify(["project-1", "project-2"]),
    );
    mockGetProject.mockImplementation(async (projectId: string) => ({
      id: projectId,
      name: projectId === "project-1" ? "项目一" : "项目二",
      rootPath: `/repo/${projectId}`,
      isFavorite: false,
    }));
  });

  afterEach(cleanupAppSidebarTest);

  it("进入项目对话后仍展示全部已打开项目，并查询全局与当前项目会话", async () => {
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/project-1"
          ? [
              {
                id: "session-project-1",
                name: "项目一会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ]
          : [],
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-project-1",
      } as AgentPageParams,
    });
    const projectGroups = await waitForProjectGroups(container);

    expect(projectGroups.map((group) => group.textContent)).toHaveLength(2);
    expect(projectGroups[0]?.textContent).toContain("项目一");
    expect(projectGroups[0]?.textContent).toContain("项目一会话");
    expect(projectGroups[1]?.textContent).toContain("项目二");
    expect(projectGroups[1]?.textContent).not.toContain("项目一会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-2",
    });
  });
});
