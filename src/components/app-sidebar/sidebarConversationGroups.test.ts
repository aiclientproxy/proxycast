import { describe, expect, it } from "vitest";
import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";
import {
  buildSidebarConversationGroups,
  flattenSidebarConversationGroups,
  type SidebarOpenedProjectSummary,
} from "./sidebarConversationGroups";

function session(
  id: string,
  overrides: Partial<AgentSessionInfo> = {},
): AgentSessionInfo {
  return {
    id,
    name: id,
    created_at: 1,
    updated_at: 1,
    archived_at: null,
    ...overrides,
  };
}

const openedProjects: SidebarOpenedProjectSummary[] = [
  {
    id: "project-a",
    name: "content-factory-app",
    rootPath: "/repo/content-factory-app",
  },
  {
    id: "project-b",
    name: "lime",
    rootPath: "/repo/lime",
  },
];

describe("sidebarConversationGroups", () => {
  it("只为已打开项目建立项目分组，并把未匹配项目的本地目录会话放回对话区", () => {
    const groups = buildSidebarConversationGroups({
      openedProjects,
      sessions: [
        session("a-1", { working_dir: "/repo/content-factory-app/" }),
        session("b-1", { working_dir: "/repo/lime" }),
        session("hidden-1", { working_dir: "/repo/hidden" }),
        session("standalone-1"),
      ],
    });

    expect(groups.projectSections).toHaveLength(2);
    expect(groups.threadSections).toEqual([]);
    expect(groups.projectSections[0]).toMatchObject({
      project: { id: "project-a", name: "content-factory-app" },
      sessions: [{ id: "a-1" }],
    });
    expect(groups.projectSections[1]).toMatchObject({
      project: { id: "project-b", name: "lime" },
      sessions: [{ id: "b-1" }],
    });
    expect(groups.standaloneSessions.map((item) => item.id)).toEqual([
      "hidden-1",
      "standalone-1",
    ]);
    expect(
      flattenSidebarConversationGroups(groups).map((item) => item.id),
    ).toEqual(["a-1", "b-1", "hidden-1", "standalone-1"]);
  });

  it("优先支持 workspace_id 匹配 current 项目，兼容 working_dir rootPath", () => {
    const groups = buildSidebarConversationGroups({
      openedProjects,
      sessions: [
        session("workspace-project", { workspace_id: "project-b" }),
        session("cwd-project", {
          workspace_id: "project-hidden",
          working_dir: "/repo/lime/",
        }),
        session("cwd-standalone", { working_dir: "/repo/other" }),
      ],
    });

    expect(groups.projectSections[1].sessions.map((item) => item.id)).toEqual([
      "workspace-project",
      "cwd-project",
    ]);
    expect(groups.standaloneSessions.map((item) => item.id)).toEqual([
      "cwd-standalone",
    ]);
  });

  it("左侧导航只构建未归档会话分组", () => {
    const groups = buildSidebarConversationGroups({
      openedProjects,
      sessions: [
        session("active", { working_dir: "/repo/content-factory-app" }),
        session("archived", {
          working_dir: "/repo/content-factory-app",
          archived_at: 1,
        }),
      ],
    });

    expect(groups.projectSections[0].sessions.map((item) => item.id)).toEqual([
      "active",
    ]);
    expect(groups.standaloneSessions).toEqual([]);
  });

  it("sectioned 会话应先按服务端顺序分组，且不再重复进入项目或对话分组", () => {
    const groups = buildSidebarConversationGroups({
      openedProjects,
      sessions: [
        session("pinned-1", {
          workspace_id: "project-a",
          section: {
            id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
            name: "Pinned",
          },
        }),
        session("pinned-2", {
          section: {
            id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
            name: "Pinned",
          },
        }),
        session("active-1", {
          working_dir: "/repo/lime",
          section: { id: "section-active", name: "Active" },
        }),
        session("project-unsectioned", { workspace_id: "project-b" }),
        session("standalone-unsectioned"),
      ],
    });

    expect(
      groups.threadSections.map(({ section, sessions }) => ({
        id: section.id,
        sessions: sessions.map((item) => item.id),
      })),
    ).toEqual([
      {
        id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
        sessions: ["pinned-1", "pinned-2"],
      },
      { id: "section-active", sessions: ["active-1"] },
    ]);
    expect(groups.projectSections[0].sessions).toEqual([]);
    expect(groups.projectSections[1].sessions.map((item) => item.id)).toEqual([
      "project-unsectioned",
    ]);
    expect(groups.standaloneSessions.map((item) => item.id)).toEqual([
      "standalone-unsectioned",
    ]);
    expect(
      flattenSidebarConversationGroups(groups).map((item) => item.id),
    ).toEqual([
      "pinned-1",
      "pinned-2",
      "active-1",
      "project-unsectioned",
      "standalone-unsectioned",
    ]);
  });

  it("应以 section catalog 保留空分组和名称顺序，并把未知 section 会话放回未分组", () => {
    const groups = buildSidebarConversationGroups({
      openedProjects,
      threadSections: [
        { id: "section-empty", name: "待处理" },
        { id: "section-active", name: "进行中" },
      ],
      sessions: [
        session("active", {
          section: { id: "section-active", name: "旧名称" },
        }),
        session("deleted-section", {
          section: { id: "section-deleted", name: "已删除" },
        }),
      ],
    });

    expect(groups.threadSections).toEqual([
      {
        section: { id: "section-empty", name: "待处理" },
        sessions: [],
      },
      {
        section: { id: "section-active", name: "进行中" },
        sessions: [expect.objectContaining({ id: "active" })],
      },
    ]);
    expect(groups.standaloneSessions.map((item) => item.id)).toEqual([
      "deleted-section",
    ]);
  });
});
