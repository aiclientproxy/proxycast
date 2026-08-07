import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import { resolveExecutableSkillId, skillExecutionApi } from "./skill-execution";

const appServerRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/appServer", () => ({
  AppServerClient: vi.fn(() => ({
    request: appServerRequestMock,
  })),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

function skillDetailMetadata(overrides: Record<string, unknown> = {}) {
  return {
    skillId: "project:writer",
    name: "writer",
    description: "生成文案",
    scope: "project",
    source: "project",
    authority: "workspace",
    enabled: true,
    interface: {
      displayName: "写作助手",
      executionMode: "prompt",
    },
    dependencies: {
      tools: [{ type: "runtime_tool", value: "Read", required: true }],
    },
    policy: {
      allowImplicitInvocation: true,
      whenToUse: "需要生成文案时",
    },
    capabilities: ["Read"],
    locator: {
      directory: "/tmp/skills/writer",
      skillFilePath: "/tmp/skills/writer/SKILL.md",
    },
    ...overrides,
  };
}

function listedSkillMetadata(overrides: Record<string, unknown> = {}) {
  return {
    name: "writer",
    description: "生成文案",
    path: "/tmp/skills/writer/SKILL.md",
    scope: "repo",
    enabled: true,
    interface: {
      displayName: "写作助手",
      shortDescription: null,
      iconSmall: null,
      iconLarge: null,
      iconSmallUrl: null,
      iconLargeUrl: null,
      brandColor: null,
      defaultPrompt: null,
    },
    dependencies: {
      tools: [{ type: "runtime_tool", value: "Read" }],
    },
    ...overrides,
  };
}

describe("skillExecutionApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    appServerRequestMock.mockReset();
  });

  it("可执行 Skill 列表应通过 App Server skills/list 读取并过滤禁用项", async () => {
    appServerRequestMock.mockResolvedValueOnce({
      result: {
        data: [
          {
            cwd: "/tmp/project",
            skills: [
              listedSkillMetadata(),
              listedSkillMetadata({ name: "disabled", enabled: false }),
            ],
            errors: [],
          },
        ],
      },
    });

    await expect(skillExecutionApi.listExecutableSkills()).resolves.toEqual([
      expect.objectContaining({
        name: "writer",
        skill_id: "project:writer",
        display_name: "写作助手",
        authority: "workspace",
        dependencies: [{ type: "runtime_tool", value: "Read", required: true }],
      }),
    ]);

    expect(appServerRequestMock).toHaveBeenCalledWith("skills/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Skill 详情应通过 App Server skill/read 读取", async () => {
    appServerRequestMock.mockResolvedValueOnce({
      result: {
        skill: {
          metadata: skillDetailMetadata(),
          markdownContent: "# Writer",
          workflowSteps: [],
        },
      },
    });

    await expect(
      skillExecutionApi.getSkillDetail("project:writer"),
    ).resolves.toEqual(
      expect.objectContaining({
        name: "writer",
        markdown_content: "# Writer",
        allowed_tools: ["Read"],
      }),
    );

    expect(appServerRequestMock).toHaveBeenCalledWith("skill/read", {
      skillId: "project:writer",
    });
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("App Server Skill 读链缺少必需 result 时不应回退 legacy", async () => {
    appServerRequestMock.mockResolvedValueOnce({ result: {} });

    await expect(skillExecutionApi.listExecutableSkills()).rejects.toThrow(
      "App Server skills/list did not return data",
    );

    appServerRequestMock.mockReset();
    appServerRequestMock.mockResolvedValueOnce({ result: {} });

    await expect(
      skillExecutionApi.getSkillDetail("project:writer"),
    ).rejects.toThrow("App Server skill/read did not return skill");

    expect(safeInvoke).not.toHaveBeenCalledWith("list_executable_skills");
    expect(safeInvoke).not.toHaveBeenCalledWith("get_skill_detail", {
      skillId: "project:writer",
    });
  });

  it("Skill 引用应优先匹配 stable id，并只允许唯一 name 解析", () => {
    const skills = [
      { skill_id: "project:writer", name: "writer" },
      { skill_id: "user:writer", name: "writer" },
      { skill_id: "app:reviewer", name: "reviewer" },
    ];

    expect(resolveExecutableSkillId(skills, "user:writer")).toBe("user:writer");
    expect(resolveExecutableSkillId(skills, "reviewer")).toBe("app:reviewer");
    expect(resolveExecutableSkillId(skills, "writer")).toBeNull();
    expect(resolveExecutableSkillId(skills, "missing")).toBeNull();
  });

  it("Skill 详情响应 identity 与请求不一致时应 fail closed", async () => {
    appServerRequestMock.mockResolvedValueOnce({
      result: {
        skill: {
          metadata: skillDetailMetadata({ skillId: "user:writer" }),
          markdownContent: "# Writer",
          workflowSteps: [],
        },
      },
    });

    await expect(
      skillExecutionApi.getSkillDetail("project:writer"),
    ).rejects.toThrow(
      "App Server skill/read returned unexpected skillId: user:writer",
    );
  });

  it("App Server Skills list metadata 缺少 name 时应 fail closed", async () => {
    appServerRequestMock.mockResolvedValueOnce({
      result: {
        data: [
          {
            cwd: "/tmp/project",
            skills: [listedSkillMetadata({ name: "" })],
            errors: [],
          },
        ],
      },
    });

    await expect(skillExecutionApi.listExecutableSkills()).rejects.toThrow(
      "skills/list data[0].skills[0].name is not a non-empty string",
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Skill 独立执行 API 不再暴露 executeSkill", () => {
    expect("executeSkill" in skillExecutionApi).toBe(false);
    expect(safeInvoke).not.toHaveBeenCalled();
    expect(appServerRequestMock).not.toHaveBeenCalled();
  });
});
