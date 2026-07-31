import { describe, expect, it } from "vitest";
import type { ExecutableSkillInfo } from "./skill-execution";
import { projectRuntimeSkillCatalog } from "./runtimeSkillCatalog";

function runtimeSkill(
  overrides: Partial<ExecutableSkillInfo> = {},
): ExecutableSkillInfo {
  return {
    skill_id: "user:article-writer",
    name: "article-writer",
    display_name: "Article Writer",
    description: "Write structured articles.",
    execution_mode: "prompt",
    has_workflow: false,
    source: "user",
    authority: "user",
    scope: "user",
    enabled: true,
    capabilities: [],
    dependencies: [],
    locator: {
      directory: "/tmp/.agents/skills/article-writer",
      skill_file_path: "/tmp/.agents/skills/article-writer/SKILL.md",
    },
    allow_implicit_invocation: true,
    ...overrides,
  };
}

describe("projectRuntimeSkillCatalog", () => {
  it("应把 current skill/list 投影为 Composer 本地 Skill", () => {
    expect(projectRuntimeSkillCatalog([runtimeSkill()])).toEqual([
      {
        key: "user:article-writer",
        name: "article-writer",
        description: "Write structured articles.",
        directory: "article-writer",
        localDirectoryPath: "/tmp/.agents/skills/article-writer",
        installed: true,
        sourceKind: "other",
        catalogSource: "user",
      },
    ]);
  });

  it("应跨平台提取目录名，并仅映射可表达的 catalog source", () => {
    const [appSkill, projectSkill, otherSkill] = projectRuntimeSkillCatalog([
      runtimeSkill({
        skill_id: "app:summary",
        source: "app",
        scope: "app",
        authority: "application",
        locator: {
          directory: "C:\\Lime\\skills\\summary\\",
          skill_file_path: "C:\\Lime\\skills\\summary\\SKILL.md",
        },
      }),
      runtimeSkill({
        skill_id: "project:review",
        source: "project",
        scope: "project",
        authority: "workspace",
      }),
      runtimeSkill({
        skill_id: "other:external",
        source: "other",
        scope: "other",
        authority: "external",
      }),
    ]);

    expect(appSkill).toMatchObject({
      directory: "summary",
      sourceKind: "builtin",
      catalogSource: undefined,
    });
    expect(projectSkill.catalogSource).toBe("project");
    expect(otherSkill.catalogSource).toBeUndefined();
  });
});
