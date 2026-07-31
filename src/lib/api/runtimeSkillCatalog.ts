import type { ExecutableSkillInfo } from "./skill-execution";
import type { Skill, SkillCatalogSource } from "./skills";

function directoryName(directory: string): string {
  const normalized = directory.replace(/[\\/]+$/, "");
  return normalized.split(/[\\/]/).pop() || directory;
}

function catalogSource(
  source: ExecutableSkillInfo["source"],
): SkillCatalogSource | undefined {
  if (source === "project" || source === "user") {
    return source;
  }
  return undefined;
}

export function projectRuntimeSkillCatalog(
  skills: readonly ExecutableSkillInfo[],
): Skill[] {
  return skills.map((skill) => ({
    key: skill.skill_id,
    name: skill.name,
    description: skill.description,
    directory: directoryName(skill.locator.directory),
    localDirectoryPath: skill.locator.directory,
    installed: true,
    sourceKind: skill.source === "app" ? "builtin" : "other",
    catalogSource: catalogSource(skill.source),
  }));
}
