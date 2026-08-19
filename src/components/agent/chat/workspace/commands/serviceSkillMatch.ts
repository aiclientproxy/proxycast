/**
 * 服务技能匹配函数（从 useWorkspaceSendActions.ts 提取）
 *
 * 纯函数，无 React 依赖。用于从技能列表中匹配特定类型的技能。
 *
 * @module serviceSkillMatch
 */

import type { ServiceSkillHomeItem } from "../../service-skills/types";

export function matchesVoiceCommandSkill(skill: ServiceSkillHomeItem): boolean {
  const searchable = [
    skill.id,
    skill.skillKey,
    skill.title,
    skill.summary,
    ...(skill.aliases ?? []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return /配音|dubbing|voice/.test(searchable);
}

export function matchesGrowthCommandSkill(
  skill: ServiceSkillHomeItem,
): boolean {
  const searchable = [
    skill.id,
    skill.skillKey,
    skill.title,
    skill.summary,
    ...(skill.aliases ?? []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return /增长|growth|account-performance|涨粉/.test(searchable);
}

export function resolveGrowthCommandServiceSkill(
  serviceSkills: ServiceSkillHomeItem[],
): ServiceSkillHomeItem | null {
  return (
    serviceSkills.find(
      (skill) =>
        (skill.id === "account-performance-tracking" ||
          skill.skillKey === "account-performance-tracking"),
    ) ||
    serviceSkills.find(
      (skill) =>
        matchesGrowthCommandSkill(skill) &&
        skill.slotSchema.some((slot) => slot.key === "platform") &&
        skill.slotSchema.some((slot) => slot.key === "account_list"),
    ) ||
    serviceSkills.find((skill) => matchesGrowthCommandSkill(skill)) ||
    null
  );
}

export function resolveVoiceCommandServiceSkill(
  serviceSkills: ServiceSkillHomeItem[],
): ServiceSkillHomeItem | null {
  return (
    serviceSkills.find(
      (skill) =>
        matchesVoiceCommandSkill(skill) &&
        skill.slotSchema.some((slot) =>
          ["reference_video", "target_language", "voice_style"].includes(
            slot.key,
          ),
        ),
    ) ||
    serviceSkills.find((skill) => matchesVoiceCommandSkill(skill)) ||
    null
  );
}

export function normalizeLocalServiceSkillExecutionKind(
  value?: string | null,
): "agent_turn" | "native_skill" | "automation_job" {
  if (value === "native_skill") {
    return "native_skill";
  }

  if (value === "automation_job") {
    return "automation_job";
  }

  return "agent_turn";
}
