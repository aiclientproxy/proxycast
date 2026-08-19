import {
  getSeededServiceSkillCatalog,
  type ServiceSkillExecutorBinding,
  type ServiceSkillItem,
} from "@/lib/api/serviceSkills";
import type { SkillCatalogExecutionKind } from "@/lib/api/skillCatalog";
import { supportsServiceSkillEntrySurface } from "./entryAdapter";
import { buildServiceSkillRecommendationBuckets } from "./recommendedServiceSkills";
import {
  getServiceSkillActionLabel,
  getServiceSkillRunnerDescription,
  getServiceSkillRunnerLabel,
  getServiceSkillRunnerTone,
} from "./skillPresentation";
import type { ServiceSkillHomeItem } from "./types";

interface ServiceSkillHomeVisibilityCandidate {
  defaultExecutorBinding: ServiceSkillExecutorBinding;
  execution?: {
    kind?: SkillCatalogExecutionKind;
  } | null;
}

interface ListFeaturedHomeServiceSkillsOptions {
  fallbackToSeeded?: boolean;
  limit?: number;
}

const DEFAULT_FEATURED_HOME_SERVICE_SKILL_LIMIT = 2;

function normalizeHomeServiceSkillBinding(
  binding: ServiceSkillExecutorBinding,
): ServiceSkillExecutorBinding {
  return binding;
}

function resolveServiceSkillExecutionKind(
  skill: Pick<ServiceSkillItem, "defaultExecutorBinding" | "executionLocation">,
): SkillCatalogExecutionKind {
  switch (normalizeHomeServiceSkillBinding(skill.defaultExecutorBinding)) {
    case "automation_job":
      return "automation_job";
    case "native_skill":
      return "native_skill";
    case "agent_turn":
      return "agent_turn";
    default:
      return "agent_turn";
  }
}

function buildSeededHomeServiceSkillItem(
  skill: ServiceSkillItem,
): ServiceSkillHomeItem {
  return {
    ...skill,
    executionKind: resolveServiceSkillExecutionKind(skill),
    badge: skill.source === "cloud_catalog" ? "云目录" : "本地技能",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: getServiceSkillRunnerLabel(skill),
    runnerTone: getServiceSkillRunnerTone(skill),
    runnerDescription: getServiceSkillRunnerDescription(skill),
    actionLabel: getServiceSkillActionLabel(skill),
    automationStatus: null,
  };
}

function listFeaturedFromCandidates(
  skills: ServiceSkillHomeItem[],
  limit: number,
): ServiceSkillHomeItem[] {
  if (limit <= 0) {
    return [];
  }

  const visibleSkills = skills.filter((skill) =>
    shouldExposeServiceSkillHomeItem(skill),
  );
  const { recentSkills, featuredSkills } =
    buildServiceSkillRecommendationBuckets(visibleSkills, {
      featuredLimit: limit,
      surface: "home",
    });

  return [...recentSkills, ...featuredSkills].slice(0, limit);
}

export function shouldExposeServiceSkillHomeItem(
  item: ServiceSkillHomeVisibilityCandidate,
): boolean {
  void item;
  return true;
}

export function listFeaturedHomeServiceSkills(
  skills: ServiceSkillHomeItem[],
  options: ListFeaturedHomeServiceSkillsOptions = {},
): ServiceSkillHomeItem[] {
  const limit = Math.max(
    options.limit ?? DEFAULT_FEATURED_HOME_SERVICE_SKILL_LIMIT,
    0,
  );
  if (limit === 0) {
    return [];
  }

  const featuredRuntimeSkills = listFeaturedFromCandidates(skills, limit);
  if (
    featuredRuntimeSkills.length >= limit ||
    options.fallbackToSeeded === false
  ) {
    return featuredRuntimeSkills;
  }

  const featuredRuntimeSkillIds = new Set(
    featuredRuntimeSkills.map((skill) => skill.id),
  );
  const seededCandidates = getSeededServiceSkillCatalog()
    .items.filter((skill) => supportsServiceSkillEntrySurface(skill, "home"))
    .filter((skill) => shouldExposeServiceSkillHomeItem(skill))
    .filter((skill) => !featuredRuntimeSkillIds.has(skill.id))
    .map((skill) => buildSeededHomeServiceSkillItem(skill));
  const remainingSlots = Math.max(limit - featuredRuntimeSkills.length, 0);

  return [
    ...featuredRuntimeSkills,
    ...listFeaturedFromCandidates(seededCandidates, remainingSlots),
  ];
}
