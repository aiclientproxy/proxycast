import type {
  SkillCatalogSceneEntry,
  SkillCatalogSkillLocator,
} from "@/lib/api/skillCatalog";
import type { ServiceSkillItem } from "@/lib/api/serviceSkills";
import type {
  BaseSetupBindingProfile,
  BaseSetupCatalogProjection,
  BaseSetupPackage,
  BaseSetupPolicyProfile,
} from "../types";

function resolveSceneExecutionKind(
  bindingProfile: BaseSetupBindingProfile,
): SkillCatalogSceneEntry["executionKind"] {
  switch (bindingProfile.bindingFamily) {
    case "native_skill":
      return "native_skill";
    case "automation_job":
      return "automation_job";
    case "cloud_scene":
      return "agent_turn";
    default:
      return "agent_turn";
  }
}

function buildLinkedSkillIndex(items: ServiceSkillItem[]): {
  byId: Map<string, string>;
  bySkillKey: Map<string, string>;
} {
  return {
    byId: new Map(items.map((item) => [item.id, item.id] as const)),
    bySkillKey: new Map(
      items
        .filter((item) => item.skillKey)
        .map((item) => [item.skillKey as string, item.id] as const),
    ),
  };
}

function resolveLinkedSkillId(
  projection: BaseSetupCatalogProjection,
  linkedSkillIndex: ReturnType<typeof buildLinkedSkillIndex>,
): string | undefined {
  return (
    linkedSkillIndex.byId.get(projection.entryKey) ??
    (projection.skillKey
      ? linkedSkillIndex.bySkillKey.get(projection.skillKey)
      : undefined) ??
    linkedSkillIndex.byId.get(projection.id)
  );
}

function createCatalogSkillLocator(input: {
  id?: string | null;
  skillKey?: string | null;
}): SkillCatalogSkillLocator | undefined {
  const name = input.skillKey?.trim() || input.id?.trim();
  return name ? { source: "catalog", name } : undefined;
}

function compileSceneProjectionEntry(params: {
  projection: BaseSetupCatalogProjection;
  bindingProfile: BaseSetupBindingProfile;
  policyProfile: BaseSetupPolicyProfile;
  linkedSkillId?: string;
}): SkillCatalogSceneEntry {
  const { projection, bindingProfile, policyProfile, linkedSkillId } = params;
  const sceneKey =
    projection.sceneBinding?.sceneKey?.trim() ||
    projection.skillKey?.trim() ||
    projection.entryKey.trim();
  const commandPrefix =
    projection.sceneBinding?.commandPrefix?.trim() || `/${sceneKey}`;

  return {
    id: `scene:${sceneKey}`,
    kind: "scene",
    title: projection.sceneBinding?.title?.trim() || projection.title,
    summary: projection.sceneBinding?.summary?.trim() || projection.summary,
    sceneKey,
    commandPrefix,
    aliases:
      projection.sceneBinding?.aliases &&
      projection.sceneBinding.aliases.length > 0
        ? [...projection.sceneBinding.aliases]
        : projection.aliases
          ? [...projection.aliases]
          : undefined,
    surfaceScopes: policyProfile.surfaceScopes
      ? [...policyProfile.surfaceScopes]
      : undefined,
    linkedSkillId,
    skillLocator: createCatalogSkillLocator({
      id: linkedSkillId,
      skillKey: projection.skillKey,
    }),
    executionKind: resolveSceneExecutionKind(bindingProfile),
    renderContract: {
      resultKind: "tool_timeline",
      detailKind: "scene_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    },
  };
}

export function compileSceneCatalogProjection(
  pkg: BaseSetupPackage,
  serviceSkillItems: ServiceSkillItem[],
): SkillCatalogSceneEntry[] {
  const bindingProfiles = new Map(
    pkg.bindingProfiles.map((profile) => [profile.id, profile] as const),
  );
  const policyProfiles = new Map(
    pkg.policyProfiles.map((profile) => [profile.id, profile] as const),
  );
  const linkedSkillIndex = buildLinkedSkillIndex(serviceSkillItems);

  return pkg.catalogProjections
    .filter((projection) => projection.targetCatalog === "scene_catalog")
    .map((projection) => {
      const bindingProfile = bindingProfiles.get(projection.bindingProfileRef);
      const policyProfile = policyProfiles.get(projection.policyProfileRef);

      if (!bindingProfile || !policyProfile) {
        throw new Error(
          `无法编译 scene projection ${projection.id}：存在未解析的引用`,
        );
      }

      return compileSceneProjectionEntry({
        projection,
        bindingProfile,
        policyProfile,
        linkedSkillId: resolveLinkedSkillId(projection, linkedSkillIndex),
      });
    });
}
