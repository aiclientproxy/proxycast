import type {
  SkillCatalogCommandEntry,
  SkillCatalogSkillLocator,
} from "@/lib/api/skillCatalog";
import type { ServiceSkillItem } from "@/lib/api/serviceSkills";
import type {
  BaseSetupBindingProfile,
  BaseSetupCatalogProjection,
  BaseSetupPackage,
  BaseSetupPolicyProfile,
} from "../types";

function resolveCommandExecutionKind(
  bindingProfile: BaseSetupBindingProfile,
  projection: BaseSetupCatalogProjection,
): NonNullable<SkillCatalogCommandEntry["binding"]>["executionKind"] {
  const explicitExecutionKind = projection.commandBinding?.executionKind;
  if (explicitExecutionKind) {
    return explicitExecutionKind;
  }

  switch (bindingProfile.bindingFamily) {
    case "native_skill":
      return "native_skill";
    case "automation_job":
      return "automation_job";
    default:
      return "agent_turn";
  }
}

function resolveCommandRenderContract(
  executionKind: NonNullable<
    SkillCatalogCommandEntry["binding"]
  >["executionKind"],
  projection: BaseSetupCatalogProjection,
): SkillCatalogCommandEntry["renderContract"] {
  if (projection.commandRenderContract) {
    return {
      ...projection.commandRenderContract,
    };
  }

  if (executionKind === "native_skill") {
    return {
      resultKind: "artifact",
      detailKind: "artifact_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    };
  }

  if (executionKind === "automation_job") {
    return {
      resultKind: "tool_timeline",
      detailKind: "task_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    };
  }

  return {
    resultKind: "tool_timeline",
    detailKind: "json",
    supportsStreaming: true,
    supportsTimeline: true,
  };
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
  const explicitSkillId = projection.commandBinding?.skillId?.trim();
  if (explicitSkillId) {
    return explicitSkillId;
  }

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

function compileCommandTriggers(
  projection: BaseSetupCatalogProjection,
): SkillCatalogCommandEntry["triggers"] {
  const seenKeys = new Set<string>();
  const triggers =
    projection.triggerHints
      ?.map((hint) => hint.trim())
      .map((hint) => {
        if (hint.startsWith("@")) {
          return {
            mode: "mention" as const,
            prefix: hint,
          };
        }

        if (hint.startsWith("/")) {
          return {
            mode: "slash" as const,
            prefix: hint,
          };
        }

        return null;
      })
      .filter(
        (
          trigger,
        ): trigger is {
          mode: "mention" | "slash";
          prefix: string;
        } => Boolean(trigger),
      )
      .filter((trigger) => {
        const dedupeKey = `${trigger.mode}:${trigger.prefix.toLowerCase()}`;
        if (seenKeys.has(dedupeKey)) {
          return false;
        }
        seenKeys.add(dedupeKey);
        return true;
      }) ?? [];

  if (triggers.length === 0) {
    throw new Error(
      `无法编译 command projection ${projection.id}：缺少可识别的 triggerHints`,
    );
  }

  return triggers;
}

function compileCommandProjectionEntry(params: {
  projection: BaseSetupCatalogProjection;
  bindingProfile: BaseSetupBindingProfile;
  policyProfile: BaseSetupPolicyProfile;
  linkedSkillId?: string;
}): SkillCatalogCommandEntry {
  const { projection, bindingProfile, policyProfile, linkedSkillId } = params;
  const commandKey = projection.skillKey?.trim() || projection.entryKey.trim();
  const executionKind = resolveCommandExecutionKind(bindingProfile, projection);
  const requestDefaults = projection.commandBinding?.requestDefaults
    ? { ...projection.commandBinding.requestDefaults }
    : undefined;
  const intentConfirmation = projection.commandBinding?.intentConfirmation
    ? { ...projection.commandBinding.intentConfirmation }
    : undefined;

  return {
    id: `command:${commandKey}`,
    kind: "command",
    title: projection.title,
    summary: projection.summary,
    commandKey,
    aliases: projection.aliases ? [...projection.aliases] : undefined,
    surfaceScopes: policyProfile.surfaceScopes
      ? [...policyProfile.surfaceScopes]
      : undefined,
    triggers: compileCommandTriggers(projection),
    binding: {
      skillId: linkedSkillId,
      skillLocator: createCatalogSkillLocator({
        id: linkedSkillId,
        skillKey: projection.skillKey,
      }),
      executionKind,
      ...(requestDefaults ? { requestDefaults } : {}),
      ...(intentConfirmation ? { intentConfirmation } : {}),
    },
    renderContract: resolveCommandRenderContract(executionKind, projection),
  };
}

export function compileCommandCatalogProjection(
  pkg: BaseSetupPackage,
  serviceSkillItems: ServiceSkillItem[],
): SkillCatalogCommandEntry[] {
  const bindingProfiles = new Map(
    pkg.bindingProfiles.map((profile) => [profile.id, profile] as const),
  );
  const policyProfiles = new Map(
    pkg.policyProfiles.map((profile) => [profile.id, profile] as const),
  );
  const linkedSkillIndex = buildLinkedSkillIndex(serviceSkillItems);

  return pkg.catalogProjections
    .filter((projection) => projection.targetCatalog === "command_catalog")
    .map((projection) => {
      const bindingProfile = bindingProfiles.get(projection.bindingProfileRef);
      const policyProfile = policyProfiles.get(projection.policyProfileRef);

      if (!bindingProfile || !policyProfile) {
        throw new Error(
          `无法编译 command projection ${projection.id}：存在未解析的引用`,
        );
      }

      return compileCommandProjectionEntry({
        projection,
        bindingProfile,
        policyProfile,
        linkedSkillId: resolveLinkedSkillId(projection, linkedSkillIndex),
      });
    });
}
