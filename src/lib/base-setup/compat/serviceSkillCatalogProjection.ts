import type {
  ServiceSkillBundleSummary,
  ServiceSkillCatalog,
  ServiceSkillExecutionLocation,
  ServiceSkillItem,
  ServiceSkillSource,
  ServiceSkillType,
} from "@/lib/api/serviceSkills";
import type {
  BaseSetupArtifactProfile,
  BaseSetupBindingProfile,
  BaseSetupBundleRef,
  BaseSetupCatalogProjection,
  BaseSetupPackage,
  BaseSetupPolicyProfile,
  BaseSetupProjectionIndex,
  BaseSetupSlotProfile,
} from "../types";

export interface CompileServiceSkillCatalogProjectionOptions {
  tenantId?: string;
  syncedAt?: string;
}

function cloneJsonValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function mapBundleSourceToServiceSkillSource(
  bundleRef: BaseSetupBundleRef,
): ServiceSkillSource {
  return bundleRef.source === "local" ? "local_custom" : "cloud_catalog";
}

function inferSkillType(
  projection: BaseSetupCatalogProjection,
  _bindingProfile: BaseSetupBindingProfile,
): ServiceSkillType {
  if (projection.skillType) {
    return projection.skillType;
  }
  return "service";
}

function normalizeExecutorBinding(
  bindingProfile: BaseSetupBindingProfile,
): ServiceSkillItem["defaultExecutorBinding"] {
  return bindingProfile.bindingFamily;
}

function inferExecutionLocation(
  projection: BaseSetupCatalogProjection,
  bindingProfile: BaseSetupBindingProfile,
): ServiceSkillExecutionLocation {
  if (bindingProfile.executionLocation === "cloud_required") {
    return "client_default";
  }
  if (bindingProfile.executionLocation) {
    return bindingProfile.executionLocation;
  }
  return "client_default";
}

function buildBundleSummary(
  pkg: BaseSetupPackage,
  projection: BaseSetupCatalogProjection,
  bundleRef: BaseSetupBundleRef,
  bindingProfile: BaseSetupBindingProfile,
  artifactProfile: BaseSetupArtifactProfile,
  policyProfile: BaseSetupPolicyProfile,
): ServiceSkillBundleSummary {
  const skillType = inferSkillType(projection, bindingProfile);
  const executionLocation = inferExecutionLocation(projection, bindingProfile);
  const executorBinding = normalizeExecutorBinding(bindingProfile);

  const metadata: Record<string, string> = {
    Lime_base_setup_package_id: pkg.id,
    Lime_base_setup_package_version: pkg.version,
    Lime_projection_id: projection.id,
    Lime_target_catalog: projection.targetCatalog,
    Lime_artifact_profile_ref: projection.artifactProfileRef,
    Lime_scorecard_profile_ref: projection.scorecardProfileRef,
    Lime_policy_profile_ref: projection.policyProfileRef,
    Lime_automation_profile_ref: projection.automationProfileRef ?? "",
    Lime_skill_type: skillType,
    Lime_category: projection.category,
    Lime_runner_type: bindingProfile.runnerType ?? "instant",
    Lime_execution_location: executionLocation,
    Lime_executor_binding: executorBinding,
    Lime_output_destination: artifactProfile.outputDestination ?? "",
    Lime_output_hint: projection.outputHint,
    Lime_entry_hint: projection.entryHint ?? "",
    Lime_prompt_template_key: projection.promptTemplateKey ?? "",
    Lime_theme_target: projection.themeTarget ?? "",
    Lime_composition_blueprint_ref: projection.compositionBlueprintRef ?? "",
  };

  if (policyProfile.surfaceScopes && policyProfile.surfaceScopes.length > 0) {
    metadata.Lime_surface_scopes = JSON.stringify(policyProfile.surfaceScopes);
  }

  for (const [key, value] of Object.entries(metadata)) {
    if (!value) {
      delete metadata[key];
    }
  }
  if (projection.skillBundleMetadata) {
    for (const [key, value] of Object.entries(projection.skillBundleMetadata)) {
      if (value.trim()) {
        metadata[key] = value;
      }
    }
  }

  return {
    name: projection.skillKey ?? projection.entryKey ?? bundleRef.id,
    description: projection.summary,
    compatibility: pkg.compatibility.minAppVersion,
    metadata,
    resourceSummary: {
      hasScripts: projection.skillBundleResourceSummary?.hasScripts ?? false,
      hasReferences:
        projection.skillBundleResourceSummary?.hasReferences ?? false,
      hasAssets: projection.skillBundleResourceSummary?.hasAssets ?? false,
    },
    standardCompliance: {
      isStandard: true,
    },
  };
}

function createServiceSkillItem(params: {
  pkg: BaseSetupPackage;
  projection: BaseSetupCatalogProjection;
  bundleRef: BaseSetupBundleRef;
  slotProfile: BaseSetupSlotProfile;
  bindingProfile: BaseSetupBindingProfile;
  artifactProfile: BaseSetupArtifactProfile;
  policyProfile: BaseSetupPolicyProfile;
}): ServiceSkillItem {
  const {
    pkg,
    projection,
    bundleRef,
    slotProfile,
    bindingProfile,
    artifactProfile,
    policyProfile,
  } = params;

  return {
    id: projection.id,
    skillKey: projection.skillKey ?? projection.entryKey,
    skillType: inferSkillType(projection, bindingProfile),
    title: projection.title,
    summary: projection.summary,
    entryHint: projection.entryHint,
    aliases: projection.aliases ? [...projection.aliases] : undefined,
    category: projection.category,
    outputHint: projection.outputHint,
    triggerHints: projection.triggerHints
      ? [...projection.triggerHints]
      : undefined,
    source: projection.source ?? mapBundleSourceToServiceSkillSource(bundleRef),
    runnerType: bindingProfile.runnerType ?? "instant",
    defaultExecutorBinding: normalizeExecutorBinding(bindingProfile),
    executionLocation: inferExecutionLocation(projection, bindingProfile),
    defaultArtifactKind: artifactProfile.defaultArtifactKind,
    readinessRequirements: projection.readinessRequirements
      ? cloneJsonValue(projection.readinessRequirements)
      : undefined,
    usageGuidelines: projection.usageGuidelines
      ? [...projection.usageGuidelines]
      : undefined,
    setupRequirements: projection.setupRequirements
      ? [...projection.setupRequirements]
      : undefined,
    examples: projection.examples ? [...projection.examples] : undefined,
    outputDestination: artifactProfile.outputDestination ?? undefined,
    sceneBinding: projection.sceneBinding
      ? cloneJsonValue(projection.sceneBinding)
      : undefined,
    slotSchema: slotProfile.slots.map((slot) => cloneJsonValue(slot)),
    surfaceScopes: policyProfile.surfaceScopes
      ? [...policyProfile.surfaceScopes]
      : undefined,
    promptTemplateKey: projection.promptTemplateKey,
    themeTarget: projection.themeTarget,
    skillBundle: buildBundleSummary(
      pkg,
      projection,
      bundleRef,
      bindingProfile,
      artifactProfile,
      policyProfile,
    ),
    version: projection.version ?? pkg.version,
  };
}

function buildProjectionIndex(
  projections: BaseSetupCatalogProjection[],
): BaseSetupProjectionIndex {
  return {
    artifactProfileRefsByProjectionId: Object.fromEntries(
      projections.map((projection) => [
        projection.id,
        projection.artifactProfileRef,
      ]),
    ),
    scorecardProfileRefsByProjectionId: Object.fromEntries(
      projections.map((projection) => [
        projection.id,
        projection.scorecardProfileRef,
      ]),
    ),
    policyProfileRefsByProjectionId: Object.fromEntries(
      projections.map((projection) => [
        projection.id,
        projection.policyProfileRef,
      ]),
    ),
    automationProfileRefsByProjectionId: Object.fromEntries(
      projections
        .filter((projection) => projection.automationProfileRef)
        .map((projection) => [
          projection.id,
          projection.automationProfileRef as string,
        ]),
    ),
    compositionBlueprintRefsByProjectionId: Object.fromEntries(
      projections
        .filter((projection) => projection.compositionBlueprintRef)
        .map((projection) => [
          projection.id,
          projection.compositionBlueprintRef as string,
        ]),
    ),
  };
}

export function compileServiceSkillCatalogProjection(
  pkg: BaseSetupPackage,
  options: CompileServiceSkillCatalogProjectionOptions = {},
): {
  catalog: ServiceSkillCatalog;
  projectionIndex: BaseSetupProjectionIndex;
} {
  const bundleRefs = new Map(
    pkg.bundleRefs.map((entry) => [entry.id, entry] as const),
  );
  const slotProfiles = new Map(
    pkg.slotProfiles.map((entry) => [entry.id, entry] as const),
  );
  const bindingProfiles = new Map(
    pkg.bindingProfiles.map((entry) => [entry.id, entry] as const),
  );
  const artifactProfiles = new Map(
    pkg.artifactProfiles.map((entry) => [entry.id, entry] as const),
  );
  const policyProfiles = new Map(
    pkg.policyProfiles.map((entry) => [entry.id, entry] as const),
  );

  const projections = pkg.catalogProjections.filter(
    (projection) => projection.targetCatalog === "service_skill_catalog",
  );

  const items = projections.map((projection) => {
    const bundleRef = bundleRefs.get(projection.bundleRefId);
    const slotProfile = slotProfiles.get(projection.slotProfileRef);
    const bindingProfile = bindingProfiles.get(projection.bindingProfileRef);
    const artifactProfile = artifactProfiles.get(projection.artifactProfileRef);
    const policyProfile = policyProfiles.get(projection.policyProfileRef);

    if (
      !bundleRef ||
      !slotProfile ||
      !bindingProfile ||
      !artifactProfile ||
      !policyProfile
    ) {
      throw new Error(`无法编译 projection ${projection.id}：存在未解析的引用`);
    }

    return createServiceSkillItem({
      pkg,
      projection,
      bundleRef,
      slotProfile,
      bindingProfile,
      artifactProfile,
      policyProfile,
    });
  });

  return {
    catalog: {
      version: pkg.version,
      tenantId: options.tenantId ?? "base-setup",
      syncedAt: options.syncedAt ?? new Date().toISOString(),
      items,
    },
    projectionIndex: buildProjectionIndex(projections),
  };
}
