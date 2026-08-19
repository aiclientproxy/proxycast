import { getRuntimeAppVersion } from "@/lib/appVersion";
import type {
  ServiceSkillArtifactKind,
  ServiceSkillCatalog,
  ServiceSkillAnyExecutionLocation,
  ServiceSkillAnyExecutorBinding,
  ServiceSkillPromptTemplateKey,
  ServiceSkillRunnerType,
  ServiceSkillSceneBinding,
  ServiceSkillSlotDefinition,
  ServiceSkillSource,
  ServiceSkillSurfaceScope,
  ServiceSkillType,
} from "@/lib/api/serviceSkills";
import { compileBaseSetupPackage } from "./compiler";
import { evaluateBaseSetupRollout } from "./rolloutGate";
import type {
  BaseSetupAutomationProfile,
  BaseSetupAutomationSchedulePreset,
  BaseSetupCommandIntentConfirmation,
  BaseSetupCommandExecutionKind,
  BaseSetupPackage,
  BaseSetupRenderDetailKind,
  BaseSetupRenderResultKind,
  CompiledBaseSetupPackage,
} from "./types";

type UnknownRecord = Record<string, unknown>;

export interface BaseSetupServiceSkillCatalogAdapterOptions {
  appVersion?: string;
  tenantId?: string;
  syncedAt?: string;
  seededFallbackAvailable?: boolean;
}

export interface ResolvedBaseSetupServiceSkillCatalog {
  package: BaseSetupPackage;
  compiled: CompiledBaseSetupPackage;
}

function asRecord(value: unknown): UnknownRecord | null {
  return value && typeof value === "object" ? (value as UnknownRecord) : null;
}

function readValue(record: UnknownRecord, ...keys: string[]): unknown {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }
  return undefined;
}

function readString(
  record: UnknownRecord,
  ...keys: string[]
): string | undefined {
  const value = readValue(record, ...keys);
  return typeof value === "string" && value.trim() ? value : undefined;
}

function readBoolean(
  record: UnknownRecord,
  ...keys: string[]
): boolean | undefined {
  const value = readValue(record, ...keys);
  return typeof value === "boolean" ? value : undefined;
}

function readStringArray(
  record: UnknownRecord,
  ...keys: string[]
): string[] | undefined {
  const value = readValue(record, ...keys);
  if (!Array.isArray(value)) {
    return undefined;
  }
  const strings = value.filter(
    (item): item is string => typeof item === "string",
  );
  return strings.length === value.length ? strings : undefined;
}

function readStringRecord(
  record: UnknownRecord,
  ...keys: string[]
): Record<string, string> | undefined {
  const value = readValue(record, ...keys);
  const source = asRecord(value);
  if (!source) {
    return undefined;
  }
  const output: Record<string, string> = {};
  for (const [key, item] of Object.entries(source)) {
    if (typeof item === "string") {
      output[key] = item;
    }
  }
  return Object.keys(output).length > 0 ? output : undefined;
}

function readCommandIntentConfirmation(
  record: UnknownRecord,
): BaseSetupCommandIntentConfirmation | undefined {
  const value = readValue(record, "intentConfirmation", "intent_confirmation");
  const source = asRecord(value);
  if (!source) {
    return undefined;
  }

  const id = readString(source, "id");
  const ruleKey = readString(source, "ruleKey", "rule_key");
  const confirmationKey = readString(
    source,
    "confirmationKey",
    "confirmation_key",
  );
  if (!id || !ruleKey || !confirmationKey) {
    return undefined;
  }

  const systemPromptKey = readString(
    source,
    "systemPromptKey",
    "system_prompt_key",
  );
  return {
    id,
    ruleKey,
    confirmationKey,
    ...(systemPromptKey ? { systemPromptKey } : {}),
  };
}

function readArray(
  record: UnknownRecord,
  ...keys: string[]
): unknown[] | undefined {
  const value = readValue(record, ...keys);
  return Array.isArray(value) ? value : undefined;
}

function cloneRecord<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function cloneOptionalRecord<T>(value: T | undefined): T | undefined {
  if (value === undefined) {
    return undefined;
  }
  return cloneRecord(value);
}

function toSlotDefinition(value: unknown): ServiceSkillSlotDefinition | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const key = readString(record, "key");
  const label = readString(record, "label");
  const type = readString(record, "type");
  const placeholder = readString(record, "placeholder");
  const requiredValue = readValue(record, "required");

  if (
    !key ||
    !label ||
    !type ||
    !placeholder ||
    typeof requiredValue !== "boolean"
  ) {
    return null;
  }

  const options = readArray(record, "options")
    ?.map((option) => {
      const optionRecord = asRecord(option);
      if (!optionRecord) {
        return null;
      }
      const value = readString(optionRecord, "value");
      const label = readString(optionRecord, "label");
      if (!value || !label) {
        return null;
      }
      const description = readString(optionRecord, "description");
      return {
        value,
        label,
        description,
      };
    })
    .filter((item): item is NonNullable<typeof item> => Boolean(item));

  return {
    key,
    label,
    type: type as ServiceSkillSlotDefinition["type"],
    required: requiredValue,
    placeholder,
    defaultValue: readString(record, "defaultValue", "default_value"),
    helpText: readString(record, "helpText", "help_text"),
    options: options?.length ? options : undefined,
  };
}

function toBundleRef(
  value: unknown,
): BaseSetupPackage["bundleRefs"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const source = readString(record, "source");
  const kind = readString(record, "kind");
  const pathOrUri =
    readString(record, "pathOrUri", "path_or_uri") ??
    readString(record, "path") ??
    readString(record, "uri");

  if (!id || !source || !kind || !pathOrUri) {
    return null;
  }

  return {
    id,
    source: source as BaseSetupPackage["bundleRefs"][number]["source"],
    kind: kind as BaseSetupPackage["bundleRefs"][number]["kind"],
    pathOrUri,
    versionConstraint: readString(
      record,
      "versionConstraint",
      "version_constraint",
    ),
  };
}

function toSlotProfile(
  value: unknown,
): BaseSetupPackage["slotProfiles"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const slots = readArray(record, "slots")
    ?.map((slot) => toSlotDefinition(slot))
    .filter((slot): slot is ServiceSkillSlotDefinition => Boolean(slot));

  if (!id || !slots) {
    return null;
  }

  return {
    id,
    slots,
  };
}

function toBindingProfile(
  value: unknown,
): BaseSetupPackage["bindingProfiles"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const bindingFamily = readString(record, "bindingFamily", "binding_family");
  if (!id || !bindingFamily) {
    return null;
  }

  const capabilityRefs = readStringArray(
    record,
    "capabilityRefs",
    "capability_refs",
  );
  const timeoutSecs = readValue(record, "timeoutSecs", "timeout_secs");
  const retryLimit = readValue(record, "retryLimit", "retry_limit");

  return {
    id,
    bindingFamily: bindingFamily as ServiceSkillAnyExecutorBinding,
    runnerType: readString(record, "runnerType", "runner_type") as
      | ServiceSkillRunnerType
      | undefined,
    executionLocation: readString(
      record,
      "executionLocation",
      "execution_location",
    ) as ServiceSkillAnyExecutionLocation | undefined,
    capabilityRefs,
    timeoutSecs: typeof timeoutSecs === "number" ? timeoutSecs : undefined,
    retryLimit: typeof retryLimit === "number" ? retryLimit : undefined,
  };
}

function toArtifactProfile(
  value: unknown,
): BaseSetupPackage["artifactProfiles"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const deliveryContract = readString(
    record,
    "deliveryContract",
    "delivery_contract",
  );
  const viewerKind = readString(record, "viewerKind", "viewer_kind");
  const requiredParts = readStringArray(
    record,
    "requiredParts",
    "required_parts",
  );

  if (!id || !deliveryContract || !viewerKind || !requiredParts) {
    return null;
  }

  return {
    id,
    deliveryContract:
      deliveryContract as BaseSetupPackage["artifactProfiles"][number]["deliveryContract"],
    requiredParts,
    viewerKind:
      viewerKind as BaseSetupPackage["artifactProfiles"][number]["viewerKind"],
    defaultArtifactKind: readString(
      record,
      "defaultArtifactKind",
      "default_artifact_kind",
    ) as ServiceSkillArtifactKind | undefined,
    outputDestination: readString(
      record,
      "outputDestination",
      "output_destination",
    ),
  };
}

function toScorecardProfile(
  value: unknown,
): BaseSetupPackage["scorecardProfiles"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const metrics = readStringArray(record, "metrics");
  if (!id || !metrics) {
    return null;
  }

  return {
    id,
    metrics,
    failureSignals: readStringArray(
      record,
      "failureSignals",
      "failure_signals",
    ),
  };
}

function toAutomationSchedulePreset(
  value: unknown,
): BaseSetupAutomationSchedulePreset | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }

  const kind = readString(record, "kind");
  const slotKey = readString(record, "slotKey", "slot_key");

  if (kind === "every") {
    const everySecs = readValue(record, "everySecs", "every_secs");
    if (typeof everySecs !== "number" || !Number.isFinite(everySecs)) {
      return undefined;
    }

    return {
      kind,
      everySecs,
      slotKey,
    };
  }

  if (kind === "cron") {
    const cronExpr = readString(record, "cronExpr", "cron_expr");
    if (!cronExpr) {
      return undefined;
    }

    return {
      kind,
      cronExpr,
      cronTz: readString(record, "cronTz", "cron_tz"),
      slotKey,
    };
  }

  if (kind === "at") {
    const at = readString(record, "at");
    if (!at) {
      return undefined;
    }

    return {
      kind,
      at,
      slotKey,
    };
  }

  return undefined;
}

function toAutomationProfile(
  value: unknown,
): BaseSetupAutomationProfile | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  if (!id) {
    return null;
  }

  const deliveryRecord = asRecord(readValue(record, "delivery"));
  const deliveryMode = readString(deliveryRecord ?? {}, "mode");

  return {
    id,
    enabledByDefault: readBoolean(
      record,
      "enabledByDefault",
      "enabled_by_default",
    ),
    schedule: toAutomationSchedulePreset(readValue(record, "schedule")),
    delivery:
      deliveryRecord && deliveryMode
        ? {
            mode: deliveryMode as "none" | "announce",
            channel: readString(deliveryRecord, "channel") as
              | "webhook"
              | "telegram"
              | "local_file"
              | "google_sheets"
              | undefined,
            target: readString(deliveryRecord, "target"),
            outputSchema: readString(
              deliveryRecord,
              "outputSchema",
              "output_schema",
            ) as "text" | "json" | "table" | "csv" | "links" | undefined,
            outputFormat: readString(
              deliveryRecord,
              "outputFormat",
              "output_format",
            ) as "text" | "json" | undefined,
            bestEffort: readBoolean(
              deliveryRecord,
              "bestEffort",
              "best_effort",
            ),
          }
        : undefined,
    maxRetries:
      typeof readValue(record, "maxRetries", "max_retries") === "number"
        ? (readValue(record, "maxRetries", "max_retries") as number)
        : undefined,
  };
}

function toPolicyProfile(
  value: unknown,
): BaseSetupPackage["policyProfiles"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  if (!id) {
    return null;
  }

  return {
    id,
    enabled: readBoolean(record, "enabled"),
    surfaceScopes: readStringArray(
      record,
      "surfaceScopes",
      "surface_scopes",
    ) as ServiceSkillSurfaceScope[] | undefined,
    rolloutStage: readString(
      record,
      "rolloutStage",
      "rollout_stage",
    ) as BaseSetupPackage["policyProfiles"][number]["rolloutStage"],
  };
}

function toCompositionBlueprint(
  value: unknown,
): NonNullable<BaseSetupPackage["compositionBlueprints"]>[number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  if (!id) {
    return null;
  }

  const deliveryContractRecord = asRecord(
    readValue(record, "deliveryContract", "delivery_contract"),
  );
  const steps = readArray(record, "steps")
    ?.map((step) => {
      const stepRecord = asRecord(step);
      if (!stepRecord) {
        return null;
      }
      const id = readString(stepRecord, "id");
      if (!id) {
        return null;
      }
      return {
        id,
        bindingProfileRef: readString(
          stepRecord,
          "bindingProfileRef",
          "binding_profile_ref",
        ),
      };
    })
    .filter((step): step is NonNullable<typeof step> => Boolean(step));

  return {
    id,
    artifactProfileRef: readString(
      record,
      "artifactProfileRef",
      "artifact_profile_ref",
    ),
    deliveryContract: deliveryContractRecord
      ? {
          requiredParts:
            readStringArray(
              deliveryContractRecord,
              "requiredParts",
              "required_parts",
            ) ?? [],
        }
      : undefined,
    steps,
  };
}

function toCatalogProjection(
  value: unknown,
): BaseSetupPackage["catalogProjections"][number] | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const targetCatalog = readString(record, "targetCatalog", "target_catalog");
  const entryKey = readString(record, "entryKey", "entry_key");
  const title = readString(record, "title");
  const summary = readString(record, "summary");
  const category = readString(record, "category");
  const outputHint = readString(record, "outputHint", "output_hint");
  const bundleRefId = readString(record, "bundleRefId", "bundle_ref_id");
  const slotProfileRef = readString(
    record,
    "slotProfileRef",
    "slot_profile_ref",
  );
  const bindingProfileRef = readString(
    record,
    "bindingProfileRef",
    "binding_profile_ref",
  );
  const artifactProfileRef = readString(
    record,
    "artifactProfileRef",
    "artifact_profile_ref",
  );
  const scorecardProfileRef = readString(
    record,
    "scorecardProfileRef",
    "scorecard_profile_ref",
  );
  const policyProfileRef = readString(
    record,
    "policyProfileRef",
    "policy_profile_ref",
  );

  if (
    !id ||
    !targetCatalog ||
    !entryKey ||
    !title ||
    !summary ||
    !category ||
    !outputHint ||
    !bundleRefId ||
    !slotProfileRef ||
    !bindingProfileRef ||
    !artifactProfileRef ||
    !scorecardProfileRef ||
    !policyProfileRef
  ) {
    return null;
  }

  const readinessRequirements = asRecord(
    readValue(record, "readinessRequirements", "readiness_requirements"),
  );
  const commandBindingRecord = asRecord(
    readValue(record, "commandBinding", "command_binding"),
  );
  const commandRenderContractRecord = asRecord(
    readValue(record, "commandRenderContract", "command_render_contract"),
  );
  const skillBundleResourceSummaryRecord = asRecord(
    readValue(
      record,
      "skillBundleResourceSummary",
      "skill_bundle_resource_summary",
    ),
  );

  return {
    id,
    targetCatalog:
      targetCatalog as BaseSetupPackage["catalogProjections"][number]["targetCatalog"],
    entryKey,
    title,
    summary,
    category,
    outputHint,
    bundleRefId,
    slotProfileRef,
    bindingProfileRef,
    artifactProfileRef,
    scorecardProfileRef,
    policyProfileRef,
    automationProfileRef: readString(
      record,
      "automationProfileRef",
      "automation_profile_ref",
    ),
    compositionBlueprintRef: readString(
      record,
      "compositionBlueprintRef",
      "composition_blueprint_ref",
    ),
    skillKey: readString(record, "skillKey", "skill_key"),
    entryHint: readString(record, "entryHint", "entry_hint"),
    aliases: readStringArray(record, "aliases"),
    triggerHints: readStringArray(record, "triggerHints", "trigger_hints"),
    source: readString(record, "source") as ServiceSkillSource | undefined,
    skillType: readString(record, "skillType", "skill_type") as
      | ServiceSkillType
      | undefined,
    readinessRequirements: readinessRequirements
      ? {
          requiresModel: readBoolean(
            readinessRequirements,
            "requiresModel",
            "requires_model",
          ),
          requiresBrowser: readBoolean(
            readinessRequirements,
            "requiresBrowser",
            "requires_browser",
          ),
          requiresSkillKey: readString(
            readinessRequirements,
            "requiresSkillKey",
            "requires_skill_key",
          ),
          requiresProject: readBoolean(
            readinessRequirements,
            "requiresProject",
            "requires_project",
          ),
        }
      : undefined,
    usageGuidelines: readStringArray(
      record,
      "usageGuidelines",
      "usage_guidelines",
    ),
    setupRequirements: readStringArray(
      record,
      "setupRequirements",
      "setup_requirements",
    ),
    examples: readStringArray(record, "examples"),
    promptTemplateKey: readString(
      record,
      "promptTemplateKey",
      "prompt_template_key",
    ) as ServiceSkillPromptTemplateKey | undefined,
    themeTarget: readString(record, "themeTarget", "theme_target"),
    sceneBinding: cloneOptionalRecord(
      readValue(record, "sceneBinding", "scene_binding"),
    ) as ServiceSkillSceneBinding | undefined,
    commandBinding: commandBindingRecord
      ? (() => {
          const requestDefaults = readStringRecord(
            commandBindingRecord,
            "requestDefaults",
            "request_defaults",
          );
          const intentConfirmation =
            readCommandIntentConfirmation(commandBindingRecord);
          return {
            skillId: readString(commandBindingRecord, "skillId", "skill_id"),
            executionKind: readString(
              commandBindingRecord,
              "executionKind",
              "execution_kind",
            ) as BaseSetupCommandExecutionKind | undefined,
            ...(requestDefaults ? { requestDefaults } : {}),
            ...(intentConfirmation ? { intentConfirmation } : {}),
          };
        })()
      : undefined,
    commandRenderContract: commandRenderContractRecord
      ? {
          resultKind: readString(
            commandRenderContractRecord,
            "resultKind",
            "result_kind",
          ) as BaseSetupRenderResultKind,
          detailKind: readString(
            commandRenderContractRecord,
            "detailKind",
            "detail_kind",
          ) as BaseSetupRenderDetailKind,
          supportsStreaming: readBoolean(
            commandRenderContractRecord,
            "supportsStreaming",
            "supports_streaming",
          ),
          supportsTimeline: readBoolean(
            commandRenderContractRecord,
            "supportsTimeline",
            "supports_timeline",
          ),
        }
      : undefined,
    skillBundleMetadata: readStringRecord(
      record,
      "skillBundleMetadata",
      "skill_bundle_metadata",
    ),
    skillBundleResourceSummary: skillBundleResourceSummaryRecord
      ? {
          hasScripts: readBoolean(
            skillBundleResourceSummaryRecord,
            "hasScripts",
            "has_scripts",
          ),
          hasReferences: readBoolean(
            skillBundleResourceSummaryRecord,
            "hasReferences",
            "has_references",
          ),
          hasAssets: readBoolean(
            skillBundleResourceSummaryRecord,
            "hasAssets",
            "has_assets",
          ),
        }
      : undefined,
    version: readString(record, "version"),
  };
}

export function parseBaseSetupPackage(value: unknown): BaseSetupPackage | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const id = readString(record, "id");
  const version = readString(record, "version");
  const title = readString(record, "title");
  const summary = readString(record, "summary");
  const bundleRefs = readArray(record, "bundleRefs", "bundle_refs")
    ?.map((entry) => toBundleRef(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const catalogProjections = readArray(
    record,
    "catalogProjections",
    "catalog_projections",
  )
    ?.map((entry) => toCatalogProjection(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const slotProfiles = readArray(record, "slotProfiles", "slot_profiles")
    ?.map((entry) => toSlotProfile(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const bindingProfiles = readArray(
    record,
    "bindingProfiles",
    "binding_profiles",
  )
    ?.map((entry) => toBindingProfile(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const artifactProfiles = readArray(
    record,
    "artifactProfiles",
    "artifact_profiles",
  )
    ?.map((entry) => toArtifactProfile(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const scorecardProfiles = readArray(
    record,
    "scorecardProfiles",
    "scorecard_profiles",
  )
    ?.map((entry) => toScorecardProfile(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const automationProfiles = readArray(
    record,
    "automationProfiles",
    "automation_profiles",
  )
    ?.map((entry) => toAutomationProfile(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const policyProfiles = readArray(record, "policyProfiles", "policy_profiles")
    ?.map((entry) => toPolicyProfile(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const compositionBlueprints = readArray(
    record,
    "compositionBlueprints",
    "composition_blueprints",
  )
    ?.map((entry) => toCompositionBlueprint(entry))
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));
  const compatibilityRecord = asRecord(readValue(record, "compatibility"));

  if (
    !id ||
    !version ||
    !title ||
    !summary ||
    !bundleRefs ||
    !catalogProjections ||
    !slotProfiles ||
    !bindingProfiles ||
    !artifactProfiles ||
    !scorecardProfiles ||
    !policyProfiles ||
    !compatibilityRecord
  ) {
    return null;
  }

  const minAppVersion = readString(
    compatibilityRecord,
    "minAppVersion",
    "min_app_version",
  );
  const requiredKernelCapabilities = readStringArray(
    compatibilityRecord,
    "requiredKernelCapabilities",
    "required_kernel_capabilities",
  );
  if (!minAppVersion || !requiredKernelCapabilities) {
    return null;
  }

  return {
    id,
    version,
    title,
    summary,
    bundleRefs,
    catalogProjections,
    slotProfiles,
    bindingProfiles,
    compositionBlueprints,
    artifactProfiles,
    scorecardProfiles,
    automationProfiles: automationProfiles ?? [],
    policyProfiles,
    compatibility: {
      minAppVersion,
      requiredKernelCapabilities,
      seededFallback: readBoolean(
        compatibilityRecord,
        "seededFallback",
        "seeded_fallback",
      ),
      compatCatalogProjection: readBoolean(
        compatibilityRecord,
        "compatCatalogProjection",
        "compat_catalog_projection",
      ),
    },
  };
}

export function parseServiceSkillCatalogFromBaseSetupPackage(
  value: unknown,
  options: BaseSetupServiceSkillCatalogAdapterOptions = {},
): ServiceSkillCatalog | null {
  const resolved = resolveBaseSetupServiceSkillCatalog(value, options);
  return resolved?.compiled.serviceSkillCatalogProjection ?? null;
}

export function resolveBaseSetupServiceSkillCatalog(
  value: unknown,
  options: BaseSetupServiceSkillCatalogAdapterOptions = {},
): ResolvedBaseSetupServiceSkillCatalog | null {
  const pkg = parseBaseSetupPackage(value);
  if (!pkg) {
    return null;
  }

  const rollout = evaluateBaseSetupRollout(pkg, {
    appVersion: options.appVersion ?? getRuntimeAppVersion(),
    seededFallbackAvailable: options.seededFallbackAvailable ?? true,
  });

  if (rollout.decision !== "accept") {
    return null;
  }

  return {
    package: pkg,
    compiled: compileBaseSetupPackage(pkg, {
      tenantId: options.tenantId,
      syncedAt: options.syncedAt,
    }),
  };
}
