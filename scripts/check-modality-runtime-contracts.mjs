#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const CONTRACT_PATH = "src/lib/governance/modalityRuntimeContracts.json";
const CAPABILITY_MATRIX_PATH =
  "src/lib/governance/modalityCapabilityMatrix.json";
const ARTIFACT_GRAPH_PATH = "src/lib/governance/modalityArtifactGraph.json";
const EXECUTION_PROFILE_PATH =
  "src/lib/governance/modalityExecutionProfiles.json";
const REQUIRED_DOCS = [
  "internal/roadmap/warp/runtime-fact-map.md",
  "internal/roadmap/warp/contract-schema.md",
  "internal/roadmap/warp/capability-matrix.md",
  "internal/roadmap/warp/execution-profile.md",
  "internal/roadmap/warp/artifact-graph.md",
  "internal/roadmap/warp/entry-binding-inventory.md",
  "internal/roadmap/warp/task-index-inventory.md",
  "internal/roadmap/warp/evolution-guide.md",
];
const LIFECYCLES = new Set(["current", "compat", "deprecated", "dead"]);
const MODALITIES = new Set([
  "text",
  "image",
  "audio",
  "video",
  "browser",
  "document",
  "code",
  "mixed",
]);
const CAPABILITY_GROUPS = new Set(["model", "tool", "runtime", "policy"]);
const PERMISSIONS = new Set([
  "read_files",
  "write_artifacts",
  "execute_commands",
  "call_mcp",
  "web_search",
  "browser_control",
  "media_upload",
  "service_api_call",
  "local_cli",
  "request_user_input",
]);
const ARTIFACT_KINDS = new Set([
  "image_task",
  "image_output",
  "audio_task",
  "audio_output",
  "transcript",
  "browser_session",
  "browser_snapshot",
  "pdf_extract",
  "report_document",
  "presentation_document",
  "webpage_artifact",
  "generic_file",
]);
const VIEWER_SURFACES = new Set([
  "image_workbench",
  "audio_player",
  "transcript_viewer",
  "browser_replay_viewer",
  "document_viewer",
  "report_viewer",
  "presentation_viewer",
  "webpage_viewer",
  "generic_file_viewer",
]);
const EVIDENCE_EVENTS = new Set([
  "candidate_set_resolved",
  "runtime_identity_bound",
  "routing_decision_made",
  "routing_not_possible",
  "single_candidate_capability_gap",
  "model_routing_decision",
  "execution_profile_decision",
  "executor_invoked",
  "artifact_written",
  "browser_action_requested",
  "browser_observation_recorded",
  "file_read_authorized",
  "web_search_performed",
]);
const LIMECORE_POLICY_REFS = new Set([
  "client_skills",
  "client_scenes",
  "model_catalog",
  "provider_offer",
  "gateway_policy",
  "scene_policy",
  "tenant_feature_flags",
  "audit_config",
]);
const EXECUTOR_KINDS = new Set([
  "skill",
  "tool",
  "service_skill",
  "browser",
  "gateway",
  "scene_cloud",
  "local_cli",
  "app_server",
  "workflow",
]);
const FAILURE_REASONS = new Set([
  "permission_denied",
  "capability_gap",
  "executor_error",
  "provider_error",
  "observation_unavailable",
  "file_unavailable",
  "source_unavailable",
]);
const PROFILE_WRITE_MODES = new Set([
  "domain_task_artifact",
  "domain_document_artifact",
  "runtime_observation_trace",
  "timeline_backed_document_artifact",
  "document_or_compat_file_artifact",
]);
const ARTIFACT_IMPLEMENTATION_STATUSES = new Set([
  "current",
  "partial",
  "planned",
]);
const REQUIRED_ARTIFACT_INDEX_FIELDS = new Set([
  "task_id",
  "contract_key",
  "artifact_kind",
  "status",
  "created_at",
  "updated_at",
]);
const MEDIA_TASK_ARTIFACT_KINDS_REQUIRING_PHASE8_FIELDS = new Set([
  "image_task",
  "image_output",
  "audio_task",
  "audio_output",
  "transcript",
]);
const REQUIRED_MEDIA_TASK_PHASE8_INDEX_FIELDS = new Set([
  "entry_key",
  "thread_id",
  "turn_id",
  "content_id",
  "modality",
  "skill_id",
  "model_id",
  "cost_state",
  "limit_state",
  "estimated_cost_class",
  "limit_event_kind",
  "quota_low",
  "executor_kind",
  "executor_binding_key",
  "limecore_policy_snapshot_status",
]);
const ENTRY_KINDS = new Set([
  "command",
  "button_action",
  "scene",
  "implicit_context",
]);
const FORBIDDEN_ENTRY_FIELDS = new Set([
  "truth_source",
  "artifact_kinds",
  "viewer_surface",
  "evidence_events",
  "executor_binding",
  "routing_slot",
]);
const PHASE7_REQUIRED_ENTRY_BINDINGS = new Map([
  ["image_generation", ["at_image_command"]],
  ["browser_control", ["at_browser_command"]],
  ["pdf_extract", ["at_pdf_read_command"]],
  ["voice_generation", ["at_voice_command"]],
  ["audio_transcription", ["at_transcription_command"]],
  ["web_research", ["at_search_command"]],
  ["text_transform", ["at_summary_command"]],
]);
const METADATA_ONLY_ROUTE_EXECUTION_STATUS = "metadata_only";
const ROUTE_EXECUTION_EVIDENCE_EVENTS = new Set(["executor_invoked"]);

function collectUniqueObjects(errors, collection, keyName, label) {
  const result = new Map();
  pushIf(
    errors,
    !isNonEmptyArray(collection),
    `${label} must be a non-empty array`,
  );
  if (!Array.isArray(collection)) {
    return result;
  }

  collection.forEach((item, index) => {
    const prefix = `${label}[${index}]`;
    pushIf(errors, !isPlainObject(item), `${prefix} must be an object`);
    if (!isPlainObject(item)) {
      return;
    }

    const key = item[keyName];
    pushIf(
      errors,
      !isNonEmptyString(key),
      `${prefix}.${keyName} must be a non-empty string`,
    );
    if (!isNonEmptyString(key)) {
      return;
    }
    pushIf(
      errors,
      result.has(key),
      `${prefix}.${keyName} is duplicated: ${key}`,
    );
    result.set(key, item);
  });

  return result;
}

function readJson(filePath) {
  const absolutePath = path.resolve(process.cwd(), filePath);
  return JSON.parse(fs.readFileSync(absolutePath, "utf8"));
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function isNonEmptyArray(value) {
  return Array.isArray(value) && value.length > 0;
}

function pushIf(errors, condition, message) {
  if (condition) {
    errors.push(message);
  }
}

function validateEnumArray(
  errors,
  contractKey,
  fieldName,
  values,
  allowedValues,
) {
  pushIf(
    errors,
    !isNonEmptyArray(values),
    `${contractKey}.${fieldName} must be a non-empty array`,
  );
  if (!Array.isArray(values)) {
    return;
  }

  for (const value of values) {
    pushIf(
      errors,
      !allowedValues.has(value),
      `${contractKey}.${fieldName} contains unknown value: ${String(value)}`,
    );
  }
}

function validateStringArray(
  errors,
  contractKey,
  fieldName,
  values,
  options = {},
) {
  const { allowEmpty = false } = options;
  const invalidArray = allowEmpty
    ? !Array.isArray(values)
    : !isNonEmptyArray(values);
  pushIf(
    errors,
    invalidArray,
    `${contractKey}.${fieldName} must be ${allowEmpty ? "an array" : "a non-empty array"}`,
  );
  if (!Array.isArray(values)) {
    return;
  }
  values.forEach((value, index) => {
    pushIf(
      errors,
      !isNonEmptyString(value),
      `${contractKey}.${fieldName}[${index}] must be a non-empty string`,
    );
  });
}

function looksLikeEntryKey(contractKey) {
  return (
    contractKey.startsWith("@") ||
    contractKey.startsWith("/") ||
    contractKey.startsWith("at_") ||
    contractKey.includes("@") ||
    /[\u4e00-\u9fff]/u.test(contractKey)
  );
}

function validateExecutor(errors, contract) {
  const contractKey = contract.contract_key;
  const executor = contract.executor_binding;
  if (
    contract.route_execution_status === METADATA_ONLY_ROUTE_EXECUTION_STATUS &&
    executor === undefined
  ) {
    return;
  }
  pushIf(
    errors,
    !isPlainObject(executor),
    `${contractKey}.executor_binding must be an object`,
  );
  if (!isPlainObject(executor)) {
    return;
  }

  pushIf(
    errors,
    !EXECUTOR_KINDS.has(executor.executor_kind),
    `${contractKey}.executor_binding.executor_kind is unknown: ${String(executor.executor_kind)}`,
  );
  for (const fieldName of ["binding_key", "current_path"]) {
    pushIf(
      errors,
      !isNonEmptyString(executor[fieldName]),
      `${contractKey}.executor_binding.${fieldName} must be a non-empty string`,
    );
  }
  for (const fieldName of [
    "supports_progress",
    "supports_cancel",
    "supports_resume",
    "supports_artifact",
  ]) {
    pushIf(
      errors,
      typeof executor[fieldName] !== "boolean",
      `${contractKey}.executor_binding.${fieldName} must be boolean`,
    );
  }
  validateEnumArray(
    errors,
    contractKey,
    "executor_binding.failure_mapping",
    executor.failure_mapping,
    FAILURE_REASONS,
  );
}

function validateEntryBindings(errors, contract) {
  const contractKey = contract.contract_key;
  pushIf(
    errors,
    !Array.isArray(contract.bound_entries),
    `${contractKey}.bound_entries must be an array`,
  );
  if (!Array.isArray(contract.bound_entries)) {
    return;
  }

  contract.bound_entries.forEach((entry, index) => {
    const prefix = `${contractKey}.bound_entries[${index}]`;
    pushIf(errors, !isPlainObject(entry), `${prefix} must be an object`);
    if (!isPlainObject(entry)) {
      return;
    }
    for (const fieldName of [
      "entry_key",
      "display_name",
      "launch_metadata_path",
      "entry_source",
    ]) {
      pushIf(
        errors,
        !isNonEmptyString(entry[fieldName]),
        `${prefix}.${fieldName} must be a non-empty string`,
      );
    }
    pushIf(
      errors,
      !ENTRY_KINDS.has(entry.entry_kind),
      `${prefix}.entry_kind is unknown: ${String(entry.entry_kind)}`,
    );
    pushIf(
      errors,
      isNonEmptyString(entry.launch_metadata_path) &&
        !entry.launch_metadata_path.startsWith("harness."),
      `${prefix}.launch_metadata_path must point to harness metadata`,
    );
    validateStringArray(
      errors,
      contractKey,
      `bound_entries[${index}].default_input_mapping`,
      entry.default_input_mapping,
    );
    validateStringArray(
      errors,
      contractKey,
      `bound_entries[${index}].entry_visibility_policy`,
      entry.entry_visibility_policy,
    );

    for (const fieldName of Object.keys(entry)) {
      pushIf(
        errors,
        FORBIDDEN_ENTRY_FIELDS.has(fieldName),
        `${prefix}.${fieldName} is a bottom-layer field and must stay on the contract`,
      );
    }
  });
}

function validatePhase7EntryBindingCoverage(errors, registry) {
  if (!Array.isArray(registry.contracts)) {
    return { entryBindingCount: 0 };
  }

  let entryBindingCount = 0;
  const globalEntryKeys = new Map();

  for (const contract of registry.contracts) {
    if (!isPlainObject(contract)) {
      continue;
    }

    const contractKey = contract.contract_key;
    const entries = Array.isArray(contract.bound_entries)
      ? contract.bound_entries.filter(isPlainObject)
      : [];
    const entryKeysForContract = new Set(
      entries
        .map((entry) => entry.entry_key)
        .filter((entryKey) => isNonEmptyString(entryKey)),
    );
    entryBindingCount += entries.length;

    if (contract.lifecycle === "current") {
      pushIf(
        errors,
        entries.length === 0,
        `current contract ${contractKey}.bound_entries must include at least one Phase 7 entry binding`,
      );
    }

    for (const entry of entries) {
      const entryKey = entry.entry_key;
      const prefix = `${contractKey}.bound_entries.${entryKey || "<missing>"}`;
      if (!isNonEmptyString(entryKey)) {
        continue;
      }

      pushIf(
        errors,
        globalEntryKeys.has(entryKey),
        `${prefix} duplicates entry_key already owned by ${globalEntryKeys.get(entryKey)}`,
      );
      globalEntryKeys.set(entryKey, contractKey);

      if (isNonEmptyString(entry.entry_source)) {
        pushIf(
          errors,
          !entryKeysForContract.has(entry.entry_source),
          `${prefix}.entry_source must reference an entry_key on the same contract, not ${entry.entry_source}`,
        );
      }

      if (entry.entry_kind === "scene") {
        const policyRefs = new Set(
          Array.isArray(contract.limecore_policy_refs)
            ? contract.limecore_policy_refs
            : [],
        );
        pushIf(
          errors,
          !policyRefs.has("client_scenes") && !policyRefs.has("scene_policy"),
          `${prefix} is a scene entry but contract.limecore_policy_refs does not include client_scenes or scene_policy`,
        );
      }
    }
  }

  for (const [
    contractKey,
    requiredEntryKeys,
  ] of PHASE7_REQUIRED_ENTRY_BINDINGS) {
    const contract = registry.contracts.find(
      (candidate) =>
        isPlainObject(candidate) && candidate.contract_key === contractKey,
    );
    pushIf(
      errors,
      !contract,
      `Phase 7 required contract is missing: ${contractKey}`,
    );
    if (!contract) {
      continue;
    }

    const entryKeys = new Set(
      Array.isArray(contract.bound_entries)
        ? contract.bound_entries
            .map((entry) => entry?.entry_key)
            .filter((entryKey) => isNonEmptyString(entryKey))
        : [],
    );
    for (const requiredEntryKey of requiredEntryKeys) {
      pushIf(
        errors,
        !entryKeys.has(requiredEntryKey),
        `Phase 7 required entry binding ${contractKey}.${requiredEntryKey} is missing`,
      );
    }
  }

  return { entryBindingCount };
}

function validateCapabilityMatrix(matrix) {
  const errors = [];
  pushIf(errors, matrix.version !== 1, "capabilityMatrix.version must be 1");
  pushIf(
    errors,
    matrix.status !== "current",
    "capabilityMatrix.status must be current",
  );
  pushIf(
    errors,
    !isNonEmptyString(matrix.owner),
    "capabilityMatrix.owner must be set",
  );

  const capabilityMap = collectUniqueObjects(
    errors,
    matrix.capabilities,
    "key",
    "capabilityMatrix.capabilities",
  );
  const modelRoleMap = collectUniqueObjects(
    errors,
    matrix.model_roles,
    "slot",
    "capabilityMatrix.model_roles",
  );

  for (const [key, capability] of capabilityMap.entries()) {
    pushIf(
      errors,
      !LIFECYCLES.has(capability.lifecycle),
      `capability ${key}.lifecycle is unknown: ${String(capability.lifecycle)}`,
    );
    pushIf(
      errors,
      !CAPABILITY_GROUPS.has(capability.group),
      `capability ${key}.group is unknown: ${String(capability.group)}`,
    );
    pushIf(
      errors,
      !isNonEmptyString(capability.description),
      `capability ${key}.description must be set`,
    );
    validateStringArray(
      errors,
      key,
      "routing_sources",
      capability.routing_sources,
    );
    pushIf(
      errors,
      !isNonEmptyString(capability.capability_gap_code),
      `capability ${key}.capability_gap_code must be set`,
    );
    validateEnumArray(
      errors,
      key,
      "evidence_events",
      capability.evidence_events,
      EVIDENCE_EVENTS,
    );
  }

  for (const [slot, role] of modelRoleMap.entries()) {
    pushIf(
      errors,
      !LIFECYCLES.has(role.lifecycle),
      `model role ${slot}.lifecycle is unknown: ${String(role.lifecycle)}`,
    );
    validateEnumArray(
      errors,
      slot,
      "capability_keys",
      role.capability_keys,
      new Set(capabilityMap.keys()),
    );
    validateStringArray(errors, slot, "fallback_slots", role.fallback_slots, {
      allowEmpty: true,
    });
    if (Array.isArray(role.fallback_slots)) {
      for (const fallbackSlot of role.fallback_slots) {
        pushIf(
          errors,
          !modelRoleMap.has(fallbackSlot),
          `model role ${slot}.fallback_slots contains unknown slot: ${fallbackSlot}`,
        );
      }
    }
  }

  return {
    errors,
    capabilityKeys: new Set(capabilityMap.keys()),
    modelRoleSlots: new Set(modelRoleMap.keys()),
  };
}

function validateArtifactGraph(graph) {
  const errors = [];
  pushIf(errors, graph.version !== 1, "artifactGraph.version must be 1");
  pushIf(
    errors,
    graph.status !== "current",
    "artifactGraph.status must be current",
  );
  pushIf(
    errors,
    !isNonEmptyString(graph.owner),
    "artifactGraph.owner must be set",
  );

  const artifactMap = collectUniqueObjects(
    errors,
    graph.artifact_kinds,
    "kind",
    "artifactGraph.artifact_kinds",
  );

  for (const kind of ARTIFACT_KINDS) {
    pushIf(
      errors,
      !artifactMap.has(kind),
      `artifactGraph.artifact_kinds is missing known artifact kind: ${kind}`,
    );
  }

  for (const [kind, artifact] of artifactMap.entries()) {
    pushIf(
      errors,
      !ARTIFACT_KINDS.has(kind),
      `artifact ${kind}.kind is unknown: ${kind}`,
    );
    pushIf(
      errors,
      !LIFECYCLES.has(artifact.lifecycle),
      `artifact ${kind}.lifecycle is unknown: ${String(artifact.lifecycle)}`,
    );
    pushIf(
      errors,
      !MODALITIES.has(artifact.modality),
      `artifact ${kind}.modality is unknown: ${String(artifact.modality)}`,
    );
    pushIf(
      errors,
      !ARTIFACT_IMPLEMENTATION_STATUSES.has(artifact.implementation_status),
      `artifact ${kind}.implementation_status is unknown: ${String(artifact.implementation_status)}`,
    );
    validateStringArray(errors, kind, "truth_sources", artifact.truth_sources);
    validateEnumArray(
      errors,
      kind,
      "viewer_surfaces",
      artifact.viewer_surfaces,
      VIEWER_SURFACES,
    );
    validateEnumArray(
      errors,
      kind,
      "evidence_events",
      artifact.evidence_events,
      EVIDENCE_EVENTS,
    );
    validateStringArray(
      errors,
      kind,
      "task_index_fields",
      artifact.task_index_fields,
    );
    if (Array.isArray(artifact.task_index_fields)) {
      const seenIndexFields = new Set();
      artifact.task_index_fields.forEach((fieldName, index) => {
        if (!isNonEmptyString(fieldName)) {
          return;
        }
        pushIf(
          errors,
          seenIndexFields.has(fieldName),
          `artifact ${kind}.task_index_fields[${index}] is duplicated: ${fieldName}`,
        );
        seenIndexFields.add(fieldName);
      });
    }
    validateStringArray(
      errors,
      kind,
      "current_contracts",
      artifact.current_contracts,
      { allowEmpty: true },
    );
    pushIf(
      errors,
      !isNonEmptyString(artifact.notes),
      `artifact ${kind}.notes must be set`,
    );

    if (
      artifact.implementation_status === "current" ||
      artifact.implementation_status === "partial"
    ) {
      const indexFields = new Set(
        Array.isArray(artifact.task_index_fields)
          ? artifact.task_index_fields
          : [],
      );
      for (const requiredField of REQUIRED_ARTIFACT_INDEX_FIELDS) {
        pushIf(
          errors,
          !indexFields.has(requiredField),
          `artifact ${kind}.task_index_fields must include ${requiredField}`,
        );
      }
      if (MEDIA_TASK_ARTIFACT_KINDS_REQUIRING_PHASE8_FIELDS.has(kind)) {
        for (const requiredField of REQUIRED_MEDIA_TASK_PHASE8_INDEX_FIELDS) {
          pushIf(
            errors,
            !indexFields.has(requiredField),
            `artifact ${kind}.task_index_fields must include ${requiredField} for Phase 8 media task index projection`,
          );
        }
      }
    }
  }

  return {
    errors,
    artifactMap,
  };
}

function hasIntersection(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right)) {
    return false;
  }
  const rightSet = new Set(right);
  return left.some((value) => rightSet.has(value));
}

function validateContractArtifactGraph(errors, contract, artifactGraphRefs) {
  const contractKey = contract.contract_key;
  if (!Array.isArray(contract.artifact_kinds)) {
    return;
  }

  for (const artifactKind of contract.artifact_kinds) {
    const artifact = artifactGraphRefs.artifactMap.get(artifactKind);
    if (!artifact) {
      errors.push(
        `${contractKey}.artifact_kinds references artifact not defined in graph: ${artifactKind}`,
      );
      continue;
    }

    pushIf(
      errors,
      Array.isArray(artifact.current_contracts) &&
        artifact.current_contracts.length > 0 &&
        !artifact.current_contracts.includes(contractKey),
      `${contractKey}.artifact_kinds contains ${artifactKind}, but artifactGraph.${artifactKind}.current_contracts does not include ${contractKey}`,
    );
    pushIf(
      errors,
      !hasIntersection(contract.truth_source, artifact.truth_sources),
      `${contractKey}.artifact_kinds ${artifactKind} has no truth_source intersection with artifact graph`,
    );
    pushIf(
      errors,
      !hasIntersection(contract.viewer_surface, artifact.viewer_surfaces),
      `${contractKey}.artifact_kinds ${artifactKind} has no viewer_surface intersection with artifact graph`,
    );
    pushIf(
      errors,
      !hasIntersection(contract.evidence_events, artifact.evidence_events),
      `${contractKey}.artifact_kinds ${artifactKind} has no evidence_events intersection with artifact graph`,
    );
    pushIf(
      errors,
      contract.route_execution_status ===
        METADATA_ONLY_ROUTE_EXECUTION_STATUS &&
        containsAny(artifact.evidence_events, ROUTE_EXECUTION_EVIDENCE_EVENTS),
      `${contractKey}.artifact_kinds ${artifactKind} must not declare executor_invoked in artifact graph while route_execution_status is metadata_only`,
    );
  }
}

function resolveExecutorAdapterKey(executor) {
  if (!isPlainObject(executor)) {
    return null;
  }
  if (
    !isNonEmptyString(executor.executor_kind) ||
    !isNonEmptyString(executor.binding_key)
  ) {
    return null;
  }
  return `${executor.executor_kind}:${executor.binding_key}`;
}

function validateSubset(errors, label, actualValues, requiredValues) {
  const actualSet = new Set(Array.isArray(actualValues) ? actualValues : []);
  for (const requiredValue of requiredValues || []) {
    pushIf(
      errors,
      !actualSet.has(requiredValue),
      `${label} must include ${requiredValue}`,
    );
  }
}

function containsAny(values, disallowedValues) {
  if (!Array.isArray(values)) {
    return false;
  }
  return values.some((value) => disallowedValues.has(value));
}

function validateMetadataOnlyRouteExecution(
  errors,
  contractKey,
  contract,
  profile,
  adapter,
) {
  if (
    contract.route_execution_status !== METADATA_ONLY_ROUTE_EXECUTION_STATUS
  ) {
    return;
  }

  pushIf(
    errors,
    !isNonEmptyString(contract.route_execution_exit_condition),
    `${contractKey}.route_execution_exit_condition must be set while route_execution_status is metadata_only`,
  );
  pushIf(
    errors,
    containsAny(contract.evidence_events, ROUTE_EXECUTION_EVIDENCE_EVENTS),
    `${contractKey}.evidence_events must not include executor_invoked while route_execution_status is metadata_only`,
  );

  if (profile) {
    pushIf(
      errors,
      profile.lifecycle === "current",
      `execution profile ${profile.profile_key} must not be current while ${contractKey}.route_execution_status is metadata_only`,
    );
    pushIf(
      errors,
      containsAny(profile.evidence_events, ROUTE_EXECUTION_EVIDENCE_EVENTS),
      `execution profile ${profile.profile_key}.evidence_events must not include executor_invoked while ${contractKey}.route_execution_status is metadata_only`,
    );
  }

  if (adapter) {
    pushIf(
      errors,
      adapter.lifecycle === "current",
      `executor adapter ${adapter.adapter_key} must not be current while ${contractKey}.route_execution_status is metadata_only`,
    );
    pushIf(
      errors,
      containsAny(adapter.evidence_events, ROUTE_EXECUTION_EVIDENCE_EVENTS),
      `executor adapter ${adapter.adapter_key}.evidence_events must not include executor_invoked while ${contractKey}.route_execution_status is metadata_only`,
    );
  }
}

function validateMetadataOnlyProfileAdapters(
  errors,
  contractKey,
  contract,
  profile,
  adapterMap,
) {
  if (
    contract.route_execution_status !== METADATA_ONLY_ROUTE_EXECUTION_STATUS ||
    !isPlainObject(profile) ||
    !Array.isArray(profile.executor_adapter_keys)
  ) {
    return;
  }

  for (const adapterKey of profile.executor_adapter_keys) {
    if (!isNonEmptyString(adapterKey)) {
      continue;
    }
    validateMetadataOnlyRouteExecution(
      errors,
      contractKey,
      contract,
      null,
      adapterMap.get(adapterKey) ?? null,
    );
  }
}

function validateArtifactPolicy(errors, label, policy, artifactGraphRefs) {
  pushIf(
    errors,
    !isPlainObject(policy),
    `${label}.artifact_policy must be an object`,
  );
  if (!isPlainObject(policy)) {
    return;
  }
  pushIf(
    errors,
    !PROFILE_WRITE_MODES.has(policy.write_mode),
    `${label}.artifact_policy.write_mode is unknown: ${String(policy.write_mode)}`,
  );
  validateEnumArray(
    errors,
    label,
    "artifact_policy.artifact_kinds",
    policy.artifact_kinds,
    ARTIFACT_KINDS,
  );
  if (Array.isArray(policy.artifact_kinds)) {
    for (const artifactKind of policy.artifact_kinds) {
      pushIf(
        errors,
        !artifactGraphRefs.artifactMap.has(artifactKind),
        `${label}.artifact_policy.artifact_kinds references artifact not defined in graph: ${artifactKind}`,
      );
    }
  }
  validateEnumArray(
    errors,
    label,
    "artifact_policy.viewer_surfaces",
    policy.viewer_surfaces,
    VIEWER_SURFACES,
  );
}

function validateExecutionProfiles(
  registry,
  profiles,
  matrixRefs,
  artifactGraphRefs,
) {
  const errors = [];
  pushIf(errors, profiles.version !== 1, "executionProfiles.version must be 1");
  pushIf(
    errors,
    profiles.status !== "current",
    "executionProfiles.status must be current",
  );
  pushIf(
    errors,
    !isNonEmptyString(profiles.owner),
    "executionProfiles.owner must be set",
  );

  const contractMap = collectUniqueObjects(
    errors,
    registry.contracts,
    "contract_key",
    "runtimeContracts.contracts",
  );
  const profileMap = collectUniqueObjects(
    errors,
    profiles.profiles,
    "profile_key",
    "executionProfiles.profiles",
  );
  const adapterMap = collectUniqueObjects(
    errors,
    profiles.executor_adapters,
    "adapter_key",
    "executionProfiles.executor_adapters",
  );

  for (const [adapterKey, adapter] of adapterMap.entries()) {
    pushIf(
      errors,
      !LIFECYCLES.has(adapter.lifecycle),
      `executor adapter ${adapterKey}.lifecycle is unknown: ${String(adapter.lifecycle)}`,
    );
    pushIf(
      errors,
      !EXECUTOR_KINDS.has(adapter.executor_kind),
      `executor adapter ${adapterKey}.executor_kind is unknown: ${String(adapter.executor_kind)}`,
    );
    pushIf(
      errors,
      !isNonEmptyString(adapter.binding_key),
      `executor adapter ${adapterKey}.binding_key must be set`,
    );
    pushIf(
      errors,
      isNonEmptyString(adapter.executor_kind) &&
        isNonEmptyString(adapter.binding_key) &&
        adapterKey !== `${adapter.executor_kind}:${adapter.binding_key}`,
      `executor adapter ${adapterKey}.adapter_key must equal executor_kind:binding_key`,
    );
    validateEnumArray(
      errors,
      adapterKey,
      "supported_contracts",
      adapter.supported_contracts,
      new Set(contractMap.keys()),
    );
    for (const fieldName of [
      "supports_progress",
      "supports_cancel",
      "supports_resume",
      "supports_artifact",
    ]) {
      pushIf(
        errors,
        typeof adapter[fieldName] !== "boolean",
        `executor adapter ${adapterKey}.${fieldName} must be boolean`,
      );
    }
    validateEnumArray(
      errors,
      adapterKey,
      "artifact_output_kinds",
      adapter.artifact_output_kinds,
      ARTIFACT_KINDS,
    );
    validateEnumArray(
      errors,
      adapterKey,
      "permission_requirements",
      adapter.permission_requirements,
      PERMISSIONS,
    );
    validateStringArray(
      errors,
      adapterKey,
      "credential_requirements",
      adapter.credential_requirements,
      { allowEmpty: true },
    );
    validateEnumArray(
      errors,
      adapterKey,
      "failure_mapping",
      adapter.failure_mapping,
      FAILURE_REASONS,
    );
    validateEnumArray(
      errors,
      adapterKey,
      "evidence_events",
      adapter.evidence_events,
      EVIDENCE_EVENTS,
    );
    pushIf(
      errors,
      !isNonEmptyString(adapter.notes),
      `executor adapter ${adapterKey}.notes must be set`,
    );
  }

  const supportedContractsByProfile = new Set();
  for (const [profileKey, profile] of profileMap.entries()) {
    pushIf(
      errors,
      !LIFECYCLES.has(profile.lifecycle),
      `execution profile ${profileKey}.lifecycle is unknown: ${String(profile.lifecycle)}`,
    );
    validateEnumArray(
      errors,
      profileKey,
      "supported_contracts",
      profile.supported_contracts,
      new Set(contractMap.keys()),
    );
    validateEnumArray(
      errors,
      profileKey,
      "model_role_slots",
      profile.model_role_slots,
      matrixRefs.modelRoleSlots,
    );
    validateEnumArray(
      errors,
      profileKey,
      "permission_profile_keys",
      profile.permission_profile_keys,
      PERMISSIONS,
    );
    validateEnumArray(
      errors,
      profileKey,
      "executor_adapter_keys",
      profile.executor_adapter_keys,
      new Set(adapterMap.keys()),
    );
    validateArtifactPolicy(
      errors,
      profileKey,
      profile.artifact_policy,
      artifactGraphRefs,
    );
    validateEnumArray(
      errors,
      profileKey,
      "limecore_policy_refs",
      profile.limecore_policy_refs,
      LIMECORE_POLICY_REFS,
    );
    pushIf(
      errors,
      !isNonEmptyString(profile.user_lock_policy),
      `execution profile ${profileKey}.user_lock_policy must be set`,
    );
    validateStringArray(
      errors,
      profileKey,
      "fallback_behavior",
      profile.fallback_behavior,
    );
    validateEnumArray(
      errors,
      profileKey,
      "evidence_events",
      profile.evidence_events,
      EVIDENCE_EVENTS,
    );
    validateStringArray(
      errors,
      profileKey,
      "audit_fields",
      profile.audit_fields,
    );
    pushIf(
      errors,
      !isNonEmptyString(profile.notes),
      `execution profile ${profileKey}.notes must be set`,
    );

    for (const contractKey of profile.supported_contracts || []) {
      const contract = contractMap.get(contractKey);
      if (!contract) {
        continue;
      }
      supportedContractsByProfile.add(contractKey);
      const adapterKey = resolveExecutorAdapterKey(contract.executor_binding);
      validateSubset(
        errors,
        `execution profile ${profileKey}.model_role_slots`,
        profile.model_role_slots,
        [contract.routing_slot],
      );
      validateSubset(
        errors,
        `execution profile ${profileKey}.permission_profile_keys`,
        profile.permission_profile_keys,
        contract.permission_profile_keys,
      );
      validateSubset(
        errors,
        `execution profile ${profileKey}.limecore_policy_refs`,
        profile.limecore_policy_refs,
        contract.limecore_policy_refs,
      );
      validateSubset(
        errors,
        `execution profile ${profileKey}.artifact_policy.artifact_kinds`,
        profile.artifact_policy?.artifact_kinds,
        contract.artifact_kinds,
      );
      validateSubset(
        errors,
        `execution profile ${profileKey}.artifact_policy.viewer_surfaces`,
        profile.artifact_policy?.viewer_surfaces,
        contract.viewer_surface,
      );
      if (adapterKey) {
        validateSubset(
          errors,
          `execution profile ${profileKey}.executor_adapter_keys`,
          profile.executor_adapter_keys,
          [adapterKey],
        );
      }
      validateMetadataOnlyRouteExecution(
        errors,
        contractKey,
        contract,
        profile,
        adapterKey ? adapterMap.get(adapterKey) : null,
      );
      validateMetadataOnlyProfileAdapters(
        errors,
        contractKey,
        contract,
        profile,
        adapterMap,
      );
    }
  }

  for (const [contractKey, contract] of contractMap.entries()) {
    const adapterKey = resolveExecutorAdapterKey(contract.executor_binding);
    if (contract.lifecycle !== "current") {
      validateMetadataOnlyRouteExecution(
        errors,
        contractKey,
        contract,
        null,
        adapterKey ? adapterMap.get(adapterKey) : null,
      );
      continue;
    }
    pushIf(
      errors,
      !supportedContractsByProfile.has(contractKey),
      `current contract ${contractKey} must be covered by an execution profile`,
    );
    const adapter = adapterKey ? adapterMap.get(adapterKey) : null;
    pushIf(
      errors,
      !adapter,
      `current contract ${contractKey}.executor_binding must be defined in executionProfiles.executor_adapters: ${String(adapterKey)}`,
    );
    if (!adapter) {
      continue;
    }
    validateSubset(
      errors,
      `executor adapter ${adapterKey}.supported_contracts`,
      adapter.supported_contracts,
      [contractKey],
    );
    for (const fieldName of [
      "supports_progress",
      "supports_cancel",
      "supports_resume",
      "supports_artifact",
    ]) {
      pushIf(
        errors,
        adapter[fieldName] !== contract.executor_binding[fieldName],
        `current contract ${contractKey}.executor_binding.${fieldName} must match executor adapter ${adapterKey}`,
      );
    }
    validateSubset(
      errors,
      `executor adapter ${adapterKey}.artifact_output_kinds`,
      adapter.artifact_output_kinds,
      contract.artifact_kinds,
    );
    validateSubset(
      errors,
      `executor adapter ${adapterKey}.permission_requirements`,
      adapter.permission_requirements,
      contract.permission_profile_keys,
    );
    validateSubset(
      errors,
      `executor adapter ${adapterKey}.failure_mapping`,
      adapter.failure_mapping,
      contract.executor_binding.failure_mapping,
    );
  }

  return {
    errors,
    profileCount: profileMap.size,
    adapterCount: adapterMap.size,
  };
}

function validateContractRegistry(registry, matrixRefs, artifactGraphRefs) {
  const errors = [];
  pushIf(errors, registry.version !== 1, "registry.version must be 1");
  pushIf(
    errors,
    registry.status !== "current",
    "registry.status must be current",
  );
  pushIf(
    errors,
    !isNonEmptyString(registry.owner),
    "registry.owner must be set",
  );
  pushIf(
    errors,
    !isNonEmptyArray(registry.contracts),
    "registry.contracts must be a non-empty array",
  );
  if (!Array.isArray(registry.contracts)) {
    return errors;
  }

  const contractKeys = new Set();
  for (const contract of registry.contracts) {
    const contractKey = contract?.contract_key ?? "<missing>";
    pushIf(errors, !isPlainObject(contract), "each contract must be an object");
    if (!isPlainObject(contract)) {
      continue;
    }

    pushIf(
      errors,
      !isNonEmptyString(contract.contract_key),
      "contract.contract_key must be a non-empty string",
    );
    pushIf(
      errors,
      isNonEmptyString(contract.contract_key) &&
        looksLikeEntryKey(contract.contract_key),
      `${contractKey}.contract_key must be a bottom-layer key, not an entry key`,
    );
    pushIf(
      errors,
      contractKeys.has(contract.contract_key),
      `${contractKey}.contract_key is duplicated`,
    );
    contractKeys.add(contract.contract_key);

    pushIf(
      errors,
      !LIFECYCLES.has(contract.lifecycle),
      `${contractKey}.lifecycle is unknown: ${String(contract.lifecycle)}`,
    );
    pushIf(
      errors,
      !MODALITIES.has(contract.modality),
      `${contractKey}.modality is unknown: ${String(contract.modality)}`,
    );
    validateStringArray(
      errors,
      contractKey,
      "runtime_identity",
      contract.runtime_identity,
    );
    validateStringArray(
      errors,
      contractKey,
      "input_context_kinds",
      contract.input_context_kinds,
    );
    validateEnumArray(
      errors,
      contractKey,
      "required_capabilities",
      contract.required_capabilities,
      matrixRefs.capabilityKeys,
    );
    validateEnumArray(
      errors,
      contractKey,
      "permission_profile_keys",
      contract.permission_profile_keys,
      PERMISSIONS,
    );
    pushIf(
      errors,
      !isNonEmptyString(contract.routing_slot),
      `${contractKey}.routing_slot must be set`,
    );
    pushIf(
      errors,
      isNonEmptyString(contract.routing_slot) &&
        !matrixRefs.modelRoleSlots.has(contract.routing_slot),
      `${contractKey}.routing_slot is not defined in capability matrix: ${String(contract.routing_slot)}`,
    );
    validateExecutor(errors, contract);
    validateStringArray(
      errors,
      contractKey,
      "truth_source",
      contract.truth_source,
    );
    validateEnumArray(
      errors,
      contractKey,
      "artifact_kinds",
      contract.artifact_kinds,
      ARTIFACT_KINDS,
    );
    validateEnumArray(
      errors,
      contractKey,
      "viewer_surface",
      contract.viewer_surface,
      VIEWER_SURFACES,
    );
    validateEnumArray(
      errors,
      contractKey,
      "evidence_events",
      contract.evidence_events,
      EVIDENCE_EVENTS,
    );
    validateEnumArray(
      errors,
      contractKey,
      "limecore_policy_refs",
      contract.limecore_policy_refs,
      LIMECORE_POLICY_REFS,
    );
    validateStringArray(
      errors,
      contractKey,
      "fallback_policy",
      contract.fallback_policy,
    );

    pushIf(
      errors,
      !isPlainObject(contract.detour_policy),
      `${contractKey}.detour_policy must be an object`,
    );
    if (isPlainObject(contract.detour_policy)) {
      validateStringArray(
        errors,
        contractKey,
        "detour_policy.allowed",
        contract.detour_policy.allowed,
        { allowEmpty: true },
      );
      validateStringArray(
        errors,
        contractKey,
        "detour_policy.denied",
        contract.detour_policy.denied,
      );
    }
    pushIf(
      errors,
      !isNonEmptyString(contract.owner_surface),
      `${contractKey}.owner_surface must be set`,
    );
    validateEntryBindings(errors, contract);
    validateContractArtifactGraph(errors, contract, artifactGraphRefs);
  }

  return {
    errors,
    entryBindingReport: validatePhase7EntryBindingCoverage(errors, registry),
  };
}

function validateRequiredDocs() {
  return REQUIRED_DOCS.flatMap((docPath) => {
    const absolutePath = path.resolve(process.cwd(), docPath);
    if (fs.existsSync(absolutePath)) {
      return [];
    }
    return [`required roadmap document is missing: ${docPath}`];
  });
}

function renderSuccess(
  registry,
  matrix,
  graph,
  contractReport,
  profileReport,
) {
  const currentCount = registry.contracts.filter(
    (contract) => contract.lifecycle === "current",
  ).length;
  return [
    "[lime] modality runtime contracts OK",
    `  contracts: ${registry.contracts.length}`,
    `  current: ${currentCount}`,
    `  capabilities: ${matrix.capabilities.length}`,
    `  model roles: ${matrix.model_roles.length}`,
    `  artifact kinds: ${graph.artifact_kinds.length}`,
    `  entry bindings: ${contractReport.entryBindingReport.entryBindingCount}`,
    `  task index core fields: ${REQUIRED_ARTIFACT_INDEX_FIELDS.size}`,
    `  media phase8 index fields: ${REQUIRED_MEDIA_TASK_PHASE8_INDEX_FIELDS.size}`,
    `  execution profiles: ${profileReport.profileCount}`,
    `  executor adapters: ${profileReport.adapterCount}`,
    `  registry: ${CONTRACT_PATH}`,
    `  matrix: ${CAPABILITY_MATRIX_PATH}`,
    `  graph: ${ARTIFACT_GRAPH_PATH}`,
    `  profiles: ${EXECUTION_PROFILE_PATH}`,
  ].join("\n");
}

function main() {
  const registry = readJson(CONTRACT_PATH);
  const matrix = readJson(CAPABILITY_MATRIX_PATH);
  const graph = readJson(ARTIFACT_GRAPH_PATH);
  const profiles = readJson(EXECUTION_PROFILE_PATH);
  const matrixReport = validateCapabilityMatrix(matrix);
  const graphReport = validateArtifactGraph(graph);
  const profileReport = validateExecutionProfiles(
    registry,
    profiles,
    matrixReport,
    graphReport,
  );
  const contractReport = validateContractRegistry(
    registry,
    matrixReport,
    graphReport,
  );
  const errors = [
    ...validateRequiredDocs(),
    ...matrixReport.errors,
    ...graphReport.errors,
    ...contractReport.errors,
    ...profileReport.errors,
  ];

  if (errors.length > 0) {
    console.error("[lime] modality runtime contracts FAILED");
    for (const error of errors) {
      console.error(`  - ${error}`);
    }
    process.exit(1);
  }

  console.log(
    renderSuccess(
      registry,
      matrix,
      graph,
      contractReport,
      profileReport,
    ),
  );
}

main();
