const HASH_RE = /^sha256:[a-f0-9]{64}$/i;

function valueAtPath(root, parts) {
  let current = root;
  for (const part of parts) {
    if (!current || typeof current !== "object") return undefined;
    current = current[part];
  }
  return current;
}

function firstStringAtPaths(root, paths) {
  for (const parts of paths) {
    const value = valueAtPath(root, parts);
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function firstBoolAtPaths(root, paths) {
  for (const parts of paths) {
    const value = valueAtPath(root, parts);
    if (typeof value === "boolean") return value;
  }
  return false;
}

function firstNumberAtPaths(root, paths) {
  for (const parts of paths) {
    const value = valueAtPath(root, parts);
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function firstOptionalBoolAtPaths(root, paths) {
  for (const parts of paths) {
    const value = valueAtPath(root, parts);
    if (typeof value === "boolean") return value;
  }
  return undefined;
}

function firstArrayAtPaths(root, paths) {
  for (const parts of paths) {
    const value = valueAtPath(root, parts);
    if (Array.isArray(value)) return value;
  }
  return [];
}

function firstObjectAtPaths(root, paths) {
  for (const parts of paths) {
    const value = valueAtPath(root, parts);
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value;
    }
  }
  return null;
}

function stringField(root, keys) {
  for (const key of keys) {
    const value = root?.[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function acceptedStatus(status) {
  return new Set(["ready", "passed", "success", "ok", "completed"]).has(status);
}

function workflowAuditExported(status) {
  return status === "exported";
}

function validSha256(value) {
  return HASH_RE.test(value || "");
}

function isFixtureText(value) {
  return /fixture|localhost|127\.0\.0\.1|hostGenerationFixture|cloudReleaseFixture/i.test(
    String(value || ""),
  );
}

function stringValuesContainFixtureText(root, seen = new Set()) {
  if (typeof root === "string") {
    return isFixtureText(root);
  }
  if (!root || typeof root !== "object" || seen.has(root)) {
    return false;
  }
  seen.add(root);
  const values = Array.isArray(root) ? root : Object.values(root);
  return values.some((value) => stringValuesContainFixtureText(value, seen));
}

function workflowResumeCandidates(metadata) {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return [];
  }
  const workflowResume = metadata.workflowResume;
  return workflowResume &&
    typeof workflowResume === "object" &&
    !Array.isArray(workflowResume)
    ? [workflowResume]
    : [];
}

function workflowResumeActionResponseBindingFromObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const method = stringField(value, ["method"]);
  if (method !== "workflow/respond") {
    return null;
  }
  const actionId = stringField(value, [
    "actionId",
    "action_id",
    "requestId",
    "request_id",
  ]);
  const confirmed = value.confirmed ?? value.approved;
  const explicitDecision = stringField(value, ["decision"]);
  const decision =
    explicitDecision ||
    (confirmed === true ? "approved" : confirmed === false ? "rejected" : "");
  if (!value.metadata || typeof value.metadata !== "object") {
    return null;
  }
  for (const candidate of workflowResumeCandidates(value.metadata)) {
    const workflowRunId = stringField(candidate, [
      "workflowRunId",
      "workflow_run_id",
      "runId",
      "run_id",
    ]);
    const workflowKey = stringField(candidate, [
      "workflowKey",
      "workflow_key",
      "key",
      "workflow",
    ]);
    const stepId = stringField(candidate, ["stepId", "step_id", "id"]);
    if (workflowRunId && workflowKey && stepId) {
      return {
        actionId: actionId || null,
        decision: decision || null,
        stepId,
        workflowKey,
        workflowRunId,
      };
    }
  }
  return null;
}

function eventTypeFromRecord(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return "";
  }
  return (
    stringField(value, ["eventType", "event_type", "type", "kind"]) ||
    stringField(value.event, ["eventType", "event_type", "type", "kind"])
  );
}

function eventPayloadFromRecord(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  if (value.payload && typeof value.payload === "object") return value.payload;
  if (value.event?.payload && typeof value.event.payload === "object") {
    return value.event.payload;
  }
  return value;
}

function workflowResumeEventBinding(value) {
  const eventType = eventTypeFromRecord(value);
  if (
    eventType !== "workflow.step.resuming" &&
    eventType !== "workflow.run.resuming"
  ) {
    return null;
  }
  const payload = eventPayloadFromRecord(value);
  const workflowRunId = stringField(payload, [
    "workflowRunId",
    "workflow_run_id",
    "runId",
    "run_id",
  ]);
  const workflowKey = stringField(payload, [
    "workflowKey",
    "workflow_key",
    "key",
    "workflow",
  ]);
  const stepId = stringField(payload, ["stepId", "step_id", "id"]);
  const actionId = stringField(payload, ["actionId", "action_id"]);
  const decision = stringField(payload, ["decision"]);
  if (!workflowRunId || !workflowKey || !stepId || !actionId || !decision) {
    return null;
  }
  return {
    actionId,
    decision,
    eventType,
    stepId,
    workflowKey,
    workflowRunId,
  };
}

function collectWorkflowResumeEventBindings(root) {
  const bindings = [];
  const seen = new Set();
  function walk(value) {
    if (!value || typeof value !== "object" || seen.has(value)) return;
    seen.add(value);
    const binding = workflowResumeEventBinding(value);
    if (binding) bindings.push(binding);
    const values = Array.isArray(value) ? value : Object.values(value);
    for (const item of values) {
      walk(item);
    }
  }
  walk(root);
  return bindings;
}

function collectWorkflowResumeActionMetadataBindings(guiEvidence) {
  const binding = workflowResumeActionResponseBindingFromObject(
    guiEvidence?.runtimeActionResponse,
  );
  return binding?.actionId && binding?.decision ? [binding] : [];
}

function workflowResumeEventBindingsForAction(actionBinding, eventBindings) {
  if (!actionBinding) return [];
  return eventBindings.filter(
    (binding) =>
      binding.actionId === actionBinding.actionId &&
      binding.decision === actionBinding.decision &&
      binding.stepId === actionBinding.stepId &&
      binding.workflowKey === actionBinding.workflowKey &&
      binding.workflowRunId === actionBinding.workflowRunId,
  );
}

function workflowResumeAuditEventsPresent(bindings) {
  const hasStepResuming = bindings.some(
    (binding) => binding.eventType === "workflow.step.resuming",
  );
  const hasRunResuming = bindings.some(
    (binding) => binding.eventType === "workflow.run.resuming",
  );
  return hasStepResuming && hasRunResuming;
}

function summarizeWorkflowResumeLifecycle(guiEvidence) {
  const actionBindings =
    collectWorkflowResumeActionMetadataBindings(guiEvidence);
  const eventBindings = collectWorkflowResumeEventBindings(guiEvidence);
  const matchedActionBinding =
    actionBindings.find((binding) =>
      workflowResumeAuditEventsPresent(
        workflowResumeEventBindingsForAction(binding, eventBindings),
      ),
    ) || actionBindings[0];
  const matchingEventBindings = workflowResumeEventBindingsForAction(
    matchedActionBinding,
    eventBindings,
  );
  const auditEventsPresent = workflowResumeAuditEventsPresent(
    matchingEventBindings,
  );
  return {
    actionId: matchedActionBinding?.actionId || null,
    actionMetadataPresent: actionBindings.length > 0,
    auditEventsPresent,
    decision: matchedActionBinding?.decision || null,
    stepId: matchedActionBinding?.stepId || null,
    workflowKey: matchedActionBinding?.workflowKey || null,
    workflowRunId: matchedActionBinding?.workflowRunId || null,
  };
}

function pushRequirement(missingRequirements, code, detail) {
  if (missingRequirements.some((item) => item?.code === code)) return;
  missingRequirements.push({ code, detail });
}

export function summarizeGuiEvidence(guiEvidence) {
  const json = JSON.stringify(guiEvidence || {});
  const schemaVersion = firstStringAtPaths(guiEvidence || {}, [
    ["schemaVersion"],
  ]);
  const status = firstStringAtPaths(guiEvidence || {}, [
    ["status"],
    ["gui", "status"],
  ]);
  const cdpAttached = firstBoolAtPaths(guiEvidence || {}, [
    ["cdp", "attached"],
  ]);
  const cdpUsedRealElectron = firstBoolAtPaths(guiEvidence || {}, [
    ["cdp", "usedRealElectron"],
  ]);
  const appVersion = firstStringAtPaths(guiEvidence || {}, [
    ["installedState", "appVersion"],
    ["installed_state", "app_version"],
    ["cloudRelease", "appVersion"],
    ["cloudRelease", "app_version"],
    ["identity", "appVersion"],
    ["identity", "app_version"],
  ]);
  const sourceKind = firstStringAtPaths(guiEvidence || {}, [
    ["installedState", "sourceKind"],
    ["installed_state", "source_kind"],
    ["cloudRelease", "sourceKind"],
    ["cloudReleaseFixture", "sourceKind"],
  ]);
  const packageHash = firstStringAtPaths(guiEvidence || {}, [
    ["installedState", "packageHash"],
    ["installed_state", "package_hash"],
    ["cloudRelease", "packageHash"],
    ["cloudRelease", "package_hash"],
    ["identity", "packageHash"],
    ["identity", "package_hash"],
    ["setup", "cloudReleaseEvidence", "packageHash"],
  ]);
  const manifestHash = firstStringAtPaths(guiEvidence || {}, [
    ["installedState", "manifestHash"],
    ["installed_state", "manifest_hash"],
    ["cloudRelease", "manifestHash"],
    ["cloudRelease", "manifest_hash"],
    ["identity", "manifestHash"],
    ["identity", "manifest_hash"],
    ["setup", "cloudReleaseEvidence", "manifestHash"],
  ]);
  const releaseId = firstStringAtPaths(guiEvidence || {}, [
    ["installedState", "releaseId"],
    ["installed_state", "release_id"],
    ["cloudRelease", "releaseId"],
    ["cloudRelease", "release_id"],
    ["identity", "releaseId"],
    ["identity", "release_id"],
  ]);
  const signatureRef = firstStringAtPaths(guiEvidence || {}, [
    ["installedState", "signatureRef"],
    ["installed_state", "signature_ref"],
    ["cloudRelease", "signatureRef"],
    ["cloudRelease", "signature_ref"],
    ["identity", "signatureRef"],
    ["identity", "signature_ref"],
    ["setup", "cloudReleaseEvidence", "signatureRef"],
  ]);
  const signaturePolicy = firstStringAtPaths(guiEvidence || {}, [
    ["signaturePolicy"],
    ["signature_policy"],
    ["installedState", "signaturePolicy"],
    ["installed_state", "signature_policy"],
    ["cloudRelease", "signaturePolicy"],
    ["cloudRelease", "signature_policy"],
    ["cloudReleaseFixture", "signaturePolicy"],
    ["setup", "cloudReleaseEvidence", "signaturePolicy"],
  ]);
  const signatureVerificationStatus = firstStringAtPaths(guiEvidence || {}, [
    ["signatureVerificationStatus"],
    ["signature_verification_status"],
    ["cloudRelease", "signatureVerificationStatus"],
    ["cloudRelease", "signature_verification_status"],
    ["cloudReleaseFixture", "signatureVerificationStatus"],
    ["installedState", "signatureVerificationStatus"],
    ["installed_state", "signature_verification_status"],
    ["setup", "cloudReleaseEvidence", "signatureVerificationStatus"],
  ]);
  const cloudReleaseEvidenceStatus = firstStringAtPaths(guiEvidence || {}, [
    ["cloudReleaseEvidenceStatus"],
    ["cloud_release_evidence_status"],
    ["evidenceStatus"],
    ["evidence_status"],
    ["cloudRelease", "status"],
    ["cloudReleaseEvidence", "status"],
    ["cloudReleaseFixture", "evidenceStatus"],
    ["installedState", "cloudReleaseEvidenceStatus"],
    ["installedState", "evidenceStatus"],
    ["installed_state", "cloud_release_evidence_status"],
    ["setup", "cloudReleaseEvidence", "status"],
  ]);
  const packageVerificationStatus = firstStringAtPaths(guiEvidence || {}, [
    ["packageVerificationStatus"],
    ["package_verification_status"],
    ["cloudRelease", "packageVerificationStatus"],
    ["cloudRelease", "package_verification_status"],
    ["cloudReleaseFixture", "packageVerificationStatus"],
    ["installedState", "packageVerificationStatus"],
    ["installed_state", "package_verification_status"],
    ["setup", "cloudReleaseEvidence", "packageVerificationStatus"],
  ]);
  const packageHashMatched =
    firstOptionalBoolAtPaths(guiEvidence || {}, [
      ["packageHashMatched"],
      ["package_hash_matched"],
      ["cloudRelease", "packageHashMatched"],
      ["cloudRelease", "package_hash_matched"],
      ["cloudReleaseFixture", "packageHashMatched"],
      ["installedState", "packageHashMatched"],
      ["installed_state", "package_hash_matched"],
      ["setup", "cloudReleaseEvidence", "packageHashMatched"],
    ]) === true;
  const manifestHashMatched =
    firstOptionalBoolAtPaths(guiEvidence || {}, [
      ["manifestHashMatched"],
      ["manifest_hash_matched"],
      ["cloudRelease", "manifestHashMatched"],
      ["cloudRelease", "manifest_hash_matched"],
      ["cloudReleaseFixture", "manifestHashMatched"],
      ["installedState", "manifestHashMatched"],
      ["installed_state", "manifest_hash_matched"],
      ["setup", "cloudReleaseEvidence", "manifestHashMatched"],
    ]) === true;
  const hostManagedGenerationStatus = firstStringAtPaths(guiEvidence || {}, [
    ["readModel", "hostManagedGenerationStatus"],
    ["currentTurn", "hostManagedGenerationStatus"],
    ["threadRead", "hostManagedGenerationStatus"],
    [
      "contentFactoryArticleWorkspaceReadModel",
      "workerArticleObject",
      "hostManagedGenerationStatus",
    ],
  ]);
  const workflowFactsHidden =
    firstOptionalBoolAtPaths(guiEvidence || {}, [
      ["contentFactoryArticleWorkspaceWorkflowFactsHidden"],
      ["assertions", "contentFactoryArticleWorkspaceWorkflowFactsHidden"],
    ]) === true;
  const workflowJsonlPath = firstStringAtPaths(guiEvidence || {}, [
    ["eventLogs", "workflowJsonl"],
    ["workflowJsonl"],
    ["workflowJsonlPath"],
  ]);
  const workflowJsonlEventCount =
    firstNumberAtPaths(guiEvidence || {}, [
      ["eventLogs", "workflowJsonlEventCount"],
      ["workflowJsonlEventCount"],
    ]) ?? 0;
  const workflowJsonlPresent = Boolean(
    workflowJsonlPath && workflowJsonlEventCount > 0,
  );
  const workflowAuditStatus = firstStringAtPaths(guiEvidence || {}, [
    ["evidenceExport", "workflowAudit", "status"],
    ["observabilitySummary", "workflow_audit", "status"],
    ["evidencePack", "observabilitySummary", "workflow_audit", "status"],
  ]);
  const workflowAuditSource = firstStringAtPaths(guiEvidence || {}, [
    ["evidenceExport", "workflowAudit", "source"],
    ["observabilitySummary", "workflow_audit", "source"],
    ["evidencePack", "observabilitySummary", "workflow_audit", "source"],
  ]);
  const workflowAuditEventCount =
    firstNumberAtPaths(guiEvidence || {}, [
      ["evidenceExport", "workflowAudit", "eventCount"],
      ["observabilitySummary", "workflow_audit", "eventCount"],
      ["evidencePack", "observabilitySummary", "workflow_audit", "eventCount"],
    ]) ?? 0;
  const workflowAuditMetadataOnly =
    firstOptionalBoolAtPaths(guiEvidence || {}, [
      ["evidenceExport", "workflowAudit", "metadataOnly"],
      ["observabilitySummary", "workflow_audit", "metadataOnly"],
      [
        "evidencePack",
        "observabilitySummary",
        "workflow_audit",
        "metadataOnly",
      ],
    ]) === true;
  const workflowAuditRawContentIncluded = firstOptionalBoolAtPaths(
    guiEvidence || {},
    [
      ["evidenceExport", "workflowAudit", "rawContentIncluded"],
      ["observabilitySummary", "workflow_audit", "rawContentIncluded"],
      [
        "evidencePack",
        "observabilitySummary",
        "workflow_audit",
        "rawContentIncluded",
      ],
    ],
  );
  const workflowAuditRawContentExcluded =
    workflowAuditRawContentIncluded === false;
  const workflowAuditRedactionPolicy = firstStringAtPaths(guiEvidence || {}, [
    ["evidenceExport", "workflowAudit", "redactionPolicy"],
    ["observabilitySummary", "workflow_audit", "redactionPolicy"],
    [
      "evidencePack",
      "observabilitySummary",
      "workflow_audit",
      "redactionPolicy",
    ],
  ]);
  const workflowAuditRedactionPolicyEventCount =
    firstNumberAtPaths(guiEvidence || {}, [
      ["evidenceExport", "workflowAudit", "redactionPolicyEventCount"],
      ["observabilitySummary", "workflow_audit", "redactionPolicyEventCount"],
      [
        "evidencePack",
        "observabilitySummary",
        "workflow_audit",
        "redactionPolicyEventCount",
      ],
    ]) ?? 0;
  const workflowAuditExportReady =
    workflowAuditExported(workflowAuditStatus) &&
    workflowAuditSource === "workflow-events.jsonl" &&
    workflowAuditEventCount > 0 &&
    workflowAuditMetadataOnly &&
    workflowAuditRawContentExcluded &&
    workflowAuditRedactionPolicy === "workflow_audit_metadata_only" &&
    workflowAuditRedactionPolicyEventCount > 0;
  const generatedArticleMarkerClean =
    firstOptionalBoolAtPaths(guiEvidence || {}, [
      ["readModel", "generatedArticleMarkerClean"],
      ["assertions", "generatedArticleMarkerClean"],
    ]) === true;
  const articleDraftDocumentLength =
    firstNumberAtPaths(guiEvidence || {}, [
      ["readModel", "articleDraftDocumentLength"],
      ["articleDraftDocumentLength"],
    ]) ?? 0;
  const articleDraftDocumentPresent =
    articleDraftDocumentLength > 0 ||
    firstBoolAtPaths(guiEvidence || {}, [
      ["readModel", "articleDraftDocumentPresent"],
      ["assertions", "articleDraftDocumentPresent"],
    ]);
  const liveProviderUsed =
    firstBoolAtPaths(guiEvidence || {}, [
      ["liveProviderUsed"],
      ["providerEvidence", "liveProviderUsed"],
      ["assertions", "liveProviderUsed"],
    ]) === true;
  const turnStartViaElectronIpc = firstBoolAtPaths(guiEvidence || {}, [
    ["assertions", "turnStartViaElectronIpc"],
    ["runtime", "turnStartViaElectronIpc"],
    ["trace", "turnStartViaElectronIpc"],
  ]);
  const turnStartTrace = firstObjectAtPaths(guiEvidence || {}, [
    ["trace", "turnStartTrace"],
  ]);
  const turnStartTraceMatched = Boolean(
    turnStartTrace?.matched === true &&
    turnStartTrace?.sessionMatched === true &&
    turnStartTrace?.command === "app_server_handle_json_lines" &&
    turnStartTrace?.transport === "electron-ipc" &&
    turnStartTrace?.status === "success" &&
    turnStartTrace?.method === "turn/start",
  );
  const appServerHandleJsonLinesSeen = firstBoolAtPaths(guiEvidence || {}, [
    ["trace", "appServerHandleJsonLinesSeen"],
    ["assertions", "appServerHandleJsonLinesSeen"],
  ]);
  const appServerMethodsSeen = firstArrayAtPaths(guiEvidence || {}, [
    ["trace", "appServerMethodsSeen"],
    ["assertions", "appServerMethodsSeen"],
  ]).filter((value) => typeof value === "string" && value.trim());
  const currentAppServerMethodsSeen =
    appServerMethodsSeen.includes("thread/read") &&
    appServerMethodsSeen.includes("evidence/export") &&
    appServerMethodsSeen.includes("turn/start");
  const hostGenerationFixture = firstObjectAtPaths(guiEvidence || {}, [
    ["hostGenerationFixture"],
    ["host_generation_fixture"],
  ]);
  const cloudReleaseFixture = firstObjectAtPaths(guiEvidence || {}, [
    ["cloudReleaseFixture"],
    ["cloud_release_fixture"],
  ]);
  const fixtureLike = Boolean(
    hostGenerationFixture ||
    cloudReleaseFixture ||
    stringValuesContainFixtureText(guiEvidence || {}),
  );
  const workflowResumeLifecycle = summarizeWorkflowResumeLifecycle(guiEvidence);
  const cloudReleaseRuntimeVerified =
    sourceKind === "cloud_release" &&
    validSha256(packageHash) &&
    validSha256(manifestHash) &&
    Boolean(releaseId) &&
    Boolean(signatureRef) &&
    signaturePolicy === "required" &&
    signatureVerificationStatus === "verified" &&
    cloudReleaseEvidenceStatus === "ready" &&
    packageVerificationStatus === "verified" &&
    packageHashMatched &&
    manifestHashMatched;
  return {
    articleDraftDocumentPresent,
    articleDraftDocumentLength,
    appVersion: appVersion || null,
    cloudReleaseEvidenceStatus: cloudReleaseEvidenceStatus || null,
    cloudReleaseRuntimeVerified,
    cdpAttached,
    cdpUsedRealElectron,
    collectorSchemaValid:
      schemaVersion === "content-factory-production-gui-evidence.v1",
    currentAppServerMethodsSeen,
    fixtureLike,
    generatedArticleMarkerClean,
    hostManagedGenerationStatus: hostManagedGenerationStatus || null,
    liveProviderUsed,
    ready:
      acceptedStatus(status) &&
      schemaVersion === "content-factory-production-gui-evidence.v1" &&
      cdpAttached &&
      cdpUsedRealElectron &&
      cloudReleaseRuntimeVerified &&
      hostManagedGenerationStatus === "completed" &&
      articleDraftDocumentPresent &&
      generatedArticleMarkerClean &&
      workflowFactsHidden &&
      workflowJsonlPresent &&
      workflowAuditExportReady &&
      workflowResumeLifecycle.actionMetadataPresent &&
      workflowResumeLifecycle.auditEventsPresent &&
      liveProviderUsed &&
      turnStartViaElectronIpc &&
      turnStartTraceMatched &&
      appServerHandleJsonLinesSeen &&
      currentAppServerMethodsSeen &&
      !fixtureLike,
    appServerHandleJsonLinesSeen,
    manifestHashMatched,
    manifestHash: validSha256(manifestHash) ? manifestHash : null,
    packageHashMatched,
    packageHash: validSha256(packageHash) ? packageHash : null,
    packageVerificationStatus: packageVerificationStatus || null,
    releaseId: releaseId || null,
    signaturePolicy: signaturePolicy || null,
    signatureRef: signatureRef || null,
    signatureVerificationStatus: signatureVerificationStatus || null,
    sourceKind: sourceKind || null,
    status: status || "missing",
    statusReady: acceptedStatus(status),
    turnStartViaElectronIpc,
    turnStartTraceMatched,
    workflowAuditEventCount,
    workflowAuditExportReady,
    workflowAuditMetadataOnly,
    workflowAuditRawContentExcluded,
    workflowAuditRedactionPolicy: workflowAuditRedactionPolicy || null,
    workflowAuditRedactionPolicyEventCount,
    workflowAuditSource: workflowAuditSource || null,
    workflowAuditStatus: workflowAuditStatus || null,
    workflowFactsHidden,
    workflowJsonlEventCount,
    workflowJsonlPresent,
    workflowResumeLifecycle,
  };
}

export function appendGuiEvidenceRequirements(
  missingRequirements,
  inputGuiEvidence,
  guiEvidence,
  catalog = null,
  preflight = null,
  fetchCloud = null,
) {
  if (!inputGuiEvidence) {
    pushRequirement(
      missingRequirements,
      "production_gui_evidence_missing",
      "Real Lime Desktop GUI install/run evidence is required.",
    );
    return;
  }
  if (!guiEvidence.statusReady) {
    pushRequirement(
      missingRequirements,
      "production_gui_evidence_not_ready",
      "GUI evidence status must be passed/ready/completed.",
    );
  }
  if (guiEvidence.sourceKind !== "cloud_release") {
    pushRequirement(
      missingRequirements,
      "production_gui_not_cloud_release",
      "GUI evidence must run the cloud_release installed app.",
    );
  }
  if (!guiEvidence.packageHash) {
    pushRequirement(
      missingRequirements,
      "production_gui_package_hash_missing",
      "GUI evidence must include the installed cloud_release packageHash.",
    );
  }
  if (!guiEvidence.manifestHash) {
    pushRequirement(
      missingRequirements,
      "production_gui_manifest_hash_missing",
      "GUI evidence must include the installed cloud_release manifestHash.",
    );
  }
  if (!guiEvidence.releaseId) {
    pushRequirement(
      missingRequirements,
      "production_gui_release_id_missing",
      "GUI evidence must include the installed cloud_release releaseId.",
    );
  }
  if (!guiEvidence.signatureRef) {
    pushRequirement(
      missingRequirements,
      "production_gui_signature_ref_missing",
      "GUI evidence must include the installed cloud_release signatureRef.",
    );
  }
  if (
    guiEvidence.appVersion &&
    catalog?.version &&
    guiEvidence.appVersion !== catalog.version
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_catalog_version_mismatch",
      "GUI evidence appVersion must match the production catalog version.",
    );
  }
  if (
    guiEvidence.packageHash &&
    catalog?.packageHash &&
    guiEvidence.packageHash !== catalog.packageHash
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_catalog_package_hash_mismatch",
      "GUI evidence packageHash must match the production catalog packageHash.",
    );
  }
  if (
    guiEvidence.manifestHash &&
    catalog?.manifestHash &&
    guiEvidence.manifestHash !== catalog.manifestHash
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_catalog_manifest_hash_mismatch",
      "GUI evidence manifestHash must match the production catalog manifestHash.",
    );
  }
  if (
    guiEvidence.releaseId &&
    catalog?.releaseId &&
    guiEvidence.releaseId !== catalog.releaseId
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_catalog_release_id_mismatch",
      "GUI evidence releaseId must match the production catalog releaseId.",
    );
  }
  if (
    guiEvidence.signatureRef &&
    catalog?.signatureRef &&
    guiEvidence.signatureRef !== catalog.signatureRef
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_catalog_signature_ref_mismatch",
      "GUI evidence signatureRef must match the production catalog signatureRef.",
    );
  }
  if (
    guiEvidence.packageHash &&
    preflight?.packageHash &&
    guiEvidence.packageHash !== preflight.packageHash
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_preflight_package_hash_mismatch",
      "GUI evidence packageHash must match the production preflight packageHash.",
    );
  }
  if (
    guiEvidence.manifestHash &&
    preflight?.manifestHash &&
    guiEvidence.manifestHash !== preflight.manifestHash
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_preflight_manifest_hash_mismatch",
      "GUI evidence manifestHash must match the production preflight manifestHash.",
    );
  }
  if (
    guiEvidence.signatureRef &&
    preflight?.signatureRef &&
    guiEvidence.signatureRef !== preflight.signatureRef
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_preflight_signature_ref_mismatch",
      "GUI evidence signatureRef must match the production preflight app.signature.yaml signatureRef.",
    );
  }
  if (
    guiEvidence.packageHash &&
    fetchCloud?.packageHash &&
    guiEvidence.packageHash !== fetchCloud.packageHash
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_fetch_cloud_package_hash_mismatch",
      "GUI evidence packageHash must match fetchCloud packageHash.",
    );
  }
  if (
    guiEvidence.manifestHash &&
    fetchCloud?.manifestHash &&
    guiEvidence.manifestHash !== fetchCloud.manifestHash
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_fetch_cloud_manifest_hash_mismatch",
      "GUI evidence manifestHash must match fetchCloud manifestHash.",
    );
  }
  if (
    guiEvidence.signatureRef &&
    fetchCloud?.signatureRef &&
    guiEvidence.signatureRef !== fetchCloud.signatureRef
  ) {
    pushRequirement(
      missingRequirements,
      "production_gui_fetch_cloud_signature_ref_mismatch",
      "GUI evidence signatureRef must match fetchCloud signatureRef.",
    );
  }
  if (guiEvidence.signaturePolicy !== "required") {
    pushRequirement(
      missingRequirements,
      "production_gui_signature_policy_not_required",
      "GUI evidence must prove cloud_release signaturePolicy is required before worker launch.",
    );
  }
  if (guiEvidence.signatureVerificationStatus !== "verified") {
    pushRequirement(
      missingRequirements,
      "production_gui_signature_not_verified",
      "GUI evidence must prove signature verification status is verified.",
    );
  }
  if (guiEvidence.cloudReleaseEvidenceStatus !== "ready") {
    pushRequirement(
      missingRequirements,
      "production_gui_release_evidence_not_ready",
      "GUI evidence must prove installed cloudReleaseEvidence.status is ready.",
    );
  }
  if (guiEvidence.packageVerificationStatus !== "verified") {
    pushRequirement(
      missingRequirements,
      "production_gui_package_verification_not_verified",
      "GUI evidence must prove packageVerificationStatus is verified.",
    );
  }
  if (!guiEvidence.packageHashMatched) {
    pushRequirement(
      missingRequirements,
      "production_gui_package_hash_not_matched",
      "GUI evidence must prove the installed package hash matched the catalog descriptor.",
    );
  }
  if (!guiEvidence.manifestHashMatched) {
    pushRequirement(
      missingRequirements,
      "production_gui_manifest_hash_not_matched",
      "GUI evidence must prove the installed manifest hash matched the catalog descriptor.",
    );
  }
  if (guiEvidence.hostManagedGenerationStatus !== "completed") {
    pushRequirement(
      missingRequirements,
      "production_host_generation_not_completed",
      "GUI/read model must show hostManagedGeneration completed.",
    );
  }
  if (!guiEvidence.collectorSchemaValid) {
    pushRequirement(
      missingRequirements,
      "production_gui_collector_schema_missing",
      "GUI evidence must be produced by the production GUI evidence collector schema.",
    );
  }
  if (!guiEvidence.cdpAttached || !guiEvidence.cdpUsedRealElectron) {
    pushRequirement(
      missingRequirements,
      "production_gui_cdp_evidence_missing",
      "GUI evidence must prove it was collected from a real Electron renderer through CDP.",
    );
  }
  if (!guiEvidence.generatedArticleMarkerClean) {
    pushRequirement(
      missingRequirements,
      "production_generated_article_marker_unclean",
      "GUI read model must prove the generated article is not fixture-marked.",
    );
  }
  if (!guiEvidence.liveProviderUsed || guiEvidence.fixtureLike) {
    pushRequirement(
      missingRequirements,
      "production_host_generation_not_live",
      "Production evidence must prove live Provider generation, not a localhost fixture.",
    );
  }
  if (guiEvidence.fixtureLike) {
    pushRequirement(
      missingRequirements,
      "fixture_cloud_release_not_allowed",
      "GUI evidence contains fixture/localhost markers and cannot close production.",
    );
  }
  if (!guiEvidence.workflowJsonlPresent) {
    pushRequirement(
      missingRequirements,
      "production_workflow_jsonl_missing",
      "GUI evidence must include workflow-events.jsonl audit output with at least one parsed event.",
    );
  }
  if (!guiEvidence.workflowAuditExportReady) {
    pushRequirement(
      missingRequirements,
      "production_workflow_audit_export_missing",
      "GUI evidence must include App Server evidence/export workflow_audit summary from workflow-events.jsonl.",
    );
  }
  if (guiEvidence.workflowAuditEventCount <= 0) {
    pushRequirement(
      missingRequirements,
      "production_workflow_audit_export_empty",
      "App Server evidence/export workflow_audit summary must include at least one workflow audit event.",
    );
  }
  if (!guiEvidence.workflowAuditMetadataOnly) {
    pushRequirement(
      missingRequirements,
      "production_workflow_audit_not_metadata_only",
      "App Server evidence/export workflow_audit summary must prove metadataOnly=true.",
    );
  }
  if (!guiEvidence.workflowAuditRawContentExcluded) {
    pushRequirement(
      missingRequirements,
      "production_workflow_audit_raw_content_included",
      "App Server evidence/export workflow_audit summary must prove rawContentIncluded=false.",
    );
  }
  if (
    guiEvidence.workflowAuditRedactionPolicy !==
      "workflow_audit_metadata_only" ||
    guiEvidence.workflowAuditRedactionPolicyEventCount <= 0
  ) {
    pushRequirement(
      missingRequirements,
      "production_workflow_audit_redaction_policy_missing",
      "App Server evidence/export workflow_audit summary must prove workflow_audit_metadata_only redaction coverage.",
    );
  }
  if (
    !guiEvidence.workflowResumeLifecycle.actionMetadataPresent ||
    !guiEvidence.workflowResumeLifecycle.auditEventsPresent
  ) {
    pushRequirement(
      missingRequirements,
      "production_workflow_resume_lifecycle_missing",
      "GUI evidence must prove typed action response metadata with workflowResume and workflow.step/run.resuming audit events.",
    );
  }
  if (!guiEvidence.workflowFactsHidden) {
    pushRequirement(
      missingRequirements,
      "production_workflow_facts_visible",
      "Right-side Article Editor must not display workflow steps.",
    );
  }
  if (!guiEvidence.turnStartViaElectronIpc) {
    pushRequirement(
      missingRequirements,
      "production_gui_turn_start_not_electron_ipc",
      "GUI evidence must prove turn/start went through Electron IPC.",
    );
  }
  if (!guiEvidence.turnStartTraceMatched) {
    pushRequirement(
      missingRequirements,
      "production_gui_turn_start_trace_missing",
      "GUI evidence must include a matched Electron IPC turn-start trace for the target session.",
    );
  }
  if (!guiEvidence.appServerHandleJsonLinesSeen) {
    pushRequirement(
      missingRequirements,
      "production_gui_app_server_json_rpc_missing",
      "GUI evidence must prove app_server_handle_json_lines was used for current App Server JSON-RPC.",
    );
  }
  if (!guiEvidence.currentAppServerMethodsSeen) {
    pushRequirement(
      missingRequirements,
      "production_gui_current_app_server_methods_missing",
      "GUI evidence must include current App Server method trace for turn/start, read, and evidence/export.",
    );
  }
  if (!guiEvidence.articleDraftDocumentPresent) {
    pushRequirement(
      missingRequirements,
      "production_article_draft_document_missing",
      "GUI/read model must include article-draft-document output.",
    );
  }
}
