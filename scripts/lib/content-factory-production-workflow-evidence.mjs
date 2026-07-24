import fs from "node:fs";

function isRecord(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readString(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function parseJson(raw, fallback = null) {
  try {
    return JSON.parse(String(raw ?? ""));
  } catch {
    return fallback;
  }
}

function workflowResumeCandidates(metadata) {
  if (!isRecord(metadata)) return [];
  return isRecord(metadata.workflowResume) ? [metadata.workflowResume] : [];
}

export function workflowResumeEvidenceFromMetadata(metadata) {
  for (const candidate of workflowResumeCandidates(metadata)) {
    const workflowRunId = readString(
      candidate.workflowRunId,
      candidate.workflow_run_id,
      candidate.runId,
      candidate.run_id,
    );
    const workflowKey = readString(
      candidate.workflowKey,
      candidate.workflow_key,
      candidate.key,
      candidate.workflow,
    );
    const stepId = readString(
      candidate.stepId,
      candidate.step_id,
      candidate.id,
    );
    if (workflowRunId && workflowKey && stepId) {
      return { stepId, workflowKey, workflowRunId };
    }
  }
  return null;
}

export function projectAppServerParamsForEvidence(params = {}) {
  if (!isRecord(params)) return {};
  return {
    actionId: readString(params.actionId, params.action_id) || null,
    approved:
      typeof params.approved === "boolean" ? params.approved : undefined,
    confirmed:
      typeof params.confirmed === "boolean" ? params.confirmed : undefined,
    decision: readString(params.decision) || null,
    requestId: readString(params.requestId, params.request_id) || null,
    sessionId: readString(params.sessionId, params.session_id) || null,
    threadId: readString(params.threadId) || null,
    turnId: readString(params.turnId, params.turn_id) || null,
    workflowResume: workflowResumeEvidenceFromMetadata(params.metadata),
  };
}

function workflowResumeBindingFromRequest(request) {
  if (!isRecord(request?.params)) return null;
  const params = request.params;
  const actionId = readString(
    params.request_id,
    params.requestId,
    params.actionId,
  );
  const explicitDecision = readString(params.decision);
  const confirmed = params.confirmed ?? params.approved;
  const decision =
    explicitDecision ||
    (confirmed === true ? "approved" : confirmed === false ? "rejected" : "");
  for (const candidate of workflowResumeCandidates(params.metadata)) {
    const workflowRunId = readString(
      candidate.workflowRunId,
      candidate.workflow_run_id,
      candidate.runId,
      candidate.run_id,
    );
    const workflowKey = readString(
      candidate.workflowKey,
      candidate.workflow_key,
      candidate.key,
      candidate.workflow,
    );
    const stepId = readString(
      candidate.stepId,
      candidate.step_id,
      candidate.id,
    );
    if (workflowRunId && workflowKey && stepId && actionId && decision) {
      return {
        actionId,
        decision,
        method: request.method,
        stepId,
        workflowKey,
        workflowRunId,
      };
    }
  }
  return null;
}

export function workflowResumeBindingsFromTrace(traceEntries) {
  return traceEntries
    .flatMap((entry) => entry.appServerRequests || [])
    .filter((request) => request.method === "workflow/respond")
    .map(workflowResumeBindingFromRequest)
    .filter(Boolean);
}

function eventTypeFromRecord(record) {
  return readString(
    record?.eventType,
    record?.event_type,
    record?.type,
    record?.kind,
    record?.event?.eventType,
    record?.event?.event_type,
    record?.event?.type,
  );
}

function eventPayloadFromRecord(record) {
  if (isRecord(record?.payload)) return record.payload;
  if (isRecord(record?.event?.payload)) return record.event.payload;
  return isRecord(record) ? record : {};
}

export function workflowResumeEventBinding(record) {
  const eventType = eventTypeFromRecord(record);
  if (
    eventType !== "workflow.step.resuming" &&
    eventType !== "workflow.run.resuming"
  ) {
    return null;
  }
  const payload = eventPayloadFromRecord(record);
  const binding = {
    actionId: readString(payload.actionId, payload.action_id),
    decision: readString(payload.decision),
    eventType,
    stepId: readString(payload.stepId, payload.step_id, payload.id),
    workflowKey: readString(
      payload.workflowKey,
      payload.workflow_key,
      payload.key,
      payload.workflow,
    ),
    workflowRunId: readString(
      payload.workflowRunId,
      payload.workflow_run_id,
      payload.runId,
      payload.run_id,
    ),
  };
  return Object.values(binding).every(Boolean) ? binding : null;
}

export function readWorkflowJsonlEvents(filePath) {
  if (!filePath) return [];
  if (!fs.existsSync(filePath)) return [];
  return fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => parseJson(line, null))
    .filter(Boolean);
}

function workflowResumeEventsForBinding(binding, eventBindings) {
  if (!binding) return [];
  return eventBindings.filter(
    (event) =>
      event.actionId === binding.actionId &&
      event.decision === binding.decision &&
      event.stepId === binding.stepId &&
      event.workflowKey === binding.workflowKey &&
      event.workflowRunId === binding.workflowRunId,
  );
}

export function summarizeWorkflowResumeLifecycle(traceBindings, eventBindings) {
  const matched =
    traceBindings.find((binding) => {
      const matches = workflowResumeEventsForBinding(binding, eventBindings);
      return (
        matches.some((event) => event.eventType === "workflow.step.resuming") &&
        matches.some((event) => event.eventType === "workflow.run.resuming")
      );
    }) || traceBindings[0];
  const matchingEvents = workflowResumeEventsForBinding(matched, eventBindings);
  return {
    actionId: matched?.actionId || null,
    auditEventsPresent:
      matchingEvents.some(
        (event) => event.eventType === "workflow.step.resuming",
      ) &&
      matchingEvents.some(
        (event) => event.eventType === "workflow.run.resuming",
      ),
    actionMetadataPresent: traceBindings.length > 0,
    decision: matched?.decision || null,
    stepId: matched?.stepId || null,
    workflowKey: matched?.workflowKey || null,
    workflowRunId: matched?.workflowRunId || null,
  };
}
