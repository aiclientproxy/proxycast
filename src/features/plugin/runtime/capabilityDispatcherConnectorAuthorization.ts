import type { LimeAppSdk } from "../sdk/CapabilityHost";
import type { PluginTaskRecord } from "../types";
import type { PluginHostBridgeCapabilityRequest } from "./hostBridge";
import { sanitizeExecutionRequestInput } from "./capabilityDispatcherExecutionInput";
import { isRecord, readString } from "./capabilityDispatcherRecord";
import type {
  ConnectorAuthorizationAgentTaskRequest,
  ConnectorAuthorizationHandoffProjection,
  ConnectorAuthorizationPolicyProjection,
  ConnectorAuthorizationRequestEnvelope,
  ConnectorRuntimeFactsInternalProjection,
  ConnectorRuntimeFactsProjection,
  ConnectorSecretDeliveryInternalFact,
  ConnectorSecretDeliveryProjection,
  RuntimeConnectorAuthorizationProjection,
  RuntimeConnectorProjection,
} from "./capabilityDispatcherRuntimeTypes";

function buildConnectorAuthorizationPolicy(
  reason: string,
): ConnectorAuthorizationPolicyProjection {
  return {
    owner: "lime_connector_policy",
    scope: "plugin_session",
    approvalRequired: true,
    mutationExposed: false,
    tokenExposed: false,
    secretBinding: "host_managed",
    sessionScoped: true,
    reason,
  };
}

export function buildConnectorAuthorizationRequestEnvelope(
  request: PluginHostBridgeCapabilityRequest,
  connectorId: string,
  input: Record<string, unknown>,
  reason: string,
): ConnectorAuthorizationRequestEnvelope {
  const taskId = readString(input.taskId);
  const sessionId = readString(input.sessionId);
  const envelope: ConnectorAuthorizationRequestEnvelope = {
    capability: "lime.connectors",
    method: "requestAuth",
    appId: request.appId,
    connectorId,
    input: sanitizeExecutionRequestInput(input),
    reason,
    policy: buildConnectorAuthorizationPolicy(reason),
  };
  if (request.entryKey) {
    envelope.entryKey = request.entryKey;
  }
  if (taskId) {
    envelope.taskId = taskId;
  }
  if (sessionId) {
    envelope.sessionId = sessionId;
  }
  if (request.idempotencyKey) {
    envelope.idempotencyKey = request.idempotencyKey;
  }
  return envelope;
}

function buildConnectorAuthorizationHandoffTaskRequest(
  authorizationRequest: ConnectorAuthorizationRequestEnvelope,
): ConnectorAuthorizationAgentTaskRequest {
  return {
    title: `Connector authorization · ${authorizationRequest.connectorId}`,
    taskKind: "plugin.connector_authorization",
    idempotencyKey:
      authorizationRequest.idempotencyKey ??
      `${authorizationRequest.appId}:${
        authorizationRequest.entryKey ?? "default"
      }:lime.connectors:requestAuth:${authorizationRequest.connectorId}`,
    prompt: [
      "【Plugin Connector Authorization Request】",
      `Connector: ${authorizationRequest.connectorId}`,
      "",
      "请由 Lime Host / Connector policy owner 创建或恢复 host-managed 授权绑定。",
      "不要要求 Plugin 输入或保存 OAuth token、refresh token、API key 或 provider secret。",
      "如果需要用户登录或授权，请通过 Host / Cloud Overlay 的授权流程发起，不要把 secret 明文写入任务结果。",
    ].join("\n"),
    input: {
      authorizationRequest,
    },
    expectedOutput: {
      kind: "connector_authorization_request",
      connectorId: authorizationRequest.connectorId,
      secretBinding: "host_managed",
      tokenExposed: false,
      source: "lime_connector_policy",
    },
    requiredCapabilities: ["lime.connectors"],
    capabilityHints: [
      "lime.connectors",
      `connector:${authorizationRequest.connectorId}`,
    ],
    humanReview: true,
    queueIfBusy: true,
    metadata: {
      plugin_connector_authorization: {
        version: "p18.7-e4",
        source: "host_bridge_authorization_gate",
        request: authorizationRequest,
      },
    },
  };
}

function readConnectorAuthorizationRequestFromResponse(
  response: Record<string, unknown>,
): ConnectorAuthorizationRequestEnvelope | null {
  const authorizationGate = isRecord(response.authorizationGate)
    ? response.authorizationGate
    : null;
  const request = authorizationGate?.request;
  return isRecord(request) &&
    request.capability === "lime.connectors" &&
    request.method === "requestAuth" &&
    typeof request.appId === "string" &&
    typeof request.connectorId === "string" &&
    isRecord(request.policy)
    ? (request as unknown as ConnectorAuthorizationRequestEnvelope)
    : null;
}

export async function attachConnectorAuthorizationHandoff(
  response: Record<string, unknown>,
  resolveSdk?: () => LimeAppSdk,
): Promise<Record<string, unknown>> {
  if (!resolveSdk) {
    return response;
  }
  const authorizationGate = isRecord(response.authorizationGate)
    ? response.authorizationGate
    : {};
  const authorizationRequest =
    readConnectorAuthorizationRequestFromResponse(response);
  if (!authorizationRequest) {
    return response;
  }
  try {
    const task = await resolveSdk().agent.startTask(
      buildConnectorAuthorizationHandoffTaskRequest(authorizationRequest),
    );
    const handoff: ConnectorAuthorizationHandoffProjection = {
      status: "accepted",
      owner: "lime_connector_policy",
      source: "lime.agent.startTask",
      taskId: task.taskId,
      traceId: task.traceId,
      taskKind: task.taskKind,
      taskStatus: task.status,
    };
    return {
      ...response,
      authorizationGate: {
        ...authorizationGate,
        handoff,
      },
    };
  } catch {
    const handoff: ConnectorAuthorizationHandoffProjection = {
      status: "not_started",
      owner: "lime_connector_policy",
      source: "lime.agent.startTask",
      reason: "connector_authorization_handoff_failed",
    };
    return {
      ...response,
      authorizationGate: {
        ...authorizationGate,
        handoff,
      },
    };
  }
}

export function isHostFixtureConnectorAction(
  connectorId: string,
  actionId?: string,
): boolean {
  return (
    connectorId === "lime_fixture" &&
    (actionId === undefined ||
      actionId === "recordMutation" ||
      actionId === "record_mutation")
  );
}

function readTaskConnectorAuthorizationRequest(
  task: PluginTaskRecord,
): Record<string, unknown> | null {
  if (task.taskKind !== "plugin.connector_authorization") {
    return null;
  }
  if (isRecord(task.input) && isRecord(task.input.authorizationRequest)) {
    return task.input.authorizationRequest;
  }
  if (
    isRecord(task.result) &&
    isRecord(task.result.plugin_connector_authorization) &&
    isRecord(task.result.plugin_connector_authorization.request)
  ) {
    return task.result.plugin_connector_authorization.request;
  }
  return null;
}

function buildConnectorAuthorizationProjection(
  task: PluginTaskRecord,
): RuntimeConnectorAuthorizationProjection | null {
  const request = readTaskConnectorAuthorizationRequest(task);
  const connectorId =
    readString(request?.connectorId) ??
    (isRecord(task.expectedOutput)
      ? readString(task.expectedOutput.connectorId)
      : undefined);
  if (!connectorId) {
    return null;
  }

  return {
    connectorId,
    actionId: readString(request?.action),
    taskId: task.taskId,
    taskStatus: task.status,
    startedAt: task.startedAt,
    finishedAt: task.finishedAt,
    reason: readString(request?.reason),
    secretBinding: "host_managed",
    tokenExposed: false,
    sessionScoped: true,
    source: "plugin_connector_authorization_task",
    secretDelivery: buildHostManagedSecretDeliveryFact(
      connectorId,
      readString(request?.action),
      task.taskId,
      task.status,
    ),
  };
}

export function buildConnectorAuthorizationProjections(
  tasks: PluginTaskRecord[],
): RuntimeConnectorAuthorizationProjection[] {
  return tasks
    .map(buildConnectorAuthorizationProjection)
    .filter((item): item is RuntimeConnectorAuthorizationProjection =>
      Boolean(item),
    )
    .sort((left, right) => left.connectorId.localeCompare(right.connectorId));
}

export function buildConnectorRuntimeFacts(
  connectorId: string,
  connector?: RuntimeConnectorProjection,
  authorizationRequest?: RuntimeConnectorAuthorizationProjection,
  fixtureActionId?: string,
  options: { exposeSecretLeaseRef?: boolean } = {},
):
  | ConnectorRuntimeFactsProjection
  | ConnectorRuntimeFactsInternalProjection
  | undefined {
  if (isHostFixtureConnectorAction(connectorId, fixtureActionId)) {
    return {
      connectorId,
      status: "authorized",
      authorizationStatus: "authorized",
      source: "host_fixture_connector",
      actionIds: fixtureActionId ? [fixtureActionId] : ["recordMutation"],
      secretBinding: "host_managed",
      tokenExposed: false,
    };
  }

  if (!connector && authorizationRequest?.taskStatus !== "succeeded") {
    return undefined;
  }

  const authorizationStatus =
    authorizationRequest?.taskStatus === "succeeded"
      ? "authorized"
      : "observed";
  const taskIds = Array.from(
    new Set([
      ...(connector?.taskIds ?? []),
      ...(authorizationRequest ? [authorizationRequest.taskId] : []),
    ]),
  );
  const secretDelivery = buildHostManagedSecretDeliveryFact(
    connectorId,
    fixtureActionId,
    authorizationRequest?.taskId,
    authorizationRequest?.taskStatus,
    options,
  );

  return {
    connectorId,
    status: connector ? "observed" : "authorized",
    authorizationStatus,
    source:
      connector && authorizationRequest
        ? "mixed"
        : (connector?.source ?? "plugin_connector_authorization_task"),
    actionIds: connector?.actionIds,
    runIds: connector?.runIds,
    taskIds: taskIds.length > 0 ? taskIds : undefined,
    secretBinding: "host_managed",
    tokenExposed: false,
    ...(secretDelivery ? { secretDelivery } : {}),
  };
}

function buildHostManagedSecretDeliveryFact(
  connectorId: string,
  actionId: string | undefined,
  authorizationTaskId: string | undefined,
  authorizationTaskStatus: PluginTaskRecord["status"] | undefined,
  options: { exposeSecretLeaseRef?: boolean } = {},
):
  | ConnectorSecretDeliveryProjection
  | ConnectorSecretDeliveryInternalFact
  | undefined {
  if (authorizationTaskStatus !== "succeeded" || !authorizationTaskId) {
    return undefined;
  }
  const normalizedActionId = actionId?.trim() || "default";
  const leaseRef = [
    "secret-lease://connector",
    encodeURIComponent(connectorId),
    encodeURIComponent(normalizedActionId),
    encodeURIComponent(authorizationTaskId),
  ].join("/");
  const fact: ConnectorSecretDeliveryProjection = {
    status: "ready",
    binding: "host_managed",
    source: "host_managed_secret_delivery_fact",
    target: "cloud_overlay_worker",
    leaseObserved: true,
    leaseRefExposed: false,
    leaseHandleStatus: "host_managed",
    credentialMaterialExposed: false,
    tokenExposed: false,
  };
  return options.exposeSecretLeaseRef ? { ...fact, leaseRef } : fact;
}
