import type {
  AppServerAgentSessionReadResponse,
  AppServerAgentSessionStatus,
  AppServerAgentTurn,
  AppServerAgentTurnStatus,
} from "@/lib/api/appServer";
import { normalizeThreadReadModel } from "./normalizers";
import type {
  AgentRuntimeProfileStatus,
  AgentRuntimeThreadReadModel,
  AgentRuntimeThreadTurnProfileView,
} from "./sessionTypes";

export type AppServerAgentSessionReadProjectionInput =
  AppServerAgentSessionReadResponse & {
    detail?: unknown;
  };

export function projectAppServerSessionReadToThreadReadModel(
  response: AppServerAgentSessionReadProjectionInput,
): AgentRuntimeThreadReadModel {
  const detailThreadRead = normalizeThreadReadModel(
    readDetailThreadRead(response.detail),
  );
  const sessionBusinessObjectRefMetadata =
    readSessionBusinessObjectRefMetadata(response);
  const hasDetailSessionBusinessObjectRefMetadata = detailThreadRead
    ? Object.prototype.hasOwnProperty.call(
        detailThreadRead,
        "session_business_object_ref_metadata",
      )
    : false;
  const sessionStatus = profileStatusFromSessionStatus(response.session.status);
  const protocolTurns = response.turns.map(projectAppServerTurn);
  const projected: AgentRuntimeThreadReadModel = {
    ...(detailThreadRead ?? {}),
    thread_id: response.session.threadId,
    status: detailThreadRead?.status ?? sessionStatus,
    profile_status: detailThreadRead?.profile_status ?? sessionStatus,
    active_turn_id: normalizeActiveTurnId(detailThreadRead?.active_turn_id),
    turns:
      detailThreadRead?.turns && detailThreadRead.turns.length > 0
        ? detailThreadRead.turns
        : protocolTurns,
    pending_requests: detailThreadRead?.pending_requests ?? [],
    incidents: detailThreadRead?.incidents ?? [],
    updated_at: response.session.updatedAt,
  };
  const projectedSessionBusinessObjectRefMetadata =
    hasDetailSessionBusinessObjectRefMetadata
      ? (detailThreadRead?.session_business_object_ref_metadata ?? null)
      : sessionBusinessObjectRefMetadata;
  if (
    projectedSessionBusinessObjectRefMetadata ||
    hasDetailSessionBusinessObjectRefMetadata
  ) {
    projected.session_business_object_ref_metadata =
      projectedSessionBusinessObjectRefMetadata;
  }
  return normalizeThreadReadModel(projected) as AgentRuntimeThreadReadModel;
}

function normalizeActiveTurnId(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function projectAppServerTurn(
  turn: AppServerAgentTurn,
): AgentRuntimeThreadTurnProfileView {
  return {
    turn_id: turn.turnId,
    status: profileStatusFromTurnStatus(turn.status),
    native_status: turn.status,
  };
}

function profileStatusFromSessionStatus(
  status: AppServerAgentSessionStatus,
): AgentRuntimeProfileStatus {
  switch (status) {
    case "idle":
      return "idle";
    case "running":
      return "running";
    case "waitingAction":
      return "blocked";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "canceled":
      return "cancelled";
  }
}

function profileStatusFromTurnStatus(
  status: AppServerAgentTurnStatus,
): AgentRuntimeProfileStatus {
  switch (status) {
    case "accepted":
    case "running":
      return "running";
    case "queued":
      return "queued";
    case "waitingAction":
      return "blocked";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "canceled":
      return "cancelled";
  }
}

function readDetailThreadRead(
  detail: unknown,
): AgentRuntimeThreadReadModel | null {
  const detailRecord = asRecord(detail);
  const threadRead =
    detailRecord?.thread_read ?? detailRecord?.threadRead ?? null;
  return asRecord(threadRead) as AgentRuntimeThreadReadModel | null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readSessionBusinessObjectRefMetadata(
  response: AppServerAgentSessionReadProjectionInput,
): Record<string, unknown> | null {
  return asRecord(response.session.businessObjectRef?.metadata);
}
