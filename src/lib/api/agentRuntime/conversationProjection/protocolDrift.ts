import type { AgentThreadItem } from "../../agentProtocol";
import type {
  ConversationProjectionDiagnostic,
  ConversationProjectionSource,
} from "./contracts";

export const RENDER_PROJECTION_REFERENCE_REVISION =
  "4c43465133428898aa84f0bfc02c306ed65fb66a";

export function buildUnknownItemProtocolDriftDiagnostic(params: {
  eventId?: string;
  item: Extract<AgentThreadItem, { type: "unknown_item" }>;
  method?: string;
  protocolRevision?: string;
  source: ConversationProjectionSource;
}): ConversationProjectionDiagnostic {
  return {
    code: "protocol_drift",
    source: params.source,
    event_id: params.eventId,
    thread_id: params.item.thread_id,
    turn_id: params.item.turn_id,
    item_id: params.item.id,
    protocol_revision:
      params.protocolRevision ?? RENDER_PROJECTION_REFERENCE_REVISION,
    protocol_method: params.method ?? sourceMethod(params.source),
    upstream_type: params.item.upstream_type,
    field_names: [...params.item.field_names],
    message: `Unsupported upstream item ${params.item.upstream_type}.`,
  };
}

function sourceMethod(source: ConversationProjectionSource): string {
  switch (source) {
    case "read":
      return "thread/read";
    case "replay":
      return "replay";
    case "live":
      return "item/unknown";
  }
}
