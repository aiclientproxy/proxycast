import { DecisionPanel } from "../DecisionPanel";
import { InputbarApprovalPrompt } from "../Inputbar/components/InputbarApprovalPrompt";
import {
  McpServerElicitationForm,
  type McpServerElicitationFormSubmission,
} from "../McpServerElicitationForm";
import type {
  PendingInteractionResponse,
  PendingInteractionResponseResult,
  TypedPendingInteraction,
} from "@/lib/api/agentRuntime/pendingInteractionController";
import { selectActivePendingInteraction } from "./pendingInteractionSelection";

export interface PendingInteractionLayerProps {
  interactions: readonly TypedPendingInteraction[];
  threadId?: string | null;
  onRespond: (
    response: PendingInteractionResponse,
  ) =>
    | PendingInteractionResponseResult
    | Promise<PendingInteractionResponseResult>;
}

/** Composer 上方唯一的 pending interaction 展示层。 */
export function PendingInteractionLayer({
  interactions,
  threadId,
  onRespond,
}: PendingInteractionLayerProps) {
  const interaction = selectActivePendingInteraction(interactions, threadId);
  if (!interaction) {
    return null;
  }

  return (
    <div
      className="w-full min-w-0"
      data-testid="pending-interaction-layer"
      data-interaction-id={interaction.id}
      data-interaction-kind={interaction.kind}
    >
      {interaction.kind === "approval" ? (
        <InputbarApprovalPrompt
          request={interaction.payload.request}
          onSubmit={async (response) => {
            await onRespond({
              interactionId: interaction.id,
              kind: "approval",
              response,
            });
          }}
        />
      ) : interaction.kind === "request_user_input" ? (
        <DecisionPanel
          request={interaction.payload.request}
          onSubmit={async (response) => {
            await onRespond({
              confirmed: response.confirmed !== false,
              interactionId: interaction.id,
              kind: "request_user_input",
              response: response.response,
              userData: response.userData,
            });
          }}
        />
      ) : (
        <McpServerElicitationForm
          request={{
            key: interaction.id,
            params: {
              message: interaction.payload.message,
              mode: "form",
              requestedSchema: interaction.payload.requestedSchema,
              serverName: interaction.payload.serverName,
              threadId: interaction.thread_id,
              turnId: interaction.turn_id ?? null,
            },
          }}
          onSubmit={(submission) =>
            onRespond(toMcpResponse(interaction.id, submission))
          }
        />
      )}
    </div>
  );
}

function toMcpResponse(
  interactionId: string,
  submission: McpServerElicitationFormSubmission,
): PendingInteractionResponse {
  return submission.action === "accept"
    ? {
        action: "accept",
        content: submission.content,
        interactionId,
        kind: "mcp_elicitation",
      }
    : {
        action: submission.action,
        interactionId,
        kind: "mcp_elicitation",
      };
}
