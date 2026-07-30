import { useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import {
  Check,
  FolderLock,
  Globe2,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { DecisionPanel } from "../DecisionPanel";
import { InputbarApprovalPrompt } from "../Inputbar/components/InputbarApprovalPrompt";
import {
  McpServerElicitationForm,
  type McpServerElicitationFormSubmission,
} from "../McpServerElicitationForm";
import type {
  PendingPermissionsApprovalInteraction,
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
      ) : interaction.kind === "permissions_approval" ? (
        <PermissionsApprovalPanel
          interaction={interaction}
          onRespond={onRespond}
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

type PermissionsApprovalDecision = Extract<
  PendingInteractionResponse,
  { kind: "permissions_approval" }
>["decision"];

function PermissionsApprovalPanel({
  interaction,
  onRespond,
}: {
  interaction: PendingPermissionsApprovalInteraction;
  onRespond: PendingInteractionLayerProps["onRespond"];
}) {
  const { t } = useTranslation("agent");
  const [submitting, setSubmitting] =
    useState<PermissionsApprovalDecision | null>(null);
  const submittingRef = useRef(false);
  const { cwd, environmentId, permissions, reason } = interaction.payload;

  const submit = async (decision: PermissionsApprovalDecision) => {
    if (submittingRef.current) {
      return;
    }
    submittingRef.current = true;
    setSubmitting(decision);
    try {
      const result = await onRespond({
        decision,
        interactionId: interaction.id,
        kind: "permissions_approval",
      });
      if (!result.accepted) {
        submittingRef.current = false;
        setSubmitting(null);
      }
    } catch (error) {
      submittingRef.current = false;
      setSubmitting(null);
      throw error;
    }
  };

  return (
    <section
      className="flex max-h-[min(28rem,55vh)] w-full flex-col overflow-hidden rounded-md border border-amber-200 bg-background shadow-sm shadow-slate-950/5"
      data-testid="permissions-approval-panel"
      data-interaction-id={interaction.id}
      aria-label={String(t("agentChat.permissionsApproval.title"))}
    >
      <header className="flex items-start gap-3 border-b border-border px-4 py-3">
        <span className="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-amber-50 text-amber-700">
          <ShieldAlert className="h-4 w-4" aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-foreground">
            {t("agentChat.permissionsApproval.title")}
          </h2>
          <p
            className="mt-0.5 whitespace-pre-wrap text-xs leading-5 text-muted-foreground"
            data-testid="permissions-approval-reason"
          >
            {reason?.trim() ||
              t("agentChat.permissionsApproval.reasonFallback")}
          </p>
        </div>
      </header>

      <div className="min-h-0 overflow-y-auto px-4 py-3">
        <dl className="grid min-w-0 gap-x-5 gap-y-3 text-xs sm:grid-cols-2">
          <PermissionFact
            label={String(t("agentChat.permissionsApproval.cwd"))}
            testId="permissions-approval-cwd"
          >
            <code className="break-all font-mono text-foreground">{cwd}</code>
          </PermissionFact>
          <PermissionFact
            label={String(t("agentChat.permissionsApproval.environment"))}
            testId="permissions-approval-environment"
          >
            <span className="break-all text-foreground">
              {environmentId?.trim() ||
                t("agentChat.permissionsApproval.environmentFallback")}
            </span>
          </PermissionFact>
          <PermissionFact
            icon={<Globe2 className="h-3.5 w-3.5" aria-hidden="true" />}
            label={String(t("agentChat.permissionsApproval.network"))}
            testId="permissions-approval-network"
          >
            <span className="text-foreground">
              {permissions.network?.enabled === true
                ? t("agentChat.permissionsApproval.network.enable")
                : permissions.network?.enabled === false
                  ? t("agentChat.permissionsApproval.network.disable")
                  : t("agentChat.permissionsApproval.noChange")}
            </span>
          </PermissionFact>
          <PermissionFact
            icon={<FolderLock className="h-3.5 w-3.5" aria-hidden="true" />}
            label={String(t("agentChat.permissionsApproval.fileSystem"))}
            testId="permissions-approval-file-system"
          >
            <FileSystemPermissionDiff fileSystem={permissions.fileSystem} />
          </PermissionFact>
        </dl>
      </div>

      <footer className="flex flex-wrap justify-end gap-2 border-t border-border px-4 py-3">
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="text-rose-700 hover:bg-rose-50 hover:text-rose-800"
          data-permission-decision="decline"
          disabled={submitting !== null}
          onClick={() => void submit("decline")}
        >
          {submitting === "decline" ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <X className="mr-2 h-4 w-4" aria-hidden />
          )}
          {submitting === "decline"
            ? t("agentChat.permissionsApproval.action.submitting")
            : t("agentChat.permissionsApproval.action.decline")}
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          data-permission-decision="grant_session"
          disabled={submitting !== null}
          onClick={() => void submit("grant_session")}
        >
          {submitting === "grant_session" ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <ShieldCheck className="mr-2 h-4 w-4" aria-hidden />
          )}
          {submitting === "grant_session"
            ? t("agentChat.permissionsApproval.action.submitting")
            : t("agentChat.permissionsApproval.action.grantSession")}
        </Button>
        <Button
          type="button"
          size="sm"
          data-permission-decision="grant_turn"
          disabled={submitting !== null}
          onClick={() => void submit("grant_turn")}
        >
          {submitting === "grant_turn" ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <Check className="mr-2 h-4 w-4" aria-hidden />
          )}
          {submitting === "grant_turn"
            ? t("agentChat.permissionsApproval.action.submitting")
            : t("agentChat.permissionsApproval.action.grantTurn")}
        </Button>
      </footer>
    </section>
  );
}

function PermissionFact({
  children,
  icon,
  label,
  testId,
}: {
  children: ReactNode;
  icon?: ReactNode;
  label: string;
  testId: string;
}) {
  return (
    <div className="min-w-0" data-testid={testId}>
      <dt className="mb-1 flex items-center gap-1.5 font-medium text-muted-foreground">
        {icon}
        {label}
      </dt>
      <dd className="min-w-0 leading-5">{children}</dd>
    </div>
  );
}

function FileSystemPermissionDiff({
  fileSystem,
}: {
  fileSystem: PendingPermissionsApprovalInteraction["payload"]["permissions"]["fileSystem"];
}) {
  const { t } = useTranslation("agent");
  if (!fileSystem) {
    return (
      <span className="text-foreground">
        {t("agentChat.permissionsApproval.noChange")}
      </span>
    );
  }

  const rows = [
    {
      label: t("agentChat.permissionsApproval.fileSystem.read"),
      value: formatPathList(fileSystem.read),
    },
    {
      label: t("agentChat.permissionsApproval.fileSystem.write"),
      value: formatPathList(fileSystem.write),
    },
    ...(fileSystem.globScanMaxDepth === null ||
    fileSystem.globScanMaxDepth === undefined
      ? []
      : [
          {
            label: t(
              "agentChat.permissionsApproval.fileSystem.globScanMaxDepth",
            ),
            value: String(fileSystem.globScanMaxDepth),
          },
        ]),
  ];

  return (
    <div className="space-y-1.5 text-foreground">
      {rows.map((row) => (
        <p className="break-words" key={String(row.label)}>
          <span className="text-muted-foreground">{row.label}: </span>
          <code className="break-all font-mono">{row.value}</code>
        </p>
      ))}
      {fileSystem.entries?.map((entry, index) => (
        <p className="break-words" key={`${entry.access}:${index}`}>
          <span className="text-muted-foreground">
            {t("agentChat.permissionsApproval.fileSystem.entry", {
              access: t(
                `agentChat.permissionsApproval.fileSystem.access.${entry.access}`,
              ),
            })}
            :{" "}
          </span>
          <code className="break-all font-mono">
            {formatSandboxPath(entry.path)}
          </code>
        </p>
      ))}
    </div>
  );
}

function formatPathList(paths?: null | string[]): string {
  if (paths === null || paths === undefined) {
    return "-";
  }
  return paths.length > 0 ? paths.join(", ") : "[]";
}

function formatSandboxPath(
  path: NonNullable<
    NonNullable<
      PendingPermissionsApprovalInteraction["payload"]["permissions"]["fileSystem"]
    >["entries"]
  >[number]["path"],
): string {
  switch (path.type) {
    case "path":
      return path.path;
    case "glob_pattern":
      return path.pattern;
    case "special": {
      const value = path.value;
      switch (value.kind) {
        case "root":
          return "/";
        case "minimal":
          return "$MINIMAL";
        case "project_roots":
          return value.subpath
            ? `$PROJECT_ROOTS/${value.subpath}`
            : "$PROJECT_ROOTS";
        case "tmpdir":
          return "$TMPDIR";
        case "slash_tmp":
          return "/tmp";
        case "unknown":
          return value.subpath ? `${value.path}/${value.subpath}` : value.path;
      }
    }
  }
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
