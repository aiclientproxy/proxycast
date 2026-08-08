import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Square, Terminal } from "lucide-react";
import type { CodingWorkbenchView } from "@limecloud/agent-runtime-projection";
import type { AgentRuntimeFileCheckpointThreadSummary } from "@/lib/api/agentRuntime/sessionTypes";
import type { ActionRequired, ConfirmResponse } from "../types";
import { CodingWorkbenchActionPanel } from "./CodingWorkbenchActionPanel";
import { CodingWorkbenchDiagnosticPanel } from "./CodingWorkbenchDiagnosticPanel";
import { CodingWorkbenchRecoveryPanel } from "./CodingWorkbenchRecoveryPanel";
import {
  buildCodingWorkbenchRecoveryView,
  type CodingWorkbenchRecoveryContext,
} from "./codingWorkbenchRecovery";
import { CodingStatusBadge } from "./codingWorkbenchStatus";
import { statusLabelKey } from "./codingWorkbenchStatusModel";

interface CodingWorkbenchOutputPanelProps {
  codingView: CodingWorkbenchView;
  fileCheckpointSummary?: AgentRuntimeFileCheckpointThreadSummary | null;
  submittedActionsInFlight?: readonly ActionRequired[];
  processControls?: CodingWorkbenchCommandProcessControls;
  onRespondToAction?: (response: ConfirmResponse) => void | Promise<void>;
  onSubmitRecoveryPrompt?: (
    prompt: string,
    context?: CodingWorkbenchRecoveryContext,
  ) => void | Promise<boolean> | boolean;
}

export interface CodingWorkbenchCommandProcessControls {
  onTerminateBackgroundTerminal?: (
    itemId: string,
  ) => void | Promise<unknown>;
}

export function CodingWorkbenchOutputPanel({
  codingView,
  fileCheckpointSummary,
  submittedActionsInFlight,
  processControls,
  onRespondToAction,
  onSubmitRecoveryPrompt,
}: CodingWorkbenchOutputPanelProps) {
  const { t } = useTranslation("agent");
  const recoveryView = buildCodingWorkbenchRecoveryView({
    codingView,
    fileCheckpointSummary,
    copy: {
      intro: t("agentChat.canvasWorkbench.coding.recovery.prompt.intro"),
      requirements: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.requirements",
      ),
      failedCommand: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.failedCommand",
      ),
      failedTest: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.failedTest",
      ),
      failedPatch: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.failedPatch",
      ),
      diagnostic: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.diagnostic",
      ),
      preview: t("agentChat.canvasWorkbench.coding.recovery.prompt.preview"),
      relatedFiles: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.relatedFiles",
      ),
      latestCheckpoint: t(
        "agentChat.canvasWorkbench.coding.recovery.prompt.latestCheckpoint",
      ),
    },
  });
  const hasCommands = codingView.commands.length > 0;
  const hasTests = codingView.tests.length > 0;
  const hasActions = codingView.actions.length > 0;
  const hasDiagnostics = codingView.diagnostics.length > 0;

  if (
    !recoveryView &&
    !hasCommands &&
    !hasTests &&
    !hasActions &&
    !hasDiagnostics
  ) {
    return (
      <div
        data-testid="coding-workbench-output-projection"
        className="p-5 text-sm text-slate-500"
      >
        {t("agentChat.canvasWorkbench.coding.outputs.empty")}
      </div>
    );
  }

  return (
    <div
      data-testid="coding-workbench-output-projection"
      className="flex h-full min-h-0 flex-col gap-4 overflow-auto bg-white p-4"
    >
      {recoveryView ? (
        <CodingWorkbenchRecoveryPanel
          recoveryView={recoveryView}
          onSubmitRecoveryPrompt={onSubmitRecoveryPrompt}
        />
      ) : null}

      {hasCommands ? (
        <section className="space-y-2" data-testid="coding-workbench-commands">
          <h3 className="text-xs font-semibold text-slate-500">
            {t("agentChat.canvasWorkbench.coding.outputs.commands")}
          </h3>
          <div className="space-y-2">
            {codingView.commands.map((command) => {
              return (
                <article
                  key={command.commandId}
                  data-testid="coding-workbench-command"
                  data-command-id={command.commandId}
                  className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm shadow-slate-950/5"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-mono text-sm text-slate-900">
                        {command.commandSummary ||
                          command.canonicalCommand ||
                          command.command ||
                          command.title}
                      </div>
                      {command.command &&
                      command.command !== command.commandSummary &&
                      command.command !== command.canonicalCommand ? (
                        <div className="mt-1 truncate text-xs text-slate-500">
                          {command.command}
                        </div>
                      ) : null}
                      {command.cwd ? (
                        <div className="mt-1 truncate text-xs text-slate-500">
                          {command.cwd}
                        </div>
                      ) : null}
                    </div>
                    <CodingStatusBadge
                      status={command.status}
                      label={t(statusLabelKey(command.status))}
                    />
                  </div>
                  {command.preview ? (
                    <pre className="mt-3 max-h-28 overflow-hidden rounded-md bg-slate-950 p-2 text-xs text-slate-100">
                      {command.preview}
                    </pre>
                  ) : null}
                  {command.processId ? (
                    <CodingWorkbenchCommandProcessRow
                      itemId={command.commandId}
                      processId={command.processId}
                      executionProcessStatus={command.executionProcessStatus}
                      executionSurface={command.executionSurface}
                      processControls={processControls}
                    />
                  ) : null}
                </article>
              );
            })}
          </div>
        </section>
      ) : null}

      {hasTests ? (
        <section className="space-y-2" data-testid="coding-workbench-tests">
          <h3 className="text-xs font-semibold text-slate-500">
            {t("agentChat.canvasWorkbench.coding.outputs.tests")}
          </h3>
          <div className="space-y-2">
            {codingView.tests.map((test) => {
              return (
                <article
                  key={test.testRunId}
                  data-testid="coding-workbench-test"
                  data-test-run-id={test.testRunId}
                  className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm shadow-slate-950/5"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-slate-900">
                        {test.commandSummary ||
                          test.canonicalCommand ||
                          test.suite ||
                          test.title}
                      </div>
                      <div className="mt-1 text-xs text-slate-500">
                        {t(
                          "agentChat.canvasWorkbench.coding.outputs.testStats",
                          {
                            passed: test.passed ?? 0,
                            failed: test.failed ?? 0,
                          },
                        )}
                      </div>
                    </div>
                    <CodingStatusBadge
                      status={test.status}
                      label={t(statusLabelKey(test.status))}
                    />
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      ) : null}

      {hasActions ? (
        <CodingWorkbenchActionPanel
          actions={codingView.actions}
          submittedActionsInFlight={submittedActionsInFlight}
          onRespondToAction={onRespondToAction}
        />
      ) : null}

      {hasDiagnostics ? (
        <CodingWorkbenchDiagnosticPanel diagnostics={codingView.diagnostics} />
      ) : null}
    </div>
  );
}

function CodingWorkbenchCommandProcessRow({
  itemId,
  processId,
  executionProcessStatus,
  executionSurface,
  processControls,
}: {
  itemId: string;
  processId: string;
  executionProcessStatus?: string;
  executionSurface?: string;
  processControls?: CodingWorkbenchCommandProcessControls;
}) {
  const { t } = useTranslation("agent");
  const live = isLiveExecutionProcessStatus(executionProcessStatus);
  const hasControls =
    live && Boolean(processControls?.onTerminateBackgroundTerminal);

  return (
    <div
      className="mt-3 space-y-2"
      data-testid="coding-workbench-command-process"
    >
      <div className="flex min-h-8 items-center justify-between gap-3 rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-xs text-slate-600">
        <div className="flex min-w-0 items-center gap-2">
          <Terminal className="h-3.5 w-3.5 shrink-0 text-slate-500" />
          <span className="truncate">
            {t("agentChat.canvasWorkbench.coding.outputs.processId", {
              id: processId,
            })}
          </span>
          {executionProcessStatus ? (
            <span className="shrink-0 rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[11px] text-slate-500">
              {t("agentChat.canvasWorkbench.coding.outputs.processStatus", {
                status: executionProcessStatus,
              })}
            </span>
          ) : null}
          {executionSurface ? (
            <span className="hidden shrink-0 rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[11px] text-slate-500 sm:inline-flex">
              {t("agentChat.canvasWorkbench.coding.outputs.processSurface", {
                surface: executionSurface,
              })}
            </span>
          ) : null}
        </div>
        {hasControls ? (
          <CodingWorkbenchCommandProcessButtons
            itemId={itemId}
            processId={processId}
            processControls={processControls}
          />
        ) : null}
      </div>
    </div>
  );
}

function CodingWorkbenchCommandProcessButtons({
  itemId,
  processId,
  processControls,
}: {
  itemId: string;
  processId: string;
  processControls?: CodingWorkbenchCommandProcessControls;
}) {
  const { t } = useTranslation("agent");
  const [pendingAction, setPendingAction] = useState<string | null>(null);

  const run = async (
    action: string,
    handler: (() => void | Promise<unknown>) | undefined,
  ) => {
    if (!handler || pendingAction) return;
    setPendingAction(action);
    try {
      await handler();
    } finally {
      setPendingAction(null);
    }
  };

  return (
    <div
      className="flex shrink-0 items-center gap-1"
      data-testid="coding-workbench-command-process-controls"
    >
      {processControls?.onTerminateBackgroundTerminal ? (
        <ProcessControlButton
          label={t(
            "agentChat.canvasWorkbench.coding.outputs.processTerminateAria",
            { id: processId },
          )}
          pending={pendingAction === "terminate"}
          disabled={Boolean(pendingAction)}
          onClick={() =>
            run("terminate", () =>
              processControls.onTerminateBackgroundTerminal?.(itemId),
            )
          }
        >
          <Square className="h-3.5 w-3.5" />
        </ProcessControlButton>
      ) : null}
    </div>
  );
}

function ProcessControlButton({
  label,
  pending,
  disabled,
  onClick,
  children,
}: {
  label: string;
  pending: boolean;
  disabled: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-600 transition hover:border-slate-300 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <span className={pending ? "animate-spin" : ""}>{children}</span>
    </button>
  );
}

function isLiveExecutionProcessStatus(status?: string): boolean {
  const normalized = status?.trim().toLowerCase();
  return normalized === "running" || normalized === "starting";
}
