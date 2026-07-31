import React from "react";
import { AlertTriangle, Loader2, Square } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { InputbarRuntimeStatusLineModel } from "../utils/inputbarRuntimeStatusLine";
import type { AgentRuntimeStatus } from "../types";

function normalizeRuntimeStatusMetaText(value?: string | null): string {
  return (value || "").trim().replace(/\s+/g, " ");
}

function truncateRuntimeStatusText(value: string, maxLength = 96): string {
  const normalized = normalizeRuntimeStatusMetaText(value);
  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${Array.from(normalized).slice(0, maxLength).join("")}...`;
}

export const MessageRuntimeStatusPill: React.FC<{
  status: AgentRuntimeStatus;
}> = ({ status }) => {
  const { t } = useTranslation("agent");
  const failed = status.phase === "failed";
  const cancelled = status.phase === "cancelled";
  const ToneIcon = failed ? AlertTriangle : cancelled ? Square : Loader2;
  const titleText = normalizeRuntimeStatusMetaText(status.title);
  const detailText = normalizeRuntimeStatusMetaText(status.detail);
  const checkpointsText = (status.checkpoints || [])
    .map((item) => normalizeRuntimeStatusMetaText(item))
    .filter(Boolean)
    .slice(0, 2)
    .join(" · ");
  const tooltip = [titleText, detailText, checkpointsText]
    .filter(
      (item, index, array) => Boolean(item) && array.indexOf(item) === index,
    )
    .join("\n");

  return (
    <div
      data-testid="message-runtime-status-pill"
      data-status={status.phase}
      aria-label={tooltip || undefined}
      className={[
        "inline-flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] leading-none",
        failed
          ? "border-rose-200 bg-rose-50 text-rose-700"
          : cancelled
            ? "border-slate-200 bg-slate-50 text-slate-600"
            : "border-sky-200 bg-sky-50 text-sky-700",
      ].join(" ")}
    >
      <ToneIcon
        className={[
          "h-3.5 w-3.5 shrink-0",
          failed || cancelled ? "" : "animate-spin",
        ].join(" ")}
      />
      <span className="truncate">
        {titleText || t("agentChat.messageList.firstTokenStatus.routing.title")}
      </span>
    </div>
  );
};

export const AssistantFirstTokenRuntimeStatus: React.FC<{
  status?: AgentRuntimeStatus | null;
}> = ({ status }) => {
  const { t } = useTranslation("agent");
  const phase = status?.phase || "submitted";
  const title = truncateRuntimeStatusText(
    t(`agentChat.messageList.firstTokenStatus.${phase}.title`),
    48,
  );
  const detail = truncateRuntimeStatusText(
    t(`agentChat.messageList.firstTokenStatus.${phase}.detail`),
    120,
  );
  const accessibleLabel = [title, detail]
    .filter((item, index, array) => Boolean(item) && array.indexOf(item) === index)
    .join("，");

  return (
    <div
      data-testid="assistant-first-token-runtime-status"
      data-status={phase}
      className="inline-flex max-w-full items-center gap-1.5 rounded-md px-2 py-[3px] text-[11px] font-medium leading-4 text-slate-500"
      aria-live="polite"
      aria-label={accessibleLabel || undefined}
    >
      <Loader2
        className="h-3 w-3 shrink-0 animate-spin text-emerald-600 motion-reduce:animate-none"
        aria-hidden
      />
      <span className="truncate">{title}</span>
    </div>
  );
};

export const AssistantStreamingInlineIndicator: React.FC<{
  runtime: InputbarRuntimeStatusLineModel;
}> = ({ runtime }) => {
  const { t } = useTranslation("agent");
  const status = runtime.status === "queued" ? "queued" : "running";
  const isQueued = status === "queued";
  const labelKey =
    status === "running" &&
    runtime.batchDescriptor &&
    runtime.batchDescriptor.hasRunning !== true
      ? "synthesizing"
      : status;

  return (
    <div
      data-testid="assistant-streaming-inline-indicator"
      data-status={status}
      role="status"
      aria-live="polite"
      className={[
        "inline-flex items-center gap-1.5 rounded-md px-2 py-[3px] text-[11px] font-medium leading-4",
        isQueued
          ? "bg-slate-100 text-slate-500"
          : "bg-slate-100 text-slate-600",
      ].join(" ")}
    >
      <Loader2
        className={[
          "h-3 w-3 shrink-0 animate-spin motion-reduce:animate-none",
          isQueued ? "text-slate-400" : "text-emerald-600",
        ].join(" ")}
        aria-hidden
      />
      <span>{t(`agentChat.messageList.streamingInline.${labelKey}`)}</span>
    </div>
  );
};
