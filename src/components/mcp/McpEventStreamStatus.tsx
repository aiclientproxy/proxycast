import { Activity, CircleCheck, CircleX, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import {
  selectMcpEventStreams,
  type McpEventStreamStateMap,
} from "@/lib/mcp/eventStreamProjection";

interface McpEventStreamStatusProps {
  eventStreams: McpEventStreamStateMap;
}

export function McpEventStreamStatus({
  eventStreams,
}: McpEventStreamStatusProps) {
  const { t } = useTranslation("settings");
  const streams = selectMcpEventStreams(eventStreams);

  if (streams.length === 0) {
    return null;
  }

  const activeCount = streams.filter(
    (stream) => stream.phase === "active",
  ).length;

  return (
    <section
      className="border-b border-slate-200/80 bg-slate-50/70 px-5 py-4"
      data-testid="mcp-event-stream-status"
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl border border-violet-100 bg-violet-50 text-violet-700">
            <Activity className="h-4 w-4" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">
              {t("settings.mcpPage.runtime.eventStream.title")}
            </p>
            <p className="text-xs text-slate-500">
              {t("settings.mcpPage.runtime.eventStream.summary", {
                active: activeCount,
                total: streams.length,
              })}
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-2 lg:grid-cols-2">
        {streams.map((stream) => {
          const active = stream.phase === "active";
          const StatusIcon = active ? CircleCheck : CircleX;
          const phaseLabel = active
            ? stream.reconnectCount > 0
              ? t("settings.mcpPage.runtime.eventStream.phase.reconnected")
              : t("settings.mcpPage.runtime.eventStream.phase.active")
            : t("settings.mcpPage.runtime.eventStream.phase.terminated");

          return (
            <article
              key={stream.subscriptionId}
              className={cn(
                "rounded-2xl border px-3 py-3",
                active
                  ? "border-emerald-200 bg-emerald-50/80"
                  : "border-slate-200 bg-white",
              )}
              data-mcp-event-stream-subscription-id={stream.subscriptionId}
              data-mcp-event-stream-phase={stream.phase}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate font-mono text-xs font-semibold text-slate-800">
                    {stream.subscriptionId}
                  </p>
                  <p className="mt-1 flex items-center gap-1.5 text-xs font-medium text-slate-700">
                    <StatusIcon className="h-3.5 w-3.5" />
                    {phaseLabel}
                  </p>
                </div>
                {stream.reconnectCount > 0 && (
                  <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-[11px] font-medium text-sky-700">
                    <RefreshCw className="h-3 w-3" />
                    {t("settings.mcpPage.runtime.eventStream.reconnects", {
                      count: stream.reconnectCount,
                    })}
                  </span>
                )}
              </div>
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-500">
                <span>
                  {t("settings.mcpPage.runtime.eventStream.lastEvent", {
                    method:
                      stream.lastEventName ?? stream.lastEventMethod ?? "-",
                  })}
                </span>
                <span>
                  {t("settings.mcpPage.runtime.eventStream.eventCount", {
                    count: stream.eventCount,
                  })}
                </span>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
