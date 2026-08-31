import { Activity, Bot, PlugZap, Sparkles } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  summarizeCanonicalChildThreads,
  type CanonicalChildThreadSummary,
} from "../projection/canonicalChildThreadSummary";

export interface ThreadActivityPanelProps {
  canonicalChildren?: readonly CanonicalChildThreadSummary[];
  threadItems?: readonly AgentThreadItem[];
  toolInventory?: AgentRuntimeToolInventory | null;
  onOpenSubagentSession?: (threadId: string) => void;
}

function compactList(values: readonly string[], limit = 6): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))].slice(
    0,
    limit,
  );
}

export function ThreadActivityPanel({
  canonicalChildren = [],
  threadItems = [],
  toolInventory = null,
  onOpenSubagentSession,
}: ThreadActivityPanelProps) {
  const { t } = useTranslation("agent");
  const childCounts = summarizeCanonicalChildThreads([...canonicalChildren]);
  const subagentItems = threadItems.filter(
    (item) => item.type === "subagent_activity",
  );
  const mcpServers = compactList(toolInventory?.mcp_servers ?? []);
  const mcpToolCount = toolInventory?.counts.mcp_tool_visible_total ?? 0;
  const skillTools = compactList(
    (toolInventory?.catalog_tools ?? [])
      .filter((tool) => tool.capabilities.includes("skill_execution"))
      .map((tool) => tool.name),
  );
  const hasAnyActivity =
    canonicalChildren.length > 0 ||
    subagentItems.length > 0 ||
    mcpServers.length > 0 ||
    skillTools.length > 0;

  return (
    <section
      className="flex h-full min-h-0 flex-col overflow-y-auto bg-[color:var(--lime-surface)] px-4 py-4"
      data-testid="thread-activity-panel"
    >
      <header className="flex items-start justify-between gap-3 border-b border-[color:var(--lime-surface-border)] pb-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-semibold text-[color:var(--lime-text-strong)]">
            <Activity className="h-4 w-4" aria-hidden="true" />
            <span>{t("agentChat.activityPanel.title")}</span>
          </div>
          <p className="mt-1 text-xs leading-5 text-[color:var(--lime-text-muted)]">
            {t("agentChat.activityPanel.description")}
          </p>
        </div>
        <Badge variant="outline" className="shrink-0">
          {childCounts.active + subagentItems.length}
        </Badge>
      </header>

      {!hasAnyActivity ? (
        <div
          className="flex flex-1 items-center justify-center py-10 text-center text-sm text-[color:var(--lime-text-muted)]"
          data-testid="thread-activity-empty"
        >
          {t("agentChat.activityPanel.empty")}
        </div>
      ) : (
        <div className="space-y-4 pt-4">
          {canonicalChildren.length > 0 || subagentItems.length > 0 ? (
            <ActivitySection
              icon={<Bot className="h-4 w-4" aria-hidden="true" />}
              title={t("agentChat.activityPanel.subagents")}
              count={childCounts.total + subagentItems.length}
            >
              <div className="space-y-2">
                {canonicalChildren.map((child) => (
                  <div
                    key={child.threadId}
                    className="flex items-center gap-2 rounded-lg border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-subtle)] px-3 py-2"
                    data-testid="thread-activity-subagent"
                  >
                    <span className="min-w-0 flex-1 truncate text-sm text-[color:var(--lime-text-strong)]">
                      {child.name}
                    </span>
                    <Badge variant={child.status === "errored" ? "destructive" : "secondary"}>
                      {t(`agentChat.activityPanel.status.${child.status}` as never, {
                        defaultValue: child.status,
                      })}
                    </Badge>
                    {onOpenSubagentSession && child.status !== "notFound" ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => onOpenSubagentSession(child.threadId)}
                      >
                        {t("agentChat.activityPanel.open")}
                      </Button>
                    ) : null}
                  </div>
                ))}
                {subagentItems.slice(-4).map((item) => (
                  <div
                    key={item.id}
                    className="rounded-lg border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-subtle)] px-3 py-2"
                    data-testid="thread-activity-item"
                  >
                    <div className="flex items-center justify-between gap-2 text-sm text-[color:var(--lime-text-strong)]">
                      <span className="truncate">{item.title || t("agentChat.activityPanel.subagent")}</span>
                      <Badge variant="outline">{item.status_label}</Badge>
                    </div>
                    {item.summary ? (
                      <p className="mt-1 line-clamp-2 text-xs text-[color:var(--lime-text-muted)]">
                        {item.summary}
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            </ActivitySection>
          ) : null}

          {mcpServers.length > 0 ? (
            <ActivitySection
              icon={<PlugZap className="h-4 w-4" aria-hidden="true" />}
              title={t("agentChat.activityPanel.mcp")}
              count={mcpToolCount}
            >
              <div className="flex flex-wrap gap-2">
                {mcpServers.map((server) => (
                  <Badge key={server} variant="outline">
                    {server}
                  </Badge>
                ))}
              </div>
            </ActivitySection>
          ) : null}

          {skillTools.length > 0 ? (
            <ActivitySection
              icon={<Sparkles className="h-4 w-4" aria-hidden="true" />}
              title={t("agentChat.activityPanel.skills")}
              count={skillTools.length}
            >
              <div className="flex flex-wrap gap-2">
                {skillTools.map((skill) => (
                  <Badge key={skill} variant="secondary">
                    {skill}
                  </Badge>
                ))}
              </div>
            </ActivitySection>
          ) : null}
        </div>
      )}
    </section>
  );
}

function ActivitySection({
  icon,
  title,
  count,
  children,
}: {
  icon: ReactNode;
  title: string;
  count: number;
  children: ReactNode;
}) {
  return (
    <section data-testid="thread-activity-section">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-[color:var(--lime-text-muted)]">
        {icon}
        <span>{title}</span>
        <Badge variant="outline" className="ml-auto text-[10px]">
          {count}
        </Badge>
      </div>
      {children}
    </section>
  );
}
