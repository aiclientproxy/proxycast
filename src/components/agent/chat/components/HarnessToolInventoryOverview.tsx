import { Loader2, Wrench } from "lucide-react";
import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { RuntimeToolAvailability } from "../utils/runtimeToolAvailability";
import { InventoryStatCard } from "./HarnessStatusPanelPrimitives";
import { agentText } from "./harnessPanelText";
import type { RuntimeToolCapabilityGap } from "./HarnessToolInventoryTypes";
import {
  formatRuntimeToolAvailabilitySourceLabel,
  type ToolInventorySourceStats,
} from "./harnessStatusPanelViewModel";

interface HarnessToolInventoryOverviewProps {
  toolInventory?: AgentRuntimeToolInventory | null;
  toolInventoryLoading: boolean;
  toolInventoryError: string | null;
  runtimeToolVisibleTotal: number;
  runtimeToolTotal: number;
  onRefreshToolInventory?: () => void;
  toolInventorySourceStats: ToolInventorySourceStats;
  toolInventoryWarnings: AgentRuntimeToolInventory["warnings"];
  runtimeToolAvailability: RuntimeToolAvailability;
  runtimeToolCapabilityGaps: RuntimeToolCapabilityGap[];
}

export function HarnessToolInventoryOverview({
  toolInventory,
  toolInventoryLoading,
  toolInventoryError,
  runtimeToolVisibleTotal,
  runtimeToolTotal,
  onRefreshToolInventory,
  toolInventorySourceStats,
  toolInventoryWarnings,
  runtimeToolAvailability,
  runtimeToolCapabilityGaps,
}: HarnessToolInventoryOverviewProps) {
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-2">
          {toolInventory ? (
            <>
              <Badge variant="secondary">
                {agentText(
                  "agentChat.harness.generated.82aec0037c",
                  "caller：",
                )}
                {toolInventory.request?.caller || "未知"}
              </Badge>
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.9d523f85db",
                  "工作台：",
                )}
                {toolInventory.request?.surface?.workbench ? "开启" : "关闭"}
              </Badge>
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.e96cdaadb1",
                  "默认允许：",
                )}
                {toolInventory.counts.default_allowed_total}
              </Badge>
            </>
          ) : (
            <Badge variant="outline">
              {agentText(
                "agentChat.harness.generated.0f4d7157ea",
                "等待工具库存",
              )}
            </Badge>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {onRefreshToolInventory ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="gap-2"
              aria-label={agentText(
                "agentChat.harness.generated.908fe49fe3",
                "刷新工具库存",
              )}
              onClick={onRefreshToolInventory}
            >
              {toolInventoryLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wrench className="h-4 w-4" />
              )}
              {agentText("agentChat.harness.generated.f79c583e24", "刷新库存")}
            </Button>
          ) : null}
        </div>
      </div>

      {toolInventoryLoading ? (
        <div className="flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-3 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          {agentText(
            "agentChat.harness.generated.713fb7c6d1",
            "正在读取当前工具库存与权限策略...",
          )}
        </div>
      ) : null}

      {toolInventoryError ? (
        <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-3 text-sm text-destructive">
          {toolInventoryError}
        </div>
      ) : null}

      {toolInventory ? (
        <>
          <div className="grid min-w-0 gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))]">
            <InventoryStatCard
              title={agentText(
                "agentChat.harness.generated.c4740e4ca2",
                "Runtime",
              )}
              value={`${runtimeToolVisibleTotal}`}
              hint={`可见 / 总数 ${runtimeToolVisibleTotal} / ${runtimeToolTotal}`}
            />
            <InventoryStatCard
              title={agentText(
                "agentChat.harness.generated.4a88d27bba",
                "Catalog",
              )}
              value={`${toolInventory.counts.catalog_total}`}
              hint={`现役 ${toolInventory.counts.catalog_current_total} · 兼容 ${toolInventory.counts.catalog_compat_total}`}
            />
            <InventoryStatCard
              title={agentText(
                "agentChat.harness.generated.1fd6a805da",
                "Native Tools",
              )}
              value={`${toolInventory.counts.native_visible_total}`}
              hint={`可见 / 总数 ${toolInventory.counts.native_visible_total} / ${toolInventory.counts.native_total}`}
            />
            <InventoryStatCard
              title={agentText(
                "agentChat.harness.generated.659087d3ca",
                "Extension",
              )}
              value={`${toolInventory.counts.extension_tool_visible_total}`}
              hint={`可见 / 总数 ${toolInventory.counts.extension_tool_visible_total} / ${toolInventory.counts.extension_tool_total}`}
            />
            <InventoryStatCard
              title={agentText("agentChat.harness.generated.21593b807a", "MCP")}
              value={`${toolInventory.counts.mcp_tool_visible_total}`}
              hint={`服务 ${toolInventory.counts.mcp_server_total} · 工具 ${toolInventory.counts.mcp_tool_total}`}
            />
          </div>

          <div className="grid min-w-0 gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,11rem),1fr))]">
            {(
              [
                ["default", "默认策略"],
                ["persisted", "持久化覆盖"],
                ["runtime", "运行时覆盖"],
              ] as const
            ).map(([source, label]) => (
              <InventoryStatCard
                key={source}
                title={label}
                value={`${toolInventorySourceStats[source]}`}
                hint="按 warning / restriction / sandbox 三字段累计"
              />
            ))}
          </div>

          {toolInventoryWarnings.length > 0 ? (
            <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-3">
              <div className="text-sm font-medium text-amber-900">
                {agentText(
                  "agentChat.harness.generated.9dd4dc2098",
                  "库存告警",
                )}
              </div>
              <div className="mt-2 space-y-1 text-xs text-amber-800">
                {toolInventoryWarnings.map((warning, index) => (
                  <div key={`${warning}-${index}`}>{warning}</div>
                ))}
              </div>
            </div>
          ) : null}

          <RuntimeToolCapabilitySummary
            runtimeToolAvailability={runtimeToolAvailability}
            runtimeToolCapabilityGaps={runtimeToolCapabilityGaps}
          />
        </>
      ) : null}
    </>
  );
}

function RuntimeToolCapabilitySummary({
  runtimeToolAvailability,
  runtimeToolCapabilityGaps,
}: {
  runtimeToolAvailability: RuntimeToolAvailability;
  runtimeToolCapabilityGaps: RuntimeToolCapabilityGap[];
}) {
  return (
    <div
      className="space-y-3"
      data-testid="harness-runtime-tool-capability-summary"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-medium text-foreground">
          {agentText(
            "agentChat.harness.generated.b8f1306458",
            "Runtime 能力摘要",
          )}
        </div>
        <Badge
          variant={runtimeToolAvailability.known ? "secondary" : "outline"}
          data-testid="harness-runtime-tool-capability-source"
        >
          {runtimeToolAvailability.known
            ? `来源 ${formatRuntimeToolAvailabilitySourceLabel(
                runtimeToolAvailability.source,
              )}`
            : "Runtime 工具面未就绪"}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-2">
        <Badge
          variant={runtimeToolAvailability.webSearch ? "secondary" : "outline"}
          data-testid="harness-runtime-tool-capability-web-search"
        >
          {runtimeToolAvailability.webSearch
            ? "WebSearch 已接通"
            : "WebSearch 未接通"}
        </Badge>
        <Badge
          variant={
            runtimeToolAvailability.subagentCore ? "secondary" : "outline"
          }
          data-testid="harness-runtime-tool-capability-subagent-core"
        >
          {runtimeToolAvailability.subagentCore
            ? "子任务核心 tools 已接通"
            : `子任务核心 tools 缺 ${runtimeToolAvailability.missingSubagentCoreTools.length} 项`}
        </Badge>
        <Badge
          variant={
            runtimeToolAvailability.subagentTeamTools ? "secondary" : "outline"
          }
          data-testid="harness-runtime-tool-capability-team"
        >
          {runtimeToolAvailability.subagentTeamTools
            ? "Subagents 协作 tools 已接通"
            : `Subagents 协作 tools 缺 ${runtimeToolAvailability.missingSubagentTeamTools.length} 项`}
        </Badge>
        <Badge
          variant={
            runtimeToolAvailability.planRuntime ? "secondary" : "outline"
          }
          data-testid="harness-runtime-tool-capability-plan"
        >
          {runtimeToolAvailability.planRuntime
            ? "Plan current tool 已接通"
            : `Plan current tool 缺 ${runtimeToolAvailability.missingPlanTools.length} 项`}
        </Badge>
      </div>
      {runtimeToolAvailability.known ? (
        runtimeToolCapabilityGaps.length > 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-muted/30 p-3 text-sm text-muted-foreground">
            <div className="font-medium text-foreground">
              {agentText(
                "agentChat.harness.generated.ab8deade1a",
                "当前 runtime current surface 仍有缺口",
              )}
            </div>
            <div className="mt-2 space-y-2">
              {runtimeToolCapabilityGaps.map((gap) => (
                <div key={gap.key}>
                  <span className="font-medium text-foreground">
                    {gap.title}
                  </span>
                  <span>：</span>
                  <span>{gap.missing.join(" / ")}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="rounded-xl border border-emerald-200/70 bg-emerald-50/60 p-3 text-sm text-emerald-900">
            {agentText(
              "agentChat.harness.generated.ff5e6ffa0a",
              "当前 runtime current surface 已覆盖 WebSearch、子任务、Subagents 协作与 Plan 主链。",
            )}
          </div>
        )
      ) : (
        <div className="rounded-xl border border-dashed border-border bg-muted/30 p-3 text-sm text-muted-foreground">
          {agentText(
            "agentChat.harness.generated.9ef6c1f213",
            "当前 inventory 尚未提供可用 runtime tool surface，暂时只能回看 registry/raw inventory。",
          )}
        </div>
      )}
    </div>
  );
}
