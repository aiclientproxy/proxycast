import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { getMcpInnerToolName } from "@/lib/api/mcp";
import { Badge } from "@/components/ui/badge";
import { agentText } from "./harnessPanelText";
import { formatExtensionSourceKindLabel } from "./harnessStatusPanelViewModel";

interface HarnessExtensionToolInventorySectionsProps {
  toolInventoryExtensionSurfaces: AgentRuntimeToolInventory["extension_surfaces"];
  toolInventoryExtensionTools: AgentRuntimeToolInventory["extension_tools"];
  toolInventoryMcpTools: AgentRuntimeToolInventory["mcp_tools"];
}

export function HarnessExtensionToolInventorySections({
  toolInventoryExtensionSurfaces,
  toolInventoryExtensionTools,
  toolInventoryMcpTools,
}: HarnessExtensionToolInventorySectionsProps) {
  return (
    <>
      {toolInventoryExtensionSurfaces.length > 0 ? (
        <ExtensionSurfaceSection entries={toolInventoryExtensionSurfaces} />
      ) : null}

      {toolInventoryExtensionTools.length > 0 ? (
        <ExtensionToolSection entries={toolInventoryExtensionTools} />
      ) : null}

      {toolInventoryMcpTools.length > 0 ? (
        <McpToolSection entries={toolInventoryMcpTools} />
      ) : null}
    </>
  );
}

function ExtensionSurfaceSection({
  entries,
}: {
  entries: AgentRuntimeToolInventory["extension_surfaces"];
}) {
  return (
    <div className="space-y-3">
      <div className="text-sm font-medium text-foreground">
        {agentText(
          "agentChat.harness.generated.0fec57f640",
          "Extension Surfaces",
        )}
      </div>
      {entries.map((entry) => (
        <div
          key={entry.extension_name}
          className="rounded-xl border border-border bg-background p-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              {entry.extension_name}
            </span>
            <Badge variant="outline">
              {formatExtensionSourceKindLabel(entry.source_kind)}
            </Badge>
            {entry.deferred_loading ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.714ae55e88",
                  "Deferred",
                )}
              </Badge>
            ) : null}
            {entry.allowed_caller ? (
              <Badge variant="secondary">
                {agentText(
                  "agentChat.harness.generated.82aec0037c",
                  "caller：",
                )}
                {entry.allowed_caller}
              </Badge>
            ) : null}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {entry.description}
          </div>
          <div className="mt-2 grid min-w-0 gap-2 text-xs text-muted-foreground [grid-template-columns:repeat(auto-fit,minmax(min(100%,10rem),1fr))]">
            <div>
              {agentText(
                "agentChat.harness.generated.9f11b02c89",
                "可用工具：",
              )}
              {entry.available_tools.length}
            </div>
            <div>
              {agentText(
                "agentChat.harness.generated.6dc2d6edaa",
                "常驻工具：",
              )}
              {entry.always_expose_tools.length}
            </div>
            <div>
              {agentText("agentChat.harness.generated.809f7ca51e", "已加载：")}
              {entry.loaded_tools.length}
            </div>
            <div>
              {agentText("agentChat.harness.generated.baf59a0b05", "可搜索：")}
              {entry.searchable_tools.length}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ExtensionToolSection({
  entries,
}: {
  entries: AgentRuntimeToolInventory["extension_tools"];
}) {
  return (
    <div className="space-y-3">
      <div className="text-sm font-medium text-foreground">
        {agentText("agentChat.harness.generated.d2f47a899a", "Extension Tools")}
      </div>
      {entries.map((entry) => (
        <div
          key={entry.name}
          className="rounded-xl border border-border bg-background p-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              {entry.name}
            </span>
            <Badge variant="outline">{entry.status}</Badge>
            <Badge variant="outline">
              {formatExtensionSourceKindLabel(entry.source_kind)}
            </Badge>
            {entry.visible_in_context ? (
              <Badge variant="secondary">
                {agentText(
                  "agentChat.harness.generated.87dfd0b2c8",
                  "上下文可见",
                )}
              </Badge>
            ) : null}
            {!entry.caller_allowed ? (
              <Badge variant="destructive">
                {agentText(
                  "agentChat.harness.generated.8a1c797eb2",
                  "Caller 拒绝",
                )}
              </Badge>
            ) : null}
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
            {entry.extension_name ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.81ca3b433b",
                  "extension：",
                )}
                {entry.extension_name}
              </Badge>
            ) : null}
            {entry.allowed_caller ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.82aec0037c",
                  "caller：",
                )}
                {entry.allowed_caller}
              </Badge>
            ) : null}
            {entry.deferred_loading ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.714ae55e88",
                  "Deferred",
                )}
              </Badge>
            ) : null}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {entry.description}
          </div>
        </div>
      ))}
    </div>
  );
}

function McpToolSection({
  entries,
}: {
  entries: AgentRuntimeToolInventory["mcp_tools"];
}) {
  return (
    <div className="space-y-3">
      <div className="text-sm font-medium text-foreground">
        {agentText("agentChat.harness.generated.1fa4eaed37", "MCP Tools")}
      </div>
      {entries.map((entry) => (
        <div
          key={`${entry.server_name}:${entry.name}`}
          className="rounded-xl border border-border bg-background p-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              {getMcpInnerToolName(entry.name, entry.server_name)}
            </span>
            <Badge variant="outline">{entry.server_name}</Badge>
            {entry.visible_in_context ? (
              <Badge variant="secondary">
                {agentText(
                  "agentChat.harness.generated.87dfd0b2c8",
                  "上下文可见",
                )}
              </Badge>
            ) : null}
            {entry.always_visible ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.6aec99f141",
                  "Always Visible",
                )}
              </Badge>
            ) : null}
            {entry.deferred_loading ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.714ae55e88",
                  "Deferred",
                )}
              </Badge>
            ) : null}
            {!entry.caller_allowed ? (
              <Badge variant="destructive">
                {agentText(
                  "agentChat.harness.generated.8a1c797eb2",
                  "Caller 拒绝",
                )}
              </Badge>
            ) : null}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {entry.description}
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
            {entry.allowed_callers.length > 0 ? (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.e8835d1775",
                  "callers：",
                )}
                {entry.allowed_callers.join(", ")}
              </Badge>
            ) : (
              <Badge variant="outline">
                {agentText(
                  "agentChat.harness.generated.1ba5809394",
                  "callers：全部",
                )}
              </Badge>
            )}
            {entry.tags.map((tag) => (
              <Badge
                key={`${entry.server_name}:${entry.name}:${tag}`}
                variant="outline"
              >
                {tag}
              </Badge>
            ))}
            <Badge variant="outline">
              {agentText(
                "agentChat.harness.generated.d434319af0",
                "input_examples：",
              )}
              {entry.input_examples_count}
            </Badge>
          </div>
        </div>
      ))}
    </div>
  );
}
