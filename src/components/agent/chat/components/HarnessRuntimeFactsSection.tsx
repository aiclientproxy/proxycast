import { Badge } from "@/components/ui/badge";
import {
  HarnessStatusSection as Section,
  type HarnessSectionKey,
} from "./HarnessStatusSectionFrame";
import { agentText } from "./harnessPanelText";
import type { HarnessRuntimeFactSummary } from "./harnessStatusPanelViewModel";

interface HarnessRuntimeFactsSectionProps {
  runtimeFactSummary: HarnessRuntimeFactSummary;
  registerSectionRef: (
    key: HarnessSectionKey,
    node: HTMLElement | null,
  ) => void;
}

export function HarnessRuntimeFactsSection({
  runtimeFactSummary,
  registerSectionRef,
}: HarnessRuntimeFactsSectionProps) {
  return (
    <Section
      sectionKey="runtime-facts"
      title={agentText("agentChat.harness.generated.33926941a3", "运行时事实")}
      badge="current"
      registerRef={registerSectionRef}
    >
      <div className="space-y-3 rounded-2xl border border-sky-200 bg-sky-50/80 px-4 py-4">
        <div className="grid min-w-0 gap-2 text-xs text-muted-foreground [grid-template-columns:repeat(auto-fit,minmax(min(100%,9rem),1fr))]">
          <div>
            <span className="font-medium text-foreground">
              {agentText("agentChat.harness.runtimeFacts.thread", "线程")}
            </span>
            <span className="ml-1 break-all font-mono text-foreground">
              {runtimeFactSummary.threadId}
            </span>
          </div>
          <div>
            <span className="font-medium text-foreground">
              {agentText("agentChat.harness.runtimeFacts.status", "状态")}
            </span>
            <span className="ml-1 text-foreground">
              {runtimeFactSummary.status ?? "unknown"}
            </span>
          </div>
          <div>
            <span className="font-medium text-foreground">
              {agentText(
                "agentChat.harness.runtimeFacts.turnItem",
                "Turn / Item",
              )}
            </span>
            <span className="ml-1 font-mono text-foreground">
              {runtimeFactSummary.turnCount} / {runtimeFactSummary.itemCount}
            </span>
          </div>
          <div>
            <span className="font-medium text-foreground">
              {agentText(
                "agentChat.harness.runtimeFacts.pendingRequests",
                "待处理请求",
              )}
            </span>
            <span className="ml-1 font-mono text-foreground">
              {runtimeFactSummary.pendingRequestCount}
            </span>
          </div>
          <div>
            <span className="font-medium text-foreground">
              {agentText(
                "agentChat.harness.runtimeFacts.artifactEvidenceRefs",
                "Artifact / Evidence ref",
              )}
            </span>
            <span className="ml-1 font-mono text-foreground">
              {runtimeFactSummary.artifactCount} /{" "}
              {runtimeFactSummary.evidenceRefCount}
            </span>
          </div>
        </div>

        {runtimeFactSummary.decisionReason ? (
          <div className="text-sm text-slate-700">
            <span className="font-medium text-foreground">
              {agentText(
                "agentChat.harness.generated.6418acb28a",
                "决策原因：",
              )}
            </span>
            {runtimeFactSummary.decisionReason}
          </div>
        ) : null}

        {runtimeFactSummary.fallbackChain.length > 0 ? (
          <div className="text-sm text-slate-700">
            <span className="font-medium text-foreground">
              {agentText("agentChat.harness.generated.7dda9f12bc", "回退链：")}
            </span>
            {runtimeFactSummary.fallbackChain.join(" → ")}
          </div>
        ) : null}

        {runtimeFactSummary.oemPolicy ? (
          <RuntimeOemPolicyBlock oemPolicy={runtimeFactSummary.oemPolicy} />
        ) : null}
      </div>
    </Section>
  );
}

function RuntimeOemPolicyBlock({
  oemPolicy,
}: {
  oemPolicy: NonNullable<HarnessRuntimeFactSummary["oemPolicy"]>;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {oemPolicy.locked ? (
          <Badge
            variant="outline"
            className="border-amber-300 bg-white text-amber-700"
          >
            {agentText(
              "agentChat.harness.generated.8bda487786",
              "品牌云端托管锁定",
            )}
          </Badge>
        ) : null}
        {oemPolicy.quotaLow ? (
          <Badge
            variant="outline"
            className="border-orange-300 bg-white text-orange-700"
          >
            {agentText(
              "agentChat.harness.generated.f90b84fdd1",
              "品牌云端额度偏低",
            )}
          </Badge>
        ) : null}
        {oemPolicy.canInvoke === false ? (
          <Badge
            variant="outline"
            className="border-rose-300 bg-white text-rose-700"
          >
            {agentText(
              "agentChat.harness.generated.b5034aede8",
              "品牌云端当前不可调用",
            )}
          </Badge>
        ) : null}
        {oemPolicy.fallbackToLocalAllowed === true ? (
          <Badge
            variant="outline"
            className="border-emerald-300 bg-white text-emerald-700"
          >
            {agentText(
              "agentChat.harness.generated.b9044a46f4",
              "允许回退本地",
            )}
          </Badge>
        ) : null}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {oemPolicy.defaultModel || oemPolicy.selectedModel ? (
          <span>
            {agentText(
              "agentChat.harness.generated.ba5a177dbc",
              "品牌云端模型",
            )}{" "}
            {oemPolicy.defaultModel || oemPolicy.selectedModel}
          </span>
        ) : null}
        {oemPolicy.quotaStatus ? (
          <span>
            {agentText("agentChat.harness.generated.9689f96384", "额度状态")}{" "}
            {oemPolicy.quotaStatus}
          </span>
        ) : null}
        {oemPolicy.offerState ? (
          <span>
            {agentText("agentChat.harness.generated.eb4e63ff82", "策略状态")}
            {oemPolicy.offerState}
          </span>
        ) : null}
        {oemPolicy.providerSource ? (
          <span>
            {agentText("agentChat.harness.generated.c63f79e636", "来源")}
            {oemPolicy.providerSource}
          </span>
        ) : null}
        {oemPolicy.providerKey ? (
          <span>
            {agentText(
              "agentChat.harness.generated.2684f75e20",
              "Provider Key",
            )}{" "}
            {oemPolicy.providerKey}
          </span>
        ) : null}
        {oemPolicy.tenantId ? (
          <span>
            {agentText("agentChat.harness.generated.cc04fa896e", "租户")}
            {oemPolicy.tenantId}
          </span>
        ) : null}
      </div>
    </div>
  );
}
