import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  type CapabilityDraftRegistrationApprovalRequest,
  type CapabilityDraftRegistrationVerificationGate,
  type CapabilityDraftVerificationEvidence,
  type WorkspaceRegisteredSkillRecord,
} from "@/lib/api/capabilityDrafts";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { cn } from "@/lib/utils";

interface WorkspaceRegisteredSummaryCopy {
  bindingBlockedFallback: string;
  bindingCandidateFallback: string;
  bindingPending: string;
  permissionEmpty: string;
  resourceEmpty: string;
  standardPassed: string;
  standardPending: string;
  formatStandardIssueCount: (count: number) => string;
}

interface ReadonlyHttpApprovalPreviewCopy {
  notGenerated: string;
  notRecorded: string;
  previewOnly: string;
}

function summarizePermissionSummary(
  skill: WorkspaceRegisteredSkillRecord,
  copy: WorkspaceRegisteredSummaryCopy,
) {
  if (skill.permissionSummary.length === 0) {
    return copy.permissionEmpty;
  }
  return skill.permissionSummary.slice(0, 2).join(" / ");
}

function summarizeResourceSummary(
  skill: WorkspaceRegisteredSkillRecord,
  copy: WorkspaceRegisteredSummaryCopy,
) {
  const resources = [
    skill.resourceSummary.hasScripts ? "scripts" : null,
    skill.resourceSummary.hasReferences ? "references" : null,
    skill.resourceSummary.hasAssets ? "assets" : null,
  ].filter((item): item is string => Boolean(item));

  return resources.length > 0 ? resources.join(" / ") : copy.resourceEmpty;
}

function summarizeStandardCompliance(
  skill: WorkspaceRegisteredSkillRecord,
  copy: WorkspaceRegisteredSummaryCopy,
) {
  if (skill.standardCompliance.validationErrors.length > 0) {
    return copy.formatStandardIssueCount(
      skill.standardCompliance.validationErrors.length,
    );
  }
  return skill.standardCompliance.isStandard
    ? copy.standardPassed
    : copy.standardPending;
}

function summarizeBindingStatus(
  binding: AgentRuntimeWorkspaceSkillBinding | undefined,
  copy: WorkspaceRegisteredSummaryCopy,
) {
  if (!binding) {
    return copy.bindingPending;
  }
  if (binding.binding_status === "blocked") {
    return binding.binding_status_reason || copy.bindingBlockedFallback;
  }
  return binding.binding_status_reason || copy.bindingCandidateFallback;
}

const REGISTRATION_EVIDENCE_LABELS: Record<string, string> = {
  credentialReferenceId: "凭证引用",
  endpointSource: "Endpoint",
  evidenceSchema: "证据 Schema",
  method: "方法",
  policyPath: "Policy",
  preflightMode: "Preflight",
};

const READONLY_HTTP_PREFLIGHT_CHECK_ID = "readonly_http_execution_preflight";

function formatRegistrationEvidenceKey(
  key: string,
  labels: Record<string, string>,
): string {
  return labels[key] ?? REGISTRATION_EVIDENCE_LABELS[key] ?? key;
}

function formatRegistrationEvidenceValue(
  evidence: CapabilityDraftVerificationEvidence,
) {
  return evidence.value.trim().replace(/\s+/g, " ");
}

function findRegistrationEvidenceValue(
  gate: CapabilityDraftRegistrationVerificationGate,
  key: string,
  copy: ReadonlyHttpApprovalPreviewCopy,
) {
  const evidence = gate.evidence.find((item) => item.key === key);
  return evidence
    ? formatRegistrationEvidenceValue(evidence)
    : copy.notRecorded;
}

function buildReadonlyHttpApprovalPreview(
  gate?: CapabilityDraftRegistrationVerificationGate,
  approvalRequest?: CapabilityDraftRegistrationApprovalRequest,
  copy?: ReadonlyHttpApprovalPreviewCopy,
) {
  if (!gate) {
    return null;
  }
  const fallback: ReadonlyHttpApprovalPreviewCopy = copy ?? {
    notGenerated: "未生成",
    notRecorded: "未记录",
    previewOnly: "preview_only",
  };

  return {
    approvalId: approvalRequest?.approvalId ?? fallback.notGenerated,
    createdAt: approvalRequest?.createdAt ?? fallback.notRecorded,
    status: approvalRequest?.status ?? fallback.previewOnly,
    credentialReferenceId:
      approvalRequest?.credentialReferenceId ??
      findRegistrationEvidenceValue(gate, "credentialReferenceId", fallback),
    endpointSource:
      approvalRequest?.endpointSource ??
      findRegistrationEvidenceValue(gate, "endpointSource", fallback),
    evidenceSchema:
      approvalRequest?.evidenceSchema.join(",") ??
      findRegistrationEvidenceValue(gate, "evidenceSchema", fallback),
    method:
      approvalRequest?.method ??
      findRegistrationEvidenceValue(gate, "method", fallback),
    policyPath:
      approvalRequest?.policyPath ??
      findRegistrationEvidenceValue(gate, "policyPath", fallback),
    consumptionGate: approvalRequest?.consumptionGate ?? null,
    credentialResolver: approvalRequest?.credentialResolver ?? null,
    consumptionInputSchema: approvalRequest?.consumptionInputSchema ?? null,
    sessionInputIntake: approvalRequest?.sessionInputIntake ?? null,
    sessionInputSubmissionContract:
      approvalRequest?.sessionInputSubmissionContract ?? null,
  };
}

export function WorkspaceRegisteredSkillRegistrationDetails({
  skill,
  binding,
}: {
  skill: WorkspaceRegisteredSkillRecord;
  binding?: AgentRuntimeWorkspaceSkillBinding;
}) {
  const { t } = useTranslation("agent");
  const summaryCopy = useMemo<WorkspaceRegisteredSummaryCopy>(
    () => ({
      bindingBlockedFallback: t(
        "capabilityDraft.registeredPanel.summary.bindingBlockedFallback",
        "Runtime binding 当前被 gate 阻断。",
      ),
      bindingCandidateFallback: t(
        "capabilityDraft.registeredPanel.summary.bindingCandidateFallback",
        "已具备后续 runtime binding 候选资格，但当前仍未进入默认工具面。",
      ),
      bindingPending: t(
        "capabilityDraft.registeredPanel.summary.bindingPending",
        "等待 runtime binding readiness 盘点。",
      ),
      formatStandardIssueCount: (count) =>
        t("capabilityDraft.registeredPanel.summary.standardIssueCount", {
          defaultValue: "标准检查仍有 {{count}} 个问题",
          count,
        }),
      permissionEmpty: t(
        "capabilityDraft.registeredPanel.summary.permissionEmpty",
        "未声明额外权限，默认停留在只读发现与注册审计。",
      ),
      resourceEmpty: t(
        "capabilityDraft.registeredPanel.summary.resourceEmpty",
        "纯 Skill 说明",
      ),
      standardPassed: t(
        "capabilityDraft.registeredPanel.summary.standardPassed",
        "Agent Skills 标准通过",
      ),
      standardPending: t(
        "capabilityDraft.registeredPanel.summary.standardPending",
        "Agent Skills 标准状态待确认",
      ),
    }),
    [t],
  );
  const evidenceLabels = useMemo<Record<string, string>>(
    () => ({
      credentialReferenceId: t(
        "capabilityDraft.registeredPanel.evidence.credentialReferenceId",
        REGISTRATION_EVIDENCE_LABELS.credentialReferenceId,
      ),
      endpointSource: t(
        "capabilityDraft.registeredPanel.evidence.endpointSource",
        REGISTRATION_EVIDENCE_LABELS.endpointSource,
      ),
      evidenceSchema: t(
        "capabilityDraft.registeredPanel.evidence.evidenceSchema",
        REGISTRATION_EVIDENCE_LABELS.evidenceSchema,
      ),
      method: t(
        "capabilityDraft.registeredPanel.evidence.method",
        REGISTRATION_EVIDENCE_LABELS.method,
      ),
      policyPath: t(
        "capabilityDraft.registeredPanel.evidence.policyPath",
        REGISTRATION_EVIDENCE_LABELS.policyPath,
      ),
      preflightMode: t(
        "capabilityDraft.registeredPanel.evidence.preflightMode",
        REGISTRATION_EVIDENCE_LABELS.preflightMode,
      ),
    }),
    [t],
  );
  const approvalPreviewCopy = useMemo<ReadonlyHttpApprovalPreviewCopy>(
    () => ({
      notGenerated: t(
        "capabilityDraft.registeredPanel.approval.notGenerated",
        "未生成",
      ),
      notRecorded: t(
        "capabilityDraft.registeredPanel.approval.notRecorded",
        "未记录",
      ),
      previewOnly: t(
        "capabilityDraft.registeredPanel.approval.previewOnly",
        "preview_only",
      ),
    }),
    [t],
  );
  const preflightGate = skill.registration.verificationGates?.find(
    (gate) => gate.checkId === READONLY_HTTP_PREFLIGHT_CHECK_ID,
  );
  const approvalRequest = skill.registration.approvalRequests?.find(
    (request) => request.sourceCheckId === READONLY_HTTP_PREFLIGHT_CHECK_ID,
  );
  const approvalPreview = buildReadonlyHttpApprovalPreview(
    preflightGate,
    approvalRequest,
    approvalPreviewCopy,
  );

  return (
    <details className="mt-3 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-[11px] leading-5 text-slate-500">
      <summary className="cursor-pointer select-none text-[12px] font-medium text-slate-700">
        {t("capabilityDraft.registeredPanel.technicalDetails", "技术详情")}
      </summary>
      <div className="mt-2 space-y-1">
        <div>
          <span className="font-medium text-slate-700">
            {t(
              "capabilityDraft.registeredPanel.card.field.directory",
              "目录：",
            )}
          </span>
          {skill.directory}
        </div>
        <div>
          <span className="font-medium text-slate-700">
            {t("capabilityDraft.registeredPanel.card.field.source", "来源：")}
          </span>
          {skill.registration.sourceDraftId}
          {skill.registration.sourceVerificationReportId
            ? ` / ${skill.registration.sourceVerificationReportId}`
            : ""}
        </div>
        <div>
          <span className="font-medium text-slate-700">
            {t(
              "capabilityDraft.registeredPanel.card.field.permission",
              "权限：",
            )}
          </span>
          {summarizePermissionSummary(skill, summaryCopy)}
        </div>
        <div>
          <span className="font-medium text-slate-700">
            {t("capabilityDraft.registeredPanel.card.field.resource", "资源：")}
          </span>
          {summarizeResourceSummary(skill, summaryCopy)}
        </div>
        <div>
          <span className="font-medium text-slate-700">
            {t("capabilityDraft.registeredPanel.card.field.standard", "标准：")}
          </span>
          {summarizeStandardCompliance(skill, summaryCopy)}
        </div>
        <div>
          <span className="font-medium text-slate-700">
            {t(
              "capabilityDraft.registeredPanel.card.field.runtimeBinding",
              "运行绑定：",
            )}
          </span>
          {summarizeBindingStatus(binding, summaryCopy)}
        </div>
        <div className="text-sky-700">
          {t(
            "capabilityDraft.registeredPanel.card.field.nextGate",
            "下一道 gate：",
          )}
          {binding?.next_gate ||
            t(
              "capabilityDraft.registeredPanel.card.nextGateFallback",
              "manual_runtime_enable / Query Loop metadata / tool_runtime 授权裁剪",
            )}
        </div>
      </div>
      {preflightGate ? (
        <div className="mt-3 rounded-2xl border border-sky-100 bg-white px-3 py-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-[11px] font-semibold text-slate-800">
              {t(
                "capabilityDraft.registeredPanel.provenance.title",
                "注册 provenance",
              )}
            </span>
            <span className="text-[10px] leading-4 text-sky-700">
              {preflightGate.label || preflightGate.checkId}
            </span>
          </div>
          <div className="mt-2 grid gap-1.5 sm:grid-cols-2">
            {preflightGate.evidence.slice(0, 6).map((evidence) => (
              <div
                key={`${preflightGate.checkId}:${evidence.key}`}
                className="rounded-xl border border-sky-100 bg-sky-50 px-2.5 py-1.5"
              >
                <div className="text-[10px] leading-4 text-slate-400">
                  {formatRegistrationEvidenceKey(evidence.key, evidenceLabels)}
                </div>
                <div className="truncate font-mono text-[10px] leading-4 text-slate-700">
                  {formatRegistrationEvidenceValue(evidence)}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {approvalPreview ? (
        <div className="mt-3 rounded-2xl border border-amber-100 bg-amber-50 px-3 py-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-[11px] font-semibold text-amber-900">
              {t(
                "capabilityDraft.registeredPanel.approval.title",
                "Session approval request artifact",
              )}
            </span>
            <span className="rounded-full border border-amber-200 bg-white px-2 py-0.5 text-[10px] font-medium text-amber-700">
              {t("capabilityDraft.registeredPanel.approval.statusSummary", {
                defaultValue: "{{status}} / 未执行 / 未保存凭证",
                status: approvalPreview.status,
              })}
            </span>
          </div>
          <p className="mt-1.5 text-[11px] leading-5 text-amber-800">
            {t(
              "capabilityDraft.registeredPanel.approval.description",
              "真实 API 执行前必须先消费这条授权请求 artifact；当前只持久化审计入口，不保存 token，也不发请求。",
            )}
          </p>
          {approvalPreview.consumptionGate ? (
            <div className="mt-2 rounded-xl border border-amber-200 bg-white px-2.5 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[10px] font-semibold text-amber-700">
                  {t(
                    "capabilityDraft.registeredPanel.approval.consumptionGate.title",
                    "消费门禁",
                  )}
                </span>
                <span className="font-mono text-[10px] text-slate-700">
                  {approvalPreview.consumptionGate.status}
                </span>
              </div>
              <p className="mt-1 text-[10px] leading-4 text-amber-800">
                {approvalPreview.consumptionGate.blockedReason}
              </p>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {approvalPreview.consumptionGate.requiredInputs.map((input) => (
                  <span
                    key={input}
                    className="rounded-full border border-amber-100 bg-amber-50 px-2 py-0.5 font-mono text-[10px] text-amber-800"
                  >
                    {input}
                  </span>
                ))}
              </div>
              <div className="mt-1.5 text-[10px] leading-4 text-slate-600">
                {t(
                  "capabilityDraft.registeredPanel.approval.flag.runtimeExecution",
                  {
                    defaultValue: "runtimeExecution={{value}}",
                    value: String(
                      approvalPreview.consumptionGate.runtimeExecutionEnabled,
                    ),
                  },
                )}{" "}
                /{" "}
                {t(
                  "capabilityDraft.registeredPanel.approval.flag.credentialStorage",
                  {
                    defaultValue: "credentialStorage={{value}}",
                    value: String(
                      approvalPreview.consumptionGate.credentialStorageEnabled,
                    ),
                  },
                )}
              </div>
            </div>
          ) : null}
          {approvalPreview.credentialResolver ? (
            <div className="mt-2 rounded-xl border border-amber-200 bg-white px-2.5 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[10px] font-semibold text-amber-700">
                  {t(
                    "capabilityDraft.registeredPanel.approval.credentialResolver.title",
                    "Session credential resolver",
                  )}
                </span>
                <span className="font-mono text-[10px] text-slate-700">
                  {approvalPreview.credentialResolver.status}
                </span>
              </div>
              <p className="mt-1 text-[10px] leading-4 text-amber-800">
                {approvalPreview.credentialResolver.blockedReason}
              </p>
              <div className="mt-1.5 grid gap-1 sm:grid-cols-2">
                {[
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.reference",
                      "Reference",
                    ),
                    approvalPreview.credentialResolver.referenceId,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.scope",
                      "Scope",
                    ),
                    approvalPreview.credentialResolver.scope,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.source",
                      "Source",
                    ),
                    approvalPreview.credentialResolver.source,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.secret",
                      "Secret",
                    ),
                    approvalPreview.credentialResolver.secretMaterialStatus,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.tokenPersisted",
                      "tokenPersisted",
                    ),
                    String(approvalPreview.credentialResolver.tokenPersisted),
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.runtimeInjection",
                      "runtimeInjection",
                    ),
                    String(
                      approvalPreview.credentialResolver
                        .runtimeInjectionEnabled,
                    ),
                  ],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-lg border border-amber-100 bg-amber-50 px-2 py-1"
                  >
                    <span className="text-[10px] text-amber-600">{label}</span>
                    <span className="ml-1 break-words font-mono text-[10px] text-slate-700">
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {approvalPreview.consumptionInputSchema ? (
            <div className="mt-2 rounded-xl border border-amber-200 bg-white px-2.5 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[10px] font-semibold text-amber-700">
                  {t(
                    "capabilityDraft.registeredPanel.approval.inputSchema.title",
                    "Approval consumption input schema",
                  )}
                </span>
                <span className="font-mono text-[10px] text-slate-700">
                  {approvalPreview.consumptionInputSchema.schemaId}
                </span>
              </div>
              <p className="mt-1 text-[10px] leading-4 text-amber-800">
                {approvalPreview.consumptionInputSchema.blockedReason}
              </p>
              <div className="mt-1.5 text-[10px] leading-4 text-slate-600">
                {t(
                  "capabilityDraft.registeredPanel.approval.flag.uiSubmission",
                  {
                    defaultValue: "uiSubmission={{value}}",
                    value: String(
                      approvalPreview.consumptionInputSchema
                        .uiSubmissionEnabled,
                    ),
                  },
                )}{" "}
                /{" "}
                {t(
                  "capabilityDraft.registeredPanel.approval.flag.runtimeExecution",
                  {
                    defaultValue: "runtimeExecution={{value}}",
                    value: String(
                      approvalPreview.consumptionInputSchema
                        .runtimeExecutionEnabled,
                    ),
                  },
                )}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {approvalPreview.consumptionInputSchema.fields.map((field) => (
                  <span
                    key={field.key}
                    className="rounded-full border border-amber-100 bg-amber-50 px-2 py-0.5 font-mono text-[10px] text-amber-800"
                    title={field.description}
                  >
                    {field.key}:{field.kind}
                    {field.required
                      ? t(
                          "capabilityDraft.registeredPanel.approval.suffix.required",
                          ":required",
                        )
                      : ""}
                    {field.secret
                      ? t(
                          "capabilityDraft.registeredPanel.approval.suffix.secret",
                          ":secret",
                        )
                      : ""}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
          {approvalPreview.sessionInputIntake ? (
            <div className="mt-2 rounded-xl border border-amber-200 bg-white px-2.5 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[10px] font-semibold text-amber-700">
                  {t(
                    "capabilityDraft.registeredPanel.approval.sessionInputIntake.title",
                    "Session input intake",
                  )}
                </span>
                <span className="font-mono text-[10px] text-slate-700">
                  {approvalPreview.sessionInputIntake.status}
                </span>
              </div>
              <p className="mt-1 text-[10px] leading-4 text-amber-800">
                {approvalPreview.sessionInputIntake.blockedReason}
              </p>
              <div className="mt-1.5 grid gap-1 sm:grid-cols-2">
                {[
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.schema",
                      "Schema",
                    ),
                    approvalPreview.sessionInputIntake.schemaId,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.scope",
                      "Scope",
                    ),
                    approvalPreview.sessionInputIntake.scope,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.credential",
                      "Credential",
                    ),
                    approvalPreview.sessionInputIntake.credentialReferenceId,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.secret",
                      "Secret",
                    ),
                    approvalPreview.sessionInputIntake.secretMaterialStatus,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.endpointPersisted",
                      "endpointPersisted",
                    ),
                    String(
                      approvalPreview.sessionInputIntake.endpointInputPersisted,
                    ),
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.tokenPersisted",
                      "tokenPersisted",
                    ),
                    String(approvalPreview.sessionInputIntake.tokenPersisted),
                  ],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-lg border border-amber-100 bg-amber-50 px-2 py-1"
                  >
                    <span className="text-[10px] text-amber-600">{label}</span>
                    <span className="ml-1 break-words font-mono text-[10px] text-slate-700">
                      {value}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-1.5 text-[10px] leading-4 text-slate-600">
                {t(
                  "capabilityDraft.registeredPanel.approval.flag.uiSubmission",
                  {
                    defaultValue: "uiSubmission={{value}}",
                    value: String(
                      approvalPreview.sessionInputIntake.uiSubmissionEnabled,
                    ),
                  },
                )}{" "}
                /{" "}
                {t(
                  "capabilityDraft.registeredPanel.approval.flag.runtimeExecution",
                  {
                    defaultValue: "runtimeExecution={{value}}",
                    value: String(
                      approvalPreview.sessionInputIntake
                        .runtimeExecutionEnabled,
                    ),
                  },
                )}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {approvalPreview.sessionInputIntake.missingFieldKeys.map(
                  (fieldKey) => (
                    <span
                      key={fieldKey}
                      className="rounded-full border border-amber-100 bg-amber-50 px-2 py-0.5 font-mono text-[10px] text-amber-800"
                    >
                      {t(
                        "capabilityDraft.registeredPanel.approval.prefix.missing",
                        "missing:",
                      )}
                      {fieldKey}
                    </span>
                  ),
                )}
              </div>
            </div>
          ) : null}
          {approvalPreview.sessionInputSubmissionContract ? (
            <div className="mt-2 rounded-xl border border-amber-200 bg-white px-2.5 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[10px] font-semibold text-amber-700">
                  {t(
                    "capabilityDraft.registeredPanel.approval.sessionSubmissionContract.title",
                    "Session submission contract",
                  )}
                </span>
                <span className="font-mono text-[10px] text-slate-700">
                  {approvalPreview.sessionInputSubmissionContract.status}
                </span>
              </div>
              <p className="mt-1 text-[10px] leading-4 text-amber-800">
                {approvalPreview.sessionInputSubmissionContract.blockedReason}
              </p>
              <div className="mt-1.5 grid gap-1 sm:grid-cols-2">
                {[
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.mode",
                      "Mode",
                    ),
                    approvalPreview.sessionInputSubmissionContract.mode,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.retention",
                      "Retention",
                    ),
                    approvalPreview.sessionInputSubmissionContract
                      .valueRetention,
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.submitHandler",
                      "submitHandler",
                    ),
                    String(
                      approvalPreview.sessionInputSubmissionContract
                        .submissionHandlerEnabled,
                    ),
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.secretAccepted",
                      "secretAccepted",
                    ),
                    String(
                      approvalPreview.sessionInputSubmissionContract
                        .secretMaterialAccepted,
                    ),
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.evidenceRequired",
                      "evidenceRequired",
                    ),
                    String(
                      approvalPreview.sessionInputSubmissionContract
                        .evidenceCaptureRequired,
                    ),
                  ],
                  [
                    t(
                      "capabilityDraft.registeredPanel.approval.label.runtimeExecution",
                      "runtimeExecution",
                    ),
                    String(
                      approvalPreview.sessionInputSubmissionContract
                        .runtimeExecutionEnabled,
                    ),
                  ],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-lg border border-amber-100 bg-amber-50 px-2 py-1"
                  >
                    <span className="text-[10px] text-amber-600">{label}</span>
                    <span className="ml-1 break-words font-mono text-[10px] text-slate-700">
                      {value}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {approvalPreview.sessionInputSubmissionContract.validationRules.map(
                  (rule) => (
                    <span
                      key={rule.fieldKey}
                      className="rounded-full border border-amber-100 bg-amber-50 px-2 py-0.5 font-mono text-[10px] text-amber-800"
                      title={rule.rule}
                    >
                      {t(
                        "capabilityDraft.registeredPanel.approval.prefix.validate",
                        "validate:",
                      )}
                      {rule.fieldKey}:{rule.kind}
                      {rule.required
                        ? t(
                            "capabilityDraft.registeredPanel.approval.suffix.required",
                            ":required",
                          )
                        : ""}
                    </span>
                  ),
                )}
              </div>
            </div>
          ) : null}
          <div className="mt-2 grid gap-1.5 sm:grid-cols-2">
            {[
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.approvalId",
                  "Approval ID",
                ),
                value: approvalPreview.approvalId,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.status",
                  "状态",
                ),
                value: approvalPreview.status,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.endpoint",
                  "Endpoint",
                ),
                value: approvalPreview.endpointSource,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.method",
                  "方法",
                ),
                value: approvalPreview.method,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.credentialReference",
                  "凭证引用",
                ),
                value: approvalPreview.credentialReferenceId,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.policy",
                  "Policy",
                ),
                value: approvalPreview.policyPath,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.createdAt",
                  "创建时间",
                ),
                value: approvalPreview.createdAt,
              },
              {
                label: t(
                  "capabilityDraft.registeredPanel.approval.label.evidenceSchema",
                  "证据 Schema",
                ),
                value: approvalPreview.evidenceSchema,
                wide: true,
              },
            ].map(({ label, value, wide }) => (
              <div
                key={label}
                className={cn(
                  "rounded-xl border border-amber-100 bg-white px-2.5 py-1.5",
                  wide && "sm:col-span-2",
                )}
              >
                <div className="text-[10px] leading-4 text-amber-600">
                  {label}
                </div>
                <div className="break-words font-mono text-[10px] leading-4 text-slate-700">
                  {value}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </details>
  );
}
