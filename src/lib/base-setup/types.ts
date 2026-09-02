import type {
  ServiceSkillArtifactKind,
  ServiceSkillCatalog,
  ServiceSkillAnyExecutionLocation,
  ServiceSkillAnyExecutorBinding,
  ServiceSkillPromptTemplateKey,
  ServiceSkillReadinessRequirements,
  ServiceSkillRunnerType,
  ServiceSkillSceneBinding,
  ServiceSkillBundleResourceSummary,
  ServiceSkillSlotDefinition,
  ServiceSkillSource,
  ServiceSkillSurfaceScope,
  ServiceSkillType,
} from "@/lib/api/serviceSkills";

export type AutomationOutputFormat = "text" | "json";
export type AutomationOutputSchema =
  | "text"
  | "json"
  | "table"
  | "csv"
  | "links";

export const BASE_SETUP_TARGET_CATALOGS = [
  "skill_catalog",
  "service_skill_catalog",
  "scene_catalog",
  "command_catalog",
] as const;

export const BASE_SETUP_VIEWER_KINDS = [
  "artifact_bundle",
  "document",
  "table_report",
] as const;

export const BASE_SETUP_DELIVERY_CONTRACTS = [
  "artifact_bundle",
  "project_pack",
  "table_report",
] as const;

export const BASE_SETUP_POLICY_ROLLOUT_STAGES = [
  "seeded",
  "bootstrap",
  "limited",
  "general",
] as const;

export const BASE_SETUP_ALLOWED_BINDING_FAMILIES = [
  "agent_turn",
  "automation_job",
  "native_skill",
] as const;

export const BASE_SETUP_ALLOWED_KERNEL_CAPABILITIES = [
  ...BASE_SETUP_ALLOWED_BINDING_FAMILIES,
  "workspace_storage",
  "artifact_viewer",
  "timeline",
] as const;

export const BASE_SETUP_ALLOWED_COMMAND_EXECUTION_KINDS = [
  "agent_turn",
  "automation_job",
  "native_skill",
  "task_queue",
  "server_api",
  "cli",
] as const;

export const BASE_SETUP_ALLOWED_RENDER_RESULT_KINDS = [
  "text",
  "tool_timeline",
  "image_gallery",
  "artifact",
  "form",
  "table_report",
] as const;

export const BASE_SETUP_ALLOWED_RENDER_DETAIL_KINDS = [
  "json",
  "task_detail",
  "artifact_detail",
  "media_detail",
  "scene_detail",
] as const;

export type BaseSetupTargetCatalog =
  (typeof BASE_SETUP_TARGET_CATALOGS)[number];
export type BaseSetupViewerKind = (typeof BASE_SETUP_VIEWER_KINDS)[number];
export type BaseSetupDeliveryContract =
  (typeof BASE_SETUP_DELIVERY_CONTRACTS)[number];
export type BaseSetupPolicyRolloutStage =
  (typeof BASE_SETUP_POLICY_ROLLOUT_STAGES)[number];
export type BaseSetupAllowedBindingFamily =
  (typeof BASE_SETUP_ALLOWED_BINDING_FAMILIES)[number];
export type BaseSetupAllowedKernelCapability =
  (typeof BASE_SETUP_ALLOWED_KERNEL_CAPABILITIES)[number];
export type BaseSetupCommandExecutionKind =
  (typeof BASE_SETUP_ALLOWED_COMMAND_EXECUTION_KINDS)[number];
export type BaseSetupRenderResultKind =
  (typeof BASE_SETUP_ALLOWED_RENDER_RESULT_KINDS)[number];
export type BaseSetupRenderDetailKind =
  (typeof BASE_SETUP_ALLOWED_RENDER_DETAIL_KINDS)[number];

export interface BaseSetupBundleRef {
  id: string;
  source: "builtin" | "remote" | "local";
  pathOrUri: string;
  kind: "skill_bundle" | "local_bundle";
  versionConstraint?: string;
}

export interface BaseSetupSlotProfile {
  id: string;
  slots: ServiceSkillSlotDefinition[];
}

export interface BaseSetupBindingProfile {
  id: string;
  bindingFamily: ServiceSkillAnyExecutorBinding;
  runnerType?: ServiceSkillRunnerType;
  executionLocation?: ServiceSkillAnyExecutionLocation;
  capabilityRefs?: string[];
  timeoutSecs?: number;
  retryLimit?: number;
}

export interface BaseSetupArtifactProfile {
  id: string;
  deliveryContract: BaseSetupDeliveryContract;
  requiredParts: string[];
  viewerKind: BaseSetupViewerKind;
  defaultArtifactKind?: ServiceSkillArtifactKind;
  outputDestination?: string;
}

export interface BaseSetupScorecardProfile {
  id: string;
  metrics: string[];
  failureSignals?: string[];
}

export interface BaseSetupAutomationScheduleEveryPreset {
  kind: "every";
  everySecs: number;
  slotKey?: string;
}

export interface BaseSetupAutomationScheduleCronPreset {
  kind: "cron";
  cronExpr: string;
  cronTz?: string;
  slotKey?: string;
}

export interface BaseSetupAutomationScheduleAtPreset {
  kind: "at";
  at: string;
  slotKey?: string;
}

export type BaseSetupAutomationSchedulePreset =
  | BaseSetupAutomationScheduleEveryPreset
  | BaseSetupAutomationScheduleCronPreset
  | BaseSetupAutomationScheduleAtPreset;

export interface BaseSetupAutomationDeliveryPreset {
  mode: "none" | "announce";
  channel?: "webhook" | "telegram" | "local_file" | "google_sheets";
  target?: string;
  outputSchema?: AutomationOutputSchema;
  outputFormat?: AutomationOutputFormat;
  bestEffort?: boolean;
}

export interface BaseSetupAutomationProfile {
  id: string;
  enabledByDefault?: boolean;
  schedule?: BaseSetupAutomationSchedulePreset;
  delivery?: BaseSetupAutomationDeliveryPreset;
  maxRetries?: number;
}

export interface BaseSetupPolicyProfile {
  id: string;
  enabled?: boolean;
  surfaceScopes?: ServiceSkillSurfaceScope[];
  rolloutStage?: BaseSetupPolicyRolloutStage;
}

export interface BaseSetupCompositionDeliveryContract {
  requiredParts: string[];
}

export interface BaseSetupCommandBinding {
  skillId?: string;
  executionKind?: BaseSetupCommandExecutionKind;
  requestDefaults?: Record<string, string>;
  intentConfirmation?: BaseSetupCommandIntentConfirmation;
}

export interface BaseSetupCommandIntentConfirmation {
  id: string;
  ruleKey: string;
  confirmationKey: string;
  systemPromptKey?: string;
}

export interface BaseSetupRenderContract {
  resultKind: BaseSetupRenderResultKind;
  detailKind: BaseSetupRenderDetailKind;
  supportsStreaming?: boolean;
  supportsTimeline?: boolean;
}

export interface BaseSetupCompositionStep {
  id: string;
  bindingProfileRef?: string;
}

export interface BaseSetupCompositionBlueprint {
  id: string;
  artifactProfileRef?: string;
  deliveryContract?: BaseSetupCompositionDeliveryContract;
  steps?: BaseSetupCompositionStep[];
}

export interface BaseSetupCatalogProjection {
  id: string;
  targetCatalog: BaseSetupTargetCatalog;
  entryKey: string;
  title: string;
  summary: string;
  category: string;
  outputHint: string;
  bundleRefId: string;
  slotProfileRef: string;
  bindingProfileRef: string;
  artifactProfileRef: string;
  scorecardProfileRef: string;
  policyProfileRef: string;
  automationProfileRef?: string;
  compositionBlueprintRef?: string;
  skillKey?: string;
  entryHint?: string;
  aliases?: string[];
  triggerHints?: string[];
  source?: ServiceSkillSource;
  skillType?: ServiceSkillType;
  readinessRequirements?: ServiceSkillReadinessRequirements;
  usageGuidelines?: string[];
  setupRequirements?: string[];
  examples?: string[];
  promptTemplateKey?: ServiceSkillPromptTemplateKey;
  themeTarget?: string;
  sceneBinding?: ServiceSkillSceneBinding;
  commandBinding?: BaseSetupCommandBinding;
  commandRenderContract?: BaseSetupRenderContract;
  version?: string;
  skillBundleMetadata?: Record<string, string>;
  skillBundleResourceSummary?: Partial<ServiceSkillBundleResourceSummary>;
}

export interface BaseSetupCompatibility {
  minAppVersion: string;
  requiredKernelCapabilities: string[];
  seededFallback?: boolean;
  compatCatalogProjection?: boolean;
}

export interface BaseSetupPackage {
  id: string;
  version: string;
  title: string;
  summary: string;
  bundleRefs: BaseSetupBundleRef[];
  catalogProjections: BaseSetupCatalogProjection[];
  slotProfiles: BaseSetupSlotProfile[];
  bindingProfiles: BaseSetupBindingProfile[];
  compositionBlueprints?: BaseSetupCompositionBlueprint[];
  artifactProfiles: BaseSetupArtifactProfile[];
  scorecardProfiles: BaseSetupScorecardProfile[];
  automationProfiles?: BaseSetupAutomationProfile[];
  policyProfiles: BaseSetupPolicyProfile[];
  compatibility: BaseSetupCompatibility;
}

export type BaseSetupValidationLevel = "L0" | "L1" | "L2";
export type BaseSetupValidationSeverity = "error" | "warning";

export interface BaseSetupValidationIssue {
  level: BaseSetupValidationLevel;
  severity: BaseSetupValidationSeverity;
  code: string;
  message: string;
  path?: string;
}

export interface BaseSetupValidationResult {
  ok: boolean;
  issues: BaseSetupValidationIssue[];
}

export interface BaseSetupProjectionIndex {
  artifactProfileRefsByProjectionId: Record<string, string>;
  scorecardProfileRefsByProjectionId: Record<string, string>;
  policyProfileRefsByProjectionId: Record<string, string>;
  automationProfileRefsByProjectionId: Record<string, string>;
  compositionBlueprintRefsByProjectionId: Record<string, string>;
}

export interface CompiledBaseSetupPackage {
  packageId: string;
  packageVersion: string;
  serviceSkillCatalogProjection: ServiceSkillCatalog;
  projectionIndex: BaseSetupProjectionIndex;
}

export type BaseSetupRolloutDecision =
  | "accept"
  | "fallback_seeded"
  | "reject_invalid_package"
  | "reject_upgrade_required";

export interface BaseSetupRolloutResult {
  decision: BaseSetupRolloutDecision;
  reason?: string;
}

export interface BaseSetupRolloutInput {
  appVersion: string;
  seededFallbackAvailable: boolean;
  supportedBindingFamilies?: readonly ServiceSkillAnyExecutorBinding[];
  supportedViewerKinds?: readonly BaseSetupViewerKind[];
  supportedKernelCapabilities?: readonly string[];
  validationResult?: BaseSetupValidationResult;
}
