#!/usr/bin/env tsx

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export type I18nRoadmapReadinessReportFormat = "text" | "json";

type I18nRoadmapCheckStatus = "failed" | "passed";
type I18nRoadmapPhaseStatus = "incomplete" | "ready" | "ready-with-known-gaps";

interface I18nRoadmapEvidenceRef {
  exists: boolean;
  path: string;
  schemaVersion: string | null;
}

interface I18nRoadmapCheck {
  evidencePath: string;
  id: string;
  notes: string[];
  requirement: string;
  signals: Record<string, unknown>;
  status: I18nRoadmapCheckStatus;
}

interface I18nRoadmapKnownGap {
  evidencePath: string;
  id: string;
  phase: string;
  severity: "blocker" | "decision" | "follow-up";
  summary: string;
}

interface I18nRoadmapPhase {
  acceptance: I18nRoadmapCheck[];
  deliverables: I18nRoadmapCheck[];
  id: "P0" | "P1" | "P2" | "P3" | "P4";
  knownGaps: I18nRoadmapKnownGap[];
  status: I18nRoadmapPhaseStatus;
  summary: {
    acceptanceFailedCount: number;
    acceptancePassedCount: number;
    deliverableFailedCount: number;
    deliverablePassedCount: number;
    knownGapCount: number;
  };
  title: string;
}

export interface I18nRoadmapReadinessReport {
  evidence: {
    bundleStrategy: I18nRoadmapEvidenceRef;
    contentTargetLanguageBoundary: I18nRoadmapEvidenceRef;
    i18nextCliParityBenchmark: I18nRoadmapEvidenceRef;
    languageBoundary: I18nRoadmapEvidenceRef;
    p4Readiness: I18nRoadmapEvidenceRef;
    patchRetirementGate: I18nRoadmapEvidenceRef;
    sourceLocaleExport: I18nRoadmapEvidenceRef;
    translationCoverage: I18nRoadmapEvidenceRef;
    translationPrPack: I18nRoadmapEvidenceRef;
  };
  phases: I18nRoadmapPhase[];
  repoRoot: string;
  schemaVersion: string;
  summary: {
    acceptanceFailedCount: number;
    acceptancePassedCount: number;
    deliverableFailedCount: number;
    deliverablePassedCount: number;
    knownGapCount: number;
    missingEvidenceCount: number;
    overallStatus: I18nRoadmapPhaseStatus;
    phaseCount: number;
    phaseIncompleteCount: number;
    phaseReadyCount: number;
    phaseReadyWithKnownGapsCount: number;
  };
}

interface CliOptions {
  format: I18nRoadmapReadinessReportFormat;
  output?: string;
  repoRoot: string;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_REPO_ROOT = path.resolve(__dirname, "../..");

const EVIDENCE_PATHS = {
  bundleStrategy: "internal/roadmap/i18n/evidence/bundle-strategy-report.json",
  contentTargetLanguageBoundary:
    "internal/roadmap/i18n/evidence/content-target-language-boundary-report.json",
  i18nextCliParityBenchmark:
    "internal/roadmap/i18n/evidence/i18next-cli-parity-benchmark.json",
  languageBoundary:
    "internal/roadmap/i18n/evidence/language-boundary-report.json",
  p4Readiness: "internal/roadmap/i18n/evidence/p4-readiness-report.json",
  patchRetirementGate:
    "internal/roadmap/i18n/evidence/patch-retirement-gate-report.json",
  sourceLocaleExport:
    "internal/roadmap/i18n/evidence/source-locale-export.json",
  translationCoverage:
    "internal/roadmap/i18n/evidence/translation-coverage-report.json",
  translationPrPack: "internal/roadmap/i18n/evidence/translation-pr-pack.json",
} as const;

const SOURCE_PATHS = {
  appSidebarTest: "src/components/AppSidebar.test.tsx",
  appConfigTypes: "src/lib/api/appConfigTypes.ts",
  appearanceSettings: "src/components/settings-v2/general/appearance/index.tsx",
  createI18n: "src/i18n/createI18n.ts",
  glossary: "internal/roadmap/i18n/glossary.md",
  legacyPatchProvider: "src/i18n/legacy-patch/I18nPatchProvider.tsx",
  loadNamespaceTest: "src/i18n/__tests__/loadNamespace.test.ts",
  locales: "src/i18n/locales.ts",
  localesTest: "src/i18n/__tests__/locales.test.ts",
  packageJson: "package.json",
  prTemplate: ".github/pull_request_template.md",
  qualityPlanner: "scripts/quality-task-planner.mjs",
} as const;

const CORE_NAMESPACES = [
  "common",
  "navigation",
  "settings",
  "workspace",
  "errors",
] as const;

const P2_NAMESPACES = [
  "agent",
  "agentInputbar",
  "agentRuntime",
  "agentSkills",
  "workspace",
  "errors",
] as const;

function normalizePath(filePath: string): string {
  return filePath.split(path.sep).join("/");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function fileExists(filePath: string): boolean {
  return fs.existsSync(filePath) && fs.statSync(filePath).isFile();
}

function readTextIfExists(filePath: string): string {
  return fileExists(filePath) ? fs.readFileSync(filePath, "utf8") : "";
}

function readJsonObject(filePath: string): Record<string, unknown> | null {
  if (!fileExists(filePath)) {
    return null;
  }
  const parsed = JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
  return isRecord(parsed) ? parsed : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readBoolean(record: Record<string, unknown>, key: string): boolean {
  return record[key] === true;
}

function readNumber(record: Record<string, unknown>, key: string): number {
  return typeof record[key] === "number" ? record[key] : 0;
}

function readNestedRecord(
  record: Record<string, unknown> | null,
  keys: string[],
): Record<string, unknown> {
  let current: unknown = record;
  for (const key of keys) {
    if (!isRecord(current)) {
      return {};
    }
    current = current[key];
  }
  return isRecord(current) ? current : {};
}

function readEvidence(
  repoRoot: string,
  relativePath: string,
): { ref: I18nRoadmapEvidenceRef; report: Record<string, unknown> | null } {
  const report = readJsonObject(path.join(repoRoot, relativePath));
  return {
    ref: {
      exists: Boolean(report),
      path: relativePath,
      schemaVersion: readString(report?.schemaVersion),
    },
    report,
  };
}

function buildCheck(
  id: string,
  requirement: string,
  evidencePath: string,
  passed: boolean,
  signals: Record<string, unknown>,
  notes: string[] = [],
): I18nRoadmapCheck {
  return {
    evidencePath,
    id,
    notes,
    requirement,
    signals,
    status: passed ? "passed" : "failed",
  };
}

function countPassed(checks: I18nRoadmapCheck[]): number {
  return checks.filter((check) => check.status === "passed").length;
}

function getPackageScripts(
  packageJson: Record<string, unknown> | null,
): Record<string, unknown> {
  return isRecord(packageJson?.scripts) ? packageJson.scripts : {};
}

function hasScript(
  packageJson: Record<string, unknown> | null,
  scriptName: string,
): boolean {
  return typeof getPackageScripts(packageJson)[scriptName] === "string";
}

function getNamespaceNames(
  report: Record<string, unknown> | null,
): Set<string> {
  const namespaces = Array.isArray(report?.namespaces) ? report.namespaces : [];
  return new Set(
    namespaces
      .map((namespace) =>
        isRecord(namespace) ? readString(namespace.namespace) : null,
      )
      .filter((namespace): namespace is string => Boolean(namespace)),
  );
}

function hasNamespaces(
  report: Record<string, unknown> | null,
  namespaces: readonly string[],
): boolean {
  const namespaceSet = getNamespaceNames(report);
  return namespaces.every((namespace) => namespaceSet.has(namespace));
}

function sourceExportContains(
  report: Record<string, unknown> | null,
  fragment: string,
): boolean {
  if (!report) {
    return false;
  }
  return JSON.stringify(report).includes(fragment);
}

function readCoverageSummary(
  report: Record<string, unknown> | null,
): Record<string, unknown> {
  return readNestedRecord(report, ["coverage", "summary"]);
}

function readTranslationCoverageLocales(
  report: Record<string, unknown> | null,
): string[] {
  return Array.isArray(report?.locales)
    ? report.locales.filter(
        (locale): locale is string => typeof locale === "string",
      )
    : [];
}

function readLanguageCategoryCount(
  report: Record<string, unknown> | null,
  category: string,
): number {
  const summary = readNestedRecord(report, ["summary"]);
  const summaries = Array.isArray(summary.categorySummaries)
    ? summary.categorySummaries
    : [];

  for (const item of summaries) {
    if (isRecord(item) && item.category === category) {
      return typeof item.count === "number" ? item.count : 0;
    }
  }

  return 0;
}

function uniquePositiveCounts(counts: Record<string, number>): {
  distinct: number[];
  values: Record<string, number>;
} {
  const values = Object.fromEntries(
    Object.entries(counts).filter(([, value]) => value > 0),
  );
  return {
    distinct: Array.from(new Set(Object.values(values))).sort((a, b) => a - b),
    values,
  };
}

function buildPhase(
  id: I18nRoadmapPhase["id"],
  title: string,
  deliverables: I18nRoadmapCheck[],
  acceptance: I18nRoadmapCheck[],
  knownGaps: I18nRoadmapKnownGap[] = [],
): I18nRoadmapPhase {
  const deliverablePassedCount = countPassed(deliverables);
  const acceptancePassedCount = countPassed(acceptance);
  const deliverableFailedCount = deliverables.length - deliverablePassedCount;
  const acceptanceFailedCount = acceptance.length - acceptancePassedCount;
  const status: I18nRoadmapPhaseStatus =
    deliverableFailedCount > 0 || acceptanceFailedCount > 0
      ? "incomplete"
      : knownGaps.length > 0
        ? "ready-with-known-gaps"
        : "ready";

  return {
    acceptance,
    deliverables,
    id,
    knownGaps,
    status,
    summary: {
      acceptanceFailedCount,
      acceptancePassedCount,
      deliverableFailedCount,
      deliverablePassedCount,
      knownGapCount: knownGaps.length,
    },
    title,
  };
}

export function analyzeI18nRoadmapReadinessReport(
  options: Pick<CliOptions, "repoRoot">,
): I18nRoadmapReadinessReport {
  const repoRoot = path.resolve(options.repoRoot || DEFAULT_REPO_ROOT);
  const evidence = {
    bundleStrategy: readEvidence(repoRoot, EVIDENCE_PATHS.bundleStrategy),
    contentTargetLanguageBoundary: readEvidence(
      repoRoot,
      EVIDENCE_PATHS.contentTargetLanguageBoundary,
    ),
    i18nextCliParityBenchmark: readEvidence(
      repoRoot,
      EVIDENCE_PATHS.i18nextCliParityBenchmark,
    ),
    languageBoundary: readEvidence(repoRoot, EVIDENCE_PATHS.languageBoundary),
    p4Readiness: readEvidence(repoRoot, EVIDENCE_PATHS.p4Readiness),
    patchRetirementGate: readEvidence(
      repoRoot,
      EVIDENCE_PATHS.patchRetirementGate,
    ),
    sourceLocaleExport: readEvidence(
      repoRoot,
      EVIDENCE_PATHS.sourceLocaleExport,
    ),
    translationCoverage: readEvidence(
      repoRoot,
      EVIDENCE_PATHS.translationCoverage,
    ),
    translationPrPack: readEvidence(repoRoot, EVIDENCE_PATHS.translationPrPack),
  };
  const evidenceRefs = Object.fromEntries(
    Object.entries(evidence).map(([key, value]) => [key, value.ref]),
  ) as I18nRoadmapReadinessReport["evidence"];

  const packageJson = readJsonObject(
    path.join(repoRoot, SOURCE_PATHS.packageJson),
  );
  const localesText = readTextIfExists(
    path.join(repoRoot, SOURCE_PATHS.locales),
  );
  const createI18nText = readTextIfExists(
    path.join(repoRoot, SOURCE_PATHS.createI18n),
  );
  const appSidebarTestText = readTextIfExists(
    path.join(repoRoot, SOURCE_PATHS.appSidebarTest),
  );
  const appearanceSettingsText = readTextIfExists(
    path.join(repoRoot, SOURCE_PATHS.appearanceSettings),
  );
  const qualityPlannerText = readTextIfExists(
    path.join(repoRoot, SOURCE_PATHS.qualityPlanner),
  );

  const translationCoverageSummary = readNestedRecord(
    evidence.translationCoverage.report,
    ["summary"],
  );
  const detailedCoverageSummary = readCoverageSummary(
    evidence.translationCoverage.report,
  );
  const sourceExportSummary = readNestedRecord(
    evidence.sourceLocaleExport.report,
    ["summary"],
  );
  const translationPrPackSummary = readNestedRecord(
    evidence.translationPrPack.report,
    ["summary"],
  );
  const bundleSummary = readNestedRecord(evidence.bundleStrategy.report, [
    "summary",
  ]);
  const languageBoundarySummary = readNestedRecord(
    evidence.languageBoundary.report,
    ["summary"],
  );
  const contentBoundarySummary = readNestedRecord(
    evidence.contentTargetLanguageBoundary.report,
    ["summary"],
  );
  const p4Summary = readNestedRecord(evidence.p4Readiness.report, ["summary"]);
  const patchGateReport = evidence.patchRetirementGate.report ?? {};
  const patchGatePatch = readNestedRecord(evidence.patchRetirementGate.report, [
    "patch",
  ]);
  const patchGateLegacy = readNestedRecord(
    evidence.patchRetirementGate.report,
    ["legacy"],
  );
  const supportedCoverageLocales = readTranslationCoverageLocales(
    evidence.translationCoverage.report,
  );
  const sourceLocale = readString(
    evidence.sourceLocaleExport.report?.sourceLocale,
  );
  const coreNamespacesReady = hasNamespaces(
    evidence.sourceLocaleExport.report,
    CORE_NAMESPACES,
  );
  const p2NamespacesReady = hasNamespaces(
    evidence.sourceLocaleExport.report,
    P2_NAMESPACES,
  );
  const translationCoverageReady =
    evidence.translationCoverage.ref.exists &&
    !readBoolean(translationCoverageSummary, "hasIssues") &&
    readNumber(detailedCoverageSummary, "missingKeyCount") === 0 &&
    readNumber(detailedCoverageSummary, "extraKeyCount") === 0;
  const sourceKeyCounts = uniquePositiveCounts({
    bundleStrategy: readNumber(bundleSummary, "sourceLocaleKeyCount"),
    sourceLocaleExport: readNumber(sourceExportSummary, "sourceKeyCount"),
    translationCoverage: readNumber(
      translationCoverageSummary,
      "sourceKeyCount",
    ),
    translationPrPack: readNumber(translationPrPackSummary, "sourceKeyCount"),
  });
  const p4KnownGaps = Array.isArray(evidence.p4Readiness.report?.knownGaps)
    ? evidence.p4Readiness.report.knownGaps.filter(isRecord).map(
        (gap): I18nRoadmapKnownGap => ({
          evidencePath:
            readString(gap.evidencePath) || EVIDENCE_PATHS.p4Readiness,
          id: readString(gap.id) || "p4-known-gap",
          phase: "P4",
          severity:
            gap.severity === "decision" || gap.severity === "follow-up"
              ? gap.severity
              : "follow-up",
          summary: readString(gap.summary) || "P4 仍有已知后续缺口。",
        }),
      )
    : [];

  const p0 = buildPhase(
    "P0",
    "骨架与兼容层",
    [
      buildCheck(
        "locale-registry-normalize",
        "新增 locale registry 与 normalize。",
        SOURCE_PATHS.locales,
        localesText.includes("SUPPORTED_LOCALES") &&
          localesText.includes("normalizeLocale") &&
          localesText.includes("normalizeLocalePreference"),
        {
          hasNormalizeLocale: localesText.includes("normalizeLocale"),
          hasNormalizeLocalePreference: localesText.includes(
            "normalizeLocalePreference",
          ),
          hasSupportedLocales: localesText.includes("SUPPORTED_LOCALES"),
        },
      ),
      buildCheck(
        "create-i18n-provider-resources",
        "新增 createI18n、provider、核心 resources。",
        SOURCE_PATHS.createI18n,
        createI18nText.includes("initReactI18next") &&
          createI18nText.includes("loadBundledI18nResources") &&
          coreNamespacesReady,
        {
          coreNamespaces: CORE_NAMESPACES,
          hasBundledResources: createI18nText.includes(
            "loadBundledI18nResources",
          ),
          hasInitReactI18next: createI18nText.includes("initReactI18next"),
        },
      ),
      buildCheck(
        "legacy-patch-boundary",
        "将现有 Patch Layer 移到 legacy-patch 或明确标注 legacy boundary。",
        SOURCE_PATHS.legacyPatchProvider,
        fileExists(path.join(repoRoot, SOURCE_PATHS.legacyPatchProvider)),
        {
          legacyPatchProviderExists: fileExists(
            path.join(repoRoot, SOURCE_PATHS.legacyPatchProvider),
          ),
        },
      ),
      buildCheck(
        "config-language-compat",
        "Config.language 读取兼容 zh / en，写入新 locale。",
        SOURCE_PATHS.locales,
        localesText.includes('lower === "zh"') &&
          localesText.includes('lower === "en"') &&
          fileExists(path.join(repoRoot, SOURCE_PATHS.appConfigTypes)),
        {
          appConfigTypesExists: fileExists(
            path.join(repoRoot, SOURCE_PATHS.appConfigTypes),
          ),
          hasEnCompat: localesText.includes('lower === "en"'),
          hasZhCompat: localesText.includes('lower === "zh"'),
        },
      ),
      buildCheck(
        "document-lang-dir-sync",
        "document.documentElement.lang 与 dir 同步。",
        SOURCE_PATHS.createI18n,
        createI18nText.includes("document.documentElement.lang") &&
          createI18nText.includes("document.documentElement.dir") &&
          createI18nText.includes("resolveDocumentDirection"),
        {
          syncsDir: createI18nText.includes("document.documentElement.dir"),
          syncsLang: createI18nText.includes("document.documentElement.lang"),
          usesDirectionHelper: createI18nText.includes(
            "resolveDocumentDirection",
          ),
        },
      ),
      buildCheck(
        "i18n-check-entry",
        "修复或替换失效的 detect-translations* 脚本入口。",
        SOURCE_PATHS.packageJson,
        hasScript(packageJson, "i18n:check") &&
          hasScript(packageJson, "detect-translations"),
        {
          hasDetectTranslations: hasScript(packageJson, "detect-translations"),
          hasI18nCheck: hasScript(packageJson, "i18n:check"),
        },
      ),
    ],
    [
      buildCheck(
        "zh-cn-en-us-startup-resource-ready",
        "App 能以 zh-CN / en-US 启动。",
        EVIDENCE_PATHS.translationCoverage,
        supportedCoverageLocales.includes("en-US") &&
          sourceLocale === "zh-CN" &&
          translationCoverageReady,
        {
          sourceLocale,
          supportedCoverageLocales,
          translationCoverageReady,
        },
      ),
      buildCheck(
        "legacy-patch-fallback-kept",
        "未迁移页面仍可由 Patch Layer 兜底。",
        SOURCE_PATHS.locales,
        localesText.includes("toLegacyPatchLanguage") &&
          fileExists(path.join(repoRoot, SOURCE_PATHS.legacyPatchProvider)),
        {
          hasLegacyMapper: localesText.includes("toLegacyPatchLanguage"),
          legacyPatchProviderExists: fileExists(
            path.join(repoRoot, SOURCE_PATHS.legacyPatchProvider),
          ),
        },
      ),
      buildCheck(
        "provider-startup-no-suspense",
        "新 provider 不造成首屏白屏或明显闪烁。",
        SOURCE_PATHS.createI18n,
        createI18nText.includes("useSuspense: false") &&
          createI18nText.includes("fallbackLng"),
        {
          hasFallbackLng: createI18nText.includes("fallbackLng"),
          useSuspenseFalse: createI18nText.includes("useSuspense: false"),
        },
        [
          "这是静态启动配置证据，真实 GUI 启动由 P1 / P3 smoke evidence 继续约束。",
        ],
      ),
      buildCheck(
        "normalize-fallback-unit-tests",
        "单测覆盖 normalize 与 fallback。",
        SOURCE_PATHS.localesTest,
        readTextIfExists(
          path.join(repoRoot, SOURCE_PATHS.localesTest),
        ).includes("normalizeLocale") &&
          readTextIfExists(
            path.join(repoRoot, SOURCE_PATHS.loadNamespaceTest),
          ).includes("fallback"),
        {
          loadNamespaceTestHasFallback: readTextIfExists(
            path.join(repoRoot, SOURCE_PATHS.loadNamespaceTest),
          ).includes("fallback"),
          localesTestHasNormalize: readTextIfExists(
            path.join(repoRoot, SOURCE_PATHS.localesTest),
          ).includes("normalizeLocale"),
        },
      ),
    ],
  );

  const p1 = buildPhase(
    "P1",
    "设置页与主导航迁移",
    [
      buildCheck(
        "settings-language-selector-key-based",
        "设置页语言选择器迁移到 key-based i18n。",
        SOURCE_PATHS.appearanceSettings,
        appearanceSettingsText.includes("UI_LOCALE_OPTIONS") &&
          appearanceSettingsText.includes("useTranslation") &&
          appearanceSettingsText.includes("changeLimeLocale"),
        {
          hasChangeLimeLocale:
            appearanceSettingsText.includes("changeLimeLocale"),
          hasUiLocaleOptions:
            appearanceSettingsText.includes("UI_LOCALE_OPTIONS"),
          usesTranslation: appearanceSettingsText.includes("useTranslation"),
        },
      ),
      buildCheck(
        "shell-core-namespaces-covered",
        "侧栏、顶部栏、主导航、Workspace shell、空态、基础按钮迁移。",
        EVIDENCE_PATHS.sourceLocaleExport,
        coreNamespacesReady &&
          sourceExportContains(
            evidence.sourceLocaleExport.report,
            "workspace.",
          ) &&
          sourceExportContains(
            evidence.sourceLocaleExport.report,
            "navigation.",
          ),
        {
          coreNamespaces: CORE_NAMESPACES,
          coreNamespacesReady,
        },
      ),
      buildCheck(
        "core-source-en-us-established",
        "common、navigation、settings、workspace、errors namespace 建立 source + en-US。",
        EVIDENCE_PATHS.translationCoverage,
        translationCoverageReady &&
          supportedCoverageLocales.includes("en-US") &&
          coreNamespacesReady,
        {
          coreNamespacesReady,
          supportedCoverageLocales,
          translationCoverageReady,
        },
      ),
      buildCheck(
        "ui-regression-tests",
        "新增 UI 回归测试，覆盖语言选择、持久化、关键文案。",
        SOURCE_PATHS.appSidebarTest,
        appSidebarTestText.includes("changeLimeLocale") &&
          appSidebarTestText.includes("document.documentElement.lang") &&
          appSidebarTestText.includes("saveConfig"),
        {
          appSidebarTestHasLocaleChange:
            appSidebarTestText.includes("changeLimeLocale"),
          appSidebarTestHasPersistence:
            appSidebarTestText.includes("saveConfig"),
          appSidebarTestHasRootLang: appSidebarTestText.includes(
            "document.documentElement.lang",
          ),
        },
      ),
    ],
    [
      buildCheck(
        "language-switch-updates-shell",
        "切换语言后设置页、侧栏、Workspace shell 立即更新。",
        SOURCE_PATHS.appSidebarTest,
        appSidebarTestText.includes('changeLimeLocale("en-US")') &&
          appSidebarTestText.includes("document.documentElement.lang"),
        {
          testsEnglishSwitch: appSidebarTestText.includes(
            'changeLimeLocale("en-US")',
          ),
          testsRootLang: appSidebarTestText.includes(
            "document.documentElement.lang",
          ),
        },
      ),
      buildCheck(
        "language-choice-persisted",
        "重启后保留选择。",
        SOURCE_PATHS.appSidebarTest,
        appSidebarTestText.includes("saveConfig") &&
          appSidebarTestText.includes("language") &&
          appSidebarTestText.includes("en-US"),
        {
          testsEnUsPersistence:
            appSidebarTestText.includes("saveConfig") &&
            appSidebarTestText.includes("en-US"),
        },
      ),
      buildCheck(
        "workspace-gui-smoke-evidence",
        "GUI smoke 覆盖默认 workspace 准备态。",
        EVIDENCE_PATHS.patchRetirementGate,
        evidence.patchRetirementGate.ref.exists &&
          patchGateReport.retirementReady === true &&
          readNumber(patchGatePatch, "totalRuns") > 0,
        {
          patchRetirementReady: patchGateReport.retirementReady === true,
          patchTotalRuns: readNumber(patchGatePatch, "totalRuns"),
        },
        [
          "当前使用 GUI smoke 产出的 Patch gate 作为默认 workspace 冒烟代理证据；若要做最终发布审计，应补独立 workspace smoke evidence。",
        ],
      ),
    ],
  );

  const p2 = buildPhase(
    "P2",
    "Agent / Artifact / Browser / Knowledge 主路径",
    [
      buildCheck(
        "agent-namespace-migrated",
        "Agent Chat 主路径迁移 agent namespace。",
        EVIDENCE_PATHS.sourceLocaleExport,
        p2NamespacesReady &&
          sourceExportContains(
            evidence.sourceLocaleExport.report,
            "agentChat.",
          ),
        {
          p2Namespaces: P2_NAMESPACES,
          p2NamespacesReady,
        },
      ),
      buildCheck(
        "agent-response-language-metadata",
        "增加 AI response language 设置与 request metadata 注入。",
        EVIDENCE_PATHS.languageBoundary,
        readLanguageCategoryCount(
          evidence.languageBoundary.report,
          "agentResponseLanguage",
        ) > 0 &&
          sourceExportContains(
            evidence.sourceLocaleExport.report,
            "settings.appearance.responseLanguage",
          ),
        {
          agentResponseLanguageCount: readLanguageCategoryCount(
            evidence.languageBoundary.report,
            "agentResponseLanguage",
          ),
          hasSettingsCopy: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "settings.appearance.responseLanguage",
          ),
        },
      ),
      buildCheck(
        "content-target-language-boundary",
        "Artifact / 文档 / 文章 / 翻译类任务明确 content target language。",
        EVIDENCE_PATHS.contentTargetLanguageBoundary,
        evidence.contentTargetLanguageBoundary.ref.exists &&
          readNumber(contentBoundarySummary, "entryCount") > 0 &&
          readNumber(contentBoundarySummary, "unknownCount") === 0,
        {
          contentTargetEntryCount: readNumber(
            contentBoundarySummary,
            "entryCount",
          ),
          contentTargetUnknownCount: readNumber(
            contentBoundarySummary,
            "unknownCount",
          ),
        },
      ),
      buildCheck(
        "browser-environment-language-copy-boundary",
        "Browser Environment 设置页文案明确 Accept-Language 与 UI language 的差异。",
        EVIDENCE_PATHS.sourceLocaleExport,
        sourceExportContains(
          evidence.sourceLocaleExport.report,
          "Accept-Language",
        ) &&
          (sourceExportContains(
            evidence.sourceLocaleExport.report,
            "不控制 Lime 界面语言",
          ) ||
            sourceExportContains(
              evidence.sourceLocaleExport.report,
              "do not control Lime UI language",
            )),
        {
          mentionsAcceptLanguage: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "Accept-Language",
          ),
          mentionsUiLanguageBoundary: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "不控制 Lime 界面语言",
          ),
          mentionsUiLanguageBoundaryEnglish: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "do not control Lime UI language",
          ),
        },
      ),
      buildCheck(
        "knowledge-browser-artifact-main-entries",
        "Knowledge / Plugin / Browser / Artifact 主要入口迁移对应 namespace。",
        EVIDENCE_PATHS.sourceLocaleExport,
        p2NamespacesReady &&
          sourceExportContains(
            evidence.sourceLocaleExport.report,
            "workspace.browser",
          ) &&
          sourceExportContains(evidence.sourceLocaleExport.report, "artifact"),
        {
          p2NamespacesReady,
          sourceExportHasArtifactCopy: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "artifact",
          ),
          sourceExportHasBrowserCopy: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "workspace.browser",
          ),
        },
      ),
      buildCheck(
        "errors-namespace-present",
        "用户可见 toast / error 进入 errors namespace。",
        EVIDENCE_PATHS.sourceLocaleExport,
        hasNamespaces(evidence.sourceLocaleExport.report, ["errors"]) &&
          readNumber(sourceExportSummary, "sourceKeyCount") > 0,
        {
          hasErrorsNamespace: hasNamespaces(
            evidence.sourceLocaleExport.report,
            ["errors"],
          ),
          sourceKeyCount: readNumber(sourceExportSummary, "sourceKeyCount"),
        },
      ),
    ],
    [
      buildCheck(
        "ui-zh-agent-en-combo",
        "UI 中文、Agent 英文回复的组合可用。",
        EVIDENCE_PATHS.languageBoundary,
        readLanguageCategoryCount(
          evidence.languageBoundary.report,
          "agentResponseLanguage",
        ) > 0,
        {
          agentResponseLanguageCount: readLanguageCategoryCount(
            evidence.languageBoundary.report,
            "agentResponseLanguage",
          ),
        },
        [
          "该项证明 response language 与 UI locale 分离，未执行 live model 英文回复端到端。",
        ],
      ),
      buildCheck(
        "ui-en-browser-preset-boundary",
        "UI 英文、Browser preset 为日区/美区的组合不互相污染。",
        EVIDENCE_PATHS.languageBoundary,
        readLanguageCategoryCount(
          evidence.languageBoundary.report,
          "browserEnvironmentLanguage",
        ) > 0 &&
          readLanguageCategoryCount(
            evidence.languageBoundary.report,
            "uiLocale",
          ) > 0,
        {
          browserEnvironmentLanguageCount: readLanguageCategoryCount(
            evidence.languageBoundary.report,
            "browserEnvironmentLanguage",
          ),
          uiLocaleCount: readLanguageCategoryCount(
            evidence.languageBoundary.report,
            "uiLocale",
          ),
        },
      ),
      buildCheck(
        "artifact-target-language-not-ui-locale",
        "Artifact 目标语言不因 UI 切换而改变。",
        EVIDENCE_PATHS.contentTargetLanguageBoundary,
        readNumber(contentBoundarySummary, "entryCount") > 0 &&
          readNumber(contentBoundarySummary, "unknownCount") === 0,
        {
          contentTargetEntryCount: readNumber(
            contentBoundarySummary,
            "entryCount",
          ),
          contentTargetUnknownCount: readNumber(
            contentBoundarySummary,
            "unknownCount",
          ),
        },
      ),
      buildCheck(
        "error-code-translation-display",
        "关键错误能通过 error code 翻译展示。",
        EVIDENCE_PATHS.sourceLocaleExport,
        hasNamespaces(evidence.sourceLocaleExport.report, ["errors"]) &&
          sourceExportContains(evidence.sourceLocaleExport.report, "errorCode"),
        {
          hasErrorsNamespace: hasNamespaces(
            evidence.sourceLocaleExport.report,
            ["errors"],
          ),
          sourceExportHasErrorCodeCopy: sourceExportContains(
            evidence.sourceLocaleExport.report,
            "errorCode",
          ),
        },
      ),
    ],
  );

  const p3 = buildPhase(
    "P3",
    "自动化与治理",
    [
      buildCheck(
        "i18n-workflow-scripts",
        "建立 source export、missing key、unused key、protected dynamic key、coverage report。",
        SOURCE_PATHS.packageJson,
        [
          "i18n:check",
          "i18n:source-export:json",
          "i18n:translation-pr-pack:json",
          "i18n:unused:json",
          "i18n:bundle-report:json",
        ].every((scriptName) => hasScript(packageJson, scriptName)),
        {
          hasBundleReport: hasScript(packageJson, "i18n:bundle-report:json"),
          hasCheck: hasScript(packageJson, "i18n:check"),
          hasSourceExport: hasScript(packageJson, "i18n:source-export:json"),
          hasTranslationPrPack: hasScript(
            packageJson,
            "i18n:translation-pr-pack:json",
          ),
          hasUnused: hasScript(packageJson, "i18n:unused:json"),
        },
      ),
      buildCheck(
        "i18next-cli-evaluated",
        "评估官方 i18next-cli 作为抽取 / lint / locale sync / type generation 候选。",
        EVIDENCE_PATHS.i18nextCliParityBenchmark,
        evidence.i18nextCliParityBenchmark.ref.exists &&
          isRecord(evidence.i18nextCliParityBenchmark.report?.summary),
        {
          hasBenchmarkSummary: isRecord(
            evidence.i18nextCliParityBenchmark.report?.summary,
          ),
        },
      ),
      buildCheck(
        "glossary-established",
        "建立 glossary：产品名、功能名、Agent 术语、Browser Runtime 与 Plugin 术语。",
        SOURCE_PATHS.glossary,
        fileExists(path.join(repoRoot, SOURCE_PATHS.glossary)),
        {
          glossaryExists: fileExists(
            path.join(repoRoot, SOURCE_PATHS.glossary),
          ),
        },
      ),
      buildCheck(
        "translation-pr-pack-review-only",
        "自动翻译只创建 PR，不直接覆盖 source locale。",
        EVIDENCE_PATHS.translationPrPack,
        evidence.translationPrPack.ref.exists &&
          readNumber(translationPrPackSummary, "localesWithGaps") === 0 &&
          readNumber(translationPrPackSummary, "proposedEntryCount") === 0,
        {
          localesWithGaps: readNumber(
            translationPrPackSummary,
            "localesWithGaps",
          ),
          proposedEntryCount: readNumber(
            translationPrPackSummary,
            "proposedEntryCount",
          ),
        },
      ),
      buildCheck(
        "pr-template-namespace-required",
        "PR 模板要求标注新增/变更文案的 namespace。",
        SOURCE_PATHS.prTemplate,
        fileExists(path.join(repoRoot, SOURCE_PATHS.prTemplate)),
        {
          prTemplateExists: fileExists(
            path.join(repoRoot, SOURCE_PATHS.prTemplate),
          ),
        },
      ),
      buildCheck(
        "bundle-strategy-report",
        "资源规模扩大后补 bundle 体积与 chunk 策略报告。",
        EVIDENCE_PATHS.bundleStrategy,
        evidence.bundleStrategy.ref.exists &&
          readNumber(bundleSummary, "sourceLocaleKeyCount") > 0 &&
          readNumber(bundleSummary, "inlineGroupCount") > 0,
        {
          inlineGroupCount: readNumber(bundleSummary, "inlineGroupCount"),
          sourceLocaleKeyCount: readNumber(
            bundleSummary,
            "sourceLocaleKeyCount",
          ),
          totalRawBytes: readNumber(bundleSummary, "totalRawBytes"),
        },
      ),
    ],
    [
      buildCheck(
        "i18n-check-green",
        "npm run i18n:check 能在本地发现漏 key 与无效 locale。",
        EVIDENCE_PATHS.translationCoverage,
        translationCoverageReady,
        {
          hasIssues: readBoolean(translationCoverageSummary, "hasIssues"),
          issueCount: readNumber(translationCoverageSummary, "issueCount"),
          missingKeyCount: readNumber(
            detailedCoverageSummary,
            "missingKeyCount",
          ),
        },
      ),
      buildCheck(
        "quality-selector-i18n-coverage",
        "npm run verify:local 或质量选择器能覆盖 i18n 结构风险。",
        SOURCE_PATHS.qualityPlanner,
        qualityPlannerText.includes("i18n:translation-pr-pack:json") &&
          qualityPlannerText.includes("i18n:bundle-report:json") &&
          qualityPlannerText.includes("i18n:p4-readiness-report:json"),
        {
          recommendsBundleReport: qualityPlannerText.includes(
            "i18n:bundle-report:json",
          ),
          recommendsP4Readiness: qualityPlannerText.includes(
            "i18n:p4-readiness-report:json",
          ),
          recommendsTranslationPrPack: qualityPlannerText.includes(
            "i18n:translation-pr-pack:json",
          ),
        },
      ),
      buildCheck(
        "translation-pr-reviewable",
        "翻译 PR 可审阅、可回滚、不会覆盖人工修订。",
        EVIDENCE_PATHS.translationPrPack,
        evidence.translationPrPack.ref.exists &&
          readNumber(translationPrPackSummary, "proposedEntryCount") === 0 &&
          readNumber(translationPrPackSummary, "localesWithGaps") === 0,
        {
          localesWithGaps: readNumber(
            translationPrPackSummary,
            "localesWithGaps",
          ),
          proposedEntryCount: readNumber(
            translationPrPackSummary,
            "proposedEntryCount",
          ),
        },
      ),
      buildCheck(
        "automation-evidence-source-key-sync",
        "P3 自动化 evidence 必须基于同一份 source locale 快照。",
        EVIDENCE_PATHS.sourceLocaleExport,
        sourceKeyCounts.distinct.length === 1,
        {
          distinctSourceKeyCounts: sourceKeyCounts.distinct,
          sourceKeyCounts: sourceKeyCounts.values,
        },
        sourceKeyCounts.distinct.length === 1
          ? []
          : [
              "source export、coverage、translation PR pack 或 bundle report 的 source key 计数不一致；需要先刷新同源 evidence，再判断整体 PRD 完成。",
            ],
      ),
    ],
  );

  const p4 = buildPhase(
    "P4",
    "扩展与发布材料",
    [
      buildCheck(
        "p4-deliverables-ready",
        "Chrome extension、发布材料、RTL 与 app metadata 评估完成。",
        EVIDENCE_PATHS.p4Readiness,
        evidence.p4Readiness.ref.exists &&
          readBoolean(p4Summary, "deliverablesReady"),
        {
          deliverableFailedCount: readNumber(
            p4Summary,
            "deliverableFailedCount",
          ),
          deliverablePassedCount: readNumber(
            p4Summary,
            "deliverablePassedCount",
          ),
          deliverablesReady: readBoolean(p4Summary, "deliverablesReady"),
        },
      ),
    ],
    [
      buildCheck(
        "p4-acceptance-ready",
        "extension 术语、RTL 主路径和 zh-CN / en-US 发布材料验收通过。",
        EVIDENCE_PATHS.p4Readiness,
        evidence.p4Readiness.ref.exists &&
          readBoolean(p4Summary, "acceptanceReady"),
        {
          acceptanceFailedCount: readNumber(p4Summary, "acceptanceFailedCount"),
          acceptancePassedCount: readNumber(p4Summary, "acceptancePassedCount"),
          acceptanceReady: readBoolean(p4Summary, "acceptanceReady"),
        },
      ),
    ],
    p4KnownGaps,
  );

  const phases = [p0, p1, p2, p3, p4];
  const deliverablePassedCount = phases.reduce(
    (sum, phase) => sum + phase.summary.deliverablePassedCount,
    0,
  );
  const deliverableFailedCount = phases.reduce(
    (sum, phase) => sum + phase.summary.deliverableFailedCount,
    0,
  );
  const acceptancePassedCount = phases.reduce(
    (sum, phase) => sum + phase.summary.acceptancePassedCount,
    0,
  );
  const acceptanceFailedCount = phases.reduce(
    (sum, phase) => sum + phase.summary.acceptanceFailedCount,
    0,
  );
  const knownGapCount = phases.reduce(
    (sum, phase) => sum + phase.summary.knownGapCount,
    0,
  );
  const phaseReadyCount = phases.filter(
    (phase) => phase.status === "ready",
  ).length;
  const phaseReadyWithKnownGapsCount = phases.filter(
    (phase) => phase.status === "ready-with-known-gaps",
  ).length;
  const phaseIncompleteCount = phases.filter(
    (phase) => phase.status === "incomplete",
  ).length;
  const missingEvidenceCount = Object.values(evidenceRefs).filter(
    (ref) => !ref.exists,
  ).length;
  const overallStatus: I18nRoadmapPhaseStatus =
    phaseIncompleteCount > 0
      ? "incomplete"
      : knownGapCount > 0
        ? "ready-with-known-gaps"
        : "ready";

  return {
    evidence: evidenceRefs,
    phases,
    repoRoot,
    schemaVersion: "lime.i18n.roadmapReadinessReport.v1",
    summary: {
      acceptanceFailedCount,
      acceptancePassedCount,
      deliverableFailedCount,
      deliverablePassedCount,
      knownGapCount,
      missingEvidenceCount,
      overallStatus,
      phaseCount: phases.length,
      phaseIncompleteCount,
      phaseReadyCount,
      phaseReadyWithKnownGapsCount,
    },
  };
}

export function formatI18nRoadmapReadinessReport(
  report: I18nRoadmapReadinessReport,
  format: I18nRoadmapReadinessReportFormat,
): string {
  if (format === "json") {
    return `${JSON.stringify(report, null, 2)}\n`;
  }

  const lines = [
    "[i18n:roadmap] readiness report",
    `overall status: ${report.summary.overallStatus}`,
    `phases: ${report.summary.phaseReadyCount} ready / ${report.summary.phaseReadyWithKnownGapsCount} ready-with-known-gaps / ${report.summary.phaseIncompleteCount} incomplete`,
    `deliverables: ${report.summary.deliverablePassedCount} passed / ${report.summary.deliverableFailedCount} failed`,
    `acceptance: ${report.summary.acceptancePassedCount} passed / ${report.summary.acceptanceFailedCount} failed`,
    `missing evidence: ${report.summary.missingEvidenceCount}`,
    `known gaps: ${report.summary.knownGapCount}`,
    "",
    "phases:",
    ...report.phases.map(
      (phase) => `- ${phase.id} ${phase.title}: ${phase.status}`,
    ),
  ];

  const failedChecks = report.phases.flatMap((phase) =>
    [...phase.deliverables, ...phase.acceptance]
      .filter((check) => check.status === "failed")
      .map((check) => `${phase.id}/${check.id}: ${check.requirement}`),
  );
  if (failedChecks.length > 0) {
    lines.push(
      "",
      "failed checks:",
      ...failedChecks.map((check) => `- ${check}`),
    );
  }

  const knownGaps = report.phases.flatMap((phase) => phase.knownGaps);
  if (knownGaps.length > 0) {
    lines.push(
      "",
      "known gaps:",
      ...knownGaps.map((gap) => `- ${gap.phase}/${gap.id}: ${gap.summary}`),
    );
  }

  return `${lines.join("\n")}\n`;
}

function printHelp(): void {
  console.log(`Usage: tsx scripts/i18n/i18n-roadmap-readiness-report.ts [options]

聚合 i18n PRD P0-P4 的 readiness evidence。
只读生成 completion audit，不刷新底层 evidence、不修改 locale resources。

Options:
  --format json|text   输出格式，默认 text
  --output <file>      将输出写入文件
  --repo-root <path>   指定仓库根目录，默认当前仓库
  --help, -h           显示帮助
`);
}

function parseArgs(argv: string[]): CliOptions & { help?: boolean } {
  const options: CliOptions & { help?: boolean } = {
    format: "text",
    repoRoot: DEFAULT_REPO_ROOT,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg) {
      continue;
    }
    if (arg === "--format") {
      const next = argv[index + 1];
      if (next !== "json" && next !== "text") {
        throw new Error("--format 只接受 json 或 text");
      }
      options.format = next;
      index += 1;
      continue;
    }
    if (arg === "--output") {
      const next = argv[index + 1];
      if (!next) {
        throw new Error("--output 需要文件路径");
      }
      options.output = next;
      index += 1;
      continue;
    }
    if (arg === "--repo-root") {
      const next = argv[index + 1];
      if (!next) {
        throw new Error("--repo-root 需要路径");
      }
      options.repoRoot = path.resolve(next);
      index += 1;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    throw new Error(`未知参数：${arg}`);
  }

  return options;
}

export function runCli(argv: string[] = process.argv.slice(2)): number {
  const options = parseArgs(argv);
  if (options.help) {
    printHelp();
    return 0;
  }

  const report = analyzeI18nRoadmapReadinessReport({
    repoRoot: options.repoRoot,
  });
  const output = formatI18nRoadmapReadinessReport(report, options.format);

  if (options.output) {
    fs.mkdirSync(path.dirname(path.resolve(options.output)), {
      recursive: true,
    });
    fs.writeFileSync(path.resolve(options.output), output, "utf8");
  } else {
    process.stdout.write(output);
  }

  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  process.exitCode = runCli();
}
