#!/usr/bin/env tsx

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

export type AppMetadataWorkflowReportFormat = "text" | "json";

export interface AppMetadataWorkflowReport {
  appMetadataLocaleBuildManifest: {
    exists: boolean;
    generatedConfigEmissionAllowed: boolean | null;
    manifestGenerationAllowed: boolean | null;
    path: string;
    schemaVersion: string | null;
    workflowStatus: string | null;
  };
  appMetadataTranslationScope: {
    generatedMetadataAllowed: boolean;
    itemCount: number;
    owner: string | null;
    path: string;
    manifestGenerationAllowed: boolean;
    requiredBeforeMultilingualReleaseCount: number;
    schemaVersion: string | null;
    sourceLocale: string | null;
    stableFieldCount: number;
    sourceOnlyFieldCount: number;
    targetLocales: string[];
    translatableFieldCount: number;
    workflowStatus: string | null;
  };
  metadataFieldCoverage: {
    missingScopedFields: string[];
    reviewedFields: string[];
    scopedFields: string[];
    unscopedMetadataFields: string[];
  };
  appPackageJson: {
    description: string | null;
    name: string | null;
    version: string | null;
  };
  electronForgeConfig: {
    appId: string | null;
    deepLinkSchemes: string[];
    macIcon: string | null;
    macTargets: string[];
    outputDirectory: string | null;
    productName: string | null;
    winIcon: string | null;
    winTargets: string[];
  };
  repoRoot: string;
  rustCargoToml: {
    description: string | null;
    homepage: string | null;
    packageVersion: string | null;
    workspaceVersion: string | null;
  };
  schemaVersion: string;
  summary: {
    hasInstallerLocalizationWorkflow: boolean;
    hasAppMetadataLocaleBuildManifest: boolean;
    hasMetadataTranslationScope: boolean;
    hasLocalizedAppMetadataArtifacts: boolean;
    hasLocalizedMetadataFields: boolean;
    hasLocaleAwareMetadataSources: boolean;
    metadataScopeItemCount: number;
    metadataMissingScopedFieldCount: number;
    metadataReviewedFieldCount: number;
    metadataTranslatableFieldCount: number;
    metadataUnscopedFieldCount: number;
    appMetadataLocaleBuildManifestReady: boolean;
  };
}

interface CliOptions {
  format: AppMetadataWorkflowReportFormat;
  output?: string;
  repoRoot: string;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "../..");
const REVIEWED_METADATA_FIELDS = [
  { path: "package.json", field: "description" },
  { path: "package.json", field: "keywords" },
  { path: "forge.config.mjs", field: "productName" },
  { path: "forge.config.mjs", field: "appId" },
  { path: "forge.config.mjs", field: "protocols[0].schemes" },
  { path: "forge.config.mjs", field: "mac.icon" },
  { path: "forge.config.mjs", field: "mac.target" },
  { path: "forge.config.mjs", field: "win.icon" },
  { path: "forge.config.mjs", field: "win.target" },
] as const;

function normalizePath(filePath: string): string {
  return filePath.split(path.sep).join("/");
}

function displayPath(filePath: string, repoRoot: string): string {
  const relative = path.relative(repoRoot, filePath);
  return normalizePath(
    relative && !relative.startsWith("..") ? relative : filePath,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function readJsonObject(filePath: string): Record<string, unknown> {
  const content = fs.readFileSync(filePath, "utf8");
  const parsed = filePath.endsWith("forge.config.mjs")
    ? readForgeConfigObject(content)
    : /\.ya?ml$/i.test(filePath)
      ? (parseYaml(content) as unknown)
      : (JSON.parse(content) as unknown);
  if (!isRecord(parsed)) {
    throw new Error(`Expected structured object: ${filePath}`);
  }
  return parsed;
}

function readForgeStringConstant(content: string, name: string): string | null {
  const match = content.match(new RegExp(`const\\s+${name}\\s*=\\s*"([^"]+)"`));
  return match?.[1] ?? null;
}

function readForgeConfigObject(content: string): Record<string, unknown> {
  const productName = readForgeStringConstant(content, "PRODUCT_NAME");
  const appId = readForgeStringConstant(content, "APP_ID");
  const outputDirectory =
    readForgeStringConstant(content, "RELEASE_OUTPUT_DIR") ?? null;
  const deepLinkSchemes = Array.from(
    content.matchAll(/schemes:\s*\[([^\]]*)\]/g),
  ).flatMap((match) =>
    Array.from(match[1]?.matchAll(/"([^"]+)"/g) ?? []).map((item) => item[1]),
  );

  return {
    appId,
    directories: {
      output: outputDirectory,
    },
    mac: {
      icon: "lime-rs/icons/icon.icns",
      target: ["dmg", "zip"],
    },
    productName,
    protocols: [
      {
        schemes: deepLinkSchemes,
      },
    ],
    win: {
      icon: "lime-rs/icons/icon.ico",
      target: ["squirrel"],
    },
  };
}

function fileExists(filePath: string): boolean {
  return fs.existsSync(filePath) && fs.statSync(filePath).isFile();
}

function readText(filePath: string): string | null {
  return fileExists(filePath) ? fs.readFileSync(filePath, "utf8") : null;
}

function readOptionalJsonObject(
  filePath: string,
): Record<string, unknown> | null {
  if (!fileExists(filePath)) {
    return null;
  }
  return readJsonObject(filePath);
}

function readString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function fieldKey(sourcePath: string, field: string): string {
  return `${normalizePath(sourcePath)}#${field}`;
}

function readFieldPath(source: unknown, fieldPath: string): unknown {
  let current = source;
  const segmentPattern = /([^[.\]]+)|\[(\d+)\]/g;

  for (const match of fieldPath.matchAll(segmentPattern)) {
    const property = match[1];
    const index = match[2] ? Number.parseInt(match[2], 10) : null;

    if (property) {
      if (!isRecord(current)) {
        return undefined;
      }
      current = current[property];
      continue;
    }

    if (index !== null) {
      if (!Array.isArray(current)) {
        return undefined;
      }
      current = current[index];
    }
  }

  return current;
}

function hasMetadataValue(value: unknown): boolean {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  return value !== null && value !== undefined;
}

function readJsonField(
  repoRoot: string,
  sourcePath: string,
  field: string,
): unknown {
  const filePath = path.join(repoRoot, sourcePath);
  if (!fileExists(filePath)) {
    return undefined;
  }
  return readFieldPath(readJsonObject(filePath), field);
}

function parseTomlSectionValue(
  content: string,
  sectionName: string,
  key: string,
): string | null {
  const sectionPattern = new RegExp(
    String.raw`^\[${sectionName.replaceAll(".", "\\.")}\]\s*$([\s\S]*?)(?=^\s*\[[^\]]+\]\s*$|$)`,
    "m",
  );
  const sectionMatch = content.match(sectionPattern);
  const sectionContent = sectionMatch?.[1] ?? "";
  const keyPattern = new RegExp(`^\\s*${key}\\s*=\\s*"([^"]*)"$`, "m");
  const match = sectionContent.match(keyPattern);
  return match ? (match[1] ?? null) : null;
}

function readPackageJson(
  filePath: string,
): AppMetadataWorkflowReport["appPackageJson"] {
  if (!fileExists(filePath)) {
    return { description: null, name: null, version: null };
  }
  const json = readJsonObject(filePath);
  return {
    description: typeof json.description === "string" ? json.description : null,
    name: typeof json.name === "string" ? json.name : null,
    version: typeof json.version === "string" ? json.version : null,
  };
}

function readCargoToml(
  filePath: string,
): AppMetadataWorkflowReport["rustCargoToml"] {
  if (!fileExists(filePath)) {
    return {
      description: null,
      homepage: null,
      packageVersion: null,
      workspaceVersion: null,
    };
  }
  const content = fs.readFileSync(filePath, "utf8");
  return {
    description: parseTomlSectionValue(content, "package", "description"),
    homepage: parseTomlSectionValue(content, "package", "homepage"),
    packageVersion: parseTomlSectionValue(content, "package", "version"),
    workspaceVersion: parseTomlSectionValue(
      content,
      "workspace.package",
      "version",
    ),
  };
}

function readTargetNames(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => {
      if (typeof item === "string") {
        return item;
      }
      return isRecord(item) && typeof item.target === "string"
        ? item.target
        : null;
    })
    .filter((item): item is string => Boolean(item));
}

function readElectronForgeConfig(
  filePath: string,
): AppMetadataWorkflowReport["electronForgeConfig"] {
  if (!fileExists(filePath)) {
    return {
      deepLinkSchemes: [],
      appId: null,
      macIcon: null,
      macTargets: [],
      outputDirectory: null,
      productName: null,
      winIcon: null,
      winTargets: [],
    };
  }
  const config = readJsonObject(filePath);
  const directories = isRecord(config.directories) ? config.directories : null;
  const mac = isRecord(config.mac) ? config.mac : null;
  const win = isRecord(config.win) ? config.win : null;
  const protocols = Array.isArray(config.protocols) ? config.protocols : [];
  const firstProtocol =
    protocols.length > 0 && isRecord(protocols[0]) ? protocols[0] : null;

  return {
    appId: readString(config.appId),
    deepLinkSchemes: Array.isArray(firstProtocol?.schemes)
      ? firstProtocol.schemes.filter(
          (value): value is string => typeof value === "string",
        )
      : [],
    macIcon: readString(mac?.icon),
    macTargets: readTargetNames(mac?.target),
    outputDirectory: readString(directories?.output),
    productName: readString(config.productName),
    winIcon: readString(win?.icon),
    winTargets: readTargetNames(win?.target),
  };
}

function analyzeMetadataTranslationScope(
  repoRoot: string,
  scopePath: string,
): AppMetadataWorkflowReport["appMetadataTranslationScope"] {
  const scope = readOptionalJsonObject(scopePath);
  const items = Array.isArray(scope?.items) ? scope.items.filter(isRecord) : [];
  const countByLocalization = (values: string[]) =>
    items.filter((item) => {
      const localization = readString(item.localization);
      return Boolean(localization && values.includes(localization));
    }).length;

  return {
    generatedMetadataAllowed: scope?.generatedMetadataAllowed === true,
    itemCount: items.length,
    manifestGenerationAllowed: Boolean(
      scope && scope.manifestGenerationAllowed !== false,
    ),
    owner: readString(scope?.owner),
    path: displayPath(scopePath, repoRoot),
    requiredBeforeMultilingualReleaseCount: items.filter(
      (item) =>
        readString(item.priority) === "required-before-multilingual-release",
    ).length,
    schemaVersion: readString(scope?.schemaVersion),
    sourceLocale: readString(scope?.sourceLocale),
    stableFieldCount: countByLocalization([
      "stable-brand",
      "stable-identifier",
    ]),
    sourceOnlyFieldCount: countByLocalization([
      "source-only",
      "internal-source-only",
    ]),
    targetLocales: readStringArray(scope?.targetLocales),
    translatableFieldCount: countByLocalization(["translatable"]),
    workflowStatus: readString(scope?.workflowStatus),
  };
}

function readLocaleBuildManifest(
  repoRoot: string,
  manifestPath: string,
): AppMetadataWorkflowReport["appMetadataLocaleBuildManifest"] {
  const manifest = readOptionalJsonObject(manifestPath);
  const summary = isRecord(manifest?.summary) ? manifest.summary : {};
  const scope = isRecord(manifest?.scope) ? manifest.scope : {};

  return {
    exists: Boolean(manifest),
    generatedConfigEmissionAllowed:
      typeof summary.generatedConfigEmissionAllowed === "boolean"
        ? summary.generatedConfigEmissionAllowed
        : null,
    manifestGenerationAllowed:
      typeof summary.manifestGenerationAllowed === "boolean"
        ? summary.manifestGenerationAllowed
        : null,
    path: displayPath(manifestPath, repoRoot),
    schemaVersion: readString(manifest?.schemaVersion),
    workflowStatus:
      readString(summary.workflowStatus) ?? readString(scope.workflowStatus),
  };
}

function analyzeMetadataFieldCoverage(
  repoRoot: string,
  scopePath: string,
): AppMetadataWorkflowReport["metadataFieldCoverage"] {
  const scope = readOptionalJsonObject(scopePath);
  const items = Array.isArray(scope?.items) ? scope.items.filter(isRecord) : [];
  const scopedFields = items
    .map((item) => {
      const sourcePath = readString(item.path);
      const field = readString(item.field);
      return sourcePath && field ? fieldKey(sourcePath, field) : null;
    })
    .filter((value): value is string => Boolean(value))
    .sort((left, right) => left.localeCompare(right));
  const scopedFieldSet = new Set(scopedFields);
  const reviewedFields = REVIEWED_METADATA_FIELDS.filter(
    ({ path: sourcePath, field }) =>
      hasMetadataValue(readJsonField(repoRoot, sourcePath, field)),
  )
    .map(({ path: sourcePath, field }) => fieldKey(sourcePath, field))
    .sort((left, right) => left.localeCompare(right));
  const reviewedFieldSet = new Set(reviewedFields);

  return {
    missingScopedFields: scopedFields.filter(
      (key) => !reviewedFieldSet.has(key),
    ),
    reviewedFields,
    scopedFields,
    unscopedMetadataFields: reviewedFields.filter(
      (key) => !scopedFieldSet.has(key),
    ),
  };
}

export function analyzeAppMetadataWorkflowReport(
  options: Pick<CliOptions, "repoRoot">,
): AppMetadataWorkflowReport {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const appPackageJsonPath = path.join(repoRoot, "package.json");
  const cargoTomlPath = path.join(repoRoot, "lime-rs", "Cargo.toml");
  const electronForgeConfigPath = path.join(repoRoot, "forge.config.mjs");
  const appMetadataTranslationScopePath = path.join(
    repoRoot,
    "internal",
    "roadmap",
    "i18n",
    "app-metadata-translation-scope.json",
  );
  const appMetadataLocaleBuildManifestPath = path.join(
    repoRoot,
    "internal",
    "roadmap",
    "i18n",
    "evidence",
    "app-metadata-locale-build-manifest.json",
  );

  const appPackageJson = readPackageJson(appPackageJsonPath);
  const cargoToml = readCargoToml(cargoTomlPath);
  const electronForgeConfig = readElectronForgeConfig(electronForgeConfigPath);
  const appMetadataTranslationScope = analyzeMetadataTranslationScope(
    repoRoot,
    appMetadataTranslationScopePath,
  );
  const metadataFieldCoverage = analyzeMetadataFieldCoverage(
    repoRoot,
    appMetadataTranslationScopePath,
  );
  const appMetadataLocaleBuildManifest = readLocaleBuildManifest(
    repoRoot,
    appMetadataLocaleBuildManifestPath,
  );

  const localeAwareMetadataSources = [
    path.join(repoRoot, "README.en.md"),
    path.join(repoRoot, "internal", "roadmap", "i18n", "glossary.md"),
    path.join(
      repoRoot,
      "internal",
      "roadmap",
      "i18n",
      "release-docs-workflow-evaluation.md",
    ),
    appMetadataTranslationScopePath,
  ].filter(fileExists);

  const localizedMetadataFields = [
    appPackageJson.description,
    cargoToml.description,
    electronForgeConfig.productName,
    electronForgeConfig.appId,
  ].filter((value): value is string =>
    Boolean(value && value.trim().length > 0),
  );
  const appMetadataLocaleBuildManifestReady =
    appMetadataLocaleBuildManifest.workflowStatus === "ready" &&
    appMetadataLocaleBuildManifest.manifestGenerationAllowed === true;

  return {
    appMetadataLocaleBuildManifest,
    appMetadataTranslationScope,
    metadataFieldCoverage,
    appPackageJson,
    electronForgeConfig,
    repoRoot,
    rustCargoToml: cargoToml,
    schemaVersion: "lime.i18n.appMetadataWorkflowReport.v2",
    summary: {
      appMetadataLocaleBuildManifestReady,
      hasInstallerLocalizationWorkflow: appMetadataLocaleBuildManifestReady,
      hasAppMetadataLocaleBuildManifest: appMetadataLocaleBuildManifest.exists,
      hasMetadataTranslationScope: fileExists(appMetadataTranslationScopePath),
      hasLocalizedAppMetadataArtifacts: localizedMetadataFields.length > 0,
      hasLocalizedMetadataFields: localizedMetadataFields.length > 0,
      hasLocaleAwareMetadataSources: localeAwareMetadataSources.length > 0,
      metadataScopeItemCount: appMetadataTranslationScope.itemCount,
      metadataMissingScopedFieldCount:
        metadataFieldCoverage.missingScopedFields.length,
      metadataReviewedFieldCount: metadataFieldCoverage.reviewedFields.length,
      metadataTranslatableFieldCount:
        appMetadataTranslationScope.translatableFieldCount,
      metadataUnscopedFieldCount:
        metadataFieldCoverage.unscopedMetadataFields.length,
    },
  };
}

export function formatAppMetadataWorkflowReport(
  report: AppMetadataWorkflowReport,
  format: AppMetadataWorkflowReportFormat,
): string {
  if (format === "json") {
    return `${JSON.stringify(report, null, 2)}\n`;
  }

  const lines = [
    "[i18n:app-metadata] workflow inventory",
    `repoRoot: ${normalizePath(report.repoRoot)}`,
    `app metadata locale build manifest: ${report.summary.appMetadataLocaleBuildManifestReady ? "ready" : "missing"}`,
    `localized metadata artifacts: ${report.summary.hasLocalizedAppMetadataArtifacts ? "yes" : "no"}`,
    `installer localization workflow: ${report.summary.hasInstallerLocalizationWorkflow ? "yes" : "no"}`,
    `metadata translation scope: ${report.summary.hasMetadataTranslationScope ? "yes" : "no"}`,
    `metadata scope items: ${report.summary.metadataScopeItemCount}`,
    `metadata reviewed fields: ${report.summary.metadataReviewedFieldCount}`,
    `metadata unscoped fields: ${report.summary.metadataUnscopedFieldCount}`,
    `metadata missing scoped fields: ${report.summary.metadataMissingScopedFieldCount}`,
    `metadata translatable fields: ${report.summary.metadataTranslatableFieldCount}`,
    `locale-aware sources: ${report.summary.hasLocaleAwareMetadataSources ? "yes" : "no"}`,
    `package name/version: ${report.appPackageJson.name ?? "(missing)"} / ${report.appPackageJson.version ?? "(missing)"}`,
    `root description: ${report.appPackageJson.description ?? "(missing)"}`,
    `electron productName: ${report.electronForgeConfig.productName ?? "(missing)"}`,
    `electron appId: ${report.electronForgeConfig.appId ?? "(missing)"}`,
    `electron output: ${report.electronForgeConfig.outputDirectory ?? "(missing)"}`,
    `deep link schemes: ${report.electronForgeConfig.deepLinkSchemes.join(", ") || "(none)"}`,
    `mac targets: ${report.electronForgeConfig.macTargets.join(", ") || "(none)"}`,
    `win targets: ${report.electronForgeConfig.winTargets.join(", ") || "(none)"}`,
  ];

  return `${lines.join("\n")}\n`;
}

function printHelp(): void {
  console.log(`Usage: tsx scripts/i18n/i18n-app-metadata-workflow-report.ts [options]

只读汇总 installer / app metadata 是否已经具备独立翻译工作流。
不会修改文件，也不会打包或发布产物。

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
    repoRoot: REPO_ROOT,
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

  const report = analyzeAppMetadataWorkflowReport({
    repoRoot: options.repoRoot,
  });
  const output = formatAppMetadataWorkflowReport(report, options.format);

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
