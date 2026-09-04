#!/usr/bin/env tsx

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

export type I18nLanguageBoundaryReportFormat = "text" | "json";

export type I18nLanguageBoundaryCategory =
  | "agentResponseLanguage"
  | "asrLanguage"
  | "browserEnvironmentLanguage"
  | "codeLanguage"
  | "contentTargetLanguage"
  | "uiLocale"
  | "unknownLanguageLike";

export interface I18nLanguageBoundaryEntry {
  category: I18nLanguageBoundaryCategory;
  file: string;
  line: number;
  marker: string;
  text: string;
}

export interface I18nLanguageBoundaryCategorySummary {
  category: I18nLanguageBoundaryCategory;
  count: number;
}

export interface I18nLanguageBoundaryFileSummary {
  count: number;
  file: string;
}

export interface I18nLanguageBoundaryMarkerSummary {
  count: number;
  marker: string;
}

export interface I18nLanguageBoundaryReport {
  entries: I18nLanguageBoundaryEntry[];
  filters: {
    category?: I18nLanguageBoundaryCategory;
  };
  rootDir: string;
  schemaVersion: string;
  scannedFileCount: number;
  sourceDirs: string[];
  summary: {
    categorySummaries: I18nLanguageBoundaryCategorySummary[];
    entryCount: number;
    fileSummaries: I18nLanguageBoundaryFileSummary[];
    markerSummaries: I18nLanguageBoundaryMarkerSummary[];
    unknownCount: number;
  };
}

interface CliOptions {
  category?: I18nLanguageBoundaryCategory;
  format: I18nLanguageBoundaryReportFormat;
  output?: string;
  rootDir: string;
  sourceDirs: string[];
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "../..");
const DEFAULT_SOURCE_DIRS = ["src", "lime-rs"];
const SOURCE_EXTENSIONS = new Set([".rs", ".ts", ".tsx", ".js", ".jsx"]);
const SKIP_SEGMENTS = [
  "/node_modules/",
  "/src/i18n/resources/",
  "/src/i18n/legacy-patch/",
  "/lime-rs/target/",
  "/lime-rs/crates/agent-rust/",
  "/target/",
];

const LANGUAGE_MARKER_PATTERN =
  /\b(accept_language|browser_launch_language|response_language|responseLanguage|target_language|targetLanguage|preferred_language|preferredLanguage|language|languages|locale|i18n\.language|resolvedLanguage)\b/;

const LANGUAGE_BOUNDARY_CATEGORIES: I18nLanguageBoundaryCategory[] = [
  "agentResponseLanguage",
  "asrLanguage",
  "browserEnvironmentLanguage",
  "codeLanguage",
  "contentTargetLanguage",
  "uiLocale",
  "unknownLanguageLike",
];

function normalizePath(filePath: string): string {
  return filePath.split(path.sep).join("/");
}

function displayPath(rootDir: string, filePath: string): string {
  const relative = path.relative(rootDir, filePath);
  return normalizePath(
    relative && !relative.startsWith("..") ? relative : filePath,
  );
}

function isSourceFile(filePath: string): boolean {
  const normalized = normalizePath(filePath);
  if (!SOURCE_EXTENSIONS.has(path.extname(filePath))) {
    return false;
  }
  return !SKIP_SEGMENTS.some((segment) => normalized.includes(segment));
}

function collectSourceFiles(rootDir: string, sourceDirs: string[]): string[] {
  const files: string[] = [];

  const visit = (target: string) => {
    if (!fs.existsSync(target)) {
      return;
    }

    const stat = fs.statSync(target);
    if (stat.isFile()) {
      if (isSourceFile(target)) {
        files.push(path.resolve(target));
      }
      return;
    }

    if (!stat.isDirectory()) {
      return;
    }

    for (const entry of fs.readdirSync(target, { withFileTypes: true })) {
      if (
        entry.name === "node_modules" ||
        entry.name === "target" ||
        entry.name === "dist"
      ) {
        continue;
      }
      visit(path.join(target, entry.name));
    }
  };

  for (const sourceDir of sourceDirs) {
    visit(path.resolve(rootDir, sourceDir));
  }

  return Array.from(new Set(files)).sort((left, right) =>
    displayPath(rootDir, left).localeCompare(displayPath(rootDir, right)),
  );
}

function classifyLanguageMarker(
  file: string,
  lineText: string,
  marker: string,
): I18nLanguageBoundaryCategory {
  const normalizedFile = normalizePath(file);
  const normalizedFileLower = normalizedFile.toLowerCase();
  const normalizedText = lineText.toLowerCase();
  const normalizedMarker = marker.toLowerCase();

  if (
    normalizedMarker.includes("accept_language") ||
    normalizedMarker.includes("browser_launch_language") ||
    normalizedFileLower.includes("/browser_environment") ||
    normalizedFileLower.includes("/browserenvironment") ||
    normalizedFileLower.includes("/browser_runtime_cmd") ||
    normalizedFileLower.includes("/webview_cmd") ||
    normalizedFileLower.includes("/webview-api") ||
    normalizedText.includes("browser language") ||
    normalizedText.includes("accept-language")
  ) {
    return "browserEnvironmentLanguage";
  }

  if (
    normalizedMarker.includes("response_language") ||
    normalizedMarker.includes("responselanguage") ||
    normalizedText.includes("metadata.language") ||
    normalizedText.includes("response language") ||
    normalizedText.includes("response-language")
  ) {
    return "agentResponseLanguage";
  }

  if (
    normalizedFileLower.includes("/artifact/parser") ||
    normalizedFileLower.includes("/artifactplaceholder") ||
    normalizedFileLower.includes("/markdownrenderer") ||
    normalizedFileLower.includes("/messageassistantbody") ||
    normalizedFileLower.includes("/messagelist") ||
    normalizedFileLower.includes("/messagelistitem") ||
    normalizedFileLower.includes("/general_chat/") ||
    normalizedFileLower.includes("/general-chat/") ||
    normalizedFileLower.includes("/markdownpreview") ||
    normalizedFileLower.includes("/streamingrenderer") ||
    normalizedFileLower.includes("/toolcalldisplay") ||
    normalizedFileLower.includes(
      "/workspace/useworkspaceartifactpreviewactions",
    ) ||
    normalizedFileLower.includes(
      "/workspace/useworkspaceartifactworkbenchactions",
    ) ||
    normalizedFileLower.includes("/zhihurenderer") ||
    normalizedFileLower.includes("/x-article-export") ||
    normalizedText.includes("handlecodeblockclick") ||
    normalizedText.includes("shouldcollapsecodeblockinchat") ||
    normalizedText.includes("code fence") ||
    normalizedText.includes("data-language") ||
    normalizedText.includes("代码块语言") ||
    normalizedText.includes("language-") ||
    normalizedText.includes('language: "html"') ||
    normalizedText.includes('language: "json"') ||
    normalizedText.includes('language: "text"') ||
    normalizedText.includes('language: "typescript"')
  ) {
    return "codeLanguage";
  }

  if (
    normalizedMarker.includes("target_language") ||
    normalizedMarker.includes("targetlanguage") ||
    normalizedFileLower.includes("/artifact_") ||
    normalizedFileLower.includes("/artifact/") ||
    normalizedFileLower.includes("/artifact-protocol/") ||
    normalizedFileLower.includes("/agentthreadtimeline") ||
    normalizedFileLower.includes("/home/homecoverassets") ||
    normalizedFileLower.includes("/hooks/useasteragentchat.test") ||
    normalizedFileLower.includes("/creation_tools") ||
    normalizedFileLower.includes("/api/knowledge") ||
    normalizedFileLower.includes("/knowledgemocks") ||
    normalizedFileLower.includes("/knowledge/") ||
    normalizedFileLower.includes("/media/") ||
    normalizedFileLower.includes("/media_task_cmd") ||
    normalizedFileLower.includes("/artifact-document/") ||
    normalizedFileLower.includes("/artifacttimeline") ||
    normalizedFileLower.includes("/artifacttool") ||
    normalizedFileLower.includes("/artifactworkbench") ||
    normalizedFileLower.includes("/canvas") ||
    normalizedFileLower.includes("/mentioncommandreplaytext") ||
    normalizedFileLower.includes("/messageartifacts") ||
    normalizedFileLower.includes("/runtime_evidence_modality") ||
    normalizedFileLower.includes(
      "/workspace/useworkspacecanvasworkflowactions",
    ) ||
    normalizedFileLower.includes("/taskmessagepreview") ||
    normalizedFileLower.includes("/taskpreviewfromtoolresult") ||
    normalizedFileLower.includes("/translationworkbenchcommand") ||
    normalizedFileLower.includes("/workspaceartifactpreview") ||
    normalizedFileLower.includes("/workspacefilepreview") ||
    normalizedText.includes("video-dubbing-language")
  ) {
    return "contentTargetLanguage";
  }

  if (
    normalizedFileLower.includes("/crates/cli/src/main.rs") ||
    normalizedFileLower.includes("/asrprovider") ||
    normalizedFileLower.includes("/agent/chat/types") ||
    normalizedFileLower.includes("/mediataskmocks") ||
    normalizedFileLower.includes("/workspace/useworkspacesendactions") ||
    normalizedFileLower.includes("/runtime_evidence_pack_service_tests") ||
    normalizedFileLower.includes("asr_cmd") ||
    normalizedFileLower.includes("voice_model") ||
    normalizedFileLower.includes("/voice") ||
    normalizedFileLower.includes("/transcription") ||
    normalizedFileLower.includes("transcriptiontaskpreview") ||
    normalizedFileLower.includes("transcriptcorrection") ||
    normalizedText.includes("parsedcommand.language") ||
    normalizedText.includes("transcription_task") ||
    normalizedText.includes("transcription_generate") ||
    normalizedText.includes("transcript_language") ||
    normalizedMarker.includes("preferredlanguage") ||
    normalizedMarker.includes("preferred_language")
  ) {
    return "asrLanguage";
  }

  if (
    normalizedMarker === "locale" ||
    normalizedMarker === "i18n.language" ||
    normalizedMarker === "resolvedlanguage" ||
    normalizedFileLower.includes("/src/i18n/") ||
    normalizedFileLower.includes("/api-key-provider/") ||
    normalizedFileLower.includes("/appsidebar") ||
    normalizedFileLower.includes("/configsystemmocks") ||
    normalizedFileLower.includes("/crashdiagnostic") ||
    normalizedFileLower.includes("/general/appearance/") ||
    normalizedFileLower.includes("/memory/memorypage") ||
    normalizedFileLower.includes("/runtime_api.rs") ||
    normalizedFileLower.includes("/settings-v2/agent/providers/") ||
    normalizedFileLower.includes("/settings-v2/shared/language/") ||
    normalizedFileLower.includes("/appconfigtypes") ||
    normalizedFileLower.includes("config/tests.rs") ||
    normalizedFileLower.includes("config/types.rs") ||
    normalizedText.includes("interface language") ||
    normalizedText.includes("i18n: { language") ||
    normalizedText.includes("config().language") ||
    normalizedText.includes("config.language") ||
    normalizedText.includes("getfixedt(instance.language") ||
    normalizedText.includes("getfixedt(currenti18n.language") ||
    normalizedText.includes("navigator.language") ||
    normalizedText.includes("settings-v2/shared/language") ||
    normalizedText.includes("documentelement.lang")
  ) {
    return "uiLocale";
  }

  return "unknownLanguageLike";
}

function extractMarker(lineText: string): string | null {
  return lineText.match(LANGUAGE_MARKER_PATTERN)?.[1] ?? null;
}

function sortCountSummaries<T extends { count: number }>(
  left: T,
  right: T,
  tieBreaker: (item: T) => string,
): number {
  if (right.count !== left.count) {
    return right.count - left.count;
  }
  return tieBreaker(left).localeCompare(tieBreaker(right));
}

function buildFileSummaries(
  entries: I18nLanguageBoundaryEntry[],
): I18nLanguageBoundaryFileSummary[] {
  return Array.from(
    entries.reduce((counts, entry) => {
      counts.set(entry.file, (counts.get(entry.file) ?? 0) + 1);
      return counts;
    }, new Map<string, number>()),
  )
    .map(([file, count]) => ({ count, file }))
    .sort((left, right) =>
      sortCountSummaries(left, right, (item) => item.file),
    );
}

function buildMarkerSummaries(
  entries: I18nLanguageBoundaryEntry[],
): I18nLanguageBoundaryMarkerSummary[] {
  return Array.from(
    entries.reduce((counts, entry) => {
      counts.set(entry.marker, (counts.get(entry.marker) ?? 0) + 1);
      return counts;
    }, new Map<string, number>()),
  )
    .map(([marker, count]) => ({ count, marker }))
    .sort((left, right) =>
      sortCountSummaries(left, right, (item) => item.marker),
    );
}

export function analyzeI18nLanguageBoundaryReport(
  options: Pick<CliOptions, "category" | "rootDir" | "sourceDirs">,
): I18nLanguageBoundaryReport {
  const rootDir = path.resolve(options.rootDir || REPO_ROOT);
  const sourceDirs =
    options.sourceDirs.length > 0 ? options.sourceDirs : DEFAULT_SOURCE_DIRS;
  const files = collectSourceFiles(rootDir, sourceDirs);
  const entries: I18nLanguageBoundaryEntry[] = [];

  for (const file of files) {
    const lines = fs.readFileSync(file, "utf8").split(/\r?\n/);
    lines.forEach((lineText, index) => {
      const marker = extractMarker(lineText);
      if (!marker) {
        return;
      }
      const text = lineText.trim();
      entries.push({
        category: classifyLanguageMarker(file, text, marker),
        file: displayPath(rootDir, file),
        line: index + 1,
        marker,
        text: text.length > 180 ? `${text.slice(0, 177)}...` : text,
      });
    });
  }

  const filteredEntries = options.category
    ? entries.filter((entry) => entry.category === options.category)
    : entries;
  const categorySummaries = Array.from(
    filteredEntries.reduce((counts, entry) => {
      counts.set(entry.category, (counts.get(entry.category) ?? 0) + 1);
      return counts;
    }, new Map<I18nLanguageBoundaryCategory, number>()),
  )
    .map(([category, count]) => ({ category, count }))
    .sort((left, right) =>
      sortCountSummaries(left, right, (item) => item.category),
    );

  return {
    entries: filteredEntries,
    filters: options.category ? { category: options.category } : {},
    rootDir,
    schemaVersion: "lime.i18n.languageBoundaryReport.v1",
    scannedFileCount: files.length,
    sourceDirs,
    summary: {
      categorySummaries,
      entryCount: filteredEntries.length,
      fileSummaries: buildFileSummaries(filteredEntries),
      markerSummaries: buildMarkerSummaries(filteredEntries),
      unknownCount:
        categorySummaries.find(
          (item) => item.category === "unknownLanguageLike",
        )?.count ?? 0,
    },
  };
}

export function formatI18nLanguageBoundaryReport(
  report: I18nLanguageBoundaryReport,
  format: I18nLanguageBoundaryReportFormat = "text",
): string {
  if (format === "json") {
    return `${JSON.stringify(report, null, 2)}${os.EOL}`;
  }

  const lines = [
    `[i18n:language-boundary] root=${displayPath(REPO_ROOT, report.rootDir)} scanned=${report.scannedFileCount} entries=${report.summary.entryCount} unknown=${report.summary.unknownCount}${report.filters.category ? ` category=${report.filters.category}` : ""}`,
    "[i18n:language-boundary] categories:",
  ];

  for (const item of report.summary.categorySummaries) {
    lines.push(`  - ${item.category}: ${item.count}`);
  }

  if (report.summary.fileSummaries.length > 0) {
    lines.push("[i18n:language-boundary] top files:");
    for (const item of report.summary.fileSummaries.slice(0, 10)) {
      lines.push(`  - ${item.file}: ${item.count}`);
    }
  }

  if (report.summary.markerSummaries.length > 0) {
    lines.push("[i18n:language-boundary] markers:");
    for (const item of report.summary.markerSummaries.slice(0, 10)) {
      lines.push(`  - ${item.marker}: ${item.count}`);
    }
  }

  if (report.summary.unknownCount > 0) {
    lines.push("[i18n:language-boundary] unknown language-like markers:");
    for (const entry of report.entries
      .filter((item) => item.category === "unknownLanguageLike")
      .slice(0, 20)) {
      lines.push(
        `  - ${entry.file}:${entry.line} ${entry.marker} :: ${entry.text}`,
      );
    }
  }

  return `${lines.join(os.EOL)}${os.EOL}`;
}

function parseCliArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    format: "text",
    rootDir: REPO_ROOT,
    sourceDirs: [],
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];

    if (arg === "--format" && next) {
      if (next === "json" || next === "text") {
        options.format = next;
        index += 1;
        continue;
      }
      throw new Error(`Unknown or missing format value: ${next}`);
    }

    if (arg === "--category" && next) {
      if (
        LANGUAGE_BOUNDARY_CATEGORIES.includes(
          next as I18nLanguageBoundaryCategory,
        )
      ) {
        options.category = next as I18nLanguageBoundaryCategory;
        index += 1;
        continue;
      }
      throw new Error(`Unknown language boundary category: ${next}`);
    }

    if (arg === "--output" && next) {
      options.output = next;
      index += 1;
      continue;
    }

    if (arg === "--root" && next) {
      options.rootDir = next;
      index += 1;
      continue;
    }

    if (arg === "--source-dir" && next) {
      options.sourceDirs.push(next);
      index += 1;
      continue;
    }

    if (arg === "--help" || arg === "-h") {
      console.log(
        "Usage: npm run i18n:language-boundary-report -- [--format text|json] [--category <category>] [--output <path>] [--root <dir>] [--source-dir <dir>]",
      );
      process.exit(0);
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  if (options.sourceDirs.length === 0) {
    options.sourceDirs = DEFAULT_SOURCE_DIRS;
  }

  return options;
}

export function runCli(argv = process.argv.slice(2)): number {
  const options = parseCliArgs(argv);
  const report = analyzeI18nLanguageBoundaryReport(options);
  const content = formatI18nLanguageBoundaryReport(report, options.format);

  if (options.output) {
    const outputPath = path.resolve(options.output);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, content, "utf8");
  } else {
    process.stdout.write(content);
  }

  return 0;
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  try {
    process.exitCode = runCli();
  } catch (error) {
    console.error(
      `[i18n:language-boundary] 失败：${error instanceof Error ? error.message : String(error)}`,
    );
    process.exitCode = 1;
  }
}
