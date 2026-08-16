import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd, env } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();
const LIME_EXECUTION_POLICY_SOURCE = "src/lib/model/modelExecutionPolicy.ts";
const DEFAULT_CODEX_OPENAI_MODELS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/openai_models.rs";
const CODEX_OPENAI_MODELS_SOURCE =
  env.CODEX_OPENAI_MODELS_SOURCE ?? DEFAULT_CODEX_OPENAI_MODELS_SOURCE;

const CODEX_TOOL_MODE_VALUES = ["direct", "code_mode", "code_mode_only"];
const CODEX_WEB_SEARCH_TOOL_TYPE_VALUES = ["text", "text_and_image"];

const CODEX_EXECUTION_POLICY_MODEL_INFO_FIELDS = [
  "web_search_tool_type",
  "supports_image_detail_original",
  "supports_search_tool",
  "tool_mode",
];

const LIME_EXECUTION_POLICY_INPUT_FIELDS = [
  "tool_mode",
  "toolMode",
  "supports_image_detail_original",
  "supportsImageDetailOriginal",
  "supports_search_tool",
  "supportsSearchTool",
  "web_search_tool_type",
  "webSearchToolType",
];

const CODEX_FIELDS_REQUIRING_SEPARATE_OWNERS = [
  "default_reasoning_level",
  "supported_reasoning_levels",
  "visibility",
  "service_tiers",
  "default_service_tier",
  "supports_parallel_tool_calls",
  "context_window",
  "auto_compact_token_limit",
  "effective_context_window_percent",
  "input_modalities",
];

const CODEX_CURRENT_MODEL_INFO_FIELDS_REQUIRING_SEPARATE_OWNERS =
  CODEX_FIELDS_REQUIRING_SEPARATE_OWNERS.filter(
    // Current Codex fixes parallel_tool_calls=true in Prompt and no longer
    // exposes this historical capability field on ModelInfo.
    (field) => field !== "supports_parallel_tool_calls",
  );

function readRepoFile(path: string): string {
  return readFileSync(join(REPO_ROOT, path), "utf8");
}

function readCodexSource(): string | null {
  if (!existsSync(CODEX_OPENAI_MODELS_SOURCE)) {
    return null;
  }
  return readFileSync(CODEX_OPENAI_MODELS_SOURCE, "utf8");
}

function requireMatch(source: string, pattern: RegExp, label: string): string {
  const match = source.match(pattern);
  expect(match, `${label} not found`).not.toBeNull();
  return match?.groups?.body ?? match?.[1] ?? "";
}

function extractConstStringArray(source: string, name: string): string[] {
  const body = requireMatch(
    source,
    new RegExp(
      `export const ${name} = \\[\\n(?<body>[\\s\\S]*?)\\n\\] as const;`,
      "u",
    ),
    name,
  );
  return [...body.matchAll(/"([^"]+)"/gmu)].map((match) => match[1]);
}

function extractTypeFieldNames(source: string, name: string): string[] {
  const body = requireMatch(
    source,
    new RegExp(
      `export interface ${name} \\{\\n(?<body>[\\s\\S]*?)\\n\\}`,
      "u",
    ),
    name,
  );
  return [...body.matchAll(/^\s{2}([a-zA-Z_][a-zA-Z0-9_]*)\??:/gmu)].map(
    (match) => match[1],
  );
}

function extractRustPubFieldNames(source: string, name: string): string[] {
  const body = requireMatch(
    source,
    new RegExp(`pub struct ${name} \\{\\n(?<body>[\\s\\S]*?)\\n\\}`, "u"),
    name,
  );
  return [...body.matchAll(/^\s+pub\s+([a-zA-Z_][a-zA-Z0-9_]*):/gmu)].map(
    (match) => match[1],
  );
}

function extractRustEnumWireValues(source: string, name: string): string[] {
  const body = requireMatch(
    source,
    new RegExp(`pub enum ${name} \\{\\n(?<body>[\\s\\S]*?)\\n\\}`, "u"),
    name,
  );
  return [...body.matchAll(/^\s+([A-Z][a-zA-Z0-9]*),$/gmu)]
    .map((match) => match[1])
    .map(toSnakeCase);
}

function toSnakeCase(value: string): string {
  return value.replace(/([a-z0-9])([A-Z])/gu, "$1_$2").toLowerCase();
}

describe("Codex model execution policy origin", () => {
  it("Lime execution policy 枚举值与 Codex ToolMode / WebSearchToolType 保持一致", () => {
    const limeSource = readRepoFile(LIME_EXECUTION_POLICY_SOURCE);

    expect(extractConstStringArray(limeSource, "MODEL_TOOL_MODES")).toEqual(
      CODEX_TOOL_MODE_VALUES,
    );
    expect(
      extractConstStringArray(limeSource, "MODEL_WEB_SEARCH_TOOL_TYPES"),
    ).toEqual(CODEX_WEB_SEARCH_TOOL_TYPE_VALUES);

    const codexSource = readCodexSource();
    if (!codexSource) {
      return;
    }

    expect(extractRustEnumWireValues(codexSource, "ToolMode")).toEqual(
      CODEX_TOOL_MODE_VALUES,
    );
    expect(
      extractRustEnumWireValues(codexSource, "WebSearchToolType"),
    ).toEqual(CODEX_WEB_SEARCH_TOOL_TYPE_VALUES);
  });

  it("ModelExecutionPolicyInput 只接收当前已认领的 Codex ModelInfo execution 字段", () => {
    const limeSource = readRepoFile(LIME_EXECUTION_POLICY_SOURCE);

    expect(
      extractTypeFieldNames(limeSource, "ModelExecutionPolicyInput"),
    ).toEqual(LIME_EXECUTION_POLICY_INPUT_FIELDS);

    const codexSource = readCodexSource();
    if (!codexSource) {
      return;
    }

    const codexModelInfoFields = extractRustPubFieldNames(
      codexSource,
      "ModelInfo",
    );
    expect(
      codexModelInfoFields.filter((field) =>
        CODEX_EXECUTION_POLICY_MODEL_INFO_FIELDS.includes(field),
      ),
    ).toEqual(CODEX_EXECUTION_POLICY_MODEL_INFO_FIELDS);
  });

  it("Codex 其它模型字段必须留在独立 owner，不混入当前 execution policy", () => {
    const limeSource = readRepoFile(LIME_EXECUTION_POLICY_SOURCE);
    const policyFields = extractTypeFieldNames(limeSource, "ModelExecutionPolicy");

    for (const field of CODEX_FIELDS_REQUIRING_SEPARATE_OWNERS) {
      expect(
        policyFields,
        `${field} 需要独立 owner，不能顺手混入 modelExecutionPolicy`,
      ).not.toContain(field);
    }

    const codexSource = readCodexSource();
    if (!codexSource) {
      return;
    }

    const codexModelInfoFields = extractRustPubFieldNames(
      codexSource,
      "ModelInfo",
    );
    for (const field of CODEX_CURRENT_MODEL_INFO_FIELDS_REQUIRING_SEPARATE_OWNERS) {
      expect(
        codexModelInfoFields,
        `Codex ModelInfo 缺少已登记字段 ${field}`,
      ).toContain(field);
    }
  });
});
