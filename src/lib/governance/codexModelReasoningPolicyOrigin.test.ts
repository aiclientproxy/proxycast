import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd, env } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();
const LIME_REASONING_POLICY_SOURCE = "src/lib/model/modelReasoningPolicy.ts";
const DEFAULT_CODEX_OPENAI_MODELS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/openai_models.rs";
const DEFAULT_CODEX_TURN_CONTEXT_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/session/turn_context.rs";
const DEFAULT_CODEX_STEP_SETTINGS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/session/step_settings.rs";
const CODEX_OPENAI_MODELS_SOURCE =
  env.CODEX_OPENAI_MODELS_SOURCE ?? DEFAULT_CODEX_OPENAI_MODELS_SOURCE;
const CODEX_TURN_CONTEXT_SOURCE =
  env.CODEX_TURN_CONTEXT_SOURCE ?? DEFAULT_CODEX_TURN_CONTEXT_SOURCE;
const CODEX_STEP_SETTINGS_SOURCE =
  env.CODEX_STEP_SETTINGS_SOURCE ?? DEFAULT_CODEX_STEP_SETTINGS_SOURCE;

const CODEX_REASONING_EFFORT_VALUES = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
  "persistent",
];

const CODEX_REASONING_MODEL_INFO_FIELDS = [
  "default_reasoning_level",
  "supported_reasoning_levels",
  "supports_reasoning_summary_parameter",
];

const LIME_REASONING_POLICY_INPUT_FIELDS = [
  "supports_reasoning_summary_parameter",
  "supportsReasoningSummaryParameter",
  "supports_reasoning_summaries",
  "supportsReasoningSummaries",
  "default_reasoning_level",
  "defaultReasoningLevel",
  "supported_reasoning_levels",
  "supportedReasoningLevels",
];

const LIME_REASONING_POLICY_FIELDS = [
  "supports_reasoning_summaries",
  "default_reasoning_level",
  "supported_reasoning_levels",
  "supported_reasoning_efforts",
  "can_set_reasoning_effort",
];

function readRepoFile(path: string): string {
  return readFileSync(join(REPO_ROOT, path), "utf8");
}

function readIfExists(path: string): string | null {
  if (!existsSync(path)) {
    return null;
  }
  return readFileSync(path, "utf8");
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
    new RegExp(`export interface ${name} \\{\\n(?<body>[\\s\\S]*?)\\n\\}`, "u"),
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
  return [...body.matchAll(/^\s+([A-Z][a-zA-Z0-9]*),$/gmu)].map((match) =>
    match[1].toLowerCase(),
  );
}

describe("Codex model reasoning policy origin", () => {
  it("Lime 已知 reasoning effort 与 Codex 保持一致，并保留开放字符串语义", () => {
    const limeSource = readRepoFile(LIME_REASONING_POLICY_SOURCE);

    expect(
      extractConstStringArray(limeSource, "MODEL_REASONING_EFFORTS"),
    ).toEqual(CODEX_REASONING_EFFORT_VALUES);

    const codexSource = readIfExists(CODEX_OPENAI_MODELS_SOURCE);
    if (!codexSource) {
      return;
    }

    expect(extractRustEnumWireValues(codexSource, "ReasoningEffort")).toEqual(
      CODEX_REASONING_EFFORT_VALUES,
    );
    expect(codexSource).toContain("Self::Custom(effort)");
    expect(codexSource).toContain("reasoning_effort must not be empty");
  });

  it("ModelReasoningPolicyInput 只接收 Codex ModelInfo reasoning 字段", () => {
    const limeSource = readRepoFile(LIME_REASONING_POLICY_SOURCE);

    expect(
      extractTypeFieldNames(limeSource, "ModelReasoningPolicyInput"),
    ).toEqual(LIME_REASONING_POLICY_INPUT_FIELDS);
    expect(extractTypeFieldNames(limeSource, "ModelReasoningPolicy")).toEqual(
      LIME_REASONING_POLICY_FIELDS,
    );

    const codexSource = readIfExists(CODEX_OPENAI_MODELS_SOURCE);
    if (!codexSource) {
      return;
    }

    const codexModelInfoFields = extractRustPubFieldNames(
      codexSource,
      "ModelInfo",
    );
    expect(
      codexModelInfoFields.filter((field) =>
        CODEX_REASONING_MODEL_INFO_FIELDS.includes(field),
      ),
    ).toEqual(CODEX_REASONING_MODEL_INFO_FIELDS);
  });

  it("request reasoning effort 沿用 Codex default fallback 语义", () => {
    const limeSource = readRepoFile(LIME_REASONING_POLICY_SOURCE);

    expect(limeSource).toContain("hasReasoningEffortPolicy(policy)");
    expect(limeSource).toContain("return policy.default_reasoning_level");

    const codexTurnContextSource = readIfExists(CODEX_TURN_CONTEXT_SOURCE);
    const codexStepSettingsSource = readIfExists(CODEX_STEP_SETTINGS_SOURCE);
    if (!codexTurnContextSource && !codexStepSettingsSource) {
      return;
    }

    expect(
      codexStepSettingsSource?.includes(
        ".or(self.model_info.default_reasoning_level.as_ref())",
      ) ||
        codexTurnContextSource?.includes(
          ".or_else(|| self.model_info.default_reasoning_level.clone())",
        ) ||
        codexTurnContextSource?.includes(
          ".or_else(|| model_info.default_reasoning_level.clone())",
        ),
    ).toBe(true);
  });

  it("切模型 reasoning effort 沿用 Codex：保留受支持 current，否则取 supported 中位数再 fallback default", () => {
    const limeSource = readRepoFile(LIME_REASONING_POLICY_SOURCE);

    expect(limeSource).toContain("Math.floor((supported.length - 1) / 2)");
    expect(limeSource).toContain(
      "middleSupportedEffort(policy) ?? policy.default_reasoning_level",
    );

    const codexTurnContextSource = readIfExists(CODEX_TURN_CONTEXT_SOURCE);
    if (!codexTurnContextSource) {
      return;
    }

    expect(codexTurnContextSource).toContain(
      "supported_reasoning_levels.contains(current_reasoning_effort)",
    );
    expect(codexTurnContextSource).toContain(
      ".get(supported_reasoning_levels.len().saturating_sub(1) / 2)",
    );
    expect(codexTurnContextSource).toContain(
      ".or_else(|| model_info.default_reasoning_level.clone())",
    );
  });
});
