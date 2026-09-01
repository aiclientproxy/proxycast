import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd, env } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();
const LIME_REASONING_OUTPUT_POLICY_SOURCE =
  "src/lib/model/modelReasoningOutputPolicy.ts";
const DEFAULT_CODEX_OPENAI_MODELS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/openai_models.rs";
const DEFAULT_CODEX_CONFIG_TYPES_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/config_types.rs";
const DEFAULT_CODEX_TURN_CONTEXT_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/session/turn_context.rs";
const DEFAULT_CODEX_STEP_SETTINGS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/session/step_settings.rs";
const DEFAULT_CODEX_CLIENT_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/client.rs";
const CODEX_OPENAI_MODELS_SOURCE =
  env.CODEX_OPENAI_MODELS_SOURCE ?? DEFAULT_CODEX_OPENAI_MODELS_SOURCE;
const CODEX_CONFIG_TYPES_SOURCE =
  env.CODEX_CONFIG_TYPES_SOURCE ?? DEFAULT_CODEX_CONFIG_TYPES_SOURCE;
const CODEX_TURN_CONTEXT_SOURCE =
  env.CODEX_TURN_CONTEXT_SOURCE ?? DEFAULT_CODEX_TURN_CONTEXT_SOURCE;
const CODEX_STEP_SETTINGS_SOURCE =
  env.CODEX_STEP_SETTINGS_SOURCE ?? DEFAULT_CODEX_STEP_SETTINGS_SOURCE;
const CODEX_CLIENT_SOURCE =
  env.CODEX_CLIENT_SOURCE ?? DEFAULT_CODEX_CLIENT_SOURCE;

const CODEX_REASONING_SUMMARY_VALUES = ["auto", "concise", "detailed", "none"];

const CODEX_VERBOSITY_VALUES = ["low", "medium", "high"];

const CODEX_REASONING_OUTPUT_MODEL_INFO_FIELDS = [
  "default_reasoning_summary",
  "support_verbosity",
  "default_verbosity",
];

const LIME_REASONING_OUTPUT_POLICY_INPUT_FIELDS = [
  "default_reasoning_summary",
  "defaultReasoningSummary",
  "support_verbosity",
  "supportVerbosity",
  "default_verbosity",
  "defaultVerbosity",
];

const LIME_REASONING_OUTPUT_POLICY_FIELDS = [
  "default_reasoning_summary",
  "support_verbosity",
  "default_verbosity",
  "can_set_verbosity",
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

describe("Codex model reasoning output policy origin", () => {
  it("Lime reasoning summary / verbosity 枚举与 Codex config types 保持一致", () => {
    const limeSource = readRepoFile(LIME_REASONING_OUTPUT_POLICY_SOURCE);

    expect(
      extractConstStringArray(limeSource, "MODEL_REASONING_SUMMARIES"),
    ).toEqual(CODEX_REASONING_SUMMARY_VALUES);
    expect(
      extractConstStringArray(limeSource, "MODEL_VERBOSITY_LEVELS"),
    ).toEqual(CODEX_VERBOSITY_VALUES);

    const codexConfigTypesSource = readIfExists(CODEX_CONFIG_TYPES_SOURCE);
    if (!codexConfigTypesSource) {
      return;
    }

    expect(
      extractRustEnumWireValues(codexConfigTypesSource, "ReasoningSummary"),
    ).toEqual(CODEX_REASONING_SUMMARY_VALUES);
    expect(
      extractRustEnumWireValues(codexConfigTypesSource, "Verbosity"),
    ).toEqual(CODEX_VERBOSITY_VALUES);
  });

  it("ModelReasoningOutputPolicyInput 只接收 Codex ModelInfo 输出控制字段", () => {
    const limeSource = readRepoFile(LIME_REASONING_OUTPUT_POLICY_SOURCE);

    expect(
      extractTypeFieldNames(limeSource, "ModelReasoningOutputPolicyInput"),
    ).toEqual(LIME_REASONING_OUTPUT_POLICY_INPUT_FIELDS);
    expect(
      extractTypeFieldNames(limeSource, "ModelReasoningOutputPolicy"),
    ).toEqual(LIME_REASONING_OUTPUT_POLICY_FIELDS);

    const codexOpenaiModelsSource = readIfExists(CODEX_OPENAI_MODELS_SOURCE);
    if (!codexOpenaiModelsSource) {
      return;
    }

    const codexModelInfoFields = extractRustPubFieldNames(
      codexOpenaiModelsSource,
      "ModelInfo",
    );
    expect(
      codexModelInfoFields.filter((field) =>
        CODEX_REASONING_OUTPUT_MODEL_INFO_FIELDS.includes(field),
      ),
    ).toEqual(CODEX_REASONING_OUTPUT_MODEL_INFO_FIELDS);
  });

  it("request reasoning summary 沿用 Codex model default 与 none 省略语义", () => {
    const limeSource = readRepoFile(LIME_REASONING_OUTPUT_POLICY_SOURCE);

    expect(limeSource).toContain("modelSupportsReasoningSummaries");
    expect(limeSource).toContain(
      "requested ?? policy.default_reasoning_summary",
    );
    expect(limeSource).toContain('summary === "none"');

    const codexTurnContextSource = readIfExists(CODEX_TURN_CONTEXT_SOURCE);
    const codexStepSettingsSource = readIfExists(CODEX_STEP_SETTINGS_SOURCE);
    const codexClientSource = readIfExists(CODEX_CLIENT_SOURCE);
    if (
      (!codexTurnContextSource && !codexStepSettingsSource) ||
      !codexClientSource
    ) {
      return;
    }

    expect(
      codexStepSettingsSource?.includes(
        ".unwrap_or(model_info.default_reasoning_summary)",
      ) ||
        codexTurnContextSource?.includes(
          ".unwrap_or(model_info.default_reasoning_summary)",
        ),
    ).toBe(true);
    expect(codexClientSource).toContain(
      "model_info.supports_reasoning_summary_parameter",
    );
    expect(codexClientSource).toContain(
      "summary != ReasoningSummaryConfig::None",
    );
  });

  it("request verbosity 沿用 Codex support_verbosity gate 与 default fallback", () => {
    const limeSource = readRepoFile(LIME_REASONING_OUTPUT_POLICY_SOURCE);

    expect(limeSource).toContain("if (!policy.support_verbosity)");
    expect(limeSource).toContain("requested ?? policy.default_verbosity");

    const codexClientSource = readIfExists(CODEX_CLIENT_SOURCE);
    if (!codexClientSource) {
      return;
    }

    expect(codexClientSource).toContain("if model_info.support_verbosity");
    expect(codexClientSource).toContain(
      "self.state.model_verbosity.or(model_info.default_verbosity)",
    );
    expect(codexClientSource).toContain(
      "model_verbosity is set but ignored as the model does not support verbosity",
    );
  });
});
