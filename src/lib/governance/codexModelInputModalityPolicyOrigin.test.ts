import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd, env } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();
const LIME_INPUT_MODALITY_POLICY_SOURCE =
  "src/lib/model/modelInputModalityPolicy.ts";
const DEFAULT_CODEX_OPENAI_MODELS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/openai_models.rs";
const DEFAULT_CODEX_HISTORY_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/context_manager/history.rs";
const DEFAULT_CODEX_NORMALIZE_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/context_manager/normalize.rs";
const CODEX_OPENAI_MODELS_SOURCE =
  env.CODEX_OPENAI_MODELS_SOURCE ?? DEFAULT_CODEX_OPENAI_MODELS_SOURCE;
const CODEX_HISTORY_SOURCE =
  env.CODEX_HISTORY_SOURCE ?? DEFAULT_CODEX_HISTORY_SOURCE;
const CODEX_NORMALIZE_SOURCE =
  env.CODEX_NORMALIZE_SOURCE ?? DEFAULT_CODEX_NORMALIZE_SOURCE;

const CODEX_INPUT_MODALITY_VALUES = ["text", "image", "audio"];
const CODEX_DEFAULT_INPUT_MODALITY_VALUES = ["text", "image"];
const CODEX_MODEL_INFO_INPUT_FIELDS = ["input_modalities"];

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
      `export const ${name} = \\[(?<body>[\\s\\S]*?)\\] as const;`,
      "u",
    ),
    name,
  );
  return [...body.matchAll(/"([^"]+)"/gmu)].map((match) => match[1]);
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

describe("Codex model input modality policy origin", () => {
  it("Lime Codex 默认输入模态与 Codex default_input_modalities 保持一致", () => {
    const limeSource = readRepoFile(LIME_INPUT_MODALITY_POLICY_SOURCE);

    expect(
      extractConstStringArray(limeSource, "CODEX_DEFAULT_INPUT_MODALITIES"),
    ).toEqual(CODEX_DEFAULT_INPUT_MODALITY_VALUES);

    const codexSource = readIfExists(CODEX_OPENAI_MODELS_SOURCE);
    if (!codexSource) {
      return;
    }

    expect(extractRustEnumWireValues(codexSource, "InputModality")).toEqual(
      CODEX_INPUT_MODALITY_VALUES,
    );
    expect(codexSource).toContain(
      "vec![InputModality::Text, InputModality::Image]",
    );
  });

  it("ModelInputModalityPolicyInput 只接收 Codex ModelInfo input_modalities 与 camelCase 等价字段", () => {
    const limeSource = readRepoFile(LIME_INPUT_MODALITY_POLICY_SOURCE);

    expect(limeSource).toContain("input_modalities?: unknown;");
    expect(limeSource).toContain("inputModalities?: unknown;");
    expect(limeSource).toContain("modalities?: unknown;");

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
        CODEX_MODEL_INFO_INPUT_FIELDS.includes(field),
      ),
    ).toEqual(CODEX_MODEL_INFO_INPUT_FIELDS);
  });

  it("Codex 用 input_modalities 驱动 prompt history 过滤和图片工具结果降级", () => {
    const historySource = readIfExists(CODEX_HISTORY_SOURCE);
    const normalizeSource = readIfExists(CODEX_NORMALIZE_SOURCE);
    if (!historySource || !normalizeSource) {
      return;
    }

    expect(historySource).toContain(
      "for_prompt(mut self, input_modalities: &[InputModality])",
    );
    expect(historySource).toContain("self.normalize_history(input_modalities)");
    expect(normalizeSource).toMatch(
      /fn strip_images_when_unsupported\(\s*input_modalities:/u,
    );
    expect(normalizeSource).toContain(
      "input_modalities.contains(&InputModality::Image)",
    );
  });
});
