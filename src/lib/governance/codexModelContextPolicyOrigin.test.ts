import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd, env } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();
const LIME_CONTEXT_POLICY_SOURCE = "src/lib/model/modelContextPolicy.ts";
const DEFAULT_CODEX_OPENAI_MODELS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/openai_models.rs";
const DEFAULT_CODEX_TURN_CONTEXT_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/session/turn_context.rs";
const CODEX_OPENAI_MODELS_SOURCE =
  env.CODEX_OPENAI_MODELS_SOURCE ?? DEFAULT_CODEX_OPENAI_MODELS_SOURCE;
const CODEX_TURN_CONTEXT_SOURCE =
  env.CODEX_TURN_CONTEXT_SOURCE ?? DEFAULT_CODEX_TURN_CONTEXT_SOURCE;

const CODEX_CONTEXT_POLICY_MODEL_INFO_FIELDS = [
  "context_window",
  "max_context_window",
  "auto_compact_token_limit",
  "effective_context_window_percent",
];

const LIME_CONTEXT_POLICY_INPUT_FIELDS = [
  "context_window",
  "contextWindow",
  "max_context_window",
  "maxContextWindow",
  "auto_compact_token_limit",
  "autoCompactTokenLimit",
  "effective_context_window_percent",
  "effectiveContextWindowPercent",
];

const LIME_CONTEXT_POLICY_FIELDS = [
  "context_window",
  "max_context_window",
  "resolved_context_window",
  "effective_context_window_percent",
  "model_context_window",
  "auto_compact_token_limit",
];

function readRepoFile(path: string): string {
  return readFileSync(join(REPO_ROOT, path), "utf8");
}

function readExistingFile(path: string): string | null {
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

function sourceBetween(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  expect(startIndex, `${start} not found`).toBeGreaterThanOrEqual(0);
  const endIndex = source.indexOf(end, startIndex + start.length);
  expect(endIndex, `${end} not found after ${start}`).toBeGreaterThanOrEqual(0);
  return source.slice(startIndex, endIndex);
}

describe("Codex model context policy origin", () => {
  it("ModelContextPolicyInput 只接收 Codex ModelInfo context / auto compact 字段", () => {
    const limeSource = readRepoFile(LIME_CONTEXT_POLICY_SOURCE);

    expect(
      extractTypeFieldNames(limeSource, "ModelContextPolicyInput"),
    ).toEqual(LIME_CONTEXT_POLICY_INPUT_FIELDS);
    expect(extractTypeFieldNames(limeSource, "ModelContextPolicy")).toEqual(
      LIME_CONTEXT_POLICY_FIELDS,
    );

    const codexSource = readExistingFile(CODEX_OPENAI_MODELS_SOURCE);
    if (!codexSource) {
      return;
    }

    const codexModelInfoFields = extractRustPubFieldNames(
      codexSource,
      "ModelInfo",
    );
    expect(
      codexModelInfoFields.filter((field) =>
        CODEX_CONTEXT_POLICY_MODEL_INFO_FIELDS.includes(field),
      ),
    ).toEqual(CODEX_CONTEXT_POLICY_MODEL_INFO_FIELDS);
  });

  it("resolved context window 与 auto compact 上限沿用 Codex ModelInfo 语义", () => {
    const limeSource = readRepoFile(LIME_CONTEXT_POLICY_SOURCE);

    expect(limeSource).toContain("contextWindow ?? maxContextWindow");
    expect(limeSource).toContain(
      "AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR = 9",
    );
    expect(limeSource).toContain(
      "AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR = 10",
    );
    expect(limeSource).toContain("Math.min(configuredLimit, contextLimit)");

    const codexSource = readExistingFile(CODEX_OPENAI_MODELS_SOURCE);
    if (!codexSource) {
      return;
    }

    const resolvedContextWindow = sourceBetween(
      codexSource,
      "pub fn resolved_context_window(&self)",
      "pub fn auto_compact_token_limit(&self)",
    );
    const autoCompactTokenLimit = sourceBetween(
      codexSource,
      "pub fn auto_compact_token_limit(&self)",
      "    }",
    );

    expect(resolvedContextWindow).toContain(
      "self.context_window.or(self.max_context_window)",
    );
    expect(autoCompactTokenLimit).toContain("(context_window * 9) / 10");
    expect(autoCompactTokenLimit).toContain("std::cmp::min");
  });

  it("model context window 沿用 Codex effective_context_window_percent 语义", () => {
    const limeSource = readRepoFile(LIME_CONTEXT_POLICY_SOURCE);

    expect(limeSource).toContain(
      "DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 95",
    );
    expect(limeSource).toContain(
      "(resolvedContextWindow * effectiveContextWindowPercent) / 100",
    );

    const codexSource = readExistingFile(CODEX_TURN_CONTEXT_SOURCE);
    const codexOpenaiModelsSource = readExistingFile(
      CODEX_OPENAI_MODELS_SOURCE,
    );
    if (!codexSource || !codexOpenaiModelsSource) {
      return;
    }

    const modelContextWindow = sourceBetween(
      codexSource,
      "pub(crate) fn model_context_window(&self)",
      "    pub(crate) fn apps_enabled(&self)",
    );

    if (
      modelContextWindow.includes("self.model_info().usable_context_window()")
    ) {
      const usableContextWindow = sourceBetween(
        codexOpenaiModelsSource,
        "    pub fn usable_context_window(&self)",
        "    pub fn auto_compact_token_limit(&self)",
      );
      expect(usableContextWindow).toContain(
        "self.effective_context_window_percent",
      );
      expect(usableContextWindow).toContain(
        "context_window.saturating_mul(self.effective_context_window_percent) / 100",
      );
    } else {
      expect(modelContextWindow).toContain("effective_context_window_percent");
      expect(modelContextWindow).toContain(
        "context_window.saturating_mul(effective_context_window_percent) / 100",
      );
    }
  });
});
