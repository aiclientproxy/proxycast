import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd, env } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();
const LIME_NATIVE_TOOL_POLICY_SOURCE =
  "src/lib/model/modelNativeToolPolicy.ts";
const DEFAULT_CODEX_OPENAI_MODELS_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/protocol/src/openai_models.rs";
const DEFAULT_CODEX_TOOL_CONFIG_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/tools/src/tool_config.rs";
const DEFAULT_CODEX_SPEC_PLAN_SOURCE =
  "/Users/coso/Documents/dev/rust/codex/codex-rs/core/src/tools/spec_plan.rs";
const CODEX_OPENAI_MODELS_SOURCE =
  env.CODEX_OPENAI_MODELS_SOURCE ?? DEFAULT_CODEX_OPENAI_MODELS_SOURCE;
const CODEX_TOOL_CONFIG_SOURCE =
  env.CODEX_TOOL_CONFIG_SOURCE ?? DEFAULT_CODEX_TOOL_CONFIG_SOURCE;
const CODEX_SPEC_PLAN_SOURCE =
  env.CODEX_SPEC_PLAN_SOURCE ?? DEFAULT_CODEX_SPEC_PLAN_SOURCE;

const CODEX_NATIVE_TOOL_MODEL_INFO_FIELDS = [
  "shell_type",
  "apply_patch_tool_type",
  "experimental_supported_tools",
];

const LIME_NATIVE_TOOL_POLICY_INPUT_FIELDS = [
  "shell_type",
  "shellType",
  "apply_patch_tool_type",
  "applyPatchToolType",
  "experimental_supported_tools",
  "experimentalSupportedTools",
];

const LIME_NATIVE_TOOL_POLICY_FIELDS = [
  "shell_type",
  "shell_tool_enabled",
  "preferred_shell_surface",
  "apply_patch_tool_type",
  "apply_patch_tool_enabled",
  "experimental_supported_tools",
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

function sourceBetween(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  expect(startIndex, `${start} not found`).toBeGreaterThanOrEqual(0);
  const endIndex = source.indexOf(end, startIndex + start.length);
  expect(endIndex, `${end} not found after ${start}`).toBeGreaterThanOrEqual(0);
  return source.slice(startIndex, endIndex);
}

describe("Codex model native tool policy origin", () => {
  it("ModelNativeToolPolicyInput 只接收 Codex ModelInfo native tool 字段", () => {
    const limeSource = readRepoFile(LIME_NATIVE_TOOL_POLICY_SOURCE);

    expect(
      extractTypeFieldNames(limeSource, "ModelNativeToolPolicyInput"),
    ).toEqual(LIME_NATIVE_TOOL_POLICY_INPUT_FIELDS);
    expect(
      extractTypeFieldNames(limeSource, "ModelNativeToolPolicy"),
    ).toEqual(LIME_NATIVE_TOOL_POLICY_FIELDS);

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
        CODEX_NATIVE_TOOL_MODEL_INFO_FIELDS.includes(field),
      ),
    ).toEqual(CODEX_NATIVE_TOOL_MODEL_INFO_FIELDS);
  });

  it("shell_type 枚举和 default/local 映射沿用 Codex tool_config", () => {
    const limeSource = readRepoFile(LIME_NATIVE_TOOL_POLICY_SOURCE);

    expect(limeSource).toContain(
      '"default",\n  "local",\n  "unified_exec",\n  "disabled",\n  "shell_command"',
    );
    expect(limeSource).toContain('shellType === "default"');
    expect(limeSource).toContain('shellType === "local"');
    expect(limeSource).toContain('return "shell_command"');

    const codexOpenaiModelsSource = readIfExists(CODEX_OPENAI_MODELS_SOURCE);
    if (codexOpenaiModelsSource) {
      const shellType = sourceBetween(
        codexOpenaiModelsSource,
        "pub enum ConfigShellToolType",
        "pub enum ApplyPatchToolType",
      );
      expect(shellType).toContain("UnifiedExec");
      expect(shellType).toContain("Disabled");
      const hasLegacyVariants = [
        "Default",
        "Local",
        "ShellCommand",
      ].every((variant) => shellType.includes(variant));
      const hasCurrentAliases = [
        'alias = "default"',
        'alias = "local"',
        'alias = "shell_command"',
      ].every((alias) => shellType.includes(alias));
      expect(hasLegacyVariants || hasCurrentAliases).toBe(true);
    }

    const codexToolConfigSource = readIfExists(CODEX_TOOL_CONFIG_SOURCE);
    if (!codexToolConfigSource) {
      return;
    }

    if (!codexToolConfigSource.includes("shell_type_for_model_and_features")) {
      expect(codexToolConfigSource).toContain("UnifiedExecShellMode");
      return;
    }

    const shellTypeForModel = sourceBetween(
      codexToolConfigSource,
      "pub fn shell_type_for_model_and_features",
      "pub enum UnifiedExecShellMode",
    );
    expect(shellTypeForModel).toContain(
      "ConfigShellToolType::Default | ConfigShellToolType::Local",
    );
    expect(shellTypeForModel).toContain("ConfigShellToolType::ShellCommand");
    expect(shellTypeForModel).toContain("ConfigShellToolType::Disabled");
    expect(shellTypeForModel).toContain("ConfigShellToolType::UnifiedExec");
  });

  it("apply_patch_tool_type 与 experimental_supported_tools 沿用 Codex tool plan gate", () => {
    const limeSource = readRepoFile(LIME_NATIVE_TOOL_POLICY_SOURCE);

    expect(limeSource).toContain('MODEL_APPLY_PATCH_TOOL_TYPES = ["freeform"]');
    expect(limeSource).toContain("apply_patch_tool_enabled");
    expect(limeSource).toContain("experimental_supported_tools");

    const codexOpenaiModelsSource = readIfExists(CODEX_OPENAI_MODELS_SOURCE);
    if (codexOpenaiModelsSource) {
      const applyPatchToolType = sourceBetween(
        codexOpenaiModelsSource,
        "pub enum ApplyPatchToolType",
        "pub enum WebSearchToolType",
      );
      expect(applyPatchToolType).toContain("Freeform");
    }

    const codexSpecPlanSource = readIfExists(CODEX_SPEC_PLAN_SOURCE);
    if (!codexSpecPlanSource) {
      return;
    }

    expect(codexSpecPlanSource).toContain(
      "environment_mode.has_environment() && turn_context.model_info.apply_patch_tool_type.is_some()",
    );
    expect(codexSpecPlanSource).toContain(
      "registry.add(ApplyPatchHandler::new(include_environment_id));",
    );
    expect(codexSpecPlanSource).toContain(".experimental_supported_tools");
    expect(codexSpecPlanSource).toContain('tool == "test_sync_tool"');
    expect(codexSpecPlanSource).toContain("registry.add(TestSyncHandler);");
  });
});
