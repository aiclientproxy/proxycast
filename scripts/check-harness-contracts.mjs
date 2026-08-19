#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const repoRoot = process.cwd();

function readSource(relativePath) {
  return fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
}

function assertMatch(source, pattern, message, failures) {
  if (!pattern.test(source)) {
    failures.push(message);
  }
}

function assertIncludes(source, needle, message, failures) {
  if (!source.includes(needle)) {
    failures.push(message);
  }
}

function assertNotMatch(source, pattern, message, failures) {
  if (pattern.test(source)) {
    failures.push(message);
  }
}

function extractBalancedBlock(sourceCode, marker, openChar, closeChar) {
  const markerIndex = sourceCode.indexOf(marker);
  if (markerIndex < 0) {
    throw new Error(`未找到标记: ${marker}`);
  }

  const openIndex = sourceCode.indexOf(openChar, markerIndex);
  if (openIndex < 0) {
    throw new Error(`标记后未找到 ${openChar}: ${marker}`);
  }

  let depth = 0;
  let inSingleQuote = false;
  let inDoubleQuote = false;
  let inTemplateString = false;
  let inLineComment = false;
  let inBlockComment = false;
  let escaped = false;

  for (let index = openIndex; index < sourceCode.length; index += 1) {
    const currentChar = sourceCode[index];
    const nextChar = sourceCode[index + 1];

    if (inLineComment) {
      if (currentChar === "\n") {
        inLineComment = false;
      }
      continue;
    }

    if (inBlockComment) {
      if (currentChar === "*" && nextChar === "/") {
        inBlockComment = false;
        index += 1;
      }
      continue;
    }

    if (inSingleQuote) {
      if (!escaped && currentChar === "'") {
        inSingleQuote = false;
      }
      escaped = !escaped && currentChar === "\\";
      continue;
    }

    if (inDoubleQuote) {
      if (!escaped && currentChar === '"') {
        inDoubleQuote = false;
      }
      escaped = !escaped && currentChar === "\\";
      continue;
    }

    if (inTemplateString) {
      if (!escaped && currentChar === "`") {
        inTemplateString = false;
      }
      escaped = !escaped && currentChar === "\\";
      continue;
    }

    if (currentChar === "/" && nextChar === "/") {
      inLineComment = true;
      index += 1;
      continue;
    }

    if (currentChar === "/" && nextChar === "*") {
      inBlockComment = true;
      index += 1;
      continue;
    }

    if (currentChar === "'") {
      inSingleQuote = true;
      escaped = false;
      continue;
    }

    if (currentChar === '"') {
      inDoubleQuote = true;
      escaped = false;
      continue;
    }

    if (currentChar === "`") {
      inTemplateString = true;
      escaped = false;
      continue;
    }

    if (currentChar === openChar) {
      depth += 1;
      continue;
    }

    if (currentChar === closeChar) {
      depth -= 1;
      if (depth === 0) {
        return sourceCode.slice(openIndex + 1, index);
      }
    }
  }

  throw new Error(`无法提取 ${marker} 的平衡块`);
}

function main() {
  const failures = [];
  const harnessMetadataPath =
    "src/components/agent/chat/utils/harnessRequestMetadata.ts";
  const executionRuntimePath =
    "src/components/agent/chat/utils/sessionExecutionRuntime.ts";
  const executionRuntimeProjectionPath =
    "src/components/agent/chat/projection/sessionExecutionRuntimeProjection.ts";
  // 旧 backend 已随 `lime-rs/src/**` 整目录在 2026-06-10 删除；harness metadata 的 backend 事实源
  // 现在落在 `lime-rs/crates/agent/src/session_execution_runtime.rs` 等 crate 模块，
  // 字段映射的写法不再是 `("frontKey","backKey")` 元组，本脚本只继续守住前端约束。

  const harnessMetadataSource = readSource(harnessMetadataPath);
  const executionRuntimeSource = readSource(executionRuntimePath);
  const executionRuntimeProjectionSource = readSource(
    executionRuntimeProjectionPath,
  );

  const legacyKeysBlock = extractBalancedBlock(
    harnessMetadataSource,
    "const LEGACY_HARNESS_STATE_KEYS = [",
    "[",
    "]",
  );
  const metadataBuilderBlock = extractBalancedBlock(
    harnessMetadataSource,
    "const metadata: Record<string, unknown> = {",
    "{",
    "}",
  );

  const requiredMetadataKeys = [
    "preferences:",
    "browser_requirement:",
    "browser_requirement_reason:",
    "browser_launch_url:",
  ];

  const forbiddenLegacyOutputKeys = [
    "creation_mode:",
    "creationMode:",
    "chat_mode:",
    "chatMode:",
    "web_search_enabled:",
    "webSearchEnabled:",
    "thinking_enabled:",
    "thinkingEnabled:",
    "task_mode_enabled:",
    "taskModeEnabled:",
    "subagent_mode_enabled:",
    "subagentModeEnabled:",
    "turn_team_decision:",
    "turnTeamDecision:",
    "turn_team_reason:",
    "turnTeamReason:",
    "turn_team_blueprint:",
    "turnTeamBlueprint:",
    "selected_team_disabled:",
    "selectedTeamDisabled:",
    "preferred_team_preset_id:",
    "preferredTeamPresetId:",
    "selected_team_id:",
    "selectedTeamId:",
    "selected_team_source:",
    "selectedTeamSource:",
    "selected_team_label:",
    "selectedTeamLabel:",
    "selected_team_description:",
    "selectedTeamDescription:",
    "selected_team_summary:",
    "selectedTeamSummary:",
    "selected_team_roles:",
    "selectedTeamRoles:",
    "team_memory_shadow:",
    "teamMemoryShadow:",
    "browser_assist:",
    "browserAssist:",
  ];

  const requiredLegacyCleanupKeys = [
    "creation_mode",
    "chat_mode",
    "web_search_enabled",
    "thinking_enabled",
    "task_mode_enabled",
    "subagent_mode_enabled",
    "turn_team_decision",
    "turn_team_reason",
    "turn_team_blueprint",
    "selected_team_disabled",
    "preferred_team_preset_id",
    "selected_team_id",
    "selected_team_source",
    "selected_team_label",
    "selected_team_description",
    "selected_team_summary",
    "selected_team_roles",
    "team_memory_shadow",
    "browser_assist",
    "browserAssist",
  ];

  const requiredRuntimeFields = [
    "session_id:",
    "execution_strategy:",
    "recent_preferences:",
    "recent_content_id:",
  ];

  requiredMetadataKeys.forEach((key) => {
    assertIncludes(
      metadataBuilderBlock,
      key,
      `[harness-contracts] 前端 metadata builder 缺少字段: ${key}`,
      failures,
    );
  });

  forbiddenLegacyOutputKeys.forEach((key) => {
    assertNotMatch(
      metadataBuilderBlock,
      new RegExp(`\\b${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`),
      `[harness-contracts] 前端 metadata builder 仍在输出 legacy 字段: ${key}`,
      failures,
    );
  });

  requiredLegacyCleanupKeys.forEach((key) => {
    assertIncludes(
      legacyKeysBlock,
      `"${key}"`,
      `[harness-contracts] LEGACY_HARNESS_STATE_KEYS 缺少清理项: ${key}`,
      failures,
    );
  });

  assertMatch(
    metadataBuilderBlock,
    /\btask:\s*preferences\.task\b/,
    "[harness-contracts] 前端未按约定输出 current preferences.task",
    failures,
  );
  assertMatch(
    metadataBuilderBlock,
    /\bsubagent:\s*preferences\.subagent\b/,
    "[harness-contracts] 前端未按约定输出 current preferences.subagent",
    failures,
  );

  requiredRuntimeFields.forEach((field) => {
    assertIncludes(
      executionRuntimeProjectionSource,
      field,
      `[harness-contracts] execution runtime projection 缺少字段: ${field}`,
      failures,
    );
  });

  assertIncludes(
    executionRuntimeSource,
    "createSessionRecentPreferencesFromChatToolPreferences",
    "[harness-contracts] execution runtime 缺少 recent preferences 适配函数",
    failures,
  );
  assertNotMatch(
    executionRuntimeSource,
    /RecentTeamSelection|createTeamDefinitionFromExecutionRuntime|createSessionRecentTeamSelection/u,
    "[harness-contracts] execution runtime 仍保留 retired recent Team adapter",
    failures,
  );
  assertNotMatch(
    executionRuntimeProjectionSource,
    /recent_team_selection|recentTeamSelection/u,
    "[harness-contracts] execution runtime projection 仍保留 retired recent Team 字段",
    failures,
  );

  console.log("[harness-contracts] 检查文件:");
  console.log(`- ${harnessMetadataPath}`);
  console.log(`- ${executionRuntimePath}`);
  console.log(`- ${executionRuntimeProjectionPath}`);

  if (failures.length > 0) {
    console.error("\n[harness-contracts] 发现契约漂移：");
    failures.forEach((failure) => {
      console.error(`- ${failure}`);
    });
    process.exitCode = 1;
    return;
  }

  console.log("\n[harness-contracts] Harness 契约检查通过。");
}

main();
