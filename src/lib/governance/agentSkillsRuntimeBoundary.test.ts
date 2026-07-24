/* global process */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();

const RUNTIME_BACKEND_DIR = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/runtime_backend",
);
const AGENT_SRC_DIR = join(REPO_ROOT, "lime-rs/crates/agent/src");

const SNAPSHOT_OWNER_FILES = new Set([
  "lime-rs/crates/app-server/src/runtime_backend/agent_skills_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/agent_skills_telemetry.rs",
]);

const SKILL_GATE_OWNER_FILES = new Set([
  "lime-rs/crates/app-server/src/runtime_backend/skill_runtime_enable.rs",
]);

const SKILL_TOOL_WRAPPER_OWNER =
  "lime-rs/crates/agent/src/tools/skill_tool_gate.rs";
const SKILL_SEARCH_TOOL_OWNER =
  "lime-rs/crates/tool-runtime/src/skill_search.rs";

function collectRustFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    const stats = statSync(fullPath);
    if (stats.isDirectory()) {
      if (entry === "target") {
        continue;
      }
      files.push(...collectRustFiles(fullPath));
      continue;
    }
    if (fullPath.endsWith(".rs")) {
      files.push(fullPath);
    }
  }
  return files;
}

function repoRelative(path: string): string {
  return relative(REPO_ROOT, path).replace(/\\/g, "/");
}

function productionSource(path: string): string {
  const source = readFileSync(path, "utf8");
  const testIndex = source.indexOf("#[cfg(test)]");
  return testIndex >= 0 ? source.slice(0, testIndex) : source;
}

describe("agent skills runtime boundary", () => {
  it("Agent turn runtime 不应绕过 AgentSkillSnapshot 直接扫描 skill roots", () => {
    const forbiddenSnippets = [
      "get_skill_roots(",
      "load_skills_from_directory(",
      "load_skill_from_file(",
    ];
    const offenders = collectRustFiles(RUNTIME_BACKEND_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) =>
        forbiddenSnippets.some((snippet) => source.includes(snippet)),
      )
      .map(({ path }) => path);

    expect(offenders).toEqual([]);
  });

  it("SKILL.md 正文读取只能挂在 AgentSkillSnapshot 选择链之后", () => {
    const offenders = collectRustFiles(RUNTIME_BACKEND_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ path, source }) => {
        if (!source.includes("read_agent_skill_body")) {
          return false;
        }
        return !SNAPSHOT_OWNER_FILES.has(path);
      })
      .map(({ path }) => path);

    expect(offenders).toEqual([]);
  });

  it("SkillTool session allowlist 只能由 runtime enable gate 写入", () => {
    const forbiddenSnippets = [
      "set_skill_tool_session_allowed_skills(",
      "set_skill_tool_session_allowed_skill_sources(",
    ];
    const offenders = collectRustFiles(RUNTIME_BACKEND_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ path, source }) => {
        if (!forbiddenSnippets.some((snippet) => source.includes(snippet))) {
          return false;
        }
        return !SKILL_GATE_OWNER_FILES.has(path);
      })
      .map(({ path }) => path);

    expect(offenders).toEqual([]);
  });

  it("生产代码不应绕过 LimeSkillTool 直接注册原始 SkillTool", () => {
    const offenders = collectRustFiles(AGENT_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ path, source }) => {
        if (!/\bSkillTool::new\(/u.test(source)) {
          return false;
        }
        return path !== SKILL_TOOL_WRAPPER_OWNER;
      })
      .map(({ path }) => path);

    expect(offenders).toEqual([]);
  });

  it("skill_search 只能搜索 metadata，不能读取正文或写入执行 gate", () => {
    const source = productionSource(join(REPO_ROOT, SKILL_SEARCH_TOOL_OWNER));

    expect(source).toContain("search_agent_skills");
    expect(source).toContain("build_agent_skill_snapshot_from_workspace");
    expect(source).not.toContain("read_agent_skill_body");
    expect(source).not.toContain("set_skill_tool_session_allowed_skills");
    expect(source).not.toContain(
      "set_skill_tool_session_allowed_skill_sources",
    );
  });

  it("runtime backend 不应新增硬编码 Skill(name) 首刀绕过 selector", () => {
    const offenders = collectRustFiles(RUNTIME_BACKEND_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) => /Skill\([A-Za-z0-9_-]+\)/u.test(source))
      .map(({ path }) => path);

    expect(offenders).toEqual([]);
  });

  it("专家 skillRefs 必须进入 selector、telemetry 和 SkillTool gate 主链", () => {
    const contextSource = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/agent_skills_context.rs",
      ),
    );
    const telemetrySource = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/agent_skills_telemetry.rs",
      ),
    );
    const gateSource = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/skill_runtime_enable.rs",
      ),
    );

    expect(contextSource).toContain("select_expert_bound_agent_skills");
    expect(contextSource).toContain(
      "AgentSkillSelectionTrigger::ExpertBinding",
    );
    expect(contextSource).toContain("expert_bound_skill_candidates");
    expect(telemetrySource).toContain("runtime_status_events_for_agent_skills");
    expect(telemetrySource).toContain("selection.trigger");
    expect(telemetrySource).toContain("skill_body_read");
    expect(telemetrySource).toContain("skill_gate_decision");
    expect(gateSource).toContain("selected_agent_skill_names_from_request");
    expect(gateSource).toContain("set_skill_tool_session_allowed_skills");
  });
});
