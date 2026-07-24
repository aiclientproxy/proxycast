import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();
const PROMPT_ROOT = "lime-rs/crates/agent/src/prompt";
const CURRENT_OWNER = `${PROMPT_ROOT}/runtime_agents.rs`;
const RETIRED_PATHS = [
  `${PROMPT_ROOT}/builder.rs`,
  `${PROMPT_ROOT}/instruction_discovery.rs`,
  `${PROMPT_ROOT}/templates.rs`,
] as const;

function read(relativePath: string): string {
  return readFileSync(join(REPO_ROOT, relativePath), "utf8");
}

function rustSources(relativeDir: string): string[] {
  const absoluteDir = join(REPO_ROOT, relativeDir);
  return readdirSync(absoluteDir, { withFileTypes: true }).flatMap((entry) => {
    const relativePath = join(relativeDir, entry.name);
    if (entry.isDirectory()) {
      return rustSources(relativePath);
    }
    return entry.isFile() && entry.name.endsWith(".rs") ? [relativePath] : [];
  });
}

describe("Agent instruction current owner boundary", () => {
  it("旧 builder、instruction discovery 和模板保持物理删除", () => {
    for (const retiredPath of RETIRED_PATHS) {
      expect(existsSync(join(REPO_ROOT, retiredPath)), retiredPath).toBe(false);
    }

    const promptModule = read(`${PROMPT_ROOT}/mod.rs`);
    const crateRoot = read("lime-rs/crates/agent/src/lib.rs");
    for (const retiredSymbol of [
      "SystemPromptBuilder",
      "InstructionLayer",
      "InstructionSource",
      "discover_instructions",
      "clear_instruction_cache",
    ]) {
      expect(promptModule).not.toContain(retiredSymbol);
      expect(crateRoot).not.toContain(retiredSymbol);
    }
  });

  it("prompt 源码不再接受非 Codex AGENT 文件名", () => {
    for (const sourcePath of rustSources(PROMPT_ROOT)) {
      const source = read(sourcePath);
      expect(source, sourcePath).not.toContain('"AGENT.md"');
      expect(source, sourcePath).not.toContain('".agent.md"');
      expect(source, sourcePath).not.toContain('"agent.md"');
      expect(source, sourcePath).not.toContain("AGENTS.local.md");
    }
  });

  it("runtime_agents 是唯一 owner 且由生产 runtime 消费", () => {
    const owner = read(CURRENT_OWNER);
    expect(owner).toContain("AGENTS.md");
    expect(owner).toContain("AGENTS.override.md");
    expect(owner).toContain("build_runtime_agents_prompt_for_project");

    for (const consumer of [
      "lime-rs/crates/app-server/src/runtime_backend/request_context/session_config.rs",
      "lime-rs/crates/scheduler/src/task_context.rs",
    ]) {
      expect(read(consumer), consumer).toContain(
        "merge_system_prompt_with_runtime_agents_for_project",
      );
    }
  });

  it("全局 AGENTS 路径只解析 current user home，不扫描或复制旧路径", () => {
    const appPaths = read("lime-rs/crates/core/src/app_paths.rs");
    const start = appPaths.indexOf("pub fn resolve_user_memory_path()");
    const end = appPaths.indexOf("pub fn resolve_default_project_dir()", start);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);

    const resolver = appPaths.slice(start, end);
    expect(resolver).toContain(
      "Ok(user_home_dir()?.join(USER_MEMORY_FILE_NAME))",
    );
    expect(resolver).not.toContain("legacy");
    expect(resolver).not.toContain("fs::copy");
    expect(appPaths).not.toContain("LEGACY_USER_MEMORY_FILE_NAMES");
    expect(appPaths).not.toContain("best_effort_user_memory_path");
    expect(appPaths).not.toContain("AGENTS.local.md");
    expect(appPaths).not.toContain(
      "resolve_workspace_local_runtime_agents_path",
    );
    expect(appPaths).not.toContain(
      "WORKSPACE_LOCAL_RUNTIME_AGENTS_GITIGNORE_ENTRY",
    );
  });

  it("AGENTS.local.md 配置与旧模板命令保持退役", () => {
    for (const currentSurface of [
      "lime-rs/crates/core/src/config/types.rs",
      "src/lib/api/memoryConfigTypes.ts",
      "src/components/settings-v2/general/memory/index.tsx",
    ]) {
      expect(read(currentSurface), currentSurface).not.toContain(
        "project_local_memory_path",
      );
      expect(read(currentSurface), currentSurface).not.toContain(
        "AGENTS.local.md",
      );
    }

    const retiredCommands = read("scripts/check-command-contracts.mjs");
    expect(retiredCommands).toContain(
      '"memory_ensure_workspace_local_agents_gitignore"',
    );
  });
});
