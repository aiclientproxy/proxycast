/* global process */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();

// 旧 Tauri wrapper 整目录 `lime-rs/src/**` 已于 2026-06-10 物理删除。
// 历史背景：该目录是脱离 cargo 构建图的孤儿目录（workspace 只含 `crates/*`，
// `src/` 没有 `lib.rs` / `main.rs` 也没有 `[package]` 段），曾沉积约 18.7 万行
// 与 113 个 `#[tauri::command]` 标注。删除后留下这条守卫，确保任何形式的回流
// 都被阻断：新 Rust 后端能力进入 `lime-rs/crates/*`（App Server / RuntimeCore /
// services），桌面壳能力进入 Electron Desktop Host。
//
// 治理收口：
// - `tauri-wrapper-quick-cleanup-queue.md` / `tauri-wrapper-command-inventory.md`
//   / `rust-commands-current-migration-cleanup-plan.md` 标记 `superseded`。
// - `scripts/check-app-server-client-contract.mjs` / `scripts/check-command-contracts.mjs`
//   不再以 `lime-rs/src/**` 文件作为正向 invariant 来源，仅保留负向回流守卫。
// - `AGENTS.md` 第 12、13 条精简为：`lime-rs/src/**` 已删除，新 Rust 代码进 crates。

const FORBIDDEN_LEGACY_PATHS = [
  "src-tauri",
  "lime-rs/src",
  "lime-rs/src/commands",
  "lime-rs/src/services",
  "lime-rs/src/dev_bridge",
  "lime-rs/src/app",
  "lime-rs/src/agent",
  "lime-rs/src/agent_tools",
  "lime-rs/src/skills",
  "lime-rs/src/voice",
  "lime-rs/src/tray",
  "lime-rs/src/config",
  "lime-rs/src/theme",
  "lime-rs/src/tests",
  "lime-rs/src/dev_bridge.rs",
  "lime-rs/src/logger.rs",
  "lime-rs/src/profiling.rs",
  "lime-rs/src/crash_reporting.rs",
  "lime-rs/src/global_shortcut_guard.rs",
  "lime-rs/src/workspace_support.rs",
];

describe("rust commands current boundary", () => {
  it("已退役 Rust wrapper 目录及其子目录不应恢复", () => {
    const restored = FORBIDDEN_LEGACY_PATHS.filter((relativePath) =>
      existsSync(join(REPO_ROOT, relativePath)),
    );
    expect(restored, "禁止恢复 lime-rs/src/** 旧 Tauri wrapper 路径").toEqual([]);
  });
});
