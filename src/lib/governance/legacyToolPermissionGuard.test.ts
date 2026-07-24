/* global process */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();
const AGENT_LIB_RS = join(REPO_ROOT, "lime-rs/crates/agent/src/lib.rs");
const RUST_SCAN_ROOTS = [join(REPO_ROOT, "lime-rs/crates")];
const FORBIDDEN_PERMISSION_PATHS = [
  "lime-rs/crates/agent/src/tool_permissions.rs",
  "lime-rs/crates/agent/src/shell_security.rs",
  "lime-rs/crates/agent/tests/legacy_permission_surfaces.rs",
];
const EXCLUDED_RUST_DIRS = new Set(["target", "agent-rust"]);

function collectRustFiles(dir: string): string[] {
  const entries = readdirSync(dir);
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = join(dir, entry);
    const stats = statSync(fullPath);

    if (stats.isDirectory()) {
      if (EXCLUDED_RUST_DIRS.has(entry)) {
        continue;
      }
      files.push(...collectRustFiles(fullPath));
      continue;
    }

    if (!fullPath.endsWith(".rs")) {
      continue;
    }

    files.push(fullPath);
  }

  return files;
}

describe("legacy tool permission guard", () => {
  it("lime-agent 不应继续把旧权限模块挂回 lib.rs 编译图", () => {
    const content = readFileSync(AGENT_LIB_RS, "utf8");
    expect(content).not.toContain("pub mod shell_security;");
    expect(content).not.toContain("pub mod tool_permissions;");
    expect(content).not.toContain("pub use shell_security::");
    expect(content).not.toContain("pub use tool_permissions::");
    expect(content).not.toContain("mod shell_security;");
    expect(content).not.toContain("mod tool_permissions;");
  });

  it("旧权限模块及其正向测试夹具不得恢复", () => {
    const restoredPaths = FORBIDDEN_PERMISSION_PATHS.filter((path) =>
      existsSync(join(REPO_ROOT, path)),
    );

    expect(restoredPaths).toEqual([]);
  });

  it("Rust 模块不应依赖已删除的旧权限表面", () => {
    const offenders: string[] = [];
    const patterns = [
      "lime_agent::shell_security::",
      "lime_agent::tool_permissions::",
      "lime_agent::ShellSecurityChecker",
      "lime_agent::DynamicPermissionCheck",
      "lime_agent::PermissionBehavior",
    ];

    for (const root of RUST_SCAN_ROOTS) {
      const files = collectRustFiles(root);

      for (const filePath of files) {
        const content = readFileSync(filePath, "utf8");
        if (patterns.some((pattern) => content.includes(pattern))) {
          offenders.push(relative(REPO_ROOT, filePath).replace(/\\/g, "/"));
        }
      }
    }

    expect(offenders).toEqual([]);
  });
});
