#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const RETIRED_PATHS = [
  "lime-rs/crates/lime-cli",
  "packages/lime-cli-npm",
  "tools/lime-cli",
  "packages/cli/scripts/install.js",
  "packages/cli/scripts/run.js",
  "packages/cli/scripts/release-meta.js",
  "packages/cli/scripts/build-release.js",
];

const CURRENT_PATHS = [
  "lime-rs/crates/cli/Cargo.toml",
  "lime-rs/crates/cli/src/main.rs",
  "lime-rs/crates/cli/src/mcp_cmd.rs",
  "lime-rs/crates/cli/src/plugin_cmd.rs",
  "lime-rs/crates/cli/src/queue_cmd.rs",
  "lime-rs/crates/tui/Cargo.toml",
  "packages/cli/package.json",
  "packages/cli/.gitignore",
  "packages/cli/bin/lime.js",
  "packages/cli/scripts/build_npm_package.py",
  "packages/cli/scripts/README.md",
  "packages/cli/tests/npm-package.test.mjs",
  "scripts/app-server/cli-npm-gate-b.mjs",
  "scripts/app-server/cli-npm-gate-b.test.mjs",
  "scripts/app-server/cli-surface-gate-b.mjs",
  "scripts/app-server/cli-surface-gate-b.test.mjs",
  "scripts/app-server/cli-structure-inventory.mjs",
  "scripts/app-server/cli-structure-inventory.test.mjs",
  "internal/exec-plans/cli-codex-test-inventory.json",
  "internal/exec-plans/cli-structure-inventory.json",
];

const TASK_SKILLS = [
  "lime-rs/resources/default-skills/broadcast_generate/SKILL.md",
  "lime-rs/resources/default-skills/modal_resource_search/SKILL.md",
  "lime-rs/resources/default-skills/transcription_generate/SKILL.md",
  "lime-rs/resources/default-skills/typesetting/SKILL.md",
  "lime-rs/resources/default-skills/url_parse/SKILL.md",
  "lime-rs/resources/default-skills/video_generate/SKILL.md",
];

const PRODUCTION_PROJECTION_FILES = [
  "src/components/agent/chat/utils/taskPreviewImage.ts",
  "src/components/agent/chat/utils/taskPreviewVideo.ts",
  "src/components/agent/chat/workspace/generalWorkbenchHelpers.ts",
];

const RETIRED_COMMAND_PATTERN = /\blime\s+(?:task|media|skill|doctor)\b/u;

function read(repoRoot, relativePath) {
  return fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
}

export function checkCliBoundary(repoRoot = process.cwd()) {
  const failures = [];

  for (const relativePath of RETIRED_PATHS) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(`retired CLI path must stay deleted: ${relativePath}`);
    }
  }

  for (const relativePath of CURRENT_PATHS) {
    if (!fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(`current CLI path is missing: ${relativePath}`);
    }
  }

  if (failures.length > 0) {
    return failures;
  }

  const cargoManifest = read(repoRoot, "lime-rs/crates/cli/Cargo.toml");
  if (!/^name = "cli"$/mu.test(cargoManifest)) {
    failures.push('CLI crate package must be named "cli"');
  }
  if (!/^name = "lime"$/mu.test(cargoManifest)) {
    failures.push('CLI binary must be named "lime"');
  }
  for (const dependency of ["lime-media-runtime", "lime-core"]) {
    if (cargoManifest.includes(dependency)) {
      failures.push(`CLI crate must not depend on ${dependency}`);
    }
  }

  const mainSource = read(repoRoot, "lime-rs/crates/cli/src/main.rs").split(
    "#[cfg(test)]",
    1,
  )[0];
  for (const retiredVariant of ["Task(", "Media(", "Skill(", "Doctor("]) {
    if (mainSource.includes(retiredVariant)) {
      failures.push(`CLI production command enum contains ${retiredVariant}`);
    }
  }

  const npmPackage = JSON.parse(read(repoRoot, "packages/cli/package.json"));
  if (npmPackage.name !== "@limecloud/lime") {
    failures.push('CLI npm package must be named "@limecloud/lime"');
  }
  if (npmPackage.bin?.lime !== "bin/lime.js") {
    failures.push("CLI npm package must expose the lime binary");
  }
  if (npmPackage.type !== "module") {
    failures.push("CLI npm launcher package must use ESM");
  }
  if (npmPackage.packageManager !== "pnpm@9.15.9") {
    failures.push("CLI npm package manager metadata must match the workspace");
  }
  if (npmPackage.scripts?.postinstall) {
    failures.push("CLI npm package must not download binaries in postinstall");
  }
  if (npmPackage.os || npmPackage.cpu) {
    failures.push(
      "CLI npm root package must delegate platform filters to aliases",
    );
  }
  if (
    !Array.isArray(npmPackage.files) ||
    npmPackage.files.length !== 1 ||
    npmPackage.files[0] !== "bin/lime.js"
  ) {
    failures.push("CLI npm root package must publish only the launcher");
  }

  const launcher = read(repoRoot, "packages/cli/bin/lime.js");
  for (const required of [
    "@limecloud/lime-linux-x64",
    "@limecloud/lime-darwin-x64",
    "@limecloud/lime-darwin-arm64",
    "@limecloud/lime-win32-x64",
    "spawn(binaryPath, process.argv.slice(2)",
    '["SIGINT", "SIGTERM", "SIGHUP"]',
    "process.removeListener(signal, handler)",
    "process.kill(process.pid, childResult.signal)",
    "LIME_MANAGED_PACKAGE_ROOT",
  ]) {
    if (!launcher.includes(required)) {
      failures.push(
        `CLI npm launcher is missing Codex-aligned contract: ${required}`,
      );
    }
  }
  for (const forbidden of [
    "spawnSync(",
    "execFileSync(",
    "cargo run",
    "LIME_CLI_BINARY_PATH",
    "releases/download",
  ]) {
    if (launcher.includes(forbidden)) {
      failures.push(
        `CLI npm launcher restores forbidden fallback: ${forbidden}`,
      );
    }
  }

  const npmBuilder = read(
    repoRoot,
    "packages/cli/scripts/build_npm_package.py",
  );
  for (const required of [
    'NPM_NAME = "@limecloud/lime"',
    '"lime-linux-x64"',
    '"lime-darwin-x64"',
    '"lime-darwin-arm64"',
    '"lime-win32-x64"',
    'f"npm:{NPM_NAME}@{compute_platform_package_version',
    'f"app-server{suffix}"',
    'f"code-mode-host{suffix}"',
    '"windows-sandbox-setup.exe"',
    '"windows-sandbox-runner.exe"',
  ]) {
    if (!npmBuilder.includes(required)) {
      failures.push(`CLI npm staging is missing runtime contract: ${required}`);
    }
  }

  const rustCli = read(repoRoot, "lime-rs/crates/cli/src/main.rs");
  for (const required of [
    'long = "remote"',
    'long = "remote-auth-token-env"',
    "ClientSession::start_remote",
    "RemoteTransportConfig::new",
    "with_optional_auth_token",
    "requires `--remote`",
  ]) {
    if (!rustCli.includes(required)) {
      failures.push(`CLI remote transport is missing contract: ${required}`);
    }
  }

  const releaseWorkflow = read(repoRoot, ".github/workflows/release.yml");
  for (const required of [
    "publish_cli_npm:",
    "Build CLI runtime payload",
    "-p app-server --bin app-server",
    "-p code-mode-host --bin code-mode-host",
    "Package CLI npm platform tarball",
    "Publish CLI npm packages platform-first",
    'publish_tarball "$ROOT_TARBALL"',
  ]) {
    if (!releaseWorkflow.includes(required)) {
      failures.push(`CLI release workflow is missing contract: ${required}`);
    }
  }
  const platformPublish = releaseWorkflow.indexOf(
    'publish_tarball "${PLATFORM_TARBALLS[0]}"',
  );
  const rootPublish = releaseWorkflow.indexOf(
    'publish_tarball "$ROOT_TARBALL"',
  );
  if (platformPublish < 0 || rootPublish < 0 || platformPublish > rootPublish) {
    failures.push(
      "CLI release workflow must publish every platform package before root",
    );
  }

  const rootPackage = JSON.parse(read(repoRoot, "package.json"));
  if (
    rootPackage.scripts?.["smoke:cli-npm-gate-b"] !==
    "node scripts/app-server/cli-npm-gate-b.mjs"
  ) {
    failures.push("CLI npm Gate B must remain a root quality entrypoint");
  }
  const npmGate = read(repoRoot, "scripts/app-server/cli-npm-gate-b.mjs");
  for (const required of [
    "LIME_CLI_BIN: launcherPath",
    'LIME_CLI_GATE_B_USE_SIBLING_APP_SERVER: "1"',
    "APP_SERVER_BIN: packagedAppServer",
  ]) {
    if (!npmGate.includes(required)) {
      failures.push(`CLI npm Gate B is missing packaged evidence: ${required}`);
    }
  }

  for (const relativePath of [...TASK_SKILLS, ...PRODUCTION_PROJECTION_FILES]) {
    const content = read(repoRoot, relativePath);
    if (RETIRED_COMMAND_PATTERN.test(content)) {
      failures.push(
        `retired CLI task command must stay absent: ${relativePath}`,
      );
    }
  }

  for (const relativePath of TASK_SKILLS) {
    const frontmatter = read(repoRoot, relativePath).split("---", 3)[1] ?? "";
    if (/^allowed-tools:.*\bBash\b/mu.test(frontmatter)) {
      failures.push(
        `task Skill must use typed tools instead of Bash: ${relativePath}`,
      );
    }
  }

  return failures;
}

function main() {
  const failures = checkCliBoundary();
  if (failures.length > 0) {
    console.error("[cli-boundary] failed");
    for (const failure of failures) {
      console.error(`- ${failure}`);
    }
    process.exitCode = 1;
    return;
  }
  console.log(
    "[cli-boundary] ok current=cli+tui+npm-platform-packages retired=lime-cli",
  );
}

const isMainModule =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
  main();
}
