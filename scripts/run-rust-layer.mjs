#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import process from "node:process";

import {
  RUST_TEST_LAYER_NAMES,
  classifyRustTestFiles,
} from "./rust-test-layer-classifier.mjs";
import {
  DEFAULT_CHANGED_REF,
  expandWithWorkspaceDependents,
  resolvePathScopedCargoArgs,
  resolveRustPathSelection,
} from "./lib/rust-test-scope-core.mjs";
import { resolveRustyV8CargoEnv } from "./lib/rusty-v8-artifacts.mjs";

export { expandWithWorkspaceDependents, resolveRustPathSelection };

const MANIFEST_PATH = "lime-rs/Cargo.toml";
const DEFAULT_DARWIN_RUST_MIN_STACK = "8388608";
const CARGO_OPTIONS_WITH_VALUE = new Set([
  "-j",
  "-p",
  "-Z",
  "--bench",
  "--bin",
  "--color",
  "--config",
  "--example",
  "--exclude",
  "--features",
  "--jobs",
  "--manifest-path",
  "--message-format",
  "--package",
  "--profile",
  "--target",
  "--target-dir",
  "--test",
]);

function liveProviderSmokeAllowed() {
  return (
    process.env.LIME_REAL_API_TEST === "1" ||
    process.env.PROXYCAST_REAL_API_TEST === "1"
  );
}

export function resolveRustTestEnv(
  env = process.env,
  platform = process.platform,
) {
  const resolved = { ...env };
  if (platform === "darwin" && !resolved.RUST_MIN_STACK?.trim()) {
    resolved.RUST_MIN_STACK = DEFAULT_DARWIN_RUST_MIN_STACK;
  }
  return resolved;
}

function looksLikeGitRef(value) {
  return (
    value === "HEAD" ||
    value === "@" ||
    value.startsWith("refs/") ||
    value.includes("/") ||
    value.includes("..") ||
    /^[0-9a-f]{7,40}$/i.test(value)
  );
}

function printUsage() {
  console.log(`Usage:
  node scripts/run-rust-layer.mjs <layer> [options] [cargo filter/package args...] [-- test args...]

Layers:
  ${RUST_TEST_LAYER_NAMES.join(", ")}

Options:
  --list       只列出当前层 Rust 测试文件，不执行
  --json       以 JSON 输出列表
  --explain    列表输出包含分类原因
  --changed[=<ref>]
               按 Git diff 推导受影响 workspace crate，默认 ref 为 HEAD
  --related <paths...>
               按给定 Rust 路径推导受影响 workspace crate
  --help       显示帮助

说明:
  --list 会遵循相同的 Cargo package scope；默认只列 lime，传 --workspace 列全 workspace，传 -p <crate> 列目标 crate。
  --changed / --related 会先映射所属 crate，再用 cargo metadata 扩展反向依赖；Cargo.toml / Cargo.lock 等 workspace 边界会扩大到 --workspace。
  若 Rust 路径无法映射到当前 workspace crate，命令会失败，避免静默通过 0 个测试。
  macOS Cargo test worker 默认使用 8 MiB 栈；显式 RUST_MIN_STACK 会覆盖该默认值。

Examples:
  npm run test:rust:unit
  npm run test:rust:changed
  npm run test:rust:related -- lime-rs/crates/agent/src/request_tool_policy.rs
  npm run test:rust:unit -- -p lime-agent request_tool_policy
  npm run test:rust:unit -- --changed=origin/main request_tool_policy
  npm run test:rust:unit -- --workspace
  npm run test:rust:integration -- -- --nocapture
  LIME_REAL_API_TEST=1 npm run test:rust:e2e -- -- --ignored --nocapture
`);
}

export function parseArgs(argv) {
  const [layer, ...rest] = argv;
  const separatorIndex = rest.indexOf("--");
  const runnerArgs =
    separatorIndex >= 0 ? rest.slice(0, separatorIndex) : rest.slice();
  const testArgs = separatorIndex >= 0 ? rest.slice(separatorIndex + 1) : [];

  const options = {
    changed: false,
    changedRef: DEFAULT_CHANGED_REF,
    explain: false,
    json: false,
    list: false,
    related: false,
    relatedPaths: [],
  };
  const cargoArgs = [];

  for (let index = 0; index < runnerArgs.length; index += 1) {
    const arg = runnerArgs[index];
    if (arg === "--explain") {
      options.explain = true;
    } else if (arg === "--json") {
      options.json = true;
    } else if (arg === "--list") {
      options.list = true;
    } else if (arg === "--changed") {
      options.changed = true;
      const next = runnerArgs[index + 1];
      if (next && !next.startsWith("-") && looksLikeGitRef(next)) {
        options.changedRef = next;
        index += 1;
      }
    } else if (arg.startsWith("--changed=")) {
      options.changed = true;
      options.changedRef =
        arg.slice("--changed=".length) || DEFAULT_CHANGED_REF;
    } else if (arg === "--related") {
      options.related = true;
      while (runnerArgs[index + 1] && !runnerArgs[index + 1].startsWith("-")) {
        options.relatedPaths.push(runnerArgs[index + 1]);
        index += 1;
      }
    } else if (arg.startsWith("--related=")) {
      options.related = true;
      const value = arg.slice("--related=".length);
      if (value) {
        options.relatedPaths.push(
          ...value
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
        );
      }
    } else if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else {
      cargoArgs.push(arg);
    }
  }

  return { layer, options, cargoArgs, testArgs };
}

function layerCargoArgs(layer) {
  if (layer === "unit") {
    return ["test", "--manifest-path", MANIFEST_PATH, "--lib"];
  }
  if (layer === "integration") {
    return ["test", "--manifest-path", MANIFEST_PATH, "--tests"];
  }
  if (layer === "e2e") {
    return ["test", "--manifest-path", MANIFEST_PATH];
  }
  throw new Error(`Unknown Rust test layer: ${layer}`);
}

function shouldUseDefaultIgnoredTestArg(layer, testArgs) {
  return layer === "e2e" && !testArgs.some((arg) => arg.includes("ignored"));
}

export function filterEntriesForCargoArgs(entries, cargoArgs) {
  const packages = new Set();
  let allWorkspace = false;

  for (let index = 0; index < cargoArgs.length; index += 1) {
    const arg = cargoArgs[index];
    if (arg === "--workspace" || arg === "--all") {
      allWorkspace = true;
    } else if (arg === "-p" || arg === "--package") {
      if (cargoArgs[index + 1]) {
        packages.add(cargoArgs[index + 1]);
        index += 1;
      }
    } else if (arg.startsWith("--package=")) {
      packages.add(arg.slice("--package=".length));
    }
  }

  const workspaceEntries = entries.filter(
    (entry) => entry.cargoScope === "workspace",
  );

  if (allWorkspace) {
    return workspaceEntries;
  }
  if (packages.size > 0) {
    return workspaceEntries.filter((entry) => packages.has(entry.packageName));
  }

  const rootPackageEntries = workspaceEntries.filter(
    (entry) => entry.packageRoot === "lime-rs",
  );
  return rootPackageEntries.length > 0 ? rootPackageEntries : workspaceEntries;
}

export function findCargoTestFilters(cargoArgs) {
  const filters = [];
  let skipNext = false;

  for (const arg of cargoArgs) {
    if (skipNext) {
      skipNext = false;
      continue;
    }

    if (CARGO_OPTIONS_WITH_VALUE.has(arg)) {
      skipNext = true;
      continue;
    }

    if (
      [...CARGO_OPTIONS_WITH_VALUE].some((option) =>
        arg.startsWith(`${option}=`),
      )
    ) {
      continue;
    }

    if (arg.startsWith("-")) {
      continue;
    }

    filters.push(arg);
  }

  return filters;
}

export function countExecutedTestsFromCargoOutput(output) {
  const summaryPattern =
    /test result: \w+\. (?<passed>\d+) passed; (?<failed>\d+) failed; (?<ignored>\d+) ignored; (?<measured>\d+) measured; (?<filtered>\d+) filtered out;/g;
  let executed = 0;

  for (const match of output.matchAll(summaryPattern)) {
    executed +=
      Number(match.groups.passed) +
      Number(match.groups.failed) +
      Number(match.groups.measured);
  }

  return executed;
}

export function shouldFailOnZeroExecutedTests(cargoArgs, testArgs) {
  return (
    findCargoTestFilters(cargoArgs).length > 0 &&
    !testArgs.some((arg) => arg === "--list")
  );
}

function listLayerEntries(layer, options, cargoArgs) {
  const scopedEntries = filterEntriesForCargoArgs(
    classifyRustTestFiles(process.cwd()).filter(
      (entry) => entry.cargoScope === "workspace",
    ),
    cargoArgs,
  );
  const entries = scopedEntries.filter((entry) => {
    if (layer === "e2e") {
      return entry.layer === "e2e" || entry.liveGated;
    }
    return entry.layer === layer;
  });

  if (options.json) {
    console.log(JSON.stringify(entries, null, 2));
    return;
  }

  for (const entry of entries) {
    const suffix = options.explain ? ` # ${entry.reasons.join(", ")}` : "";
    console.log(`${entry.file}${suffix}`);
  }
}

function runCargo(layer, cargoArgs, testArgs) {
  const finalTestArgs = [...testArgs];
  if (shouldUseDefaultIgnoredTestArg(layer, finalTestArgs)) {
    finalTestArgs.push("--ignored");
  }

  const args = [...layerCargoArgs(layer), ...cargoArgs];
  if (finalTestArgs.length > 0) {
    args.push("--", ...finalTestArgs);
  }

  console.log(`[rust-layer] running cargo ${args.join(" ")}`);
  const failOnZeroExecutedTests = shouldFailOnZeroExecutedTests(
    cargoArgs,
    finalTestArgs,
  );
  const baseEnv = resolveRustTestEnv();
  const result = spawnSync("cargo", args, {
    encoding: failOnZeroExecutedTests ? "utf8" : undefined,
    env: {
      ...baseEnv,
      ...resolveRustyV8CargoEnv({ env: baseEnv }),
    },
    maxBuffer: 50 * 1024 * 1024,
    stdio: failOnZeroExecutedTests ? "pipe" : "inherit",
  });

  if (result.error) {
    throw result.error;
  }

  if (failOnZeroExecutedTests) {
    const stdout = result.stdout ?? "";
    const stderr = result.stderr ?? "";
    process.stderr.write(stderr);
    process.stdout.write(stdout);

    if (
      result.status === 0 &&
      countExecutedTestsFromCargoOutput(`${stdout}\n${stderr}`) === 0
    ) {
      console.error(
        `[rust-layer] cargo test filter matched no executed tests: ${findCargoTestFilters(cargoArgs).join(" ")}`,
      );
      return 1;
    }
  }

  return typeof result.status === "number" ? result.status : 1;
}

function main() {
  const { layer, options, cargoArgs, testArgs } = parseArgs(
    process.argv.slice(2),
  );

  if (options.help || !layer) {
    printUsage();
    process.exit(options.help ? 0 : 1);
  }

  if (!RUST_TEST_LAYER_NAMES.includes(layer)) {
    console.error(
      `[rust-layer] unknown layer "${layer}". Expected one of: ${RUST_TEST_LAYER_NAMES.join(", ")}`,
    );
    process.exit(1);
  }

  let scopedCargoArgs = cargoArgs;
  try {
    const scoped = resolvePathScopedCargoArgs(options, cargoArgs);
    if (scoped.skipped) {
      return;
    }
    scopedCargoArgs = scoped.cargoArgs;
  } catch (error) {
    console.error(`[rust-layer] ${error.message}`);
    process.exit(1);
  }

  if (options.list) {
    listLayerEntries(layer, options, scopedCargoArgs);
    return;
  }

  if (layer === "e2e" && !liveProviderSmokeAllowed()) {
    console.log(
      "[rust-layer] e2e/live Rust tests are gated. Set LIME_REAL_API_TEST=1 to run ignored live tests.",
    );
    return;
  }

  process.exit(runCargo(layer, scopedCargoArgs, testArgs));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
