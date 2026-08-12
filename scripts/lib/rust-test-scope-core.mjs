import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { withNativeSystemPath } from "./native-executable-env.mjs";

import { parseWorkspaceMemberRoots } from "../rust-test-layer-classifier.mjs";

export const DEFAULT_CHANGED_REF = "HEAD";
const MANIFEST_PATH = "lime-rs/Cargo.toml";

function toPosix(value) {
  return String(value || "").replaceAll(path.sep, "/");
}

function normalizeRepoPath(repoRoot, inputPath) {
  const rawPath = String(inputPath || "").trim();
  if (!rawPath) {
    return "";
  }
  const relativePath = path.isAbsolute(rawPath)
    ? path.relative(repoRoot, rawPath)
    : rawPath;
  return toPosix(path.normalize(relativePath)).replace(/^\.\//, "");
}

function isWorkspaceWideRustPath(relPath) {
  return (
    relPath === "rust-toolchain.toml" ||
    relPath === "rust-toolchain" ||
    relPath === "lime-rs" ||
    relPath === "lime-rs/" ||
    relPath === "lime-rs/Cargo.toml" ||
    relPath === "lime-rs/Cargo.lock" ||
    relPath === "lime-rs/rust-toolchain.toml" ||
    relPath === "lime-rs/rust-toolchain" ||
    relPath.startsWith("lime-rs/.cargo/")
  );
}

function isVendoredRustPath(relPath) {
  return relPath === "lime-rs/vendor" || relPath.startsWith("lime-rs/vendor/");
}

function findWorkspaceRootForPath(relPath, memberRoots) {
  const roots = Array.from(memberRoots.keys()).sort(
    (a, b) => b.length - a.length,
  );
  return roots.find(
    (root) => relPath === root || relPath.startsWith(`${root}/`),
  );
}

function isExcludedSubcrateMetadataPath(relPath, packageRoot) {
  return (
    relPath === `${packageRoot}/Cargo.toml` ||
    relPath === `${packageRoot}/Cargo.lock`
  );
}

function repoPathExists(repoRoot, relPath) {
  return fs.existsSync(path.join(repoRoot, relPath));
}

export function expandWithWorkspaceDependents(packageNames, dependencyGraph) {
  const selected = new Set(packageNames);
  const missingPackages = [...selected].filter(
    (packageName) => !dependencyGraph.has(packageName),
  );
  if (missingPackages.length > 0) {
    return {
      packages: [...selected].sort(),
      addedDependents: [],
      missingPackages,
    };
  }

  const addedDependents = new Set();
  let changed = true;
  while (changed) {
    changed = false;
    for (const [packageName, dependencies] of dependencyGraph) {
      if (selected.has(packageName)) {
        continue;
      }
      if ([...dependencies].some((dependency) => selected.has(dependency))) {
        selected.add(packageName);
        addedDependents.add(packageName);
        changed = true;
      }
    }
  }

  return {
    packages: [...selected].sort(),
    addedDependents: [...addedDependents].sort(),
    missingPackages: [],
  };
}

export function resolveRustPathSelection(
  inputPaths,
  { dependencyGraph = null, repoRoot = process.cwd() } = {},
) {
  const memberRoots = parseWorkspaceMemberRoots(repoRoot);
  const directPackages = new Set();
  const rustPaths = [];
  const skippedPaths = [];
  const errors = [];
  const workspaceReasons = [];

  for (const inputPath of inputPaths) {
    const relPath = normalizeRepoPath(repoRoot, inputPath);
    if (!relPath) {
      continue;
    }

    if (isWorkspaceWideRustPath(relPath)) {
      rustPaths.push(relPath);
      workspaceReasons.push(relPath);
      continue;
    }

    if (!relPath.startsWith("lime-rs")) {
      skippedPaths.push(relPath);
      continue;
    }

    if (isVendoredRustPath(relPath)) {
      rustPaths.push(relPath);
      workspaceReasons.push(`${relPath} (vendored Rust dependency)`);
      continue;
    }

    const packageRoot = findWorkspaceRootForPath(relPath, memberRoots);
    if (!packageRoot) {
      if (!repoPathExists(repoRoot, relPath)) {
        skippedPaths.push(relPath);
        continue;
      }
      rustPaths.push(relPath);
      errors.push(
        `${relPath}: 无法映射到 lime-rs workspace crate；请扩大到 --workspace 或手动指定 -p <crate>`,
      );
      continue;
    }

    const packageInfo = memberRoots.get(packageRoot);
    if (!packageInfo.workspaceMember) {
      if (isExcludedSubcrateMetadataPath(relPath, packageRoot)) {
        skippedPaths.push(relPath);
        continue;
      }
      rustPaths.push(relPath);
      errors.push(
        `${relPath}: ${packageInfo.packageName} 已被 lime-rs workspace exclude，不能通过根 manifest 的 -p 定向运行`,
      );
      continue;
    }

    rustPaths.push(relPath);
    directPackages.add(packageInfo.packageName);
  }

  if (errors.length > 0) {
    return {
      addedDependents: [],
      directPackages: [...directPackages].sort(),
      errors,
      packages: [],
      rustPaths,
      skippedPaths,
      workspaceReasons,
      workspaceWide: false,
    };
  }

  if (workspaceReasons.length > 0) {
    return {
      addedDependents: [],
      directPackages: [...directPackages].sort(),
      errors: [],
      packages: [],
      rustPaths,
      skippedPaths,
      workspaceReasons,
      workspaceWide: true,
    };
  }

  if (directPackages.size === 0) {
    return {
      addedDependents: [],
      directPackages: [],
      errors: [],
      packages: [],
      rustPaths,
      skippedPaths,
      workspaceReasons: [],
      workspaceWide: false,
    };
  }

  if (!dependencyGraph) {
    return {
      addedDependents: [],
      directPackages: [...directPackages].sort(),
      errors: [],
      packages: [...directPackages].sort(),
      rustPaths,
      skippedPaths,
      workspaceReasons: [],
      workspaceWide: false,
    };
  }

  const expansion = expandWithWorkspaceDependents(
    directPackages,
    dependencyGraph,
  );
  if (expansion.missingPackages.length > 0) {
    return {
      addedDependents: [],
      directPackages: [...directPackages].sort(),
      errors: [
        `cargo metadata 缺少 workspace package: ${expansion.missingPackages.join(", ")}`,
      ],
      packages: [],
      rustPaths,
      skippedPaths,
      workspaceReasons: [],
      workspaceWide: false,
    };
  }

  return {
    addedDependents: expansion.addedDependents,
    directPackages: [...directPackages].sort(),
    errors: [],
    packages: expansion.packages,
    rustPaths,
    skippedPaths,
    workspaceReasons: [],
    workspaceWide: false,
  };
}

function hasExplicitCargoPackageScope(cargoArgs) {
  return cargoArgs.some(
    (arg) =>
      arg === "--workspace" ||
      arg === "--all" ||
      arg === "-p" ||
      arg === "--package" ||
      arg.startsWith("--package="),
  );
}

function collectChangedPaths(ref, repoRoot) {
  const diffResult = spawnSync(
    "git",
    ["diff", "--name-only", "--diff-filter=ACMRTD", ref, "--"],
    {
      cwd: repoRoot,
      encoding: "utf8",
      env: withNativeSystemPath(process.env),
    },
  );
  if (diffResult.error) {
    throw diffResult.error;
  }
  if (diffResult.status !== 0) {
    throw new Error(
      (diffResult.stderr || "").trim() ||
        `git diff failed for changed ref ${ref}`,
    );
  }

  const untrackedResult = spawnSync(
    "git",
    ["ls-files", "--others", "--exclude-standard", "--", "lime-rs"],
    {
      cwd: repoRoot,
      encoding: "utf8",
      env: withNativeSystemPath(process.env),
    },
  );
  if (untrackedResult.error) {
    throw untrackedResult.error;
  }
  if (untrackedResult.status !== 0) {
    throw new Error(
      (untrackedResult.stderr || "").trim() ||
        "git ls-files failed while collecting untracked Rust paths",
    );
  }

  return [
    ...new Set(
      `${diffResult.stdout}\n${untrackedResult.stdout}`
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean),
    ),
  ];
}

function loadWorkspaceDependencyGraph(repoRoot) {
  const result = spawnSync(
    "cargo",
    ["metadata", "--manifest-path", MANIFEST_PATH, "--format-version", "1"],
    {
      cwd: repoRoot,
      encoding: "utf8",
      maxBuffer: 50 * 1024 * 1024,
    },
  );
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(
      (result.stderr || "").trim() ||
        "cargo metadata failed while resolving affected Rust crates",
    );
  }

  const metadata = JSON.parse(result.stdout);
  const workspaceMembers = new Set(metadata.workspace_members || []);
  const packageNamesById = new Map(
    metadata.packages.map((item) => [item.id, item.name]),
  );
  const graph = new Map(
    metadata.packages
      .filter((item) => workspaceMembers.has(item.id))
      .map((item) => [item.name, new Set()]),
  );

  for (const node of metadata.resolve?.nodes || []) {
    if (!workspaceMembers.has(node.id)) {
      continue;
    }
    const packageName = packageNamesById.get(node.id);
    const dependencies = graph.get(packageName);
    for (const dependency of node.deps || []) {
      if (workspaceMembers.has(dependency.pkg)) {
        dependencies.add(packageNamesById.get(dependency.pkg));
      }
    }
  }

  return graph;
}

export function resolvePathScopedCargoArgs(
  options,
  cargoArgs,
  { repoRoot = process.cwd() } = {},
) {
  if (!options.changed && !options.related) {
    return { cargoArgs, skipped: false };
  }
  if (options.changed && options.related) {
    throw new Error("--changed 与 --related 不能同时使用");
  }
  if (hasExplicitCargoPackageScope(cargoArgs)) {
    throw new Error(
      "--changed / --related 不能与 -p、--package、--workspace 或 --all 混用",
    );
  }

  const inputPaths = options.changed
    ? collectChangedPaths(options.changedRef, repoRoot)
    : options.relatedPaths;
  if (options.related && inputPaths.length === 0) {
    throw new Error("--related 需要至少传入一个路径");
  }

  let selection = resolveRustPathSelection(inputPaths, { repoRoot });
  if (
    selection.errors.length === 0 &&
    !selection.workspaceWide &&
    selection.directPackages.length > 0
  ) {
    selection = resolveRustPathSelection(inputPaths, {
      dependencyGraph: loadWorkspaceDependencyGraph(repoRoot),
      repoRoot,
    });
  }
  if (selection.errors.length > 0) {
    throw new Error(selection.errors.join("\n"));
  }
  if (selection.rustPaths.length === 0) {
    const mode = options.changed
      ? `--changed ${options.changedRef}`
      : "--related";
    console.log(`[rust-layer] ${mode} 未命中 lime-rs 路径，跳过 Rust 层测试。`);
    return { cargoArgs, skipped: true };
  }
  if (selection.workspaceWide) {
    console.log(
      `[rust-layer] ${selection.workspaceReasons.join(", ")} 触达 workspace 边界，扩大到 --workspace。`,
    );
    return { cargoArgs: ["--workspace", ...cargoArgs], skipped: false };
  }
  if (selection.packages.length === 0) {
    throw new Error(
      "Rust 路径未能映射到任何 workspace package；为避免空跑，本次失败。",
    );
  }

  const packageArgs = selection.packages.flatMap((packageName) => [
    "-p",
    packageName,
  ]);
  const dependentSuffix =
    selection.addedDependents.length > 0
      ? `；反向依赖扩展：${selection.addedDependents.join(", ")}`
      : "";
  console.log(
    `[rust-layer] scoped packages: ${selection.packages.join(", ")}${dependentSuffix}`,
  );

  return {
    cargoArgs: [...packageArgs, ...cargoArgs],
    skipped: false,
  };
}
