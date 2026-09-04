#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { readCargoVersions } from "./app-version.mjs";

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

const repoRoot = path.resolve(process.cwd());
const cargoTomlPath = path.join(repoRoot, "lime-rs", "Cargo.toml");
const packageJsonPath = path.join(repoRoot, "package.json");
const cliPackageJsonPath = path.join(repoRoot, "packages", "cli", "package.json");

const cargo = readCargoVersions(cargoTomlPath);
const packageJson = readJson(packageJsonPath);
const cliPackageJson = readJson(cliPackageJsonPath);

const sourceVersion = cargo.workspaceVersion;
const issues = [];

if (!sourceVersion) {
  issues.push("lime-rs/Cargo.toml [workspace.package].version 缺失");
}

if (
  cargo.packageSectionExists &&
  !cargo.packageVersionIsWorkspace &&
  cargo.packageVersion !== sourceVersion
) {
  issues.push(
    `lime-rs/Cargo.toml [package].version (${cargo.packageVersion ?? "missing"}) 与 workspace.version (${sourceVersion ?? "missing"}) 不一致`,
  );
}

if ((packageJson.version ?? null) !== sourceVersion) {
  issues.push(
    `package.json version (${packageJson.version ?? "missing"}) 与 workspace.version (${sourceVersion ?? "missing"}) 不一致`,
  );
}

if ((cliPackageJson.version ?? null) !== sourceVersion) {
  issues.push(
    `packages/cli/package.json version (${cliPackageJson.version ?? "missing"}) 与 workspace.version (${sourceVersion ?? "missing"}) 不一致`,
  );
}

if (issues.length > 0) {
  console.error("[lime] 应用版本一致性检查失败:");
  for (const issue of issues) {
    console.error(`- ${issue}`);
  }
  process.exit(1);
}

console.log(`[lime] 版本一致性检查通过: ${sourceVersion}`);
