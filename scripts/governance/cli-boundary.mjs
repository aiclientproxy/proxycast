#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const RETIRED_PATHS = [
  "lime-rs/crates/lime-cli",
  "packages/lime-cli-npm",
  "tools/lime-cli",
];

const CURRENT_PATHS = [
  "lime-rs/crates/cli/Cargo.toml",
  "lime-rs/crates/cli/src/main.rs",
  "lime-rs/crates/cli/src/commands.rs",
  "lime-rs/crates/tui/Cargo.toml",
  "packages/cli/package.json",
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
  if (npmPackage.bin?.lime !== "scripts/run.js") {
    failures.push("CLI npm package must expose the lime binary");
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
  console.log("[cli-boundary] ok current=cli+tui retired=lime-cli");
}

const isMainModule =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
  main();
}
