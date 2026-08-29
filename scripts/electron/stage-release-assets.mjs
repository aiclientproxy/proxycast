#!/usr/bin/env node

import {
  copyFileSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
} from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const TARGETS = {
  "aarch64-apple-darwin": {
    archLabel: "aarch64",
    forgeArchLabel: "arm64",
    installerExtensions: [".dmg"],
    archiveExtensions: [".zip"],
    metadataNames: ["RELEASES.json"],
  },
  "x86_64-apple-darwin": {
    archLabel: "x64",
    forgeArchLabel: "x64",
    installerExtensions: [".dmg"],
    archiveExtensions: [".zip"],
    metadataNames: ["RELEASES.json"],
  },
  "x86_64-pc-windows-msvc": {
    archLabel: "x64",
    forgeArchLabel: "x64",
    installerExtensions: [".exe"],
    installerBasenameIncludes: ["setup"],
    archiveExtensions: [".nupkg"],
    metadataNames: ["RELEASES"],
  },
};

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith("--")) {
      continue;
    }
    const key = item.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      args[key] = "true";
      continue;
    }
    args[key] = next;
    index += 1;
  }
  return args;
}

function normalizeVersion(value) {
  const version = String(value || "")
    .trim()
    .replace(/^v/, "");
  if (!version) {
    throw new Error("version is required");
  }
  return version;
}

function walkFiles(root) {
  const result = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const filePath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(filePath);
      } else if (entry.isFile()) {
        result.push(filePath);
      }
    }
  }
  return result.sort();
}

function isAllowedMetadataAsset(filePath, spec) {
  return spec.metadataNames.includes(path.basename(filePath));
}

function scoreByNameAndPath(filePath, spec, extensions) {
  const basename = path.basename(filePath).toLowerCase();
  if (!extensions.some((extension) => basename.endsWith(extension))) {
    return 0;
  }
  if (
    extensions === spec.installerExtensions &&
    Array.isArray(spec.installerBasenameIncludes) &&
    spec.installerBasenameIncludes.length > 0 &&
    !spec.installerBasenameIncludes.some((needle) =>
      basename.includes(needle.toLowerCase()),
    )
  ) {
    return 0;
  }
  const normalizedPath = filePath.replace(/\\/g, "/").toLowerCase();
  let score = 1;
  if (basename.includes(spec.archLabel)) {
    score += 3;
  }
  if (basename.includes(spec.forgeArchLabel)) {
    score += 3;
  }
  if (normalizedPath.includes(`-${spec.forgeArchLabel}`)) {
    score += 2;
  }
  if (basename.includes("setup")) {
    score += 1;
  }
  if (basename.includes("universal")) {
    score -= 1;
  }
  return score;
}

function describeCandidateFiles(files, spec) {
  const extensions = [
    ...spec.installerExtensions,
    ...spec.archiveExtensions,
    ...spec.metadataNames,
    ".dmg",
    ".exe",
    ".nupkg",
    ".zip",
    "RELEASES",
    "RELEASES.json",
  ];
  const seen = new Set();
  return files
    .filter((filePath) => {
      const basename = path.basename(filePath);
      const normalized = basename.toLowerCase();
      return extensions.some((extension) => {
        if (extension.startsWith(".")) {
          return normalized.endsWith(extension);
        }
        return basename === extension;
      });
    })
    .map((filePath) => path.relative(process.cwd(), filePath))
    .filter((filePath) => {
      if (seen.has(filePath)) {
        return false;
      }
      seen.add(filePath);
      return true;
    })
    .sort();
}

function candidateSummary(files, spec) {
  const candidates = describeCandidateFiles(files, spec);
  return candidates.length > 0
    ? ` Candidate files: ${candidates.join(", ")}`
    : " No installer, archive, or updater metadata candidates were present.";
}

function selectAsset(
  files,
  spec,
  extensions,
  label,
  { exactBasenames = [] } = {},
) {
  if (exactBasenames.length > 0) {
    const exact = files.filter((filePath) =>
      exactBasenames.some(
        (expectedName) =>
          path.basename(filePath).toLowerCase() === expectedName.toLowerCase(),
      ),
    );
    if (exact.length === 1) {
      return exact[0];
    }
    if (exact.length > 1) {
      throw new Error(
        `expected exactly one ${exactBasenames.join(" or ")} under Forge output; found ${exact.length}.${candidateSummary(files, spec)}`,
      );
    }
    throw new Error(
      `no ${label} asset found under Forge output. Expected exactly one ${exactBasenames.join(" or ")}.${candidateSummary(files, spec)}`,
    );
  }

  const selected = files
    .map((filePath) => ({
      filePath,
      score: scoreByNameAndPath(filePath, spec, extensions),
    }))
    .filter((item) => item.score > 0)
    .sort(
      (left, right) =>
        right.score - left.score || left.filePath.localeCompare(right.filePath),
    )[0]?.filePath;
  if (!selected) {
    throw new Error(
      `no ${label} asset found under Forge output.${candidateSummary(files, spec)}`,
    );
  }
  return selected;
}

function assertNoRetiredUpdaterAssets(files) {
  const retiredAssets = files.filter((filePath) =>
    /(?:\.app\.tar\.gz|\.sig|latest(?:-mac)?\.yml|\.blockmap|latest\.json)$/i.test(
      path.basename(filePath),
    ),
  );
  if (retiredAssets.length > 0) {
    throw new Error(
      `legacy updater assets are not allowed in Electron release staging: ${retiredAssets.join(", ")}`,
    );
  }
}

function assertNoLocalMacUpdateManifest(filePath) {
  if (path.basename(filePath) !== "RELEASES.json") {
    return;
  }

  const manifest = JSON.parse(readFileSync(filePath, "utf8"));
  const releases = Array.isArray(manifest.releases) ? manifest.releases : [];
  const localUrls = releases
    .map((release) => release?.updateTo?.url)
    .filter((url) => typeof url === "string")
    .filter((url) =>
      /^https?:\/\/(?:127\.0\.0\.1|localhost)(?::|\/)/i.test(url),
    );
  if (localUrls.length > 0) {
    throw new Error(
      `local Electron updater feed URLs are not allowed in release staging: ${localUrls.join(", ")}`,
    );
  }
}

function stageElectronReleaseAssets({
  forgeDir,
  outDir,
  targetTriple,
  version,
}) {
  const spec = TARGETS[targetTriple];
  if (!spec) {
    throw new Error(`unsupported target triple: ${targetTriple}`);
  }
  const sourceDir = path.resolve(forgeDir || "release-electron");
  const stagingDir = path.resolve(
    outDir || path.join("release-assets", targetTriple),
  );
  const normalizedVersion = normalizeVersion(version);

  const sourceStat = statSync(sourceDir, { throwIfNoEntry: false });
  if (!sourceStat?.isDirectory()) {
    throw new Error(`Electron Forge output is missing: ${sourceDir}`);
  }

  const allFiles = walkFiles(sourceDir);
  assertNoRetiredUpdaterAssets(allFiles);

  const forgeMakeDir = path.join(sourceDir, "make");
  const forgeMakeStat = statSync(forgeMakeDir, { throwIfNoEntry: false });
  if (!forgeMakeStat?.isDirectory()) {
    throw new Error(`Electron Forge make output is missing: ${forgeMakeDir}`);
  }
  const makeFiles = walkFiles(forgeMakeDir);

  const installer = selectAsset(
    makeFiles,
    spec,
    spec.installerExtensions,
    `installer for ${targetTriple}`,
    targetTriple === "x86_64-pc-windows-msvc"
      ? {
          exactBasenames: [
            `Lime-${normalizedVersion} Setup.exe`,
            `Lime-${normalizedVersion}.Setup.exe`,
          ],
        }
      : undefined,
  );
  const archive =
    spec.archiveExtensions.length > 0
      ? selectAsset(
          makeFiles,
          spec,
          spec.archiveExtensions,
          `updater archive for ${targetTriple}`,
        )
      : null;

  const metadataAssets = makeFiles.filter((filePath) =>
    isAllowedMetadataAsset(filePath, spec),
  );
  if (metadataAssets.length === 0) {
    throw new Error(
      `no update metadata asset found under Forge output for ${targetTriple}: ${spec.metadataNames.join(", ")}`,
    );
  }
  for (const metadataAsset of metadataAssets) {
    assertNoLocalMacUpdateManifest(metadataAsset);
  }

  const assets = [installer, archive, ...metadataAssets].filter(Boolean).sort();

  rmSync(stagingDir, { recursive: true, force: true });
  mkdirSync(stagingDir, { recursive: true });

  const copied = [];
  for (const asset of assets) {
    const destination = path.join(stagingDir, path.basename(asset));
    copyFileSync(asset, destination);
    copied.push({ destination, source: asset });
  }

  return copied.sort((left, right) =>
    left.destination.localeCompare(right.destination),
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const copied = stageElectronReleaseAssets({
    forgeDir: args["forge-dir"],
    outDir: args["out-dir"],
    targetTriple: args["target-triple"],
    version:
      args.version || process.env.RELEASE_TAG || process.env.GITHUB_REF_NAME,
  });
  console.log("Staged Electron release assets:");
  for (const item of copied) {
    console.log(` - ${path.relative(process.cwd(), item.destination)}`);
  }
}

const isCli =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isCli) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

export { stageElectronReleaseAssets };
