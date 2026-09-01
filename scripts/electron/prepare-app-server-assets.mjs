#!/usr/bin/env node

import { prepareElectronAppServerAssets } from "../lib/electron-app-server-assets.mjs";
import { prepareDevelopmentDesktopResources } from "../lib/electron-desktop-resources.mjs";

const result = await prepareElectronAppServerAssets();
const desktopResources = prepareDevelopmentDesktopResources({
  outputRoot: "dist-electron",
  platform: process.platform,
  arch: process.arch,
  version: result.manifest.version,
});

console.log(
  `[electron-assets] prepared app-server sidecar ${result.binaryPath}, code-mode host ${result.codeModeHostBinaryPath}, Windows sandbox runner ${result.windowsSandboxRunnerBinaryPath || "n/a"}, resource manifest ${desktopResources.manifestPath}, and ${result.manifestPath}`,
);
