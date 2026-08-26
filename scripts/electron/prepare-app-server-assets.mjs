#!/usr/bin/env node

import { prepareElectronAppServerAssets } from "../lib/electron-app-server-assets.mjs";

const result = await prepareElectronAppServerAssets();

console.log(
  `[electron-assets] prepared app-server sidecar ${result.binaryPath}, code-mode host ${result.codeModeHostBinaryPath}, Windows sandbox runner ${result.windowsSandboxRunnerBinaryPath || "n/a"}, and ${result.manifestPath}`,
);
