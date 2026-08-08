#!/usr/bin/env node

import process from "node:process";
import { run } from "./mcp-elicitation-gate-b.mjs";

run({ pluginPackage: true }).catch((error) => {
  console.error(
    `[smoke:plugin-package-electron-gate-b] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
