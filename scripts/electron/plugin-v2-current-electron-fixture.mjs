#!/usr/bin/env node

import process from "node:process";
import { run } from "./mcp-elicitation-gate-b.mjs";

run({ pluginV2: true }).catch((error) => {
  console.error(
    `[smoke:plugin-v2-current-electron-fixture] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
