import process from "node:process";

import {
  closeElectronFixture,
  launchElectronFixture,
  sleep,
} from "../electron/mcp-config-fixture-smoke.mjs";
import {
  collectProcessTreeSnapshot,
  waitForProcessIdsExit,
} from "../agent-runtime/tool-execution-soak-evidence.mjs";

function electronPid(handle) {
  if (typeof handle?.app?.process !== "function") return null;
  try {
    return handle.app.process()?.pid ?? null;
  } catch {
    return null;
  }
}

export async function closeControlledElectronFixture(handle) {
  const pid = electronPid(handle);
  if (!handle?.app) return;
  await Promise.race([closeElectronFixture(handle), sleep(5_000)]);
  if (!pid) return;
  try {
    process.kill(pid, 0);
    process.kill(pid, "SIGTERM");
    await sleep(500);
    try {
      process.kill(pid, 0);
      process.kill(pid, "SIGKILL");
    } catch {
      // The fixture exited after SIGTERM.
    }
  } catch {
    // The fixture exited during closeElectronFixture().
  }
}

export async function coldRestartControlledElectronFixture({
  appServerEnv,
  consoleErrors,
  electronHandle,
  options,
  pageErrors,
  runtimeEnv,
}) {
  const previousElectronPid = electronPid(electronHandle);
  if (!Number.isInteger(previousElectronPid)) {
    throw new Error("cold restart could not resolve the original Electron PID");
  }
  const previousProcessTree = collectProcessTreeSnapshot(
    previousElectronPid,
    "before-deepswe-cold-restart",
  );

  await closeControlledElectronFixture(electronHandle);
  const previousProcessTreeExit = await waitForProcessIdsExit(
    previousProcessTree.processes.map((entry) => entry.pid),
    { timeoutMs: Math.min(options.timeoutMs, 30_000) },
  );
  if (!previousProcessTreeExit.exited) {
    throw new Error(
      `cold restart left the previous Electron process tree alive: ${JSON.stringify(
        previousProcessTreeExit.remainingPids,
      )}`,
    );
  }

  const restartedHandle = await launchElectronFixture({
    options,
    runtimeEnv,
    appServerEnv,
    consoleErrors,
    pageErrors,
    backendMode: "runtime",
  });
  const restartedElectronPid = electronPid(restartedHandle);
  if (
    !Number.isInteger(restartedElectronPid) ||
    restartedElectronPid === previousElectronPid
  ) {
    await closeControlledElectronFixture(restartedHandle);
    throw new Error(
      `cold restart did not replace Electron PID: ${previousElectronPid} -> ${restartedElectronPid}`,
    );
  }
  const restartedProcessTree = collectProcessTreeSnapshot(
    restartedElectronPid,
    "after-deepswe-cold-restart",
  );
  const previousAppServerPids = new Set(previousProcessTree.appServerPids);
  const appServerProcessReplaced =
    previousAppServerPids.size > 0 &&
    restartedProcessTree.appServerPids.length > 0 &&
    restartedProcessTree.appServerPids.every(
      (pid) => !previousAppServerPids.has(pid),
    );
  if (!appServerProcessReplaced) {
    await closeControlledElectronFixture(restartedHandle);
    throw new Error(
      `cold restart did not replace App Server PID: ${JSON.stringify({
        before: previousProcessTree.appServerPids,
        after: restartedProcessTree.appServerPids,
      })}`,
    );
  }

  return {
    electronHandle: restartedHandle,
    lifecycle: {
      previousElectronPid,
      restartedElectronPid,
      electronProcessReplaced: true,
      appServerProcessReplaced,
      previousProcessTree,
      previousProcessTreeExit,
      restartedProcessTree,
    },
  };
}
