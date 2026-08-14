import {
  collectProcessTreeSnapshot,
  waitForProcessIdsExit,
} from "./tool-execution-soak-evidence.mjs";

export async function runManagedColdRestarts({
  app,
  appServerEnv,
  bridge,
  closeElectronApp,
  closeServer,
  consoleErrors,
  count,
  initialElectronPid,
  launchManagedElectron,
  logPrefix,
  readAgentControlDomState,
  readRestoredDomState = readAgentControlDomState,
  restoreAgentSessionRoute,
  runtimeEnv,
  sessionId,
  timeoutMs,
}) {
  const processSnapshots = [];
  const restartRecords = [];
  let currentApp = app;
  let currentPage = null;
  let previousElectronPid = initialElectronPid;
  let rendererSnapshot = null;

  for (let restartIndex = 0; restartIndex < count; restartIndex += 1) {
    console.log(
      `${logPrefix} stage=cold-restart-electron restart=${restartIndex + 1}/${count}`,
    );
    const previousProcessSnapshot = collectProcessTreeSnapshot(
      previousElectronPid,
      `pre-restart-${restartIndex + 1}`,
    );
    await closeServer(bridge?.server);
    bridge = null;
    await closeElectronApp(currentApp);
    currentApp = null;
    const previousProcessTreeExit = await waitForProcessIdsExit(
      previousProcessSnapshot.processes.map((entry) => entry.pid),
    );
    consoleErrors.length = 0;
    const restarted = await launchManagedElectron({
      appServerEnv,
      consoleErrors,
      runtimeEnv,
      timeoutMs,
    });
    currentApp = restarted.app;
    currentPage = restarted.page;
    const restartedElectronPid = currentApp.process().pid;
    rendererSnapshot = await restoreAgentSessionRoute(
      currentPage,
      sessionId,
      timeoutMs,
    );
    const restoredDom = await readRestoredDomState({
      page: currentPage,
      sessionId,
      timeoutMs,
    });
    const restartedProcessSnapshot = collectProcessTreeSnapshot(
      restartedElectronPid,
      `restart-${restartIndex + 1}`,
    );
    processSnapshots.push(restartedProcessSnapshot);
    restartRecords.push({
      restart: restartIndex + 1,
      previousElectronPid,
      restartedElectronPid,
      electronProcessReplaced:
        Number.isInteger(previousElectronPid) &&
        Number.isInteger(restartedElectronPid) &&
        previousElectronPid !== restartedElectronPid,
      previousProcessTreeExit,
      process: restartedProcessSnapshot,
      restoredDom: {
        activeSessionId: restoredDom.activeSessionId,
        typedToolRowCount: restoredDom.typedToolRows.length,
        subagentActivityRowCount: restoredDom.subagentActivityRows.length,
        finalAssistantTextVisible: restoredDom.finalAssistantTextVisible,
      },
      consoleErrorCount: consoleErrors.length,
    });
    previousElectronPid = restartedElectronPid;
  }

  return {
    app: currentApp,
    bridge,
    page: currentPage,
    processSnapshots,
    rendererSnapshot,
    restartRecords,
  };
}
