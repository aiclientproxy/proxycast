import { LIVE_TAIL_COMMIT_PROMPT } from "./claw-chat-current-fixture-constants.mjs";

function resizeSnapshots(summary) {
  const snapshots = summary.electronResizeReflowLayout?.snapshots ?? {};
  return [snapshots.wide, snapshots.compact, snapshots.restored].filter(
    Boolean,
  );
}

export function buildElectronResizeReflowScenarioAssertions({
  electronResizeReflowTurnStart,
  summary,
}) {
  const snapshots = resizeSnapshots(summary);
  const layout = summary.electronResizeReflowLayout ?? {};
  const stableActiveSurfaces = new Set(
    snapshots.map((snapshot) => snapshot?.rightSurface?.activeSurface ?? null),
  );

  return {
    electronResizeReflowPromptReachedBackend:
      electronResizeReflowTurnStart?.inputText === LIVE_TAIL_COMMIT_PROMPT,
    guiElectronResizeReflowInputSubmitted:
      summary.electronResizeReflowInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.electronResizeReflowInputSend?.clicked?.clicked === true,
    guiElectronResizeReflowCompleted:
      summary.guiElectronResizeReflowCompleted?.hasPrompt === true &&
      (summary.guiElectronResizeReflowCompleted?.hasAssistantSummary === true ||
        summary.guiElectronResizeReflowCompleted?.hasDoneText === true) &&
      summary.guiElectronResizeReflowCompleted?.textareaVisible === true &&
      summary.guiElectronResizeReflowCompleted?.textareaDisabled === false &&
      summary.guiElectronResizeReflowCompleted?.stopButtonVisible === false,
    guiElectronResizeReflowFilesSurfaceOpened:
      Boolean(
        summary.electronResizeReflowFilesSurfaceRequest?.result?.requestId,
      ) &&
      summary.electronResizeReflowFilesSurface?.stable?.activeSurface ===
        "files" &&
      summary.electronResizeReflowFilesSurface?.stable?.rootVisible === true,
    guiElectronResizeReflowViewportSnapshotsCaptured:
      layout.stableViewportCount === 3 &&
      layout.screenshotCount === 3 &&
      layout.viewports?.wide?.width === 1280 &&
      layout.viewports?.wide?.height === 820 &&
      layout.viewports?.compact?.width === 880 &&
      layout.viewports?.compact?.height === 720 &&
      layout.viewports?.restored?.width === 1280 &&
      layout.viewports?.restored?.height === 820 &&
      snapshots.map((snapshot) => snapshot?.label).join(",") ===
        "wide,compact,restored",
    guiElectronResizeReflowMessageAnchorStable:
      snapshots.length === 3 &&
      snapshots.every(
        (snapshot) =>
          snapshot.hasPrompt === true &&
          snapshot.hasFirstText === true &&
          snapshot.hasOverflowMarker === true &&
          snapshot.hasTableHeader === true &&
          snapshot.hasTableTail === true &&
          snapshot.hasDoneText === true &&
          snapshot.renderedTableCount === 1 &&
          snapshot.tableHeaderOccurrenceCount === 1 &&
          snapshot.tableTailOccurrenceCount === 1 &&
          snapshot.messageAnchorStable === true,
      ),
    guiElectronResizeReflowTableOverflowContained:
      snapshots.length === 3 &&
      snapshots.every(
        (snapshot) =>
          snapshot.table?.rendered === true &&
          snapshot.table?.hostContained === true &&
          snapshot.table?.overflowHandled === true &&
          snapshot.table?.noDocumentHorizontalOverflow === true,
      ),
    guiElectronResizeReflowInputbarAnchored:
      snapshots.length === 3 &&
      snapshots.every(
        (snapshot) =>
          snapshot.inputbarAnchored === true &&
          snapshot.textareaDisabled === false &&
          snapshot.stopButtonVisible === false,
      ),
    guiElectronResizeReflowActiveThreadHeaderStable:
      snapshots.length === 3 &&
      snapshots.every(
        (snapshot) =>
          snapshot.activeThreadHeaderStable === true &&
          snapshot.activeThreadHeader?.stable === true &&
          snapshot.activeThreadHeader?.header?.visible === true &&
          snapshot.activeThreadHeader?.context?.visible === true &&
          snapshot.activeThreadHeader?.title?.visible === true &&
          snapshot.activeThreadHeader?.actions?.visible === true &&
          snapshot.activeThreadHeader?.toolbar?.visible === true &&
          snapshot.activeThreadHeader?.taskCenterChromeShell?.visible ===
            false &&
          snapshot.activeThreadHeader?.taskCenterTabStrip?.visible === false &&
          snapshot.activeThreadHeader?.taskCenterWorkspaceBar?.visible ===
            false &&
          snapshot.activeThreadHeader?.noContextActionsOverlap === true &&
          snapshot.activeThreadHeader?.childrenContained === true &&
          snapshot.activeThreadHeader?.noCompactModeBarOverlap === true &&
          snapshot.activeThreadHeader?.compactModeBar?.expectedVisible ===
            (snapshot.label === "compact") &&
          snapshot.activeThreadHeader?.compactModeBar?.visible ===
            (snapshot.label === "compact") &&
          snapshot.activeThreadHeader?.compactModeBar?.positionStable === true,
      ),
    guiElectronResizeReflowRightSurfaceStable:
      snapshots.length === 3 &&
      stableActiveSurfaces.size === 1 &&
      stableActiveSurfaces.has("files") &&
      snapshots.every((snapshot) => snapshot.rightSurfaceStable === true) &&
      layout.snapshots?.wide?.rightSurface?.host?.visible === true &&
      layout.snapshots?.compact?.rightSurface?.host?.visible === false &&
      layout.snapshots?.restored?.rightSurface?.host?.visible === true,
    guiElectronResizeReflowNoOverlap:
      snapshots.length === 3 &&
      snapshots.every(
        (snapshot) =>
          snapshot.noOverlap === true &&
          snapshot.noTailInputOverlap === true &&
          snapshot.noTableTailInputOverlap === true &&
          snapshot.noDoneTextInputOverlap === true &&
          snapshot.noTurnGroupInputOverlap === true &&
          snapshot.noMessageRightOverlap === true &&
          snapshot.noInputRightOverlap === true &&
          snapshot.noTableRightOverlap === true,
      ),
    readModelElectronResizeReflowCompleted:
      summary.readModelElectronResizeReflowCompleted?.includesPrompt === true &&
      summary.readModelElectronResizeReflowCompleted?.latestTurnStatus ===
        "completed" &&
      summary.readModelElectronResizeReflowCompleted?.includesFirstText ===
        true &&
      summary.readModelElectronResizeReflowCompleted?.includesOverflowMarker ===
        true &&
      summary.readModelElectronResizeReflowCompleted?.includesTableHeader ===
        true &&
      summary.readModelElectronResizeReflowCompleted?.includesTableTail ===
        true &&
      summary.readModelElectronResizeReflowCompleted?.includesAssistantDone ===
        true,
    backendElectronResizeReflowRecorded:
      summary.electronResizeReflowBackendCompleted?.droppedEventType ===
        "message.delta" &&
      summary.electronResizeReflowBackendCompleted?.repairEventType ===
        "item.completed" &&
      summary.electronResizeReflowBackendCompleted?.terminalEventType ===
        "turn.completed" &&
      summary.electronResizeReflowBackendCompleted?.turnId ===
        electronResizeReflowTurnStart?.turnId,
  };
}
