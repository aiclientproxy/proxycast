import type { ReactNode } from "react";
import type {
  WorkspaceRightSurfaceLauncherProjection,
  WorkspaceRightSurfaceState,
} from "./right-surface";
import type { WorkspaceRightSurfaceCoordinatorRuntime } from "./useWorkspaceRightSurfaceCoordinatorRuntime";

export type WorkspaceConversationRightSurfaceAttentionLevel =
  | "idle"
  | "active"
  | "warning";

export interface WorkspaceConversationRightSurfaceChromeRuntime {
  content?: ReactNode;
  launchers?: readonly WorkspaceRightSurfaceLauncherProjection[];
  browserOpen?: boolean;
  onToggleBrowser?: () => void;
  filesOpen?: boolean;
  onToggleFiles?: () => void;
  traceOpen?: boolean;
  onToggleTrace?: () => void;
  shellOpen?: boolean;
  onToggleShell?: () => void;
  showHarnessToggle: boolean;
  harnessPanelVisible: boolean;
  onToggleHarnessPanel?: () => void;
  showExpertInfoToggle?: boolean;
  expertInfoPanelVisible?: boolean;
  onToggleExpertInfoPanel?: () => void;
  harnessPendingCount: number;
  harnessAttentionLevel: WorkspaceConversationRightSurfaceAttentionLevel;
  harnessToggleLabel?: string;
}

export interface WorkspaceConversationRightSurfaceSceneProps {
  rightSurfaceContent?: ReactNode;
  rightSurfaceLaunchers?: readonly WorkspaceRightSurfaceLauncherProjection[];
  rightSurfaceBrowserOpen?: boolean;
  onToggleRightSurfaceBrowser?: () => void;
  rightSurfaceFilesOpen?: boolean;
  onToggleRightSurfaceFiles?: () => void;
  rightSurfaceTraceOpen?: boolean;
  onToggleRightSurfaceTrace?: () => void;
  rightSurfaceShellOpen?: boolean;
  onToggleRightSurfaceShell?: () => void;
  showHarnessToggle: boolean;
  harnessPanelVisible: boolean;
  onToggleHarnessPanel?: () => void;
  showExpertInfoToggle?: boolean;
  expertInfoPanelVisible?: boolean;
  onToggleExpertInfoPanel?: () => void;
  harnessPendingCount: number;
  harnessAttentionLevel: WorkspaceConversationRightSurfaceAttentionLevel;
  harnessToggleLabel?: string;
}

type WorkspaceConversationRightSurfaceChromeRuntimeInput = Pick<
  WorkspaceRightSurfaceCoordinatorRuntime,
  | "rightSurfaceLaunchers"
  | "rightSurfaceState"
  | "handleToggleRightSurfaceBrowser"
  | "handleToggleRightSurfaceFiles"
  | "handleToggleRightSurfaceTrace"
  | "handleToggleRightSurfaceShell"
  | "handleToggleRightSurfaceHarness"
  | "handleToggleExpertInfoPanel"
>;

export interface BuildWorkspaceConversationRightSurfaceChromeInput {
  content?: ReactNode;
  rightSurfaceRuntime: WorkspaceConversationRightSurfaceChromeRuntimeInput;
  showHarnessToggle: boolean;
  hasExpertInfoPanel: boolean;
  expertInfoPanelVisible?: boolean;
  harnessPendingCount: number;
  harnessAttentionLevel: WorkspaceConversationRightSurfaceAttentionLevel;
  harnessToggleLabel?: string;
  suppressHarnessChrome?: boolean;
}

function isWorkspaceRightSurfaceActive(
  state: Pick<WorkspaceRightSurfaceState, "activeSurface">,
  surface: WorkspaceRightSurfaceState["activeSurface"],
): boolean {
  return state.activeSurface === surface;
}

export function buildWorkspaceConversationRightSurfaceChrome({
  content,
  rightSurfaceRuntime,
  showHarnessToggle,
  hasExpertInfoPanel,
  expertInfoPanelVisible,
  harnessPendingCount,
  harnessAttentionLevel,
  harnessToggleLabel,
  suppressHarnessChrome = false,
}: BuildWorkspaceConversationRightSurfaceChromeInput): WorkspaceConversationRightSurfaceChromeRuntime {
  const harnessChromeVisible = !suppressHarnessChrome;

  return {
    content,
    launchers: rightSurfaceRuntime.rightSurfaceLaunchers,
    browserOpen: isWorkspaceRightSurfaceActive(
      rightSurfaceRuntime.rightSurfaceState,
      "browser",
    ),
    onToggleBrowser: rightSurfaceRuntime.handleToggleRightSurfaceBrowser,
    filesOpen: isWorkspaceRightSurfaceActive(
      rightSurfaceRuntime.rightSurfaceState,
      "files",
    ),
    onToggleFiles: rightSurfaceRuntime.handleToggleRightSurfaceFiles,
    traceOpen: isWorkspaceRightSurfaceActive(
      rightSurfaceRuntime.rightSurfaceState,
      "trace",
    ),
    onToggleTrace: rightSurfaceRuntime.handleToggleRightSurfaceTrace,
    shellOpen: isWorkspaceRightSurfaceActive(
      rightSurfaceRuntime.rightSurfaceState,
      "shell",
    ),
    onToggleShell: rightSurfaceRuntime.handleToggleRightSurfaceShell,
    showHarnessToggle: harnessChromeVisible && showHarnessToggle,
    harnessPanelVisible:
      harnessChromeVisible &&
      isWorkspaceRightSurfaceActive(
        rightSurfaceRuntime.rightSurfaceState,
        "harness",
      ),
    onToggleHarnessPanel: rightSurfaceRuntime.handleToggleRightSurfaceHarness,
    showExpertInfoToggle: hasExpertInfoPanel,
    expertInfoPanelVisible,
    onToggleExpertInfoPanel: rightSurfaceRuntime.handleToggleExpertInfoPanel,
    harnessPendingCount: harnessChromeVisible ? harnessPendingCount : 0,
    harnessAttentionLevel: harnessChromeVisible
      ? harnessAttentionLevel
      : "idle",
    harnessToggleLabel: harnessChromeVisible ? harnessToggleLabel : undefined,
  };
}

export function buildWorkspaceConversationRightSurfaceSceneProps({
  rightSurfaceChrome,
  utilityActionsVisible,
}: {
  rightSurfaceChrome: WorkspaceConversationRightSurfaceChromeRuntime;
  utilityActionsVisible: boolean;
}): WorkspaceConversationRightSurfaceSceneProps {
  const visible = utilityActionsVisible;
  const open = (value: boolean | undefined) => visible && Boolean(value);
  const action = <T>(value: T | undefined) => (visible ? value : undefined);

  return {
    rightSurfaceContent: rightSurfaceChrome.content,
    rightSurfaceLaunchers: rightSurfaceChrome.launchers,
    rightSurfaceBrowserOpen: open(rightSurfaceChrome.browserOpen),
    onToggleRightSurfaceBrowser: action(rightSurfaceChrome.onToggleBrowser),
    rightSurfaceFilesOpen: open(rightSurfaceChrome.filesOpen),
    onToggleRightSurfaceFiles: action(rightSurfaceChrome.onToggleFiles),
    rightSurfaceTraceOpen: open(rightSurfaceChrome.traceOpen),
    onToggleRightSurfaceTrace: action(rightSurfaceChrome.onToggleTrace),
    rightSurfaceShellOpen: open(rightSurfaceChrome.shellOpen),
    onToggleRightSurfaceShell: action(rightSurfaceChrome.onToggleShell),
    showHarnessToggle: visible && rightSurfaceChrome.showHarnessToggle,
    harnessPanelVisible: visible && rightSurfaceChrome.harnessPanelVisible,
    onToggleHarnessPanel: action(rightSurfaceChrome.onToggleHarnessPanel),
    showExpertInfoToggle: open(rightSurfaceChrome.showExpertInfoToggle),
    expertInfoPanelVisible: open(rightSurfaceChrome.expertInfoPanelVisible),
    onToggleExpertInfoPanel: action(rightSurfaceChrome.onToggleExpertInfoPanel),
    harnessPendingCount: visible ? rightSurfaceChrome.harnessPendingCount : 0,
    harnessAttentionLevel: visible
      ? rightSurfaceChrome.harnessAttentionLevel
      : "idle",
    harnessToggleLabel: action(rightSurfaceChrome.harnessToggleLabel),
  };
}
