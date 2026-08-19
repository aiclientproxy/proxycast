import { translateCanvasWorkbenchText } from "../CanvasWorkbenchLayoutState";
import type {
  CanvasWorkbenchOpenedToolTab,
  CanvasWorkbenchTab,
  CanvasWorkbenchTranslation,
} from "../CanvasWorkbenchLayoutState";
import type {
  CanvasWorkbenchNewTabAction,
  CanvasWorkbenchTopTab,
} from "./CanvasWorkbenchTopTabs";

export interface CanvasWorkbenchToolTabProjectionInput {
  changeItemCount: number;
  documentDiffLineCount: number;
  failedChangeItemCount: number;
  utilityTabs?: {
    outputs?: boolean;
    logs?: boolean;
  };
  openedToolTabs: CanvasWorkbenchOpenedToolTab[];
  translateWorkbench: CanvasWorkbenchTranslation;
}

export interface CanvasWorkbenchToolTabProjection {
  primaryTabs: CanvasWorkbenchTopTab[];
  newTabActions: CanvasWorkbenchNewTabAction[];
}

export function buildCanvasWorkbenchToolTabProjection({
  changeItemCount,
  documentDiffLineCount,
  failedChangeItemCount,
  utilityTabs,
  openedToolTabs,
  translateWorkbench,
}: CanvasWorkbenchToolTabProjectionInput): CanvasWorkbenchToolTabProjection {
  return {
    primaryTabs: [
      {
        key: "changes",
        label: translateCanvasWorkbenchText(
          translateWorkbench,
          "agentChat.canvasWorkbench.coding.tabs.changes",
        ),
        badge:
          changeItemCount > 0
            ? changeItemCount > 99
              ? "99+"
              : String(changeItemCount)
            : documentDiffLineCount > 0
              ? String(documentDiffLineCount)
              : undefined,
        badgeTone:
          failedChangeItemCount > 0
            ? "rose"
            : changeItemCount > 0 || documentDiffLineCount > 0
              ? "sky"
              : "slate",
      },
      ...buildCodingUtilityTabs(utilityTabs, translateWorkbench),
      ...openedToolTabs.map((tab) =>
        buildCanvasWorkbenchToolTab(tab, translateWorkbench),
      ),
    ],
    newTabActions: [
      {
        key: "terminal",
        label: translateCanvasWorkbenchText(
          translateWorkbench,
          "agentChat.canvasWorkbench.newTabs.terminal",
        ),
        shortcut: "^`",
      },
      {
        key: "project-files",
        label: translateCanvasWorkbenchText(
          translateWorkbench,
          "agentChat.canvasWorkbench.newTabs.files",
        ),
        shortcut: "⌘P",
      },
    ],
  };
}

function buildCodingUtilityTabs(
  utilityTabs: CanvasWorkbenchToolTabProjectionInput["utilityTabs"],
  translateWorkbench: CanvasWorkbenchTranslation,
): CanvasWorkbenchTopTab[] {
  const tabs: CanvasWorkbenchTopTab[] = [];
  if (utilityTabs?.outputs) {
    tabs.push({
      key: "outputs",
      label: translateCanvasWorkbenchText(
        translateWorkbench,
        "agentChat.canvasWorkbench.coding.tabs.outputs",
      ),
    });
  }
  if (utilityTabs?.logs) {
    tabs.push({
      key: "logs",
      label: translateCanvasWorkbenchText(
        translateWorkbench,
        "agentChat.canvasWorkbench.coding.tabs.logs",
      ),
    });
  }
  return tabs;
}

function buildCanvasWorkbenchToolTab(
  tab: CanvasWorkbenchOpenedToolTab,
  translateWorkbench: CanvasWorkbenchTranslation,
): CanvasWorkbenchTopTab {
  const baseLabel = translateCanvasWorkbenchText(
    translateWorkbench,
    resolveCanvasWorkbenchToolTabLabelKey(tab.kind),
  );
  return {
    key: tab.id,
    label: tab.sequence > 1 ? `${baseLabel} ${tab.sequence}` : baseLabel,
    closable: true,
  };
}

function resolveCanvasWorkbenchToolTabLabelKey(
  tab: CanvasWorkbenchTab,
): string {
  if (tab === "terminal") {
    return "agentChat.canvasWorkbench.newTabs.terminalTab";
  }
  return "agentChat.canvasWorkbench.newTabs.filesTab";
}
