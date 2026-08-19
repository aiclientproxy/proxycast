import { useCallback, useRef, useState } from "react";
import {
  createCanvasWorkbenchToolTabId,
  isCanvasWorkbenchToolTab,
  resolveCanvasWorkbenchToolTabKind,
  type CanvasWorkbenchNewToolTab,
  type CanvasWorkbenchOpenedToolTab,
  type CanvasWorkbenchTab,
} from "../CanvasWorkbenchLayoutState";

export interface CanvasWorkbenchToolTabsState {
  openedToolTabs: CanvasWorkbenchOpenedToolTab[];
  openNewToolTab: (tab: CanvasWorkbenchNewToolTab) => void;
  closeToolTab: (tab: CanvasWorkbenchTab) => void;
  resolveToolTabKind: (
    tab: CanvasWorkbenchTab,
  ) => CanvasWorkbenchNewToolTab | null;
}

export function useCanvasWorkbenchToolTabsState({
  activeTab,
  setActiveTab,
}: {
  activeTab: CanvasWorkbenchTab;
  setActiveTab: (tab: CanvasWorkbenchTab) => void;
}): CanvasWorkbenchToolTabsState {
  const [openedToolTabs, setOpenedToolTabs] = useState<
    CanvasWorkbenchOpenedToolTab[]
  >([]);
  const nextToolTabSequenceRef = useRef<
    Record<CanvasWorkbenchNewToolTab, number>
  >({
    "project-files": 1,
    terminal: 1,
  });

  const createToolTab = useCallback(
    (kind: CanvasWorkbenchNewToolTab): CanvasWorkbenchOpenedToolTab => {
      const sequence = nextToolTabSequenceRef.current[kind];
      nextToolTabSequenceRef.current[kind] += 1;
      return {
        id: createCanvasWorkbenchToolTabId(kind, sequence),
        kind,
        sequence,
      };
    },
    [],
  );

  const openNewToolTab = useCallback(
    (tab: CanvasWorkbenchNewToolTab) => {
      const nextTab = createToolTab(tab);
      setOpenedToolTabs((previous) => [...previous, nextTab]);
      setActiveTab(nextTab.id);
    },
    [createToolTab, setActiveTab],
  );

  const closeToolTab = useCallback(
    (tab: CanvasWorkbenchTab) => {
      if (!isCanvasWorkbenchToolTab(tab)) {
        return;
      }

      const targetIndex = openedToolTabs.findIndex((item) => item.id === tab);
      const nextTabs = openedToolTabs.filter((item) => item.id !== tab);
      setOpenedToolTabs(nextTabs);
      if (activeTab === tab) {
        const nextActiveTab =
          nextTabs[targetIndex] ?? nextTabs[targetIndex - 1] ?? null;
        setActiveTab(nextActiveTab?.id ?? "changes");
      }
    },
    [activeTab, openedToolTabs, setActiveTab],
  );

  return {
    openedToolTabs,
    openNewToolTab,
    closeToolTab,
    resolveToolTabKind: resolveCanvasWorkbenchToolTabKind,
  };
}
