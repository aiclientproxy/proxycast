import type {
  Event as ElectronEvent,
  Input,
  MouseInputEvent,
  WebContents,
} from "electron";

export interface BrowserTabUserControlObserver {
  dispose(): void;
  runAgentInput<T>(action: () => Promise<T>): Promise<T>;
}

const USER_MOUSE_INPUT_TYPES = new Set<MouseInputEvent["type"]>([
  "contextMenu",
  "mouseDown",
  "mouseWheel",
]);

export function observeBrowserTabUserControl(
  webContents: WebContents,
  onUserInput: () => void,
): BrowserTabUserControlObserver {
  let agentInputDepth = 0;
  const handleKeyboardInput = (_event: ElectronEvent, input: Input) => {
    if (agentInputDepth === 0 && input.type === "keyDown") {
      onUserInput();
    }
  };
  const handleMouseInput = (_event: ElectronEvent, input: MouseInputEvent) => {
    if (agentInputDepth === 0 && USER_MOUSE_INPUT_TYPES.has(input.type)) {
      onUserInput();
    }
  };

  webContents.on("before-input-event", handleKeyboardInput);
  webContents.on("before-mouse-event", handleMouseInput);

  return {
    dispose() {
      webContents.off("before-input-event", handleKeyboardInput);
      webContents.off("before-mouse-event", handleMouseInput);
    },
    async runAgentInput<T>(action: () => Promise<T>): Promise<T> {
      agentInputDepth += 1;
      try {
        return await action();
      } finally {
        agentInputDepth -= 1;
      }
    },
  };
}
