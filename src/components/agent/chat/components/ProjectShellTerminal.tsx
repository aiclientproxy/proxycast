import React from "react";
import { agentText } from "./harnessPanelText";
import {
  execCommand,
  resizeCommandExec,
  subscribeCommandExecOutput,
  terminateCommandExec,
  writeCommandExec,
} from "@/lib/api/commandExec";
import "@xterm/xterm/css/xterm.css";

type XTermTerminal = import("@xterm/xterm").Terminal;
type XTermDisposable = ReturnType<XTermTerminal["onData"]>;
type XTermFitAddon = import("@xterm/addon-fit").FitAddon;
type XTermTheme = NonNullable<
  ConstructorParameters<typeof import("@xterm/xterm").Terminal>[0]
>["theme"];

export interface ProjectShellTabState {
  errorText: string | null;
  ready: boolean;
  shell: string | null;
  statusText: string;
  title: string;
}

export interface ProjectShellTerminalHandle {
  fit: () => void;
  focus: () => void;
  runCommand: (command: string) => void;
}

interface ProjectShellTerminalProps {
  active: boolean;
  projectRootPath?: string | null;
  tabId: string;
  testIdPrefix?: string;
  onStateChange: (tabId: string, state: Partial<ProjectShellTabState>) => void;
}

const FALLBACK_COLS = 120;
const FALLBACK_ROWS = 14;
const INPUT_FLUSH_DELAY_MS = 8;
const PROJECT_SHELL_THEME: XTermTheme = {
  background: "#ffffff",
  foreground: "#1f2937",
  cursor: "#111827",
  cursorAccent: "#ffffff",
  selectionBackground: "#dbeafe",
  selectionForeground: "#111827",
  selectionInactiveBackground: "#e5e7eb",
  black: "#24292f",
  red: "#d1242f",
  green: "#16a34a",
  yellow: "#ca8a04",
  blue: "#0969da",
  magenta: "#c026d3",
  cyan: "#0891b2",
  white: "#f8fafc",
  brightBlack: "#6b7280",
  brightRed: "#dc2626",
  brightGreen: "#22c55e",
  brightYellow: "#eab308",
  brightBlue: "#1d4ed8",
  brightMagenta: "#a855f7",
  brightCyan: "#06b6d4",
  brightWhite: "#ffffff",
  overviewRulerBorder: "#e2e8f0",
  scrollbarSliderBackground: "#cbd5e1",
  scrollbarSliderHoverBackground: "#94a3b8",
  scrollbarSliderActiveBackground: "#64748b",
};

function extractErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export const ProjectShellTerminal = React.forwardRef<
  ProjectShellTerminalHandle,
  ProjectShellTerminalProps
>(function ProjectShellTerminal(
  {
    active,
    projectRootPath,
    tabId,
    testIdPrefix = "task-center-shell",
    onStateChange,
  },
  ref,
) {
  const normalizedProjectRootPath = projectRootPath?.trim() || null;
  const terminalContainerRef = React.useRef<HTMLDivElement | null>(null);
  const terminalRef = React.useRef<XTermTerminal | null>(null);
  const fitAddonRef = React.useRef<XTermFitAddon | null>(null);
  const processIdRef = React.useRef<string | null>(null);
  const writeQueueRef = React.useRef<Promise<void>>(Promise.resolve());
  const terminalSizeRef = React.useRef({
    cols: FALLBACK_COLS,
    rows: FALLBACK_ROWS,
  });

  const patchState = React.useCallback(
    (state: Partial<ProjectShellTabState>) => onStateChange(tabId, state),
    [onStateChange, tabId],
  );

  const fitTerminal = React.useCallback(() => {
    const terminal = terminalRef.current;
    const fitAddon = fitAddonRef.current;
    if (!terminal || !fitAddon) return;
    fitTerminalToContainer({
      terminal,
      fitAddon,
      processIdRef,
      terminalSizeRef,
    });
  }, []);

  const writeShellData = React.useCallback((data: string) => {
    const processId = processIdRef.current;
    if (!processId) return;
    writeQueueRef.current = writeQueueRef.current
      .catch(() => undefined)
      .then(async () => {
        if (processIdRef.current !== processId) return;
        await writeCommandExec({
          processId,
          deltaBase64: encodeBase64(data),
        });
      })
      .catch((error) => {
        terminalRef.current?.writeln(
          `\r\n${agentText(
            "agentChat.navbar.shell.writeFailed",
            "写入 Shell 失败：{{message}}",
            { message: extractErrorMessage(error) },
          )}`,
        );
      });
  }, []);

  React.useImperativeHandle(
    ref,
    () => ({
      fit: fitTerminal,
      focus: () => terminalRef.current?.focus(),
      runCommand: (command: string) => {
        if (!processIdRef.current) return;
        writeShellData(`${command}\r`);
        terminalRef.current?.focus();
      },
    }),
    [fitTerminal, writeShellData],
  );

  React.useEffect(() => {
    let disposed = false;
    let terminal: XTermTerminal | null = null;
    let fitAddon: XTermFitAddon | null = null;
    let inputDisposable: XTermDisposable | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let unlisten: (() => void) | null = null;
    let processIdForBoot: string | null = null;
    let pendingInput = "";
    let inputFlushTimer: ReturnType<typeof setTimeout> | null = null;

    function flushPendingInput() {
      if (inputFlushTimer) {
        clearTimeout(inputFlushTimer);
        inputFlushTimer = null;
      }
      const data = pendingInput;
      pendingInput = "";
      if (data && !disposed) writeShellData(data);
    }

    function scheduleInputFlush() {
      if (!inputFlushTimer) {
        inputFlushTimer = setTimeout(flushPendingInput, INPUT_FLUSH_DELAY_MS);
      }
    }

    async function bootShell() {
      const container = terminalContainerRef.current;
      if (!container) return;
      patchState({
        errorText: null,
        ready: false,
        shell: null,
        statusText: agentText("agentChat.navbar.shell.connecting", "连接中"),
      });
      if (!normalizedProjectRootPath) {
        patchState({
          errorText: agentText(
            "agentChat.navbar.shell.noProjectRoot",
            "当前项目缺少本地目录",
          ),
          statusText: agentText("agentChat.navbar.shell.unavailable", "不可用"),
        });
        return;
      }

      try {
        const [{ Terminal }, { FitAddon }] = await Promise.all([
          import("@xterm/xterm"),
          import("@xterm/addon-fit"),
        ]);
        if (disposed) return;
        terminal = new Terminal({
          cols: FALLBACK_COLS,
          rows: FALLBACK_ROWS,
          cursorBlink: true,
          convertEol: true,
          disableStdin: false,
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          fontSize: 12,
          lineHeight: 1.35,
          scrollback: 2000,
          theme: PROJECT_SHELL_THEME,
        });
        fitAddon = new FitAddon();
        terminal.loadAddon(fitAddon);
        terminalRef.current = terminal;
        fitAddonRef.current = fitAddon;
        terminal.open(container);
        fitTerminalToContainer({
          terminal,
          fitAddon,
          processIdRef,
          terminalSizeRef,
        });
        if (typeof ResizeObserver !== "undefined") {
          resizeObserver = new ResizeObserver(() =>
            fitTerminalToContainer({
              terminal: terminal!,
              fitAddon: fitAddon!,
              processIdRef,
              terminalSizeRef,
            }),
          );
          resizeObserver.observe(container);
        }

        const processId = createProcessId();
        processIdForBoot = processId;
        processIdRef.current = processId;
        const shell = resolveInteractiveShell();
        const decoder = new TextDecoder();
        unlisten = subscribeCommandExecOutput(processId, (delta) => {
          if (!disposed)
            terminal?.write(
              decoder.decode(decodeBytes(delta.deltaBase64), { stream: true }),
            );
        });
        const size = terminalSizeRef.current;
        patchState({
          ready: true,
          shell: shell[0],
          statusText: agentText("agentChat.navbar.shell.connected", "已连接"),
          title: buildShellTitle(normalizedProjectRootPath),
        });
        fitTerminalToContainer({
          terminal,
          fitAddon,
          processIdRef,
          terminalSizeRef,
        });
        void execCommand({
          command: shell,
          processId,
          tty: true,
          streamStdin: true,
          streamStdoutStderr: true,
          disableOutputCap: true,
          disableTimeout: true,
          cwd: normalizedProjectRootPath,
          size,
        })
          .then((result) => {
            if (disposed) return;
            terminal?.write(decoder.decode());
            terminal?.writeln(
              `\r\n${agentText(
                "agentChat.navbar.shell.exited",
                "Shell 已退出：{{code}}",
                { code: result.exitCode },
              )}`,
            );
            patchState({
              ready: false,
              statusText: agentText(
                "agentChat.navbar.shell.exitedStatus",
                "已退出",
              ),
            });
          })
          .catch((error) => {
            if (disposed) return;
            const message = extractErrorMessage(error);
            patchState({
              ready: false,
              errorText: message,
              statusText: agentText("agentChat.navbar.shell.failed", "已断开"),
            });
            terminal?.writeln(
              `\r\n${agentText(
                "agentChat.navbar.shell.startFailed",
                "Shell 启动失败：{{message}}",
                { message },
              )}`,
            );
          });

        inputDisposable = terminal.onData((data) => {
          pendingInput += data;
          if (data.includes("\r") || data.includes("\n")) {
            flushPendingInput();
          } else {
            scheduleInputFlush();
          }
        });
      } catch (error) {
        if (disposed) return;
        const message = extractErrorMessage(error);
        patchState({
          errorText: message,
          ready: false,
          statusText: agentText("agentChat.navbar.shell.failed", "已断开"),
        });
      }
    }

    void bootShell();
    return () => {
      disposed = true;
      unlisten?.();
      inputDisposable?.dispose();
      if (inputFlushTimer) clearTimeout(inputFlushTimer);
      pendingInput = "";
      resizeObserver?.disconnect();
      if (processIdForBoot && processIdRef.current === processIdForBoot) {
        processIdRef.current = null;
        void terminateCommandExec({ processId: processIdForBoot }).catch(
          () => undefined,
        );
      }
      terminal?.dispose();
      if (terminalRef.current === terminal) terminalRef.current = null;
      if (fitAddonRef.current === fitAddon) fitAddonRef.current = null;
    };
  }, [normalizedProjectRootPath, patchState, writeShellData]);

  React.useEffect(() => {
    if (!active) return;
    const animationFrame = requestAnimationFrame(fitTerminal);
    return () => cancelAnimationFrame(animationFrame);
  }, [active, fitTerminal]);

  return (
    <div
      className={active ? "h-full w-full" : "hidden"}
      data-active={active ? "true" : "false"}
      data-testid={
        active
          ? `${testIdPrefix}-terminal-pane`
          : `${testIdPrefix}-terminal-pane-inactive`
      }
      onClick={() => terminalRef.current?.focus()}
    >
      <div
        ref={terminalContainerRef}
        className="h-full w-full bg-white [&_.xterm-rows]:!bg-white [&_.xterm-screen]:!bg-white [&_.xterm-screen]:outline-none [&_.xterm-scrollable-element]:!bg-white [&_.xterm-viewport]:!bg-white [&_.xterm]:h-full [&_.xterm]:w-full [&_.xterm]:!bg-white"
        title={agentText(
          "agentChat.navbar.shell.ready",
          "Shell 已就绪，可以输入命令",
        )}
        data-testid={
          active
            ? `${testIdPrefix}-terminal`
            : `${testIdPrefix}-terminal-hidden`
        }
      />
    </div>
  );
});

function fitTerminalToContainer({
  terminal,
  fitAddon,
  processIdRef,
  terminalSizeRef,
}: {
  terminal: XTermTerminal;
  fitAddon: XTermFitAddon;
  processIdRef: React.MutableRefObject<string | null>;
  terminalSizeRef: React.MutableRefObject<{ cols: number; rows: number }>;
}) {
  try {
    fitAddon.fit();
  } catch {
    return;
  }
  const cols = terminal.cols || FALLBACK_COLS;
  const rows = terminal.rows || FALLBACK_ROWS;
  const previous = terminalSizeRef.current;
  if (previous.cols === cols && previous.rows === rows) return;
  terminalSizeRef.current = { cols, rows };
  const processId = processIdRef.current;
  if (!processId) return;
  void resizeCommandExec({ processId, size: { cols, rows } }).catch(
    () => undefined,
  );
}

function createProcessId(): string {
  return `shell-${crypto.randomUUID()}`;
}

function resolveInteractiveShell(): string[] {
  if (/Windows/i.test(navigator.userAgent)) return ["cmd.exe", "/d"];
  return ["/bin/sh", "-i"];
}

function buildShellTitle(rootPath: string): string {
  return rootPath.split(/[\\/]/).filter(Boolean).pop() ?? "project";
}

function encodeBase64(value: string): string {
  const bytes = new TextEncoder().encode(value);
  return btoa(String.fromCharCode(...bytes));
}

function decodeBytes(value: string): Uint8Array {
  const binary = atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}
