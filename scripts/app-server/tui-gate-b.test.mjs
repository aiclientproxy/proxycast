import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import YAML from "yaml";

const gateSource = readFileSync(
  path.resolve(process.cwd(), "scripts/app-server/tui-gate-b.mjs"),
  "utf8",
);
const runtimeSource = readFileSync(
  path.resolve(process.cwd(), "lime-rs/crates/tui/src/runtime.rs"),
  "utf8",
);
const ptyTestSource = readFileSync(
  path.resolve(process.cwd(), "lime-rs/crates/tui/src/runtime_pty_tests.rs"),
  "utf8",
);

describe("TUI Gate B", () => {
  it("drives the real TUI through a portable PTY and current App Server", () => {
    expect(gateSource).toContain('LIME_TEST_TUI_GATE_B: "1"');
    expect(gateSource).toContain('"--exact"');
    expect(gateSource).toContain("writeTerminalExternalBackend");
    expect(ptyTestSource).toContain('OsString::from("tui")');
    expect(gateSource).toContain(
      "runtime::pty_tests::real_pty_restores_terminal_after_visible_turn_completion",
    );
    expect(ptyTestSource).toContain("native_pty_system()");
    expect(ptyTestSource).toContain('output.contains("\\u{1b}[?1049h")');
    expect(ptyTestSource).toContain('output.contains("\\u{1b}[?1049l")');
    expect(ptyTestSource).toContain("EDITOR_JOB_CONTROL_OK");
    expect(ptyTestSource).toContain("configure_external_editor");
    expect(ptyTestSource).not.toContain('write_all(b"\\x1b[1;1R")');
    expect(ptyTestSource).toContain("writer.write_all(&[20])");
    expect(ptyTestSource).toContain('"Ctrl+T/Esc/Q close"');
    expect(ptyTestSource).toContain('writer.write_all(b"\\x1b")');
    expect(ptyTestSource).toContain('"esc to interrupt"');
    expect(ptyTestSource).toContain('write_all(b"\\x1b[1;3A")');
    expect(ptyTestSource).toContain('"editing queued"');
    expect(gateSource).toContain('event?.type === "queue.added"');
    expect(gateSource).toContain('event?.type === "queue.removed"');
    expect(gateSource).toContain(
      'event.payload?.source === "thread/queue/delete"',
    );
    expect(gateSource).toContain(
      '"complete,approval,user-input,interrupt,failure,queue-edit"',
    );
  });

  it("does not substitute a mock backend or synthetic completion event", () => {
    expect(ptyTestSource).toContain(
      'OsString::from("--app-server-arg=external")',
    );
    expect(gateSource).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(gateSource).not.toContain("turn.final_done");
    expect(runtimeSource).not.toContain("final_done");
  });

  it("keeps Windows CLI and TUI current-path evidence in the package workflow", () => {
    const workflow = YAML.parse(
      readFileSync(
        path.resolve(process.cwd(), ".github/workflows/build-windows-test.yml"),
        "utf8",
      ),
    );
    const steps = workflow.jobs["build-windows-test"].steps;
    const gate = steps.find(
      (step) => step.name === "Run Windows CLI and TUI current-path gates",
    );
    const upload = steps.find(
      (step) => step.name === "Upload Windows CLI and TUI Gate B evidence",
    );

    expect(gate?.shell).toBe("bash");
    expect(gate?.run).toContain(
      "cargo test --manifest-path lime-rs/Cargo.toml -p cli -p tui",
    );
    expect(gate?.run).toContain(
      "cargo build --manifest-path lime-rs/Cargo.toml -p cli -p app-server",
    );
    expect(gate?.run).toContain("npm run smoke:cli-gate-b");
    expect(gate?.run).toContain("npm run smoke:tui-gate-b");
    expect(gate?.run).toContain('} 2>&1 | tee "windows-cli-tui-gate-b.log"');
    expect(upload?.if).toBe("${{ always() }}");
    expect(upload?.with?.path).toBe("windows-cli-tui-gate-b.log");
    expect(upload?.with?.["if-no-files-found"]).toBe("error");
  });
});
