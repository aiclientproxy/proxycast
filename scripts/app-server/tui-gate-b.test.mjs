import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

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
    expect(ptyTestSource).toContain("writer.write_all(&[3])");
  });

  it("does not substitute a mock backend or synthetic completion event", () => {
    expect(ptyTestSource).toContain(
      'OsString::from("--app-server-arg=external")',
    );
    expect(gateSource).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(gateSource).not.toContain("turn.final_done");
    expect(runtimeSource).not.toContain("final_done");
  });
});
