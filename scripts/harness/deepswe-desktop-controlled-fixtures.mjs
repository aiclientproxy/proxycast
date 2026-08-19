function nativeMarkerCommand(marker) {
  return `node -e "process.stdout.write('${marker}\\n')"`;
}

function writeMarkerCommand(fileName, value) {
  return `node -e "require('node:fs').writeFileSync('${fileName}', '${value}')"`;
}

const FIXTURES = {
  "happy-dom-abort-pending-body-reads": {
    primaryPath: "src/abort.ts",
    grepPattern: "pending-body-read",
    finalMarker: "DESKTOP_TYPESCRIPT_TASK_VISIBLE",
    testMarker: "DESKTOP_TYPESCRIPT_TEST_OK",
    testCommand: `node --test && ${nativeMarkerCommand("DESKTOP_TYPESCRIPT_TEST_OK")}`,
    files: {
      "package.json": '{"type":"module","scripts":{"test":"node --test"}}\n',
      "src/abort.ts": [
        'export const bodyReadState = "pending-body-read";',
        "export function shutdownBodyRead(): string {",
        '  return "pending";',
        "}",
        "",
      ].join("\n"),
      "test/abort.test.mjs": [
        'import assert from "node:assert/strict";',
        'import { readFileSync } from "node:fs";',
        'import test from "node:test";',
        "",
        'test("shutdown aborts pending body reads", () => {',
        '  const source = readFileSync(new URL("../src/abort.ts", import.meta.url), "utf8")',
        '    .replace(/^export /gmu, "")',
        '    .replace(/: string/gmu, "");',
        "  const module = new Function(`${source}; return { shutdownBodyRead };`)();",
        '  assert.equal(module.shutdownBodyRead(), "AbortError");',
        "});",
        "",
      ].join("\n"),
    },
    finalFiles: {
      "src/abort.ts": [
        'export const bodyReadState = "pending-body-read";',
        "export function shutdownBodyRead(): string {",
        '  return "AbortError";',
        "}",
        "",
      ].join("\n"),
    },
    patch: [
      "*** Begin Patch",
      "*** Update File: src/abort.ts",
      "@@",
      ' export const bodyReadState = "pending-body-read";',
      " export function shutdownBodyRead(): string {",
      '-  return "pending";',
      '+  return "AbortError";',
      " }",
      "*** End Patch",
    ].join("\n"),
    recovery: {
      approvalResumeFile: "desktop-approval-resume-marker.txt",
      approvalResumeCommand: writeMarkerCommand(
        "desktop-approval-resume-marker.txt",
        "approved",
      ),
      approvalResumeDoneText: "DESKTOP_APPROVAL_RESUME_DONE",
      cancelNoGhostWriteFile: "desktop-cancel-ghost-marker.txt",
      cancelNoGhostWriteCommand: writeMarkerCommand(
        "desktop-cancel-ghost-marker.txt",
        "ghost",
      ),
    },
  },
  "go-genai-streamed-function-args": {
    primaryPath: "stream.go",
    grepPattern: "partial-args",
    finalMarker: "DESKTOP_GO_TASK_VISIBLE",
    testMarker: "DESKTOP_GO_TEST_OK",
    testCommand: `go test ./... && ${nativeMarkerCommand("DESKTOP_GO_TEST_OK")}`,
    files: {
      "go.mod": "module desktopfixture/genai\n\ngo 1.24\n",
      "stream.go": [
        "package genai",
        "",
        'const streamState = "partial-args"',
        "",
        "func AccumulatedArgs() string {",
        '\treturn "partial"',
        "}",
        "",
      ].join("\n"),
      "stream_test.go": [
        "package genai",
        "",
        'import "testing"',
        "",
        "func TestAccumulatedArgs(t *testing.T) {",
        '\tif got := AccumulatedArgs(); got != "complete" {',
        '\t\tt.Fatalf("got %q", got)',
        "\t}",
        "}",
        "",
      ].join("\n"),
    },
    finalFiles: {
      "stream.go": [
        "package genai",
        "",
        'const streamState = "partial-args"',
        "",
        "func AccumulatedArgs() string {",
        '\treturn "complete"',
        "}",
        "",
      ].join("\n"),
    },
    patch: [
      "*** Begin Patch",
      "*** Update File: stream.go",
      "@@",
      " func AccumulatedArgs() string {",
      '-\treturn "partial"',
      '+\treturn "complete"',
      " }",
      "*** End Patch",
    ].join("\n"),
    recovery: {
      approvalResumeFile: "desktop-approval-resume-marker.txt",
      approvalResumeCommand: writeMarkerCommand(
        "desktop-approval-resume-marker.txt",
        "approved",
      ),
      approvalResumeDoneText: "DESKTOP_APPROVAL_RESUME_DONE",
      cancelNoGhostWriteFile: "desktop-cancel-ghost-marker.txt",
      cancelNoGhostWriteCommand: writeMarkerCommand(
        "desktop-cancel-ghost-marker.txt",
        "ghost",
      ),
    },
  },
  "httpx-multipart-response-parsing": {
    primaryPath: "multipart.py",
    grepPattern: "multipart-sync-before",
    finalMarker: "DESKTOP_PYTHON_TASK_VISIBLE",
    testMarker: "DESKTOP_PYTHON_TEST_OK",
    testCommand: `python3 -B -m unittest discover -s tests && ${nativeMarkerCommand("DESKTOP_PYTHON_TEST_OK")}`,
    files: {
      "multipart.py": [
        'PARSER_STATE = "multipart-sync-before"',
        "",
        "def parse_multipart():",
        '    return "before"',
        "",
      ].join("\n"),
      "tests/test_multipart.py": [
        "import unittest",
        "",
        "from multipart import parse_multipart",
        "from multipart_async import parse_multipart_async",
        "",
        "",
        "class MultipartTest(unittest.TestCase):",
        "    def test_sync_async_parity(self):",
        '        self.assertEqual(parse_multipart(), "parsed")',
        '        self.assertEqual(parse_multipart_async(), "parsed")',
        "",
        "",
        'if __name__ == "__main__":',
        "    unittest.main()",
        "",
      ].join("\n"),
    },
    finalFiles: {
      "multipart.py": [
        'PARSER_STATE = "multipart-sync-before"',
        "",
        "def parse_multipart():",
        '    return "parsed"',
        "",
      ].join("\n"),
      "multipart_async.py": [
        "def parse_multipart_async():",
        '    return "parsed"',
        "",
      ].join("\n"),
    },
    patch: [
      "*** Begin Patch",
      "*** Update File: multipart.py",
      "@@",
      " def parse_multipart():",
      '-    return "before"',
      '+    return "parsed"',
      "*** Add File: multipart_async.py",
      "+def parse_multipart_async():",
      '+    return "parsed"',
      "+",
      "*** End Patch",
    ].join("\n"),
    recovery: {
      approvalResumeFile: "desktop-approval-resume-marker.txt",
      approvalResumeCommand: writeMarkerCommand(
        "desktop-approval-resume-marker.txt",
        "approved",
      ),
      approvalResumeDoneText: "DESKTOP_APPROVAL_RESUME_DONE",
      cancelNoGhostWriteFile: "desktop-cancel-ghost-marker.txt",
      cancelNoGhostWriteCommand: writeMarkerCommand(
        "desktop-cancel-ghost-marker.txt",
        "ghost",
      ),
    },
  },
  "fd-deterministic-multi-key-sorting": {
    primaryPath: "src/lib.rs",
    grepPattern: "unstable-sort",
    finalMarker: "DESKTOP_RUST_TASK_VISIBLE",
    testMarker: "DESKTOP_RUST_TEST_OK",
    testCommand: `cargo +stable test --quiet && ${nativeMarkerCommand("DESKTOP_RUST_TEST_OK")}`,
    files: {
      "Cargo.toml": [
        "[package]",
        'name = "desktop-fd-fixture"',
        'version = "0.1.0"',
        'edition = "2024"',
        "",
      ].join("\n"),
      "src/lib.rs": [
        'pub const SORT_STATE: &str = "unstable-sort";',
        "",
        "pub fn deterministic_sort() -> bool {",
        "    false",
        "}",
        "",
        "#[cfg(test)]",
        "mod tests {",
        "    use super::*;",
        "",
        "    #[test]",
        "    fn sorting_is_deterministic() {",
        "        assert!(deterministic_sort());",
        "    }",
        "}",
        "",
      ].join("\n"),
    },
    finalFiles: {
      "src/lib.rs": [
        'pub const SORT_STATE: &str = "unstable-sort";',
        "",
        "pub fn deterministic_sort() -> bool {",
        "    true",
        "}",
        "",
        "#[cfg(test)]",
        "mod tests {",
        "    use super::*;",
        "",
        "    #[test]",
        "    fn sorting_is_deterministic() {",
        "        assert!(deterministic_sort());",
        "    }",
        "}",
        "",
      ].join("\n"),
    },
    patch: [
      "*** Begin Patch",
      "*** Update File: src/lib.rs",
      "@@",
      " pub fn deterministic_sort() -> bool {",
      "-    false",
      "+    true",
      " }",
      "*** End Patch",
    ].join("\n"),
    recovery: {
      approvalResumeFile: "desktop-approval-resume-marker.txt",
      approvalResumeCommand: writeMarkerCommand(
        "desktop-approval-resume-marker.txt",
        "approved",
      ),
      approvalResumeDoneText: "DESKTOP_APPROVAL_RESUME_DONE",
      cancelNoGhostWriteFile: "desktop-cancel-ghost-marker.txt",
      cancelNoGhostWriteCommand: writeMarkerCommand(
        "desktop-cancel-ghost-marker.txt",
        "ghost",
      ),
    },
  },
  "yjs-map-conflict-detection": {
    primaryPath: "conflict.js",
    grepPattern: "conflict-policy-allow",
    finalMarker: "DESKTOP_JAVASCRIPT_TASK_VISIBLE",
    testMarker: "DESKTOP_JAVASCRIPT_TEST_OK",
    testCommand: `node --test && ${nativeMarkerCommand("DESKTOP_JAVASCRIPT_TEST_OK")}`,
    files: {
      "package.json": '{"type":"module","scripts":{"test":"node --test"}}\n',
      "conflict.js": [
        'export const policyState = "conflict-policy-allow";',
        "",
        "export function conflictPolicy() {",
        '  return "allow";',
        "}",
        "",
      ].join("\n"),
      "conflict.test.js": [
        'import assert from "node:assert/strict";',
        'import test from "node:test";',
        'import { conflictPolicy } from "./conflict.js";',
        "",
        'test("collects deterministic conflicts", () => {',
        '  assert.equal(conflictPolicy(), "collect");',
        "});",
        "",
      ].join("\n"),
    },
    finalFiles: {
      "conflict.js": [
        'export const policyState = "conflict-policy-allow";',
        "",
        "export function conflictPolicy() {",
        '  return "collect";',
        "}",
        "",
      ].join("\n"),
    },
    patch: [
      "*** Begin Patch",
      "*** Update File: conflict.js",
      "@@",
      " export function conflictPolicy() {",
      '-  return "allow";',
      '+  return "collect";',
      " }",
      "*** End Patch",
    ].join("\n"),
    recovery: {
      approvalResumeFile: "desktop-approval-resume-marker.txt",
      approvalResumeCommand: writeMarkerCommand(
        "desktop-approval-resume-marker.txt",
        "approved",
      ),
      approvalResumeDoneText: "DESKTOP_APPROVAL_RESUME_DONE",
      cancelNoGhostWriteFile: "desktop-cancel-ghost-marker.txt",
      cancelNoGhostWriteCommand: writeMarkerCommand(
        "desktop-cancel-ghost-marker.txt",
        "ghost",
      ),
    },
  },
};

export function controlledFixtureForTask(taskId) {
  const fixture = FIXTURES[taskId];
  if (!fixture)
    throw new Error(`controlled desktop fixture not found: ${taskId}`);
  return fixture;
}

export function controlledFixtureTaskIds() {
  return Object.keys(FIXTURES);
}

export function controlledFixtureResponses(taskId) {
  const fixture = controlledFixtureForTask(taskId);
  return [
    {
      type: "tool_call",
      id: `desktop-${taskId}-read`,
      name: "Read",
      arguments: { path: fixture.primaryPath },
    },
    {
      type: "tool_call",
      id: `desktop-${taskId}-glob`,
      name: "Glob",
      arguments: { pattern: "**/*", max_results: 100 },
    },
    {
      type: "tool_call",
      id: `desktop-${taskId}-grep`,
      name: "Grep",
      arguments: {
        pattern: fixture.grepPattern,
        path: ".",
        mode: "content",
        include_hidden: true,
        max_results: 20,
      },
    },
    {
      type: "tool_call",
      id: `desktop-${taskId}-patch`,
      name: "apply_patch",
      arguments: { patch: fixture.patch },
    },
    {
      type: "tool_call",
      id: `desktop-${taskId}-test`,
      name: "exec_command",
      arguments: { cmd: fixture.testCommand, yield_time_ms: 30_000 },
    },
    { type: "text", content: fixture.finalMarker },
    {
      type: "tool_call",
      recovery: "approval_resume",
      id: `desktop-${taskId}-approval-resume`,
      name: "exec_command",
      arguments: {
        cmd: fixture.recovery.approvalResumeCommand,
        yield_time_ms: 30_000,
      },
    },
    { type: "text", content: fixture.recovery.approvalResumeDoneText },
    {
      type: "tool_call",
      recovery: "cancel_no_ghost_write",
      id: `desktop-${taskId}-cancel-no-ghost-write`,
      name: "exec_command",
      arguments: {
        cmd: fixture.recovery.cancelNoGhostWriteCommand,
        yield_time_ms: 30_000,
      },
    },
  ];
}
