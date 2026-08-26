import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  buildEvidenceSummary,
  buildMatrixResult,
  parseTestCases,
  REQUIRED_TESTS,
  RUNNER_TIMEOUT_MS,
  SETUP_TIMEOUT_MS,
  runMatrix,
} from "./windows-restricted-execution-evidence.mjs";

describe("Windows restricted execution evidence runner", () => {
  it("在非 Windows 主机 fail-closed 并写出 evidence-pending", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-windows-evidence-test-"),
    );
    const output = path.join(root, "summary.json");
    const summary = runMatrix({
      platform: "darwin",
      outputPath: output,
      now: (() => {
        let index = 0;
        return () =>
          ["2026-08-26T00:00:00.000Z", "2026-08-26T00:00:01.000Z"][index++];
      })(),
    });

    expect(summary.result).toBe("evidence-pending");
    expect(summary.failedStage).toBe("windows-runner");
    expect(summary.blockers).toHaveLength(1);
    expect(JSON.parse(fs.readFileSync(output, "utf8"))).toMatchObject({
      schemaVersion: "windows-restricted-execution-evidence-v3",
      result: "evidence-pending",
      platform: "darwin",
    });
    expect(
      fs.existsSync(path.join(root, "windows-restricted-execution.stdout.txt")),
    ).toBe(true);
    expect(
      fs.existsSync(path.join(root, "windows-restricted-execution.stderr.txt")),
    ).toBe(true);
  });

  it("在 Windows runner 记录测试结果和日志 artifact", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-windows-evidence-test-"),
    );
    const output = path.join(root, "summary.json");
    let calls = 0;
    const summary = runMatrix({
      platform: "win32",
      outputPath: output,
      provision: true,
      environment: { USERNAME: "runner", USERDOMAIN: "CI" },
      now: () => "2026-08-26T00:00:00.000Z",
      runner(command, args, options) {
        expect(command).toBe("cargo");
        expect(options.shell).toBe(false);
        calls += 1;
        if (args.includes("windows-sandbox-setup")) {
          expect(options.timeout).toBe(SETUP_TIMEOUT_MS);
          expect(options.env.LIME_AGENT_RUNTIME_ROOT).toMatch(/-agent-root$/u);
          expect(
            path
              .relative(root, options.env.LIME_AGENT_RUNTIME_ROOT)
              .startsWith(".."),
          ).toBe(true);
          return {
            status: 0,
            stdout: "windows sandbox setup completed\n",
            stderr: "",
          };
        }
        expect(args).toContain("windows_restricted_execution");
        expect(options.timeout).toBe(RUNNER_TIMEOUT_MS);
        return {
          status: 0,
          stdout:
            REQUIRED_TESTS.map((name) => `test ${name} ... ok`).join("\n") +
            `\ntest result: ok. ${REQUIRED_TESTS.length} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s\n`,
          stderr: "",
        };
      },
    });

    expect(calls).toBe(2);
    expect(summary).toMatchObject({
      result: "pass",
      failedStage: null,
      tests: { passed: REQUIRED_TESTS.length, failed: 0, ignored: 0 },
      matrix: { complete: true },
      setup: { requested: true, result: "pass", exitCode: 0 },
    });
    expect(summary.blockers).toEqual([]);
    expect(summary.artifacts).toEqual({
      stdout: "windows-restricted-execution.stdout.txt",
      stderr: "windows-restricted-execution.stderr.txt",
      setupStdout: "windows-sandbox-setup.stdout.txt",
      setupStderr: "windows-sandbox-setup.stderr.txt",
    });
  });

  it("不会把失败的 cargo 进程标记为通过", () => {
    const summary = buildEvidenceSummary({
      platform: "win32",
      startedAt: "2026-08-26T00:00:00.000Z",
      completedAt: "2026-08-26T00:00:02.000Z",
      result: "fail",
      failedStage: "test",
      exitCode: 101,
      tests: { passed: 3, failed: 1, ignored: 0 },
    });

    expect(summary.result).toBe("fail");
    expect(summary.exitCode).toBe(101);
    expect(summary.tests.failed).toBe(1);
  });

  it("解析每个 case 并拒绝缺失、重复或未知场景", () => {
    const cases = parseTestCases(
      [
        "test workspace_write_allows_workspace_and_denies_metadata_and_external_paths ... ok",
        "test restricted_execution_uses_offline_account_and_blocks_network ... ok",
        "test restricted_execution_bounds_large_output ... ok",
        "test restricted_execution_preserves_allowlisted_stdin_handle ... ignored",
        "test restricted_conpty_supports_stdin_resize_and_combined_output ... ok",
        "test world_writable_audit_reports_everyone_write_acl ... ok",
        "test terminate_ends_restricted_process_and_its_job ... ok",
        "test unexpected_case ... ok",
      ].join("\n"),
    );
    expect(cases).toHaveLength(REQUIRED_TESTS.length + 1);
    const matrix = buildMatrixResult(cases);
    expect(matrix.complete).toBe(false);
    expect(matrix.ignored).toEqual([
      "restricted_execution_preserves_allowlisted_stdin_handle",
    ]);
    expect(matrix.unexpected).toEqual(["unexpected_case"]);
    expect(matrix.missing).toEqual([]);
  });

  it("Windows 进程退出码为 0 但矩阵缺 case 时仍 fail-closed", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-windows-evidence-test-"),
    );
    const summary = runMatrix({
      platform: "win32",
      outputPath: path.join(root, "summary.json"),
      provision: true,
      environment: { USERNAME: "runner", USERDOMAIN: "CI" },
      runner() {
        return {
          status: 0,
          stdout:
            "test restricted_execution_bounds_large_output ... ok\n" +
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s\n",
          stderr: "",
        };
      },
    });

    expect(summary.result).toBe("fail");
    expect(summary.failedStage).toBe("matrix");
    expect(summary.matrix.complete).toBe(false);
    expect(summary.matrix.missing).toContain(
      "terminate_ends_restricted_process_and_its_job",
    );
  });

  it("runner 超时会保留结构化失败证据", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-windows-evidence-test-"),
    );
    let calls = 0;
    const summary = runMatrix({
      platform: "win32",
      outputPath: path.join(root, "summary.json"),
      provision: true,
      environment: { USERNAME: "runner", USERDOMAIN: "CI" },
      runner() {
        calls += 1;
        if (calls === 1) {
          return { status: 0, stdout: "setup ok", stderr: "" };
        }
        return {
          status: null,
          error: Object.assign(new Error("spawnSync cargo ETIMEDOUT"), {
            code: "ETIMEDOUT",
          }),
          stdout: "",
          stderr: "",
        };
      },
    });

    expect(summary.result).toBe("fail");
    expect(summary.failedStage).toBe("matrix");
    expect(summary.error).toContain("ETIMEDOUT");
    expect(
      JSON.parse(fs.readFileSync(path.join(root, "summary.json"), "utf8")),
    ).toMatchObject({
      result: "fail",
      error: "spawnSync cargo ETIMEDOUT",
    });
  });

  it("Windows runner 未显式 provision 时不会修改系统并 fail-closed", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-windows-evidence-test-"),
    );
    let runnerCalled = false;
    const summary = runMatrix({
      platform: "win32",
      outputPath: path.join(root, "summary.json"),
      runner() {
        runnerCalled = true;
        return { status: 0, stdout: "", stderr: "" };
      },
    });

    expect(runnerCalled).toBe(false);
    expect(summary).toMatchObject({
      result: "fail",
      failedStage: "setup",
      setup: { requested: false, result: "not-requested" },
    });
    expect(summary.blockers).toContain(
      "Windows sandbox setup was not provisioned; rerun with --provision",
    );
  });
});
