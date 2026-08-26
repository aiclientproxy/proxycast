import fs from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ensureMacBinaryRpath,
  buildSherpaArchiveExtractCommand,
  MACOS_EXECUTABLE_RPATH,
  missingSherpaRuntimeLibraries,
  readMachORpaths,
  resolveSherpaOnnxSysVersion,
  resolveSherpaRuntimePlan,
} from "./prepare-sherpa-onnx-runtime.mjs";

describe("prepare sherpa-onnx runtime", () => {
  it("从 Cargo.lock 解析 sherpa-onnx-sys 版本", () => {
    const version = resolveSherpaOnnxSysVersion(`
[[package]]
name = "other"
version = "0.1.0"

[[package]]
name = "sherpa-onnx-sys"
version = "1.13.0"
`);

    expect(version).toBe("1.13.0");
  });

  it("为 macOS arm64 解析预置运行时归档和库文件", () => {
    const plan = resolveSherpaRuntimePlan({
      repoRoot: "/repo",
      targetTriple: "aarch64-apple-darwin",
      version: "1.13.0",
    });

    expect(plan.archiveName).toBe(
      "sherpa-onnx-v1.13.0-osx-arm64-shared-lib.tar.bz2",
    );
    expect(plan.libs).toEqual([
      "libonnxruntime.1.24.4.dylib",
      "libonnxruntime.dylib",
      "libsherpa-onnx-c-api.dylib",
    ]);
    expect(plan.releaseDir).toBe(
      "/repo/lime-rs/target/aarch64-apple-darwin/release",
    );
    expect(plan.debugDirs).toEqual([
      "/repo/lime-rs/target/debug",
      "/repo/lime-rs/target/aarch64-apple-darwin/debug",
    ]);
  });

  it("不把缺少共享库的缓存目录视为已准备", () => {
    const plan = {
      libDir: "/repo/lime-rs/target/sherpa-onnx-prebuilt/runtime/lib",
      libs: ["libonnxruntime.1.24.4.dylib", "libsherpa-onnx-c-api.dylib"],
    };

    expect(
      missingSherpaRuntimeLibraries(plan, {
        exists: (filePath) => filePath.endsWith("libonnxruntime.1.24.4.dylib"),
      }),
    ).toEqual(["libsherpa-onnx-c-api.dylib"]);
  });

  it("为 Windows 解析预置运行时归档和库文件", () => {
    const plan = resolveSherpaRuntimePlan({
      repoRoot: "/repo",
      targetTriple: "x86_64-pc-windows-msvc",
      version: "1.13.0",
    });

    expect(plan.archiveName).toBe(
      "sherpa-onnx-v1.13.0-win-x64-shared-MT-Release-lib.tar.bz2",
    );
    expect(plan.libs).toEqual(["onnxruntime.dll", "sherpa-onnx-c-api.dll"]);
  });

  it("解压 sherpa 归档时使用 workspace cwd，兼容 Windows 驱动器路径", () => {
    expect(
      buildSherpaArchiveExtractCommand({
        archivePath:
          "D:\\a\\lime\\lime\\lime-rs\\target\\sherpa-onnx-prebuilt\\sherpa-onnx-v1.13.0-win-x64-shared-MT-Release-lib.tar.bz2",
        prebuiltRoot:
          "D:\\a\\lime\\lime\\lime-rs\\target\\sherpa-onnx-prebuilt",
      }),
    ).toEqual({
      args: [
        "-xjf",
        "sherpa-onnx-v1.13.0-win-x64-shared-MT-Release-lib.tar.bz2",
        "-C",
        ".",
      ],
      cwd: "D:\\a\\lime\\lime\\lime-rs\\target\\sherpa-onnx-prebuilt",
    });
  });

  it("Windows Quality 为 sherpa 准备保留足够预算并记录清理/解压阶段", () => {
    const workflow = fs.readFileSync(".github/workflows/quality.yml", "utf8");
    const windowsJob = workflow
      .split("\n  windows_shell_runtime:\n")[1]
      ?.split("\n  quality_gate:\n")[0];
    expect(windowsJob).toContain("timeout-minutes: 60");
    const securityMatrixOffset = windowsJob.indexOf(
      "Test Windows restricted execution security matrix",
    );
    const securityEvidenceUploadOffset = windowsJob.indexOf(
      "Upload Windows restricted execution evidence",
    );
    const sherpaOffset = windowsJob.indexOf("Prepare sherpa-onnx runtime");
    const sidecarBuildOffset = windowsJob.indexOf(
      "Build Windows app-server, code-mode-host, and sandbox setup sidecars",
    );
    const timeoutContractOffset = windowsJob.indexOf(
      "Test Windows command timeout contract",
    );
    expect(securityMatrixOffset).toBeGreaterThanOrEqual(0);
    expect(securityEvidenceUploadOffset).toBeGreaterThan(securityMatrixOffset);
    expect(sherpaOffset).toBeGreaterThan(securityEvidenceUploadOffset);
    expect(sidecarBuildOffset).toBeGreaterThan(sherpaOffset);
    expect(timeoutContractOffset).toBeGreaterThan(sidecarBuildOffset);

    const source = fs.readFileSync(
      "scripts/prepare-sherpa-onnx-runtime.mjs",
      "utf8",
    );
    const stages = [
      "Removing previous sherpa-onnx runtime:",
      "Removed previous sherpa-onnx runtime:",
      "Extracting sherpa-onnx archive:",
      "Extracted sherpa-onnx archive:",
    ];
    const offsets = stages.map((stage) => source.indexOf(stage));
    expect(offsets.every((offset) => offset >= 0)).toBe(true);
    expect(offsets).toEqual([...offsets].sort((left, right) => left - right));
    expect(windowsJob).toContain(
      "scripts/lib/windows-restricted-execution-evidence.mjs",
    );
    expect(windowsJob).toContain("actions/upload-artifact@v4");
  });

  it("支持显式 Rust workspace 目录，不再暴露旧目录参数口径", () => {
    const plan = resolveSherpaRuntimePlan({
      repoRoot: "/repo",
      rustWorkspaceDir: "runtime-rs",
      targetTriple: "x86_64-apple-darwin",
      version: "1.13.0",
    });

    expect(plan.releaseDir).toBe(
      "/repo/runtime-rs/target/x86_64-apple-darwin/release",
    );
    expect(plan.runtimeLibDir).toBe(
      "/repo/runtime-rs/.release-runtime-libs/x86_64-apple-darwin",
    );
  });

  it("解析 Mach-O rpath load command", () => {
    const rpaths = readMachORpaths("/repo/lime-rs/target/debug/app-server", {
      platform: "darwin",
      runner(command, args) {
        expect(command).toBe("otool");
        expect(args).toEqual(["-l", "/repo/lime-rs/target/debug/app-server"]);
        return {
          status: 0,
          stdout: `
Load command 0
      cmd LC_SEGMENT_64
Load command 1
      cmd LC_RPATH
  cmdsize 32
     path @executable_path (offset 12)
Load command 2
      cmd LC_LOAD_DYLIB
`,
        };
      },
    });

    expect(rpaths).toEqual([MACOS_EXECUTABLE_RPATH]);
  });

  it("缺少 macOS rpath 时为 app-server 二进制补 @executable_path", () => {
    const calls = [];
    const result = ensureMacBinaryRpath(
      "/repo/lime-rs/target/debug/app-server",
      {
        exists: () => true,
        getStats: () => ({ isFile: () => true, size: 1 }),
        platform: "darwin",
        runner(command, args, options) {
          calls.push([command, args, options]);
          if (command === "otool") {
            return { status: 0, stdout: "" };
          }
          if (command === "install_name_tool") {
            return { status: 0 };
          }
          throw new Error(`unexpected command: ${command}`);
        },
      },
    );

    expect(result).toMatchObject({
      checked: true,
      patched: true,
      reason: "patched",
      rpaths: [MACOS_EXECUTABLE_RPATH],
    });
    expect(calls[1]).toEqual([
      "install_name_tool",
      [
        "-add_rpath",
        MACOS_EXECUTABLE_RPATH,
        "/repo/lime-rs/target/debug/app-server",
      ],
      {
        stdio: "inherit",
        shell: false,
      },
    ]);
  });

  it("已有 macOS rpath 时不重复 patch", () => {
    const calls = [];
    const result = ensureMacBinaryRpath(
      "/repo/lime-rs/target/debug/app-server",
      {
        exists: () => true,
        getStats: () => ({ isFile: () => true, size: 1 }),
        platform: "darwin",
        runner(command) {
          calls.push(command);
          return {
            status: 0,
            stdout: `
Load command 1
      cmd LC_RPATH
  cmdsize 32
     path @executable_path (offset 12)
`,
          };
        },
      },
    );

    expect(result).toMatchObject({
      checked: true,
      patched: false,
      reason: "already-present",
    });
    expect(calls).toEqual(["otool"]);
  });

  it("并发构建替换 app-server 时不把 rpath patch race 当成永久失败", () => {
    const calls = [];
    let exists = true;
    const result = ensureMacBinaryRpath(
      "/repo/lime-rs/target/debug/app-server",
      {
        exists: () => exists,
        getStats: () => ({ isFile: () => true, size: 1 }),
        platform: "darwin",
        runner(command) {
          calls.push(command);
          if (command === "otool") {
            return { status: 0, stdout: "" };
          }
          if (command === "install_name_tool") {
            exists = false;
            return { status: 1 };
          }
          throw new Error(`unexpected command: ${command}`);
        },
      },
    );

    expect(result).toMatchObject({
      checked: false,
      patched: false,
      reason: "missing-binary-after-rpath-race",
    });
    expect(calls).toEqual(["otool", "install_name_tool"]);
  });

  it("并发构建已补 rpath 时不把 duplicate rpath 当成失败", () => {
    const calls = [];
    const result = ensureMacBinaryRpath(
      "/repo/lime-rs/target/debug/app-server",
      {
        exists: () => true,
        getStats: () => ({ isFile: () => true, size: 1 }),
        platform: "darwin",
        runner(command) {
          calls.push(command);
          if (command === "otool" && calls.length === 1) {
            return { status: 0, stdout: "" };
          }
          if (command === "install_name_tool") {
            return { status: 1 };
          }
          if (command === "otool") {
            return {
              status: 0,
              stdout: `
Load command 1
      cmd LC_RPATH
  cmdsize 32
     path @executable_path (offset 12)
`,
            };
          }
          throw new Error(`unexpected command: ${command}`);
        },
      },
    );

    expect(result).toMatchObject({
      checked: true,
      patched: false,
      reason: "already-present-after-rpath-race",
      rpaths: [MACOS_EXECUTABLE_RPATH],
    });
    expect(calls).toEqual(["otool", "install_name_tool", "otool"]);
  });

  it("空 app-server 二进制不执行 install_name_tool", () => {
    const result = ensureMacBinaryRpath(
      "/repo/lime-rs/target/debug/app-server",
      {
        exists: () => true,
        getStats: () => ({ isFile: () => true, size: 0 }),
        platform: "darwin",
        runner() {
          throw new Error("should not run");
        },
      },
    );

    expect(result).toMatchObject({
      checked: false,
      patched: false,
      reason: "empty-binary",
    });
  });
});
