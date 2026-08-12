import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  appServerBinaryName,
  buildLocalAppServer,
  codeModeHostBinaryName,
  isUsableAppServerBinary,
  localAppServerBinaryPath,
  localCodeModeHostBinaryPath,
  resolveDevAppServerBackendEnv,
  resolveDevAppServerBinary,
  resolveCargoTargetDirectory,
  shouldRebuildAppServer,
} from "./electron-dev-sidecar.mjs";

describe("electron dev sidecar", () => {
  it("按平台解析 app-server 二进制名称", () => {
    expect(appServerBinaryName("darwin")).toBe("app-server");
    expect(appServerBinaryName("linux")).toBe("app-server");
    expect(appServerBinaryName("win32")).toBe("app-server.exe");
    expect(codeModeHostBinaryName("darwin")).toBe("code-mode-host");
    expect(codeModeHostBinaryName("win32")).toBe("code-mode-host.exe");
  });

  it("默认解析仓库内 debug app-server 路径", () => {
    expect(
      localAppServerBinaryPath({
        repoRoot: "/repo/lime",
        platform: "darwin",
        targetDirectory: path.resolve("/repo/lime/lime-rs/target"),
      }),
    ).toBe(path.resolve("/repo/lime/lime-rs/target/debug/app-server"));
    expect(
      localCodeModeHostBinaryPath({
        repoRoot: "/repo/lime",
        platform: "darwin",
        targetDirectory: path.resolve("/repo/lime/lime-rs/target"),
      }),
    ).toBe(path.resolve("/repo/lime/lime-rs/target/debug/code-mode-host"));
  });

  it("读取仓库 cargo target-dir 配置", () => {
    expect(
      resolveCargoTargetDirectory({
        repoRoot: "/repo/lime",
        readConfigFile: () => '[build]\ntarget-dir = "lime-rs/target"\n',
      }),
    ).toBe(path.resolve("/repo/lime/lime-rs/target"));
  });

  it("没有 cargo target-dir 配置时回退到 lime-rs/target", () => {
    expect(
      resolveCargoTargetDirectory({
        repoRoot: "/repo/lime",
        readConfigFile: () => "[build]\n",
      }),
    ).toBe(path.resolve("/repo/lime/lime-rs/target"));
  });

  it("优先使用 APP_SERVER_BIN 环境变量", () => {
    const calls = [];
    const resolved = resolveDevAppServerBinary({
      env: { APP_SERVER_BIN: "  /custom/app-server  " },
      exists(pathValue) {
        calls.push(pathValue);
        return false;
      },
      build() {
        throw new Error("should not build");
      },
    });

    expect(resolved).toBe("/custom/app-server");
    expect(calls).toHaveLength(0);
  });

  it("本地二进制存在时直接返回，不触发 cargo build", () => {
    const builds = [];
    const prepared = [];
    const resolved = resolveDevAppServerBinary({
      env: {},
      repoRoot: "/repo/lime",
      platform: "darwin",
      exists: () => true,
      build(call) {
        builds.push(call);
      },
      prepareBinary: (...args) => prepared.push(args),
    });

    expect(resolved).toBe(
      path.resolve("/repo/lime/lime-rs/target/debug/app-server"),
    );
    expect(builds).toHaveLength(0);
    expect(prepared).toEqual([
      [
        {
          binaryPath: path.resolve(
            "/repo/lime/lime-rs/target/debug/app-server",
          ),
          platform: "darwin",
        },
      ],
      [
        {
          binaryPath: path.resolve(
            "/repo/lime/lime-rs/target/debug/code-mode-host",
          ),
          platform: "darwin",
        },
      ],
    ]);
  });

  it("本地二进制缺失时先构建再返回二进制路径", () => {
    let existsCalls = 0;
    const builds = [];
    const prepared = [];
    const resolved = resolveDevAppServerBinary({
      env: {},
      repoRoot: "/repo/lime",
      platform: "darwin",
      exists: () => {
        existsCalls += 1;
        return existsCalls > 1;
      },
      build(call) {
        builds.push(call);
      },
      prepareBinary: (...args) => prepared.push(args),
    });

    expect(resolved).toBe(
      path.resolve("/repo/lime/lime-rs/target/debug/app-server"),
    );
    expect(builds).toEqual([
      { repoRoot: "/repo/lime", platform: "darwin", env: {} },
    ]);
    expect(prepared).toEqual([
      [
        {
          binaryPath: path.resolve(
            "/repo/lime/lime-rs/target/debug/app-server",
          ),
          platform: "darwin",
        },
      ],
      [
        {
          binaryPath: path.resolve(
            "/repo/lime/lime-rs/target/debug/code-mode-host",
          ),
          platform: "darwin",
        },
      ],
    ]);
  });

  it("构建后仍缺失二进制时报错", () => {
    expect(() =>
      resolveDevAppServerBinary({
        env: {},
        repoRoot: "/repo/lime",
        platform: "darwin",
        exists: () => false,
        build() {},
      }),
    ).toThrow(/app-server sidecar binaries were not created/);
  });

  it("空 app-server 文件不算可用二进制", () => {
    expect(
      isUsableAppServerBinary("/repo/lime/lime-rs/target/debug/app-server", {
        exists: () => true,
        getStats: () => ({ isFile: () => true, size: 0 }),
      }),
    ).toBe(false);
    expect(
      isUsableAppServerBinary("/repo/lime/lime-rs/target/debug/app-server", {
        exists: () => true,
        getStats: () => ({ isFile: () => true, size: 128 }),
      }),
    ).toBe(true);
  });

  it("watcher 不应被 Cargo target 产物触发重建", () => {
    expect(
      shouldRebuildAppServer(
        "target/debug/build/mime_guess/out/mime_types_generated.rs",
      ),
    ).toBe(false);
    expect(
      shouldRebuildAppServer("crates/agent/src/tools/skill_tool_gate.rs"),
    ).toBe(true);
    expect(shouldRebuildAppServer("Cargo.toml")).toBe(true);
  });

  it("默认 dev App Server backend 使用 App Server 内部 runtime", () => {
    expect(
      resolveDevAppServerBackendEnv({
        env: {},
      }),
    ).toEqual({
      APP_SERVER_BACKEND_MODE: "runtime",
    });
  });

  it("显式 unavailable 时 dev App Server backend 不接 external", () => {
    const env = { APP_SERVER_BACKEND_MODE: "unavailable" };
    expect(
      resolveDevAppServerBackendEnv({
        env,
        backendCommand: "/repo/backend",
      }),
    ).toEqual({
      APP_SERVER_BACKEND_MODE: "unavailable",
    });
  });

  it("显式 external backend command 时保留用户配置", () => {
    expect(
      resolveDevAppServerBackendEnv({
        env: {
          APP_SERVER_BACKEND_MODE: "external",
          APP_SERVER_BACKEND_COMMAND: "/custom/backend",
          APP_SERVER_BACKEND_TIMEOUT_MS: "90000",
        },
        backendCommand: "/repo/backend",
      }),
    ).toEqual({
      APP_SERVER_BACKEND_MODE: "external",
    });
  });

  it("显式 external backend command 时可由 dev sidecar 注入命令", () => {
    expect(
      resolveDevAppServerBackendEnv({
        env: {
          APP_SERVER_BACKEND_MODE: "external",
        },
        backendCommand: "/repo/backend",
      }),
    ).toEqual({
      APP_SERVER_BACKEND_MODE: "external",
      APP_SERVER_BACKEND_COMMAND: "/repo/backend",
      APP_SERVER_BACKEND_TIMEOUT_MS: "120000",
    });
  });

  it("调用 cargo build 成组构建 app-server 与 code-mode host", () => {
    const calls = [];
    const prepared = [];
    buildLocalAppServer({
      repoRoot: "/repo/lime",
      platform: "darwin",
      arch: "arm64",
      env: { PATH: "/bin" },
      runner(command, args, options) {
        calls.push({ command, args, options });
        return { status: 0 };
      },
      prepareBinary: (...args) => prepared.push(args),
      resolveV8Env(options) {
        expect(options).toEqual({
          env: { PATH: "/bin" },
          repoRoot: "/repo/lime",
          platform: "darwin",
          arch: "arm64",
        });
        return {
          RUSTY_V8_ARCHIVE: "/cache/v8.a.gz",
          RUSTY_V8_SRC_BINDING_PATH: "/cache/src_binding.rs",
        };
      },
    });

    expect(calls).toEqual([
      {
        command: "cargo",
        args: [
          "build",
          "--manifest-path",
          path.resolve("/repo/lime/lime-rs/Cargo.toml"),
          "-p",
          "app-server",
          "--bin",
          "app-server",
          "-p",
          "tool-runtime",
          "--bin",
          "code-mode-host",
        ],
        options: {
          cwd: "/repo/lime",
          env: {
            PATH: "/bin",
            RUSTY_V8_ARCHIVE: "/cache/v8.a.gz",
            RUSTY_V8_SRC_BINDING_PATH: "/cache/src_binding.rs",
          },
          stdio: "inherit",
          shell: false,
        },
      },
    ]);
    expect(prepared).toEqual([
      [
        {
          binaryPath: path.resolve(
            "/repo/lime/lime-rs/target/debug/app-server",
          ),
          platform: "darwin",
        },
      ],
      [
        {
          binaryPath: path.resolve(
            "/repo/lime/lime-rs/target/debug/code-mode-host",
          ),
          platform: "darwin",
        },
      ],
    ]);
  });

  it("cargo build 失败时抛出错误", () => {
    expect(() =>
      buildLocalAppServer({
        repoRoot: "/repo/lime",
        env: {},
        runner: () => ({ status: 2 }),
        resolveV8Env: () => ({}),
      }),
    ).toThrow("cargo build app-server failed with 2");
  });
});
