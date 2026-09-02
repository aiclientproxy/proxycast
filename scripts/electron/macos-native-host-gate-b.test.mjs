import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";

import {
  parseArgs,
  resolveInstalledResourcesRoot,
} from "./macos-native-host-gate-b.mjs";

const roots = [];

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop(), { recursive: true, force: true });
  }
});

describe("macos-native-host-gate-b", () => {
  it("保留 help 解析，不要求在非 macOS 主机执行 Gate B", () => {
    expect(parseArgs(["--help"]).help).toBe(true);
  });

  it("从已安装 Lime.app 可执行文件解析 Resources 根目录", () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-macos-gate-b-"));
    roots.push(root);
    const executable = path.join(root, "Lime.app", "Contents", "MacOS", "Lime");
    const resources = path.join(root, "Lime.app", "Contents", "Resources");
    mkdirSync(resources, { recursive: true });
    expect(resolveInstalledResourcesRoot(executable)).toBe(resources);
  });

  it("Gate B summary 固定记录 helper 协议和候选 identity", () => {
    const source = readFileSync(
      "scripts/electron/macos-native-host-gate-b.mjs",
      "utf8",
    );
    expect(source).toContain("protocolVersion: PROTOCOL_VERSION");
    expect(source).toContain(
      "candidateRunId: process.env.LIME_GATE_RUN_ID?.trim() || null",
    );
    expect(source).toContain("permissionMode");
    expect(source).toContain("bookmark.create-resolve-start-stop");
    expect(source).toContain("window.anchor-stack-hideForTask");
    expect(source).toContain("macos-window-fixture.swift");
    expect(source).toContain(
      "electronExecutable: path.resolve(options.electronExecutable)",
    );
  });

  it("Gate B 必须经过真实 Electron preload/IPC，同时进入 App Server 和 native host", () => {
    const source = readFileSync(
      "scripts/electron/lib/macos-native-host-electron-gate-b.mjs",
      "utf8",
    );
    expect(source).toContain("launchElectronFixture");
    expect(source).toContain("window.electronAPI?.invoke");
    expect(source).toContain(
      'const APP_SERVER_COMMAND = "app_server_handle_json_lines"',
    );
    expect(source).toContain(
      'const NATIVE_HOST_COMMAND = "macos_native_host_invoke"',
    );
    expect(source).toContain(
      "Electron IPC trace did not record app_server_handle_json_lines",
    );
    expect(source).toContain("gui.visible-state");
  });

  it("主入口只负责 native 校验并委托 Electron Gate B helper", () => {
    const source = readFileSync(
      "scripts/electron/macos-native-host-gate-b.mjs",
      "utf8",
    );
    expect(source).toContain(
      'import { runMacOSNativeHostElectronGateB } from "./lib/macos-native-host-electron-gate-b.mjs"',
    );
    expect(source).toContain(
      "electronGate = await runMacOSNativeHostElectronGateB(options)",
    );
    expect(source).toContain("function permissionStatus(capabilities, key)");
    expect(source).not.toContain("launchElectronFixture");
    expect(source).not.toContain("window.electronAPI?.invoke");
  });
});
