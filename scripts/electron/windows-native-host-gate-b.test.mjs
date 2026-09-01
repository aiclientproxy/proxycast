import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";

import {
  parseArgs,
  resolveInstalledResourcesRoot,
} from "./windows-native-host-gate-b.mjs";

const roots = [];

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop(), { recursive: true, force: true });
  }
});

describe("windows-native-host-gate-b", () => {
  it("保留 help 解析，不要求在本机启动 Windows runner", () => {
    expect(parseArgs(["--help"]).help).toBe(true);
  });

  it("从已安装 Electron 可执行文件解析 resources 根目录", () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-windows-gate-b-"));
    roots.push(root);
    const executable = path.join(root, "app-1.0.0", "Lime.exe");
    const resources = path.join(root, "app-1.0.0", "resources");
    mkdirSync(resources, { recursive: true });
    expect(resolveInstalledResourcesRoot(executable)).toBe(resources);
  });
});
