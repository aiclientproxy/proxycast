import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, vi } from "vitest";
import YAML from "yaml";

import {
  buildNMinusOneLaunchEnv,
  buildWaitForWindowsProcessExitScript,
  buildWindowsRcSummary,
  compareVersions,
  findReadyElectronUpdaterPage,
  isFinalElectronRendererUrl,
  normalizeVersion,
  resolveInstalledSquirrelPaths,
  resolveSquirrelFeed,
  selectNMinusOneVersion,
  selectSquirrelInstaller,
  waitForWindowsProcessExit,
} from "./windows-squirrel-rc-smoke.mjs";

describe("Windows Squirrel RC smoke", () => {
  it("等待最终 renderer，不能把带 preload 的临时启动页当成 updater 页面", () => {
    expect(
      isFinalElectronRendererUrl(
        "file:///C:/Users/runner/AppData/Roaming/Lime/startup/main-window-startup.html",
      ),
    ).toBe(false);
    expect(
      isFinalElectronRendererUrl(
        "file:///C:/Users/runner/AppData/Local/lime/app-1.106.0/resources/app.asar/dist/index.html?nativeStartup=1",
      ),
    ).toBe(true);
    expect(isFinalElectronRendererUrl("about:blank")).toBe(false);
  });

  it("updater 页面选择跳过 bridge 已就绪但仍会导航的临时启动页", async () => {
    const startupPage = {
      evaluate: vi.fn().mockResolvedValue(true),
      url: () =>
        "file:///C:/Users/runner/AppData/Roaming/Lime/startup/main-window-startup.html",
    };
    const rendererPage = {
      evaluate: vi.fn().mockResolvedValue(true),
      url: () =>
        "file:///C:/Users/runner/AppData/Local/lime/app-1.106.0/resources/app.asar/dist/index.html?nativeStartup=1",
    };

    await expect(
      findReadyElectronUpdaterPage([startupPage, rendererPage]),
    ).resolves.toBe(rendererPage);
    expect(startupPage.evaluate).not.toHaveBeenCalled();
    expect(rendererPage.evaluate).toHaveBeenCalledTimes(1);
  });

  it("packaged N-1 启动环境移除 Electron 不支持的 NODE_OPTIONS", () => {
    const env = buildNMinusOneLaunchEnv({
      baseEnv: {
        NODE_OPTIONS: "--max-old-space-size=8192",
        PATH: "C:\\Windows\\System32",
        VITE_DEV_SERVER_URL: "http://127.0.0.1:5173",
      },
      feedUrl: "http://127.0.0.1:49152",
      userDataDir: "C:\\Temp\\lime-updater",
    });

    expect(env).not.toHaveProperty("NODE_OPTIONS");
    expect(env).not.toHaveProperty("VITE_DEV_SERVER_URL");
    expect(env).toEqual(
      expect.objectContaining({
        APP_SERVER_BIN: "",
        ELECTRON_E2E_USER_DATA_DIR: "C:\\Temp\\lime-updater",
        LIME_ELECTRON_ENABLE_DEV_UPDATER: "1",
        LIME_ELECTRON_UPDATES_URL: "http://127.0.0.1:49152",
        PATH: "C:\\Windows\\System32",
      }),
    );
  });

  it("N-1 启动前应等待安装器遗留的 Squirrel Update.exe 退出", async () => {
    const runProcessImpl = vi.fn().mockResolvedValue({ exitCode: 0 });

    await expect(
      waitForWindowsProcessExit(
        "C:\\Users\\runner\\AppData\\Local\\lime\\Update.exe",
        {
          runProcessImpl,
          timeoutMs: 12_000,
        },
      ),
    ).resolves.toEqual({
      executable: "C:\\Users\\runner\\AppData\\Local\\lime\\Update.exe",
      exitCode: 0,
      timeoutMs: 12_000,
    });

    expect(runProcessImpl).toHaveBeenCalledWith(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        buildWaitForWindowsProcessExitScript(),
      ],
      expect.objectContaining({
        env: expect.objectContaining({
          LIME_PROCESS_WAIT_TIMEOUT_MS: "12000",
          LIME_TARGET_EXECUTABLE:
            "C:\\Users\\runner\\AppData\\Local\\lime\\Update.exe",
        }),
        timeoutMs: 17_000,
      }),
    );
    expect(buildWaitForWindowsProcessExitScript()).toContain(
      "Get-CimInstance Win32_Process",
    );
    expect(buildWaitForWindowsProcessExitScript()).toContain(
      "Start-Sleep -Milliseconds 250",
    );

    const source = fs.readFileSync(
      "scripts/electron/lib/windows-squirrel-n-minus-one.mjs",
      "utf8",
    );
    expect(source.indexOf("await waitForWindowsProcessExit(")).toBeGreaterThan(
      -1,
    );
    expect(source.indexOf("await waitForWindowsProcessExit(")).toBeLessThan(
      source.indexOf("const child = spawn("),
    );
  });

  it("Squirrel Update.exe 未在截止时间退出时应 fail closed", async () => {
    const runProcessImpl = vi.fn().mockResolvedValue({ exitCode: 1 });

    await expect(
      waitForWindowsProcessExit("C:\\runner\\Update.exe", {
        runProcessImpl,
        timeoutMs: 1_000,
      }),
    ).rejects.toThrow(
      "timed out waiting for process exit at C:\\runner\\Update.exe: exit 1",
    );
  });

  it("N-1 更新应观察应用自动检查且不得主动触发第二次 native check", () => {
    const source = fs.readFileSync(
      "scripts/electron/lib/windows-squirrel-n-minus-one.mjs",
      "utf8",
    );

    expect(source).toContain('label: "N-1 automatic update check"');
    expect(source).toContain('session.stage !== "idle"');
    expect(source).not.toContain(
      'window.electronAPI.invoke("check_for_updates")',
    );
    expect(source.indexOf('label: "N-1 automatic update check"')).toBeLessThan(
      source.indexOf('label: "candidate update download terminal"'),
    );
  });

  it("只选择当前候选版本的 Forge Squirrel installer", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "squirrel-rc-assets-"));
    fs.mkdirSync(path.join(root, "make", "squirrel.windows", "x64"), {
      recursive: true,
    });
    const current = path.join(
      root,
      "make",
      "squirrel.windows",
      "x64",
      "Lime-1.2.3 Setup.exe",
    );
    fs.writeFileSync(current, "current");
    fs.writeFileSync(path.join(root, "Lime-1.2.2 Setup.exe"), "stale");

    expect(
      selectSquirrelInstaller({ installerDir: root, version: "v1.2.3" }),
    ).toBe(current);
  });

  it("接受 GitHub Release 使用的点号 Squirrel installer 名称", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "squirrel-release-assets-"),
    );
    const installer = path.join(root, "Lime-1.2.2.Setup.exe");
    fs.writeFileSync(installer, "n-minus-one");

    expect(
      selectSquirrelInstaller({ installerDir: root, version: "1.2.2" }),
    ).toBe(installer);
  });

  it("N-1 版本必须严格小于候选版本", () => {
    expect(compareVersions("1.105.0", "1.106.0")).toBe(-1);
    expect(compareVersions("1.106.0", "1.106.0")).toBe(0);
    expect(compareVersions("1.107.0", "1.106.0")).toBe(1);
  });

  it("从稳定 tag 中选择严格小于候选的最近版本", () => {
    expect(
      selectNMinusOneVersion({
        candidateVersion: "1.106.0",
        tags: ["v1.104.0", "v1.106.0", "v1.105.0", "v1.106.0-rc.1"],
      }),
    ).toBe("1.105.0");
  });

  it("候选 feed 必须由 RELEASES 精确引用 full nupkg", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "squirrel-feed-"));
    const packageName = "lime-1.2.3-full.nupkg";
    fs.writeFileSync(path.join(root, packageName), "candidate");
    fs.writeFileSync(
      path.join(root, "RELEASES"),
      `${"a".repeat(40)} ${packageName} 9\n`,
    );

    expect(resolveSquirrelFeed({ feedDir: root, version: "1.2.3" })).toEqual(
      expect.objectContaining({
        entries: [expect.objectContaining({ fileName: packageName, size: 9 })],
      }),
    );
    expect(() =>
      resolveSquirrelFeed({ feedDir: root, version: "1.2.4" }),
    ).toThrow("does not reference lime-1.2.4-full.nupkg");
  });

  it("安装路径锁定当前版本，不接受 stale app 目录", () => {
    expect(
      resolveInstalledSquirrelPaths({
        localAppData: "/runner/local-app-data",
        version: "1.2.3",
      }),
    ).toEqual({
      appDirectory: "/runner/local-app-data/lime/app-1.2.3",
      executable: "/runner/local-app-data/lime/app-1.2.3/Lime.exe",
      packageRoot: "/runner/local-app-data/lime",
      updateExecutable: "/runner/local-app-data/lime/Update.exe",
    });
  });

  it("L8 summary 不把单版本安装启动冒充 updater 或 soak", () => {
    const summary = buildWindowsRcSummary({
      assertions: { installerExitZero: true, shell01Passed: true },
      completedAt: "2026-07-17T02:00:00.000Z",
      evidence: {},
      runId: "windows-rc-1",
      startedAt: "2026-07-17T01:00:00.000Z",
      version: normalizeVersion("v1.2.3"),
    });

    expect(summary.result).toBe("pass");
    expect(summary.proofLevel).toBe("L8 platform/packaged");
    expect(summary.remainingClaims).toEqual({
      nMinusOneUpdate: "not-exercised",
      longDurationSoak: "not-exercised",
    });
  });

  it("L8 summary 只有完整 N-1 观测才能声明 updater passed", () => {
    const summary = buildWindowsRcSummary({
      assertions: {
        nMinusOneVersionOlder: true,
        nMinusOneInstalled: true,
        candidateFeedServed: true,
        updateDownloaded: true,
        updateInstallRequested: true,
        candidateInstalledByUpdater: true,
      },
      completedAt: "2026-07-17T02:00:00.000Z",
      evidence: {},
      nMinusOneRequested: true,
      runId: "windows-n-minus-one-1",
      startedAt: "2026-07-17T01:00:00.000Z",
      version: "1.2.3",
    });

    expect(summary.result).toBe("pass");
    expect(summary.remainingClaims.nMinusOneUpdate).toBe("passed");
    expect(summary.remainingClaims.longDurationSoak).toBe("not-exercised");
  });

  it("非 Windows runner 明确标记 platform evidence pending，不伪造 fail 或 pass", () => {
    const summary = buildWindowsRcSummary({
      assertions: { windowsRunner: false },
      completedAt: "2026-07-17T02:00:00.000Z",
      error: "PLT-02-windows-squirrel-rc requires a real Windows runner",
      evidence: {},
      runId: "windows-rc-macos-blocked",
      startedAt: "2026-07-17T01:00:00.000Z",
      version: "1.2.3",
    });

    expect(summary.result).toBe("evidence-pending");
    expect(summary.failedStage).toBe("windows-runner");
    expect(summary.blockers).toEqual([
      "PLT-02 requires a real Windows runner; no platform evidence was collected",
    ]);
  });

  it("安装后 Gate B 必须直启 packaged executable 并禁用源码 sidecar override", () => {
    const smoke = fs.readFileSync("scripts/electron/smoke.mjs", "utf8");

    expect(smoke).toContain("LIME_ELECTRON_SMOKE_EXECUTABLE");
    expect(smoke).toContain('{ APP_SERVER_BIN: "" }');
    expect(smoke).toMatch(
      /args:\s*packagedExecutable\s*\?\s*\["--use-mock-keychain"\]/,
    );
    expect(smoke).toContain("shell: packagedExecutable ? false : undefined");
  });

  it("Windows workflows 必须下载 N-1、运行真实更新并上传结构化证据", () => {
    const workflows = [
      {
        job: "build-windows-test",
        path: ".github/workflows/build-windows-test.yml",
      },
      { job: "build", path: ".github/workflows/release.yml" },
    ];

    for (const entry of workflows) {
      const workflow = YAML.parse(fs.readFileSync(entry.path, "utf8"));
      const steps = workflow.jobs[entry.job].steps;
      const resolveSourceRef = steps.find(
        (step) => step.name === "Resolve source ref",
      );
      const checkout = steps.find((step) => step.name === "Checkout");
      const installDependencies = steps.find(
        (step) => step.name === "Install dependencies",
      );
      const pluginPathTests = steps.find(
        (step) => step.name === "Run Windows Agent Plugin path contract tests",
      );
      const download = steps.find(
        (step) => step.name === "Download Windows N-1 Squirrel installer",
      );
      const smoke = steps.find(
        (step) => step.name === "Smoke installed Windows Squirrel candidate",
      );
      const pluginGate = steps.find(
        (step) => step.name === "Run installed Windows Agent Plugin Gate B",
      );
      const codeModeGate = steps.find(
        (step) => step.name === "Run installed Windows CodeMode Gate B",
      );
      const nativeHostGate = steps.find(
        (step) => step.name === "Run installed Windows native host Gate B",
      );
      const packagedEvidence = steps.find(
        (step) => step.name === "Validate Windows packaged Gate B evidence identity",
      );
      const upload = steps.find(
        (step) => step.name === "Upload Windows Squirrel RC evidence",
      );
      const pluginUpload = steps.find(
        (step) => step.name === "Upload Windows Agent Plugin Gate B evidence",
      );
      const pluginPathUpload = steps.find(
        (step) => step.name === "Upload Windows Agent Plugin path contract log",
      );
      const codeModeUpload = steps.find(
        (step) => step.name === "Upload Windows CodeMode Gate B evidence",
      );
      const nativeHostUpload = steps.find(
        (step) => step.name === "Upload Windows native host Gate B evidence",
      );
      const packagedEvidenceUpload = steps.find(
        (step) => step.name === "Upload Windows packaged Gate B evidence identity",
      );

      expect(download?.run).toContain("gh release download");
      expect(download?.run).toContain("selectNMinusOneVersion");
      expect(installDependencies?.run).toContain(
        "pnpm install --frozen-lockfile",
      );
      expect(installDependencies?.run).not.toContain("npm ci");
      if (entry.job === "build-windows-test") {
        expect(resolveSourceRef?.id).toBe("source");
        expect(resolveSourceRef?.env?.SOURCE_REF).toBe(
          "${{ github.event.inputs.source_ref || github.ref }}",
        );
        expect(resolveSourceRef?.run).toContain(
          "repos/$env:GITHUB_REPOSITORY/commits/$encodedRef",
        );
        expect(checkout?.with?.ref).toBe("${{ steps.source.outputs.ref }}");
        expect(pluginPathTests?.run).toContain("cargo test");
        expect(pluginPathTests?.run).toContain(
          '--manifest-path "lime-rs/Cargo.toml"',
        );
        expect(pluginPathTests?.run).toContain("-p lime-mcp");
        expect(pluginPathTests?.run).toContain("agent_plugin_config");
        expect(pluginPathTests?.run).toContain("--nocapture");
      }
      expect(smoke?.run).toContain(
        "scripts/electron/windows-squirrel-rc-smoke.mjs",
      );
      expect(smoke?.run).toContain("--candidate-feed-dir");
      expect(smoke?.run).toContain("--n-minus-one-installer-dir");
      expect(smoke?.run).toContain("--n-minus-one-version");
      expect(upload?.with?.path).toBe(".lime/qc/windows-squirrel-rc");
      if (entry.job === "build-windows-test") {
        expect(pluginGate?.run).toContain(
          "npm run smoke:plugin-package-electron-gate-b",
        );
        expect(pluginGate?.run).toContain("--electron-executable");
        expect(pluginGate?.run).toContain(
          ".lime/qc/gui-evidence/plugin-package-electron-gate-b-windows",
        );
        expect(pluginUpload?.with?.path).toBe(
          ".lime/qc/gui-evidence/plugin-package-electron-gate-b-windows",
        );
        expect(pluginPathUpload?.with?.path).toBe(
          "windows-plugin-mcp-contract-tests.log",
        );
        expect(pluginPathUpload?.with?.["if-no-files-found"]).toBe("warn");
        expect(pluginPathUpload?.if).toBe("${{ always() }}");
        expect(codeModeGate?.run).toContain(
          "npm run smoke:code-mode-electron-gate-b",
        );
        expect(codeModeGate?.run).toContain("--electron-executable");
        expect(codeModeGate?.run).toContain(
          ".lime/qc/gui-evidence/code-mode-electron-gate-b-windows",
        );
        expect(codeModeUpload?.with?.path).toBe(
          ".lime/qc/gui-evidence/code-mode-electron-gate-b-windows",
        );
      }
      expect(nativeHostGate?.run).toContain(
        "scripts/electron/windows-native-host-gate-b.mjs",
      );
      expect(nativeHostGate?.run).toContain("--electron-executable");
      expect(nativeHostGate?.run).toContain(
        ".lime/qc/gui-evidence/windows-native-host-gate-b",
      );
      expect(nativeHostUpload?.with?.path).toContain(
        "windows-native-host-gate-b",
      );
      expect(packagedEvidence?.run).toContain(
        "scripts/electron/windows-packaged-evidence.mjs",
      );
      expect(packagedEvidence?.run).toContain("--squirrel-summary");
      expect(packagedEvidence?.run).toContain("--code-mode-summary");
      expect(packagedEvidence?.run).toContain("--native-host-summary");
      expect(packagedEvidenceUpload?.with?.path).toContain(
        "windows-packaged-evidence",
      );
    }
  });

  it("Windows test workflow 必须先准备并验证 packaged sidecar，再进入安装后 Gate B", () => {
    const workflow = YAML.parse(
      fs.readFileSync(".github/workflows/build-windows-test.yml", "utf8"),
    );
    const steps = workflow.jobs["build-windows-test"].steps;
    const sherpa = steps.find(
      (step) => step.name === "Prepare sherpa-onnx runtime",
    );
    const build = steps.find(
      (step) =>
        step.name ===
        "Build Electron Windows test package with app-server and code-mode-host sidecars",
    );
    const installSmoke = steps.find(
      (step) => step.name === "Smoke installed Windows Squirrel candidate",
    );
    const pluginGate = steps.find(
      (step) => step.name === "Run installed Windows Agent Plugin Gate B",
    );
    const codeModeGate = steps.find(
      (step) => step.name === "Run installed Windows CodeMode Gate B",
    );
    const nativeHostGate = steps.find(
      (step) => step.name === "Run installed Windows native host Gate B",
    );
    const packagedEvidence = steps.find(
      (step) => step.name === "Validate Windows packaged Gate B evidence identity",
    );

    expect(sherpa).toBeDefined();
    expect(build).toBeDefined();
    expect(installSmoke).toBeDefined();
    expect(pluginGate).toBeDefined();
    expect(codeModeGate).toBeDefined();
    expect(nativeHostGate).toBeDefined();
    expect(packagedEvidence).toBeDefined();
    expect(sherpa?.run).toContain("scripts/prepare-sherpa-onnx-runtime.mjs");
    expect(sherpa?.run).toContain("x86_64-pc-windows-msvc");
    expect(build?.run).toContain("electron-forge make --platform win32");
    expect(build?.run).toContain("npm run electron:build");
    expect(build?.run).toContain(
      "scripts/electron/verify-package-resources.mjs",
    );
    expect(build?.run).toContain("--platform win32");
    expect(build?.run).toContain("scripts/electron/stage-release-assets.mjs");
    expect(installSmoke?.run).toContain(
      "scripts/electron/windows-squirrel-rc-smoke.mjs",
    );
    expect(pluginGate?.run).toContain(
      "npm run smoke:plugin-package-electron-gate-b",
    );
    expect(nativeHostGate?.run).toContain(
      "scripts/electron/windows-native-host-gate-b.mjs",
    );
    expect(packagedEvidence?.run).toContain(
      "scripts/electron/windows-packaged-evidence.mjs",
    );

    const orderedNames = steps.map((step) => step.name);
    expect(orderedNames.indexOf(sherpa.name)).toBeLessThan(
      orderedNames.indexOf(build.name),
    );
    expect(orderedNames.indexOf(build.name)).toBeLessThan(
      orderedNames.indexOf(installSmoke.name),
    );
    expect(orderedNames.indexOf(installSmoke.name)).toBeLessThan(
      orderedNames.indexOf(pluginGate.name),
    );
    expect(orderedNames.indexOf(pluginGate.name)).toBeLessThan(
      orderedNames.indexOf(codeModeGate.name),
    );
    expect(orderedNames.indexOf(codeModeGate.name)).toBeLessThan(
      orderedNames.indexOf(nativeHostGate.name),
    );
    expect(orderedNames.indexOf(nativeHostGate.name)).toBeLessThan(
      orderedNames.indexOf(packagedEvidence.name),
    );
  });
});
