import { describe, expect, it } from "vitest";

import forgeConfig, {
  ignorePackagerInput,
  macNotarizeOptions,
  macSignOptions,
  macZipConfig,
  squirrelConfig,
  updateFeedLabel,
  updateFeedUrl,
  windowsSigningOptions,
  windowsSquirrelRemoteReleasesOptions,
  windowsSquirrelRemoteReleasesUrl,
} from "../../forge.config.mjs";

describe("Electron Forge config", () => {
  it("keeps Forge official makers as the packaging fact source", () => {
    expect(forgeConfig.outDir).toBe(
      process.env.LIME_ELECTRON_FORGE_OUT_DIR || "release-electron",
    );
    expect(forgeConfig.makers.map((maker) => maker.name)).toEqual([
      "dmg",
      "zip",
      "squirrel",
    ]);
    expect(forgeConfig.makers[0].platformsToMakeOn).toEqual(["darwin"]);
    expect(forgeConfig.makers[1].platformsToMakeOn).toEqual(["darwin"]);
    expect(forgeConfig.makers[2].platformsToMakeOn).toEqual(["win32"]);
  });

  it("builds formal macOS signing and notarization options from release env", () => {
    const env = {
      APPLE_ID: "release@example.com",
      APPLE_PASSWORD: "app-specific-password",
      APPLE_SIGNING_IDENTITY: "Developer ID Application: Lime",
      APPLE_TEAM_ID: "TEAM123456",
      LIME_ELECTRON_SIGN: "1",
      LIME_MACOS_KEYCHAIN: "/tmp/lime.keychain-db",
    };

    expect(macSignOptions({ env, platform: "darwin" })).toEqual({
      continueOnError: false,
      identity: "Developer ID Application: Lime",
      identityValidation: true,
      keychain: "/tmp/lime.keychain-db",
      optionsForFile: expect.any(Function),
      preAutoEntitlements: false,
      preEmbedProvisioningProfile: false,
      strictVerify: true,
    });
    expect(
      macSignOptions({ env, platform: "darwin" }).optionsForFile(
        "release-electron/Lime-darwin-arm64/Lime.app",
      ),
    ).toEqual({
      entitlements: "lime-rs/entitlements.plist",
      hardenedRuntime: true,
      signatureFlags: ["runtime"],
    });
    expect(
      macSignOptions({ env, platform: "darwin" }).optionsForFile(
        "release-electron/Lime-darwin-arm64/Lime.app/Contents/MacOS/Lime",
      ),
    ).toBeNull();
    expect(macNotarizeOptions({ env, platform: "darwin" })).toEqual({
      appleId: "release@example.com",
      appleIdPassword: "app-specific-password",
      teamId: "TEAM123456",
    });
    expect(macSignOptions({ env, platform: "win32" })).toBeUndefined();
    expect(
      macNotarizeOptions({
        env: { ...env, APPLE_PASSWORD: "" },
        platform: "darwin",
      }),
    ).toBeUndefined();
  });

  it("ad-hoc signs local macOS packages without enabling notarization", () => {
    expect(macSignOptions({ env: {}, platform: "darwin" })).toEqual({
      continueOnError: false,
      identity: "-",
      identityValidation: false,
      optionsForFile: expect.any(Function),
      preAutoEntitlements: false,
      preEmbedProvisioningProfile: false,
      strictVerify: true,
    });
    expect(
      macSignOptions({ env: {}, platform: "darwin" }).optionsForFile(
        "release-electron/Lime-darwin-arm64/Lime.app",
      ),
    ).toEqual({
      entitlements: "lime-rs/entitlements.plist",
      hardenedRuntime: false,
      timestamp: "none",
    });
    expect(
      macSignOptions({ env: {}, platform: "darwin" }).optionsForFile(
        "release-electron/Lime-darwin-arm64/Lime.app/Contents/MacOS/Lime",
      ),
    ).toEqual({ hardenedRuntime: false, timestamp: "none" });
    expect(macNotarizeOptions({ env: {}, platform: "darwin" })).toBeUndefined();
  });

  it("runs macOS branding before signing and notarization", () => {
    expect(forgeConfig.packagerConfig.afterComplete).toBeUndefined();
    expect(forgeConfig.packagerConfig.afterCopyExtraResources).toHaveLength(1);
    expect(
      forgeConfig.packagerConfig.afterCopyExtraResources[0].length,
    ).toBeGreaterThanOrEqual(5);
  });

  it("builds updater feed URLs from explicit env or platform feed labels", () => {
    expect(updateFeedLabel("darwin", "arm64")).toBe("darwin-arm64");
    expect(updateFeedLabel("darwin", "x64")).toBe("darwin-x64");
    expect(updateFeedLabel("win32", "x64")).toBe("win32-x64");
    expect(
      updateFeedUrl("darwin", "arm64", {
        env: { LIME_UPDATES_BASE_URL: "https://updates.example/" },
      }),
    ).toBe("https://updates.example/lime/stable/darwin-arm64");
    expect(
      updateFeedUrl("win32", "x64", {
        env: { LIME_ELECTRON_UPDATES_URL: "https://feed.example/win32-x64/" },
      }),
    ).toBe("https://feed.example/win32-x64");
    expect(
      macZipConfig("x64", {
        env: { LIME_UPDATES_BASE_URL: "https://updates.example" },
      }),
    ).toEqual({
      macUpdateManifestBaseUrl:
        "https://updates.example/lime/stable/darwin-x64",
    });
  });

  it("maps Windows GitHub Actions PFX env into Forge Squirrel signing config", () => {
    const env = {
      LIME_ELECTRON_SIGN: "1",
      LIME_UPDATES_BASE_URL: "https://updates.example/",
      LIME_WINDOWS_SIGNING_CERTIFICATE_FILE: " C:/certs/lime.pfx ",
      LIME_WINDOWS_SIGNING_CERTIFICATE_PASSWORD: "secret",
    };

    expect(windowsSigningOptions({ env, platform: "win32" })).toEqual({
      certificateFile: "C:/certs/lime.pfx",
      certificatePassword: "secret",
    });
    expect(
      squirrelConfig("x64", {
        env,
        packageVersion: "9.8.7",
        platform: "win32",
      }),
    ).toEqual({
      authors: "Lime",
      certificateFile: "C:/certs/lime.pfx",
      certificatePassword: "secret",
      exe: "Lime.exe",
      name: "lime",
      noMsi: true,
      setupExe: "Lime-9.8.7 Setup.exe",
      setupIcon: "lime-rs/icons/icon.ico",
    });
    expect(windowsSigningOptions({ env, platform: "darwin" })).toEqual({});
  });

  it("keeps Windows Squirrel signing optional but paired", () => {
    expect(
      windowsSigningOptions({
        env: {
          LIME_ELECTRON_SIGN: "1",
          LIME_WINDOWS_SIGNING_CERTIFICATE_FILE: "",
          LIME_WINDOWS_SIGNING_CERTIFICATE_PASSWORD: "",
        },
        platform: "win32",
      }),
    ).toEqual({});
    expect(
      squirrelConfig("x64", {
        env: {
          LIME_ELECTRON_SIGN: "1",
          LIME_UPDATES_BASE_URL: "https://updates.example",
        },
        packageVersion: "9.8.7",
        platform: "win32",
      }),
    ).not.toHaveProperty("certificateFile");
    expect(
      windowsSigningOptions({
        env: {
          LIME_ELECTRON_SIGN: "1",
          LIME_WINDOWS_SIGNING_CERTIFICATE_FILE: "C:/certs/lime.pfx",
          LIME_WINDOWS_SIGNING_CERTIFICATE_PASSWORD: "",
        },
        platform: "win32",
      }),
    ).toEqual({});
  });

  it("keeps Windows Squirrel remote release sync opt-in", () => {
    const env = {
      LIME_ELECTRON_UPDATES_URL: "https://feed.example/win32-x64/",
      LIME_UPDATES_BASE_URL: "https://updates.example/",
      LIME_WINDOWS_SQUIRREL_REMOTE_RELEASES_URL:
        " https://updates.example/lime/stable/win32-x64/ ",
    };

    expect(windowsSquirrelRemoteReleasesUrl({ env: {} })).toBeUndefined();
    expect(windowsSquirrelRemoteReleasesUrl({ env })).toBe(
      "https://updates.example/lime/stable/win32-x64",
    );
    expect(windowsSquirrelRemoteReleasesOptions({ env })).toEqual({
      remoteReleases: "https://updates.example/lime/stable/win32-x64",
    });
    expect(
      squirrelConfig("x64", {
        env: { LIME_ELECTRON_UPDATES_URL: "https://feed.example/win32-x64/" },
        packageVersion: "9.8.7",
        platform: "win32",
      }),
    ).not.toHaveProperty("remoteReleases");
    expect(
      squirrelConfig("x64", {
        env,
        packageVersion: "9.8.7",
        platform: "win32",
      }),
    ).toMatchObject({
      remoteReleases: "https://updates.example/lime/stable/win32-x64",
    });
  });

  it("keeps required packaged app inputs while ignoring repository-only sources", () => {
    expect(ignorePackagerInput(`${process.cwd()}/package.json`)).toBe(false);
    expect(ignorePackagerInput(`${process.cwd()}/dist/index.html`)).toBe(false);
    expect(ignorePackagerInput(`${process.cwd()}/dist-electron/main`)).toBe(
      false,
    );
    expect(
      ignorePackagerInput(`${process.cwd()}/dist-electron/preload/index.js`),
    ).toBe(false);
    expect(ignorePackagerInput(`${process.cwd()}/src/App.tsx`)).toBe(true);
    expect(
      ignorePackagerInput(`${process.cwd()}/scripts/electron/smoke.mjs`),
    ).toBe(true);
  });

  it("packages the bundled Plugin marketplace beside the app-server resources", () => {
    expect(forgeConfig.packagerConfig.extraResource).toContain(
      "dist-electron/plugins",
    );
  });
});
