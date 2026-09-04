import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

function readPlistString(content, key) {
  return (
    content.match(
      new RegExp(`<key>${key}</key>\\s*<string>([^<]+)</string>`, "u"),
    )?.[1] ?? null
  );
}

export function verifyCodeSignature(
  bundlePath,
  execFileSyncImpl = execFileSync,
) {
  try {
    execFileSyncImpl("codesign", ["--verify", "--strict", bundlePath], {
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

export function readMacOSAppIdentity(
  electronExecutable,
  { applicationId, execFileSyncImpl = execFileSync },
) {
  const executable = path.resolve(electronExecutable);
  const appBundlePath = executable.match(
    /^(.*\.app)[/\\]Contents[/\\]MacOS[/\\][^/\\]+$/u,
  )?.[1];
  if (!appBundlePath) {
    throw new Error("Installed macOS executable is not inside a .app bundle");
  }
  const infoPlistPath = path.join(appBundlePath, "Contents", "Info.plist");
  if (!existsSync(infoPlistPath)) {
    throw new Error(
      `Installed macOS app Info.plist is missing: ${infoPlistPath}`,
    );
  }
  const infoPlist = readFileSync(infoPlistPath, "utf8");
  const bundleIdentifier = readPlistString(infoPlist, "CFBundleIdentifier");
  const executableName = readPlistString(infoPlist, "CFBundleExecutable");
  const version = readPlistString(infoPlist, "CFBundleShortVersionString");
  if (bundleIdentifier !== applicationId || executableName !== "Lime") {
    throw new Error("Installed macOS app bundle identity is invalid");
  }
  const signed = verifyCodeSignature(appBundlePath, execFileSyncImpl);
  if (!signed) {
    throw new Error("Installed macOS app signature is invalid");
  }
  return {
    appBundlePath,
    infoPlistPath,
    bundleIdentifier,
    executableName,
    version,
    signed,
  };
}

export function parseDeveloperIdSignatureIdentity(output, bundlePath) {
  const authority =
    String(output)
      .match(/^Authority=(.+)$/mu)?.[1]
      ?.trim() || null;
  const teamIdentifier =
    String(output)
      .match(/^TeamIdentifier=(.+)$/mu)?.[1]
      ?.trim() || null;
  if (!authority?.startsWith("Developer ID Application:")) {
    throw new Error(`Release bundle is not Developer ID signed: ${bundlePath}`);
  }
  if (!teamIdentifier || teamIdentifier === "not set") {
    throw new Error(
      `Release bundle has no signing team identity: ${bundlePath}`,
    );
  }
  return { authority, teamIdentifier };
}

function readCodeSignatureIdentity(bundlePath, spawnSyncImpl) {
  const result = spawnSyncImpl(
    "codesign",
    ["--display", "--verbose=4", bundlePath],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    throw new Error(`Unable to read code signature identity: ${bundlePath}`);
  }
  return parseDeveloperIdSignatureIdentity(
    `${result.stdout || ""}\n${result.stderr || ""}`,
    bundlePath,
  );
}

export function verifyMacOSReleaseTrust(
  appBundlePath,
  helperBundlePath,
  { execFileSyncImpl = execFileSync, spawnSyncImpl = spawnSync } = {},
) {
  execFileSyncImpl(
    "codesign",
    ["--verify", "--deep", "--strict", appBundlePath],
    { stdio: "pipe" },
  );
  execFileSyncImpl("codesign", ["--verify", "--strict", helperBundlePath], {
    stdio: "pipe",
  });
  const application = readCodeSignatureIdentity(appBundlePath, spawnSyncImpl);
  const helper = readCodeSignatureIdentity(helperBundlePath, spawnSyncImpl);
  if (application.teamIdentifier !== helper.teamIdentifier) {
    throw new Error("Application and native helper signing teams do not match");
  }
  execFileSyncImpl(
    "spctl",
    ["--assess", "--type", "execute", "--verbose=4", appBundlePath],
    { stdio: "pipe" },
  );
  execFileSyncImpl("xcrun", ["stapler", "validate", appBundlePath], {
    stdio: "pipe",
  });
  return {
    status: "passed",
    developerId: true,
    gatekeeperAccepted: true,
    stapleValid: true,
    teamIdentifier: application.teamIdentifier,
    applicationAuthority: application.authority,
    helperAuthority: helper.authority,
  };
}
