import { describe, expect, it, vi } from "vitest";

import {
  parseDeveloperIdSignatureIdentity,
  verifyCodeSignature,
  verifyMacOSReleaseTrust,
} from "./macos-release-trust.mjs";

const APP_AUTHORITY = "Developer ID Application: Lime Cloud (TEAM123456)";

describe("macOS release trust", () => {
  it("复用严格签名校验供 packaged helper 完整性判断", () => {
    const execFileSyncImpl = vi.fn();
    expect(
      verifyCodeSignature("/Applications/Lime.app", execFileSyncImpl),
    ).toBe(true);
    expect(execFileSyncImpl).toHaveBeenCalledWith(
      "codesign",
      ["--verify", "--strict", "/Applications/Lime.app"],
      { stdio: "ignore" },
    );

    execFileSyncImpl.mockImplementationOnce(() => {
      throw new Error("invalid signature");
    });
    expect(
      verifyCodeSignature("/Applications/Lime.app", execFileSyncImpl),
    ).toBe(false);
  });

  it("parses a Developer ID signature identity", () => {
    expect(
      parseDeveloperIdSignatureIdentity(
        `Authority=${APP_AUTHORITY}\nTeamIdentifier=TEAM123456\n`,
        "/Applications/Lime.app",
      ),
    ).toEqual({
      authority: APP_AUTHORITY,
      teamIdentifier: "TEAM123456",
    });
  });

  it("rejects ad-hoc or identity-less signatures", () => {
    expect(() =>
      parseDeveloperIdSignatureIdentity(
        "Signature=adhoc\nTeamIdentifier=not set\n",
        "/Applications/Lime.app",
      ),
    ).toThrow("not Developer ID signed");
  });

  it("requires app/helper team parity, Gatekeeper, and stapling", () => {
    const execFileSyncImpl = vi.fn();
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      stdout: "",
      stderr: `Authority=${APP_AUTHORITY}\nTeamIdentifier=TEAM123456\n`,
    }));

    expect(
      verifyMacOSReleaseTrust("/Applications/Lime.app", "/helper.app", {
        execFileSyncImpl,
        spawnSyncImpl,
      }),
    ).toMatchObject({
      status: "passed",
      developerId: true,
      gatekeeperAccepted: true,
      stapleValid: true,
      teamIdentifier: "TEAM123456",
    });
    expect(execFileSyncImpl).toHaveBeenCalledWith(
      "spctl",
      [
        "--assess",
        "--type",
        "execute",
        "--verbose=4",
        "/Applications/Lime.app",
      ],
      { stdio: "pipe" },
    );
    expect(execFileSyncImpl).toHaveBeenCalledWith(
      "xcrun",
      ["stapler", "validate", "/Applications/Lime.app"],
      { stdio: "pipe" },
    );
  });

  it("rejects a helper signed by another team", () => {
    const spawnSyncImpl = vi
      .fn()
      .mockReturnValueOnce({
        status: 0,
        stdout: "",
        stderr: `Authority=${APP_AUTHORITY}\nTeamIdentifier=TEAM123456\n`,
      })
      .mockReturnValueOnce({
        status: 0,
        stdout: "",
        stderr:
          "Authority=Developer ID Application: Other (OTHER12345)\n" +
          "TeamIdentifier=OTHER12345\n",
      });

    expect(() =>
      verifyMacOSReleaseTrust("/Applications/Lime.app", "/helper.app", {
        execFileSyncImpl: vi.fn(),
        spawnSyncImpl,
      }),
    ).toThrow("signing teams do not match");
  });
});
