import { createHash } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  parseRustyV8Checksums,
  resolveRustyV8CargoEnv,
  resolveRustyV8Target,
  resolveV8CrateVersion,
  rustyV8ArtifactNames,
} from "./rusty-v8-artifacts.mjs";

const temporaryDirectories = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { force: true, recursive: true });
  }
});

describe("rusty_v8 artifact supply chain", () => {
  it("maps supported desktop platforms to exact Rust targets", () => {
    expect(resolveRustyV8Target({ platform: "darwin", arch: "arm64" })).toBe(
      "aarch64-apple-darwin",
    );
    expect(resolveRustyV8Target({ platform: "win32", arch: "x64" })).toBe(
      "x86_64-pc-windows-msvc",
    );
    expect(
      resolveRustyV8Target({
        env: { CARGO_BUILD_TARGET: "x86_64-unknown-linux-musl" },
        platform: "linux",
        arch: "x64",
      }),
    ).toBe("x86_64-unknown-linux-musl");
    expect(() =>
      resolveRustyV8Target({ platform: "freebsd", arch: "x64" }),
    ).toThrow(/No sandbox-enabled rusty_v8 artifact/u);
  });

  it("reads one exact v8 version and builds sandbox asset names", () => {
    expect(
      resolveV8CrateVersion(`
[[package]]
name = "other"
version = "1.0.0"

[[package]]
name = "v8"
version = "150.4.0"
`),
    ).toBe("150.4.0");
    expect(rustyV8ArtifactNames("aarch64-apple-darwin")).toEqual({
      archive: "librusty_v8_ptrcomp_sandbox_release_aarch64-apple-darwin.a.gz",
      binding: "src_binding_ptrcomp_sandbox_release_aarch64-apple-darwin.rs",
      checksums: "rusty_v8_ptrcomp_sandbox_release_aarch64-apple-darwin.sha256",
    });
    expect(rustyV8ArtifactNames("x86_64-pc-windows-msvc").archive).toBe(
      "rusty_v8_ptrcomp_sandbox_release_x86_64-pc-windows-msvc.lib.gz",
    );
  });

  it("rejects incomplete, duplicate, and path-bearing checksum manifests", () => {
    const names = ["archive.a.gz", "binding.rs"];
    const digest = "a".repeat(64);
    expect(
      parseRustyV8Checksums(
        `${digest}  archive.a.gz\r\n${"b".repeat(64)}  binding.rs\r\n`,
        names,
      ).get("binding.rs"),
    ).toBe("b".repeat(64));
    expect(() =>
      parseRustyV8Checksums(`${digest}  archive.a.gz\n`, names),
    ).toThrow(/exactly 2/u);
    expect(() =>
      parseRustyV8Checksums(
        `${digest}  archive.a.gz\n${digest}  ..\\binding.rs\n`,
        names,
      ),
    ).toThrow(/Invalid rusty_v8 checksum/u);
  });

  it("downloads and verifies both artifacts before exposing Cargo env", () => {
    const repoRoot = temporaryDirectory();
    const cacheRoot = temporaryDirectory();
    const archive = Buffer.from("archive");
    const binding = Buffer.from("binding");
    const names = rustyV8ArtifactNames("aarch64-apple-darwin");
    mkdirSync(path.join(repoRoot, "lime-rs"));
    writeFileSync(
      path.join(repoRoot, "lime-rs", "Cargo.lock"),
      '[[package]]\nname = "v8"\nversion = "150.4.0"\n',
      { flag: "wx" },
    );
    const downloads = new Map([
      [
        names.checksums,
        Buffer.from(
          `${sha256(archive)}  ${names.archive}\n${sha256(binding)}  ${names.binding}\n`,
        ),
      ],
      [names.archive, archive],
      [names.binding, binding],
    ]);
    const env = resolveRustyV8CargoEnv({
      env: {},
      repoRoot,
      platform: "darwin",
      arch: "arm64",
      cacheRoot,
      download(url, destination) {
        writeFileSync(destination, downloads.get(path.basename(url)));
      },
    });

    expect(readFileSync(env.RUSTY_V8_ARCHIVE)).toEqual(archive);
    expect(readFileSync(env.RUSTY_V8_SRC_BINDING_PATH)).toEqual(binding);
  });
});

function temporaryDirectory() {
  const directory = mkdtempSync(path.join(os.tmpdir(), "lime-rusty-v8-test-"));
  temporaryDirectories.push(directory);
  return directory;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}
