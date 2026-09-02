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
  defaultRustyV8CacheRoot,
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
  it("uses stable OS-native cache roots", () => {
    expect(
      defaultRustyV8CacheRoot({
        env: {},
        platform: "darwin",
        homeDirectory: "/Users/tester",
      }),
    ).toBe("/Users/tester/Library/Caches/Lime/rusty-v8");
    expect(
      defaultRustyV8CacheRoot({
        env: { LOCALAPPDATA: "C:\\Users\\tester\\AppData\\Local" },
        platform: "win32",
        homeDirectory: "C:\\Users\\tester",
      }),
    ).toBe("C:\\Users\\tester\\AppData\\Local\\Lime\\Cache\\rusty-v8");
    expect(
      defaultRustyV8CacheRoot({
        env: { XDG_CACHE_HOME: "/tmp/cache" },
        platform: "linux",
        homeDirectory: "/home/tester",
      }),
    ).toBe("/tmp/cache/lime/rusty-v8");
    expect(
      defaultRustyV8CacheRoot({
        env: { LIME_RUSTY_V8_CACHE_DIR: "/tmp/lime-v8" },
        platform: "darwin",
        homeDirectory: "/Users/tester",
      }),
    ).toBe("/tmp/lime-v8");
  });

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

  it("aligns Windows Cargo builds with the artifact static CRT", () => {
    const repoRoot = temporaryDirectory();
    const cacheRoot = temporaryDirectory();
    const archive = Buffer.from("windows archive");
    const binding = Buffer.from("windows binding");
    const names = rustyV8ArtifactNames("x86_64-pc-windows-msvc");
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
      platform: "win32",
      arch: "x64",
      cacheRoot,
      download(url, destination) {
        writeFileSync(destination, downloads.get(path.basename(url)));
      },
    });

    expect(env.CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS).toBe(
      "-C target-feature=+crt-static",
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
    const downloadUrls = [];
    const env = resolveRustyV8CargoEnv({
      env: {},
      repoRoot,
      platform: "darwin",
      arch: "arm64",
      cacheRoot,
      download(url, destination) {
        downloadUrls.push(url);
        writeFileSync(destination, downloads.get(path.basename(url)));
      },
    });

    expect(readFileSync(env.RUSTY_V8_ARCHIVE)).toEqual(archive);
    expect(readFileSync(env.RUSTY_V8_SRC_BINDING_PATH)).toEqual(binding);
    expect(downloadUrls).toEqual([
      "https://github.com/openai/codex/releases/download/rusty-v8-v150.4.0/rusty_v8_ptrcomp_sandbox_release_aarch64-apple-darwin.sha256",
      "https://github.com/openai/codex/releases/download/rusty-v8-v150.4.0/librusty_v8_ptrcomp_sandbox_release_aarch64-apple-darwin.a.gz",
      "https://github.com/openai/codex/releases/download/rusty-v8-v150.4.0/src_binding_ptrcomp_sandbox_release_aarch64-apple-darwin.rs",
    ]);
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
