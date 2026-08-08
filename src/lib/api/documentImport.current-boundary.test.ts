import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";
import { readAppServerApiSources } from "../../test/appServerApiSources";

const RETIRED_IMPORT_DOCUMENT_COMMAND = "import_document";
const RETIRED_IMPORT_DOCUMENT_TO_SESSION_COMMAND = "import_document_to_session";
const CURRENT_FS_READ_FILE_METHOD = "fs/readFile";
const RETIRED_FILE_PREVIEW_METHOD = "fileSystem/readFilePreview";

const FORBIDDEN_IMPORT_DOCUMENT_SOURCES = [
  "electron/hostCommands.ts",
  "electron/ipcChannels.ts",
  "src/lib/dev-bridge/commandPolicy.ts",
  "src/lib/dev-bridge/mockPriorityCommands.ts",
  "src/lib/desktop-host/sessionFileMocks.ts",
  "lime-rs/src/app/runner.rs",
  "lime-rs/src/commands/mod.rs",
  "lime-rs/src/dev_bridge/dispatcher.rs",
  "lime-rs/src/dev_bridge/dispatcher/files.rs",
];

const RETIRED_DOCUMENT_IMPORT_WRAPPER_FILES = [
  "lime-rs/src/commands/document_import_cmd.rs",
];

function readRepoFile(path: string): string {
  return readFileSync(resolve(cwd(), path), "utf8");
}

function readOptionalRepoFile(path: string): string {
  const absolutePath = resolve(cwd(), path);
  return existsSync(absolutePath) ? readFileSync(absolutePath, "utf8") : "";
}

function expectStringLiteralAbsent(source: string, literal: string): void {
  expect(source).not.toContain(`"${literal}"`);
  expect(source).not.toContain(`'${literal}'`);
}

describe("Document Import current App Server boundary", () => {
  it("importDocument 应固定走 exact fs/readFile 文件网关", () => {
    const source = readRepoFile("src/lib/api/session-files.ts");

    expect(source).toContain('from "@/lib/api/fileBrowser"');
    expect(source).toContain("readFilePreview(filePath, 2 * 1024 * 1024)");
    expectStringLiteralAbsent(source, RETIRED_FILE_PREVIEW_METHOD);
    expectStringLiteralAbsent(source, RETIRED_IMPORT_DOCUMENT_COMMAND);
    expectStringLiteralAbsent(
      source,
      RETIRED_IMPORT_DOCUMENT_TO_SESSION_COMMAND,
    );
  });

  it("App Server protocol / client 应只保留 exact fs/readFile current 方法", () => {
    const appServerSource = readAppServerApiSources();
    const fileBrowserSource = readRepoFile("src/lib/api/fileBrowser.ts");
    const generatedClientProtocolSource = readRepoFile(
      "packages/app-server-client/src/generated/protocol-types.ts",
    );
    const rustProtocolSource = readRepoFile(
      "lime-rs/crates/app-server-protocol/src/protocol/v2/methods.rs",
    );

    expect(appServerSource).toContain("APP_SERVER_METHOD_FS_READ_FILE");
    expect(appServerSource).toContain("readFile(");
    expect(fileBrowserSource).toContain(".readFile({ path })");
    expect(generatedClientProtocolSource).toContain(
      `"${CURRENT_FS_READ_FILE_METHOD}"`,
    );
    expect(rustProtocolSource).toContain(`"${CURRENT_FS_READ_FILE_METHOD}"`);
    expectStringLiteralAbsent(appServerSource, RETIRED_FILE_PREVIEW_METHOD);
    expectStringLiteralAbsent(
      generatedClientProtocolSource,
      RETIRED_FILE_PREVIEW_METHOD,
    );
    expectStringLiteralAbsent(rustProtocolSource, RETIRED_FILE_PREVIEW_METHOD);
  });

  it("旧 Document Import facade 不应回到 Electron、DevBridge、mock 或 legacy Rust", () => {
    const restrictedSources =
      FORBIDDEN_IMPORT_DOCUMENT_SOURCES.map(readOptionalRepoFile).join("\n");

    expectStringLiteralAbsent(
      restrictedSources,
      RETIRED_IMPORT_DOCUMENT_COMMAND,
    );
    expectStringLiteralAbsent(
      restrictedSources,
      RETIRED_IMPORT_DOCUMENT_TO_SESSION_COMMAND,
    );
    expect(restrictedSources).not.toContain("document_import_cmd");
    for (const retiredPath of RETIRED_DOCUMENT_IMPORT_WRAPPER_FILES) {
      expect(existsSync(resolve(cwd(), retiredPath))).toBe(false);
    }
  });

  it("import_document_to_session 只应停留在 retired guard 中", () => {
    const source = readRepoFile("src/lib/api/session-files.ts");
    const contractSource = readRepoFile("scripts/check-command-contracts.mjs");

    expectStringLiteralAbsent(
      source,
      RETIRED_IMPORT_DOCUMENT_TO_SESSION_COMMAND,
    );
    expect(contractSource).toContain(
      `"${RETIRED_IMPORT_DOCUMENT_TO_SESSION_COMMAND}"`,
    );
  });
});
