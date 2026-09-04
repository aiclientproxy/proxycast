import { describe, expect, it } from "vitest";

import {
  inspectCargoManifests,
  inspectPackageManifests,
  scanSourceRecords,
  validateEvidenceIndex,
} from "./desktop-reference-boundary.mjs";

function validIndex() {
  return {
    schemaVersion: 1,
    productTarget: "codex-desktop",
    productChain:
      "Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI",
    planPath:
      "internal/exec-plans/codex-desktop-selective-goose-reference-plan.md",
    reference: {
      repository: "https://github.com/aaif-goose/goose",
      commit: "794b04a0b1f4c58378ef3738dade297c13690b77",
      license: "Apache-2.0",
      role: "mechanism-reference-only",
      codeCopied: false,
      dependencyAdded: false,
    },
    forbiddenOwners: [
      "goose-acp",
      "goose-session-message",
      "goose-recipe-runtime-storage",
      "goose-autonomous-default",
      "second-runtime",
      "second-catalog",
    ],
    entries: [
      {
        id: "connection-recovery",
        decision: "adopted-mechanism",
        status: "verified-local",
        targetBasis: ["codex-rust", "lime-local"],
        evidenceLevels: ["goose-static", "lime-local"],
        limeOwner: "Electron Desktop Host",
        implementationPaths: ["electron/appServerHost.ts"],
        verificationCommands: ["npm run test:contracts"],
      },
      {
        id: "parallel-runtime",
        decision: "excluded",
        status: "excluded",
        targetBasis: ["codex-rust"],
        evidenceLevels: ["goose-static", "lime-local"],
        implementationPaths: [],
        guardIds: ["acp-runtime-owner"],
      },
    ],
  };
}

describe("desktop reference boundary", () => {
  it("allows canonical owners and generic recipe metadata", () => {
    expect(
      scanSourceRecords([
        {
          path: "src/current.ts",
          content:
            "const chain = 'Thread/Turn/Item'; const recipe = metadata.recipe;",
        },
      ]),
    ).toEqual([]);
  });

  it.each([
    ["spawn_acp_sessions", "acp-runtime-owner"],
    ["class AcpConnection {}", "acp-runtime-owner"],
    ["struct RecipeRuntime {}", "recipe-runtime-owner"],
    ["const default_autonomous = true", "autonomous-default-owner"],
    ["const method = 'goose/session/start'", "foreign-protocol-method"],
    ["interface GooseMessage {}", "goose-owned-symbol"],
  ])("rejects %s", (content, expectedRule) => {
    expect(scanSourceRecords([{ path: "src/current.ts", content }])).toEqual([
      expect.objectContaining({ rule: expectedRule }),
    ]);
  });

  it("rejects parallel owner directories", () => {
    expect(
      scanSourceRecords([
        { path: "packages/acp-runtime/src/index.ts", content: "export {};" },
      ]),
    ).toEqual([
      expect.objectContaining({
        kind: "source-path",
        rule: "parallel-owner-path",
      }),
    ]);
  });

  it("rejects JavaScript and Cargo dependencies that introduce ACP or Goose", () => {
    expect(
      inspectPackageManifests([
        {
          path: "package.json",
          content: JSON.stringify({
            dependencies: { "@agentclientprotocol/sdk": "1.0.0" },
          }),
        },
      ]),
    ).toEqual([
      expect.objectContaining({ rule: "foreign-runtime-dependency" }),
    ]);

    expect(
      inspectCargoManifests([
        {
          path: "lime-rs/Cargo.toml",
          content: '[workspace.dependencies]\ngoose-ai = "1"\n',
        },
      ]),
    ).toEqual([
      expect.objectContaining({ rule: "foreign-runtime-dependency" }),
    ]);
  });

  it("requires Codex target evidence and a no-copy license boundary", () => {
    expect(
      validateEvidenceIndex(validIndex(), { pathExists: () => true }),
    ).toEqual([]);

    const invalid = validIndex();
    invalid.reference.codeCopied = true;
    invalid.reference.dependencyAdded = true;
    invalid.entries[0].targetBasis = ["goose-static"];

    expect(validateEvidenceIndex(invalid, { pathExists: () => true })).toEqual(
      expect.arrayContaining([
        expect.stringContaining("codeCopied"),
        expect.stringContaining("dependencyAdded"),
        expect.stringContaining("cannot rely on Goose alone"),
      ]),
    );
  });

  it("fails closed when indexed implementation evidence disappears", () => {
    expect(
      validateEvidenceIndex(validIndex(), {
        pathExists: (filePath) => !filePath.endsWith("appServerHost.ts"),
      }),
    ).toContain(
      "entries[0].implementationPaths is missing electron/appServerHost.ts",
    );
  });
});
