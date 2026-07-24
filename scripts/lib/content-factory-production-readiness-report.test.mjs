import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

import {
  buildContentFactoryProductionReadinessReport,
  CONTENT_FACTORY_PRODUCTION_READINESS_REPORT_FILE_NAME,
  nextActionForProductionRequirement,
} from "./content-factory-production-readiness-report.mjs";
import {
  buildContentFactoryProductionEvidenceBundle,
  CONTENT_FACTORY_PRODUCTION_EVIDENCE_BUNDLE_FILE_NAME,
} from "./content-factory-production-evidence-bundle.mjs";
import {
  CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
  CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES,
} from "./plugin-content-factory-signed-release-gate-constants.mjs";

const PACKAGE_HASH =
  "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MANIFEST_HASH =
  "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const PAYLOAD_HASH =
  "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const RELEASE_ID = "prod";
const SIGNATURE_REF = `sigstore:content-factory-app@2.2.2:${RELEASE_ID}`;
const TEST_PUBLIC_KEY_PEM = [
  "-----BEGIN PUBLIC KEY-----",
  "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwdDX0Ptde31EvvL9din4",
  "unewddOxzgn2d2EDv7SR/45RJlh6z0gSUfhz2Qh/HZo9KTHqwdQlZFxfYczBQLsv",
  "DFldL4LZ5sfIBETomBupaO4oU7cjslPpAr9lvP2ojG2rt0H3Mep/3wxPAHwXLio/",
  "rWsrQIP2zsCryE5ooIO2cLLQQ6CENHA+xciS73Deu3M2Hg+3ZQkqEPDyUrNtMcj3",
  "uBxgZyY3ka11IfvyrZe0leJLM1/w/1l9IR+m83cSxMDV7ZgbN/qt25bioe0qPxmY",
  "xQDRZ1/GyfKqNntotmYJUEjpTx60G3RDvx75XH00hX+2AQrQ3eIu+2xQ/UeJ5gw1",
  "uwIDAQAB",
  "-----END PUBLIC KEY-----",
].join("\n");

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function readyPreflight() {
  return {
    schemaVersion: 1,
    appId: "content-factory-app",
    expectedVersion: "2.2.2",
    status: "ready",
    ready: true,
    package: {
      appId: "content-factory-app",
      exists: true,
      manifestHash: MANIFEST_HASH,
      missingEntries: [],
      packageHash: PACKAGE_HASH,
      validZip: true,
      version: "2.2.2",
    },
    appServerInspect: {
      manifestHash: MANIFEST_HASH,
      packageHash: PACKAGE_HASH,
      present: true,
      sourceKind: "local_folder",
    },
    signature: {
      algorithm: "Ed25519",
      appSignatureYamlPresent: true,
      payloadHash: PAYLOAD_HASH,
      payloadHashValid: true,
      publicKeyId: "content-factory-prod-root-2026",
      schemaVersion: "plugin-cloud-release-signature/v1",
      signatureCryptographicVerificationStatus: "verified",
      signaturePayloadHashMatched: true,
      signaturePresent: true,
      signatureRef: SIGNATURE_REF,
      signedAt: "2026-07-05T00:00:00.000Z",
      trustRootPresent: true,
      trustRootPublicKeyPresent: true,
      trustRootPublicKeyId: "content-factory-prod-root-2026",
    },
    publishReadiness: {
      configured: true,
      requirements: [
        {
          configured: true,
          env: ["PLUGIN_SIGNING_PRIVATE_KEY_PEM"],
          key: "signingPrivateKey",
          value: "SECRET_SHOULD_NOT_LEAK",
        },
        {
          configured: true,
          env: ["CONTENT_FACTORY_PACKAGE_URL"],
          key: "packageUrl",
          remoteHttps: true,
          value: "https://packages.example.com/content-factory-app-2.2.2.lapp",
        },
      ],
    },
    missingRequirements: [],
  };
}

function readyCatalog() {
  return {
    apps: [
      {
        appId: "content-factory-app",
        appVersion: "2.2.2",
        identity: {
          manifestHash: MANIFEST_HASH,
          packageHash: PACKAGE_HASH,
          releaseId: RELEASE_ID,
          signatureRef: SIGNATURE_REF,
          sourceKind: "cloud_release",
          sourceUri:
            "https://packages.example.com/content-factory-app-2.2.2.lapp",
        },
        signatureProof: {
          algorithm: "Ed25519",
          payloadHash: PAYLOAD_HASH,
          publicKeyId: "content-factory-prod-root-2026",
          signature: "base64-signature",
          signedAt: "2026-07-05T00:00:00.000Z",
        },
      },
    ],
  };
}

function readyGuiEvidence() {
  return {
    schemaVersion: "content-factory-production-gui-evidence.v1",
    status: "passed",
    cdp: {
      attached: true,
      usedRealElectron: true,
    },
    installedState: {
      appVersion: "2.2.2",
      cloudReleaseEvidenceStatus: "ready",
      manifestHash: MANIFEST_HASH,
      manifestHashMatched: true,
      packageHash: PACKAGE_HASH,
      packageHashMatched: true,
      packageVerificationStatus: "verified",
      releaseId: RELEASE_ID,
      signaturePolicy: "required",
      signatureRef: SIGNATURE_REF,
      sourceKind: "cloud_release",
      signatureVerificationStatus: "verified",
    },
    readModel: {
      articleDraftDocumentLength: 3153,
      articleDraftDocumentPresent: true,
      generatedArticleMarkerClean: true,
      hostManagedGenerationStatus: "completed",
    },
    assertions: {
      appServerHandleJsonLinesSeen: true,
      articleDraftDocumentPresent: true,
      contentFactoryArticleWorkspaceWorkflowFactsHidden: true,
      liveProviderUsed: true,
      turnStartViaElectronIpc: true,
    },
    eventLogs: {
      workflowJsonl: "/tmp/content-factory-production/workflow-events.jsonl",
      workflowJsonlEventCount: 16,
    },
    evidenceExport: {
      workflowAudit: {
        eventCount: 16,
        metadataOnly: true,
        rawContentIncluded: false,
        redactionPolicy: "workflow_audit_metadata_only",
        redactionPolicyEventCount: 16,
        source: "workflow-events.jsonl",
        status: "exported",
      },
    },
    runtimeActionResponse: {
      actionId: "approve-1",
      decision: "approved",
      method: "workflow/respond",
      metadata: {
        workflowResume: {
          stepId: "review",
          workflowKey: "content_article_workflow",
          workflowRunId: "workflow-run-1",
        },
      },
    },
    workflowResumeEvents: [
      {
        eventType: "workflow.step.resuming",
        payload: {
          actionId: "approve-1",
          decision: "approved",
          stepId: "review",
          workflowKey: "content_article_workflow",
          workflowRunId: "workflow-run-1",
        },
      },
      {
        eventType: "workflow.run.resuming",
        payload: {
          actionId: "approve-1",
          decision: "approved",
          stepId: "review",
          workflowKey: "content_article_workflow",
          workflowRunId: "workflow-run-1",
        },
      },
    ],
    trace: {
      appServerHandleJsonLinesSeen: true,
      appServerMethodsSeen: ["turn/start", "thread/read", "evidence/export"],
      turnStartTrace: {
        command: "app_server_handle_json_lines",
        matched: true,
        method: "turn/start",
        sessionMatched: true,
        status: "success",
        transport: "electron-ipc",
      },
    },
  };
}

function writeReadyEvidenceDir(rootDir) {
  writeJson(
    path.join(
      rootDir,
      CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.preflight,
    ),
    readyPreflight(),
  );
  writeJson(
    path.join(
      rootDir,
      CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.catalog,
    ),
    readyCatalog(),
  );
  writeJson(
    path.join(
      rootDir,
      CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.bootstrap,
    ),
    {
      pluginSignatureTrustRoots: [
        {
          algorithm: "Ed25519",
          publicKey: TEST_PUBLIC_KEY_PEM,
          publicKeyId: "content-factory-prod-root-2026",
        },
      ],
    },
  );
  writeJson(
    path.join(
      rootDir,
      CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.fetchCloud,
    ),
    {
      descriptor: {
        manifestHash: MANIFEST_HASH,
        packageHash: PACKAGE_HASH,
        sourceUri:
          "https://packages.example.com/content-factory-app-2.2.2.lapp",
      },
      manifestHash: MANIFEST_HASH,
      manifestHashMatched: true,
      packageHash: PACKAGE_HASH,
      packageHashMatched: true,
      packageVerificationStatus: "verified",
      signatureProof: {
        algorithm: "Ed25519",
        payloadHash: PAYLOAD_HASH,
        publicKeyId: "content-factory-prod-root-2026",
        signature: "base64-signature",
        signedAt: "2026-07-05T00:00:00.000Z",
      },
      signatureRef: SIGNATURE_REF,
      signaturePolicy: "required",
      signatureVerificationStatus: "verified",
      sourceKind: "cloud_release",
      status: "ready",
    },
  );
  writeJson(
    path.join(
      rootDir,
      CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.guiEvidence,
    ),
    readyGuiEvidence(),
  );
}

function readyStudioDryRun(overrides = {}) {
  return {
    mode: "dry-run",
    plan: {
      appId: "content-factory-app",
      publishable: true,
    },
    releaseReadiness: {
      ready: true,
      blockers: [],
      warnings: [],
      checks: {
        auth: {
          apiBaseConfigured: true,
          tenantIdConfigured: true,
          tokenConfigured: true,
        },
        manifest: {
          manifestHash: MANIFEST_HASH,
          source: "app-server-inspect",
        },
        package: {
          fileCount: 44,
          packageHash: PACKAGE_HASH,
          packageName: "content-factory-app-2.2.2.lapp",
          sizeBytes: 12345,
          version: "2.2.2",
        },
        packageUrl: {
          configured: true,
          host: "packages.example.com",
          https: true,
        },
        signature: {
          algorithm: "Ed25519",
          payloadHash: PAYLOAD_HASH,
          publicKeyId: "content-factory-prod-root-2026",
          signaturePresent: true,
          signatureRef: SIGNATURE_REF,
          signedAt: "2026-07-05T00:00:00.000Z",
        },
      },
      ...overrides.releaseReadiness,
    },
    ...overrides,
  };
}

describe("content factory production readiness report", () => {
  it("is exposed as an explicit npm production readiness report", () => {
    const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
    expect(
      packageJson.scripts["plugin:content-factory-production-readiness-report"],
    ).toBe(
      "node scripts/plugin/content-factory-production-readiness-report.mjs",
    );
  });

  it("summarizes blockers without copying secret values or package URLs", () => {
    const rootDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-production-readiness-"),
    );
    const preflightPath = path.join(rootDir, "preflight.json");
    const catalogPath = path.join(rootDir, "catalog.json");
    writeJson(preflightPath, {
      ...readyPreflight(),
      ready: false,
      status: "blocked",
      signature: {
        appSignatureYamlPresent: false,
        trustRootPresent: false,
      },
      missingRequirements: [
        { code: "production_app_signature_yaml_missing" },
        { code: "production_trust_root_missing" },
        { code: "production_signature_catalog_algorithm_mismatch" },
        { code: "production_preflight_signature_ref_missing" },
      ],
    });
    writeJson(catalogPath, readyCatalog());

    const report = buildContentFactoryProductionReadinessReport({
      catalogPath,
      expectedVersion: "2.2.2",
      preflightPath,
    });

    expect(report).toMatchObject({
      ready: false,
      status: "blocked",
      preflight: {
        appSignatureYamlPresent: false,
        trustRootPresent: false,
      },
      preflightBlockers: expect.arrayContaining([
        expect.objectContaining({
          code: "production_app_signature_yaml_missing",
        }),
        expect.objectContaining({
          code: "production_trust_root_missing",
        }),
        expect.objectContaining({
          code: "production_signature_catalog_algorithm_mismatch",
          nextAction: expect.stringContaining(
            "catalog signatureProof.algorithm",
          ),
        }),
        expect.objectContaining({
          code: "production_preflight_signature_ref_missing",
          nextAction: expect.stringContaining("不要手写 ready preflight JSON"),
        }),
      ]),
      signedGate: {
        missingCodes: expect.arrayContaining([
          "production_preflight_not_ready",
        ]),
      },
      blockerPlan: {
        nextPhase: expect.objectContaining({
          id: "release_signing_and_trust",
          owner: "operator",
        }),
      },
    });
    const serialized = JSON.stringify(report);
    expect(serialized).not.toContain("SECRET_SHOULD_NOT_LEAK");
    expect(serialized).not.toContain("packages.example.com");
  });

  it("has explicit next actions for all signed gate production codes", () => {
    const implementationFiles = [
      "scripts/lib/content-factory-production-preflight-core.mjs",
      "scripts/lib/plugin-content-factory-signed-release-gate-core.mjs",
      "scripts/lib/plugin-content-factory-signed-release-gate-preflight.mjs",
      "scripts/lib/plugin-content-factory-signed-release-gate-fetch-cloud.mjs",
      "scripts/lib/plugin-content-factory-signed-release-gate-gui.mjs",
      "scripts/lib/plugin-content-factory-signed-release-gate-safety.mjs",
    ];
    const gateCodes = [
      ...new Set(
        implementationFiles.flatMap(
          (filePath) =>
            fs
              .readFileSync(filePath, "utf8")
              .match(
                /production_[a-z0-9_]+|fixture_cloud_release_not_allowed/g,
              ) || [],
        ),
      ),
    ].sort();

    for (const code of gateCodes) {
      expect(nextActionForProductionRequirement(code), code).not.toContain(
        "补齐对应 production evidence",
      );
    }
  });

  it("CLI --check accepts a fully ready five-file evidence-dir", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-production-readiness-ready-"),
    );
    const output = path.join(
      evidenceDir,
      CONTENT_FACTORY_PRODUCTION_READINESS_REPORT_FILE_NAME,
    );
    writeReadyEvidenceDir(evidenceDir);

    const result = spawnSync(
      process.execPath,
      [
        path.resolve(
          "scripts/plugin/content-factory-production-readiness-report.mjs",
        ),
        "--evidence-dir",
        evidenceDir,
        "--expected-version",
        "2.2.2",
        "--output",
        output,
        "--check",
      ],
      { encoding: "utf8", env: { PATH: process.env.PATH } },
    );

    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    expect(JSON.parse(fs.readFileSync(output, "utf8"))).toMatchObject({
      guiEvidence: {
        workflowResumeLifecycle: {
          actionMetadataPresent: true,
          auditEventsPresent: true,
        },
      },
      ready: true,
      status: "ready",
      signedGate: {
        missingCount: 0,
        ready: true,
      },
    });
  });

  it("recomputes from evidence files instead of trusting a stale ready gate result", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-production-readiness-stale-"),
    );
    writeJson(
      path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
      ),
      {
        missingRequirements: [],
        ready: true,
        status: "ready",
      },
    );

    const report = buildContentFactoryProductionReadinessReport({
      evidenceDir,
      expectedVersion: "2.2.2",
    });

    expect(report).toMatchObject({
      ready: false,
      signedGate: {
        existingResult: {
          matchesCurrentEvidence: false,
          ready: true,
          status: "ready",
        },
        resultDrift: expect.arrayContaining([
          expect.objectContaining({
            code: "production_signed_gate_result_stale",
            nextAction: expect.stringContaining("重新运行 signed release gate"),
          }),
        ]),
        missingCodes: expect.arrayContaining([
          "production_preflight_missing",
          "production_catalog_missing",
          "production_gui_evidence_missing",
        ]),
      },
      status: "blocked",
    });
  });

  it("blocks a stale gate result even when current evidence recomputes ready", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(
        os.tmpdir(),
        "content-factory-production-readiness-stale-ready-",
      ),
    );
    writeReadyEvidenceDir(evidenceDir);
    writeJson(
      path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
      ),
      {
        missingRequirements: [
          { code: "production_preflight_missing", detail: "stale" },
        ],
        ready: false,
        status: "blocked",
      },
    );

    const report = buildContentFactoryProductionReadinessReport({
      evidenceDir,
      expectedVersion: "2.2.2",
    });

    expect(report).toMatchObject({
      ready: false,
      signedGate: {
        computedReady: true,
        existingResult: {
          matchesCurrentEvidence: false,
          missingCodes: ["production_preflight_missing"],
        },
        missingCount: 0,
        ready: false,
        resultDrift: expect.arrayContaining([
          expect.objectContaining({
            code: "production_signed_gate_result_stale",
          }),
        ]),
      },
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: "production_signed_gate_result_stale",
        }),
      ]),
      status: "blocked",
    });
  });

  it("blocks a stale production evidence bundle manifest", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(
        os.tmpdir(),
        "content-factory-production-readiness-stale-bundle-",
      ),
    );
    writeReadyEvidenceDir(evidenceDir);
    writeJson(
      path.join(
        evidenceDir,
        CONTENT_FACTORY_PRODUCTION_EVIDENCE_BUNDLE_FILE_NAME,
      ),
      {
        schemaVersion: "content-factory-production-evidence-bundle.v1",
        appId: "content-factory-app",
        expectedVersion: "2.2.2",
        inputs: {
          digest:
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
          slots: {
            preflight: {
              present: true,
              sha256:
                "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            },
          },
        },
      },
    );

    const report = buildContentFactoryProductionReadinessReport({
      evidenceDir,
      expectedVersion: "2.2.2",
    });

    expect(report).toMatchObject({
      ready: false,
      evidenceBundle: {
        digestMatches: false,
        matchesCurrentEvidence: false,
        present: true,
        drift: expect.arrayContaining([
          expect.objectContaining({
            code: "production_evidence_bundle_stale",
            nextAction: expect.stringContaining("production evidence bundle"),
          }),
        ]),
        slotMismatches: expect.arrayContaining([
          expect.objectContaining({
            slot: "preflight",
            currentPresent: true,
            bundlePresent: true,
          }),
        ]),
      },
      signedGate: {
        computedReady: true,
        missingCount: 0,
      },
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: "production_evidence_bundle_stale",
        }),
      ]),
      status: "blocked",
    });
  });

  it("consumes Studio dry-run readiness and blocks release-side blockers", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-production-readiness-studio-"),
    );
    const studioDryRunPath = path.join(evidenceDir, "studio-dry-run.json");
    writeReadyEvidenceDir(evidenceDir);
    writeJson(
      studioDryRunPath,
      readyStudioDryRun({
        releaseReadiness: {
          ready: false,
          blockers: [
            { code: "production_package_url_missing" },
            { code: "production_studio_token_missing" },
          ],
          warnings: [],
        },
      }),
    );

    const report = buildContentFactoryProductionReadinessReport({
      evidenceDir,
      expectedVersion: "2.2.2",
      studioDryRunPath,
    });

    expect(report).toMatchObject({
      ready: false,
      signedGate: {
        computedReady: true,
        missingCount: 0,
      },
      studioDryRun: {
        present: true,
        ready: false,
        blockers: expect.arrayContaining([
          expect.objectContaining({
            code: "production_package_url_missing",
            nextAction: expect.stringContaining("HTTPS packageUrl"),
          }),
          expect.objectContaining({
            code: "production_studio_token_missing",
            nextAction: expect.stringContaining("开发者 token"),
          }),
        ]),
      },
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: "production_package_url_missing" }),
        expect.objectContaining({ code: "production_studio_token_missing" }),
      ]),
      status: "blocked",
    });
    const serialized = JSON.stringify(report);
    expect(serialized).not.toContain("SECRET_SHOULD_NOT_LEAK");
  });

  it("blocks when Studio dry-run package facts drift from preflight", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(
        os.tmpdir(),
        "content-factory-production-readiness-studio-drift-",
      ),
    );
    const studioDryRunPath = path.join(evidenceDir, "studio-dry-run.json");
    writeReadyEvidenceDir(evidenceDir);
    writeJson(
      studioDryRunPath,
      readyStudioDryRun({
        releaseReadiness: {
          checks: {
            ...readyStudioDryRun().releaseReadiness.checks,
            manifest: {
              manifestHash: `sha256:${"d".repeat(64)}`,
              source: "app-server-inspect",
            },
          },
        },
      }),
    );

    const report = buildContentFactoryProductionReadinessReport({
      evidenceDir,
      expectedVersion: "2.2.2",
      studioDryRunPath,
    });

    expect(report).toMatchObject({
      ready: false,
      studioDryRun: {
        present: true,
        ready: false,
        drift: expect.arrayContaining([
          expect.objectContaining({
            code: "production_studio_dry_run_manifest_hash_mismatch",
            nextAction: expect.stringContaining("Studio dry-run"),
          }),
        ]),
      },
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: "production_studio_dry_run_manifest_hash_mismatch",
        }),
      ]),
      status: "blocked",
    });
  });

  it("blocks a production evidence bundle when inputs match but bundled gate summary is stale", () => {
    const evidenceDir = fs.mkdtempSync(
      path.join(
        os.tmpdir(),
        "content-factory-production-readiness-stale-bundle-gate-",
      ),
    );
    writeReadyEvidenceDir(evidenceDir);
    buildContentFactoryProductionEvidenceBundle({
      bootstrapPath: path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.bootstrap,
      ),
      catalogPath: path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.catalog,
      ),
      expectedVersion: "2.2.2",
      fetchCloudPath: path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.fetchCloud,
      ),
      guiEvidencePath: path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.guiEvidence,
      ),
      outputDir: evidenceDir,
      preflightPath: path.join(
        evidenceDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.preflight,
      ),
    });
    const bundlePath = path.join(
      evidenceDir,
      CONTENT_FACTORY_PRODUCTION_EVIDENCE_BUNDLE_FILE_NAME,
    );
    const bundle = JSON.parse(fs.readFileSync(bundlePath, "utf8"));
    bundle.gate = {
      ...bundle.gate,
      digest:
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
      missingCodes: ["production_preflight_missing"],
      ready: false,
      status: "blocked",
    };
    writeJson(bundlePath, bundle);

    const report = buildContentFactoryProductionReadinessReport({
      evidenceDir,
      expectedVersion: "2.2.2",
    });

    expect(report).toMatchObject({
      ready: false,
      evidenceBundle: {
        digestMatches: true,
        gate: {
          digestMatches: false,
          expectedReady: true,
          matchesCurrentEvidence: false,
          missingCodes: ["production_preflight_missing"],
          ready: false,
          summaryMatches: false,
        },
        matchesCurrentEvidence: false,
        present: true,
        slotMismatches: [],
      },
      signedGate: {
        computedReady: true,
        missingCount: 0,
      },
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: "production_evidence_bundle_gate_stale",
          nextAction: expect.stringContaining("gate 摘要"),
        }),
      ]),
      status: "blocked",
    });
  });
});
