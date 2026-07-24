import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  CONTENT_FACTORY_PRODUCTION_READINESS_PIPELINE_FILE_NAME,
  runContentFactoryProductionReadinessPipeline,
} from "./content-factory-production-readiness-pipeline.mjs";

const PACKAGE_HASH =
  "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MANIFEST_HASH =
  "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const STALE_PACKAGE_HASH =
  "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const RELEASE_ID = "release_2026_07_06";
const PUBLIC_KEY_ID = "content-factory-prod-root-2026";
const SIGNATURE_REF = `sigstore:content-factory-app@2.2.2:${RELEASE_ID}`;
const PACKAGE_URL =
  "https://packages.example.com/content-factory-app-2.2.2.lapp";

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function commandValue(args, flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : "";
}

function createPipelineFixture() {
  const root = fs.mkdtempSync(
    path.join(os.tmpdir(), "content-factory-production-fetchcloud-pipeline-"),
  );
  const contentFactoryDir = path.join(root, "content-factory-app");
  fs.mkdirSync(contentFactoryDir, { recursive: true });
  writeJson(path.join(contentFactoryDir, "package.json"), {
    name: "content-factory-app",
    version: "2.2.2",
  });
  return {
    bootstrapPath: path.join(root, "bootstrap.json"),
    catalogPath: path.join(root, "catalog.json"),
    contentFactoryDir,
    fetchCloudPath: path.join(root, "stale-fetch-cloud.json"),
    guiEvidencePath: path.join(root, "gui-evidence.json"),
    outputDir: path.join(root, "out"),
    preflightScript: path.join(root, "fake-preflight.mjs"),
    studioCli: path.join(root, "fake-studio-cli.mjs"),
    studioDir: path.join(root, "studio"),
  };
}

function readyPreflight() {
  return {
    schemaVersion: 1,
    appId: "content-factory-app",
    expectedVersion: "2.2.2",
    missingRequirements: [],
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
    ready: true,
    signature: {
      algorithm: "Ed25519",
      appSignatureYamlPresent: true,
      payloadHash:
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      publicKeyId: PUBLIC_KEY_ID,
      schemaVersion: "plugin-cloud-release-signature/v1",
      signatureCryptographicVerificationStatus: "verified",
      signaturePayloadHashMatched: true,
      signaturePresent: true,
      signatureRef: SIGNATURE_REF,
      signedAt: "2026-07-06T00:00:00.000Z",
      trustRootPresent: true,
      trustRootPublicKeyId: PUBLIC_KEY_ID,
      trustRootPublicKeyPresent: true,
    },
    status: "ready",
  };
}

function studioDryRun() {
  return {
    mode: "dry-run",
    plan: {
      appId: "content-factory-app",
      channel: "stable",
      version: "2.2.2",
    },
    releaseReadiness: {
      appId: "content-factory-app",
      blockers: [],
      checks: {
        manifest: {
          manifestHash: MANIFEST_HASH,
          source: "app-server-inspect",
        },
        package: {
          packageHash: PACKAGE_HASH,
          version: "2.2.2",
        },
      },
      ready: true,
      warnings: [],
    },
  };
}

function catalogEvidence() {
  return {
    apps: [
      {
        appId: "content-factory-app",
        appVersion: "2.2.2",
        identity: {
          appVersion: "2.2.2",
          manifestHash: MANIFEST_HASH,
          packageHash: PACKAGE_HASH,
          releaseId: RELEASE_ID,
          signatureRef: SIGNATURE_REF,
          sourceKind: "cloud_release",
          sourceUri: PACKAGE_URL,
        },
        signatureProof: signatureProof(),
      },
    ],
  };
}

function bootstrapEvidence() {
  return {
    pluginSignatureTrustRoots: [
      {
        algorithm: "Ed25519",
        publicKey:
          "-----BEGIN PUBLIC KEY-----\nproduction-test-key\n-----END PUBLIC KEY-----",
        publicKeyId: PUBLIC_KEY_ID,
      },
    ],
  };
}

function signatureProof() {
  return {
    algorithm: "Ed25519",
    payloadHash:
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    publicKeyId: PUBLIC_KEY_ID,
    signature: "base64-signature",
    signedAt: "2026-07-06T00:00:00.000Z",
  };
}

function fetchCloudEvidence(packageHash = PACKAGE_HASH) {
  return {
    schemaVersion: "content-factory-fetch-cloud-evidence.v1",
    appId: "content-factory-app",
    cloudReleaseEvidence: {
      status: "ready",
    },
    descriptor: {
      manifestHash: MANIFEST_HASH,
      packageHash,
      sourceUri: PACKAGE_URL,
    },
    manifestHash: MANIFEST_HASH,
    manifestHashMatched: true,
    packageHash,
    packageHashMatched: true,
    packageVerificationStatus: "verified",
    signaturePolicy: "required",
    signatureProof: signatureProof(),
    signatureRef: SIGNATURE_REF,
    signatureVerificationStatus: "verified",
    sourceKind: "cloud_release",
    status: "ready",
  };
}

function guiEvidence() {
  return {
    schemaVersion: "content-factory-production-gui-evidence.v1",
    assertions: {
      articleDraftDocumentPresent: true,
      contentFactoryArticleWorkspaceWorkflowFactsHidden: true,
      liveProviderUsed: true,
      turnStartViaElectronIpc: true,
    },
    cdp: {
      attached: true,
      usedRealElectron: true,
    },
    eventLogs: {
      workflowJsonl: "/tmp/content-factory-production/workflow-events.jsonl",
      workflowJsonlEventCount: 16,
      workflowResumeEvents: [
        workflowResumeEvent("workflow.step.resuming"),
        workflowResumeEvent("workflow.run.resuming"),
      ],
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
      signatureVerificationStatus: "verified",
      sourceKind: "cloud_release",
    },
    readModel: {
      articleDraftDocumentLength: 3153,
      articleDraftDocumentPresent: true,
      generatedArticleMarkerClean: true,
      hostManagedGenerationStatus: "completed",
    },
    runtimeActionResponse: {
      actionId: "article-draft-review",
      confirmed: true,
      method: "workflow/respond",
      metadata: {
        workflowResume: {
          stepId: "draft",
          workflowKey: "content_article_workflow",
          workflowRunId: "turn_prod:content-article",
        },
      },
    },
    status: "passed",
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

function workflowResumeEvent(eventType) {
  return {
    eventType,
    payload: {
      actionId: "article-draft-review",
      decision: "approved",
      stepId: "draft",
      workflowKey: "content_article_workflow",
      workflowRunId: "turn_prod:content-article",
    },
  };
}

describe("content factory production readiness pipeline fetchCloud evidence", () => {
  it("uses catalog-generated fetchCloud evidence instead of a stale explicit fetchCloud path", () => {
    const fixture = createPipelineFixture();
    writeJson(fixture.catalogPath, catalogEvidence());
    writeJson(fixture.bootstrapPath, bootstrapEvidence());
    writeJson(fixture.fetchCloudPath, fetchCloudEvidence(STALE_PACKAGE_HASH));
    writeJson(fixture.guiEvidencePath, guiEvidence());
    const preflightArgsSeen = [];
    const runner = (_command, args) => {
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun()),
          stderr: "",
        };
      }
      if (args[0] === fixture.preflightScript) {
        preflightArgsSeen.push(args);
        expect(args).toContain("--fetch-cloud-from-catalog");
        expect(args).toContain("--fetch-cloud-output");
        expect(args).not.toContain("--fetch-cloud");
        writeJson(
          commandValue(args, "--fetch-cloud-output"),
          fetchCloudEvidence(),
        );
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      throw new Error(`unexpected command: ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      bootstrapPath: fixture.bootstrapPath,
      catalogPath: fixture.catalogPath,
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      fetchCloudFromCatalog: true,
      fetchCloudPath: fixture.fetchCloudPath,
      guiEvidencePath: fixture.guiEvidencePath,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(preflightArgsSeen).toHaveLength(1);
    expect(result.pipeline).toMatchObject({
      ready: true,
      status: "ready",
      fetchCloudFromCatalog: {
        executed: true,
        requested: true,
        source: "generated",
      },
    });
    expect(result.readinessReport).toMatchObject({
      ready: true,
      signedGate: {
        ready: true,
      },
    });
    expect(result.bundle.bundle.sources.fetchCloud.path).toBe(
      result.files.fetchCloud,
    );
    expect(result.bundle.bundle.sources.fetchCloud.path).not.toBe(
      fixture.fetchCloudPath,
    );
    expect(
      fs.readFileSync(
        path.join(
          fixture.outputDir,
          CONTENT_FACTORY_PRODUCTION_READINESS_PIPELINE_FILE_NAME,
        ),
        "utf8",
      ),
    ).not.toContain(STALE_PACKAGE_HASH);
  });
});
