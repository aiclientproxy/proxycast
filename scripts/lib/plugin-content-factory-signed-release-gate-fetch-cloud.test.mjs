import { describe, expect, it } from "vitest";

import { buildContentFactorySignedReleaseGate } from "./plugin-content-factory-signed-release-gate-core.mjs";

const MANIFEST_HASH =
  "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const PACKAGE_HASH =
  "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const PAYLOAD_HASH =
  "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const PACKAGE_URL =
  "https://updates.limeai.run/plugins/content-factory-app/prod/content-factory-app-2.2.2.lapp";
const RELEASE_ID = "prod";
const SIGNATURE_REF = "sigstore:content-factory-app@2.2.2:prod";

const SIGNATURE_PROOF = {
  algorithm: "Ed25519",
  payloadHash: PAYLOAD_HASH,
  publicKeyId: "content-factory-prod-root-2026",
  signature: "base64-signature",
  signedAt: "2026-07-03T00:00:00.000Z",
};

const READY_CATALOG = {
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
        sourceUri: PACKAGE_URL,
      },
      signatureProof: SIGNATURE_PROOF,
    },
  ],
};

const READY_BOOTSTRAP = {
  pluginSignatureTrustRoots: [
    {
      algorithm: "Ed25519",
      appIds: ["content-factory-app"],
      publicKeyId: "content-factory-prod-root-2026",
      revoked: false,
    },
  ],
};

const READY_PREFLIGHT = {
  appId: "content-factory-app",
  expectedVersion: "2.2.2",
  package: {
    appId: "content-factory-app",
    manifestHash: MANIFEST_HASH,
    packageHash: PACKAGE_HASH,
    validZip: true,
    version: "2.2.2",
  },
  appServerInspect: {
    manifestHash: MANIFEST_HASH,
    packageHash: PACKAGE_HASH,
    present: true,
  },
  ready: true,
  signature: {
    algorithm: "Ed25519",
    appSignatureYamlPresent: true,
    payloadHash: PAYLOAD_HASH,
    publicKeyId: "content-factory-prod-root-2026",
    signaturePresent: true,
    signatureRef: SIGNATURE_REF,
    signedAt: "2026-07-03T00:00:00.000Z",
    trustRootPresent: true,
  },
  status: "ready",
};

const READY_GUI_EVIDENCE = {
  schemaVersion: "content-factory-production-gui-evidence.v1",
  cdp: {
    attached: true,
    usedRealElectron: true,
  },
  assertions: {
    articleDraftDocumentPresent: true,
    contentFactoryArticleWorkspaceWorkflowFactsHidden: true,
    liveProviderUsed: true,
    turnStartViaElectronIpc: true,
  },
  eventLogs: {
    workflowJsonl:
      "/tmp/lime-runtime/events/sessions/session-prod/workflow-events.jsonl",
    workflowJsonlEventCount: 16,
    workflowResumeEvents: [
      {
        eventType: "workflow.step.resuming",
        payload: {
          actionId: "article-draft-review",
          decision: "approved",
          stepId: "draft",
          workflowKey: "content_article_workflow",
          workflowRunId: "turn_prod:content-article",
        },
      },
      {
        eventType: "workflow.run.resuming",
        payload: {
          actionId: "article-draft-review",
          decision: "approved",
          stepId: "draft",
          workflowKey: "content_article_workflow",
          workflowRunId: "turn_prod:content-article",
        },
      },
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
  providerEvidence: {
    liveProviderUsed: true,
    productionRoute: true,
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
  signatureVerificationStatus: "verified",
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

describe("content factory signed release gate fetchCloud evidence", () => {
  it("拒绝只写 matched=true 但缺少 fetchCloud 可审计字段的 evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: {
        manifestHashMatched: true,
        packageHashMatched: true,
        packageVerificationStatus: "verified",
        signaturePolicy: "required",
        signatureVerificationStatus: "verified",
        sourceKind: "cloud_release",
        status: "ready",
      },
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      fetchCloud: {
        ready: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_release_evidence_not_ready",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_package_hash_missing",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_manifest_hash_missing",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_package_url_missing",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_signature_ref_missing",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_signature_proof_missing",
        }),
      ]),
    });
  });

  it("拒绝 fetchCloud 具体 hash/source/signature 与 catalog 或 preflight 不一致", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: {
        cloudReleaseEvidence: {
          manifestHash:
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
          packageHash:
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
          signatureProof: {
            algorithm: "RSA-PSS-SHA256",
            payloadHash:
              "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            publicKeyId: "content-factory-prod-root-rotated",
            signature: "DRIFTED_FETCH_SIGNATURE_SHOULD_NOT_LEAK",
            signedAt: "2026-07-04T00:00:00.000Z",
          },
          signatureRef: "sigstore:content-factory-app@2.2.2:rotated",
          sourceUri:
            "https://updates.limeai.run/plugins/content-factory-app/prod-rotated/content-factory-app-2.2.2.lapp",
        },
        manifestHashMatched: true,
        packageHashMatched: true,
        packageVerificationStatus: "verified",
        signaturePolicy: "required",
        signatureVerificationStatus: "verified",
        sourceKind: "cloud_release",
        status: "ready",
      },
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      fetchCloud: {
        ready: true,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_package_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_manifest_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_package_url_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_signature_ref_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_public_key_id_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_algorithm_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_payload_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_catalog_signed_at_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_preflight_package_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_preflight_manifest_hash_mismatch",
        }),
      ]),
    });
    expect(JSON.stringify(result)).not.toContain(
      "DRIFTED_FETCH_SIGNATURE_SHOULD_NOT_LEAK",
    );
  });

  it("拒绝 fetchCloud signaturePolicy 不是 required", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: {
        manifestHashMatched: true,
        packageHashMatched: true,
        packageVerificationStatus: "verified",
        signaturePolicy: "optional",
        signatureVerificationStatus: "verified",
        sourceKind: "cloud_release",
        status: "ready",
      },
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      fetchCloud: {
        ready: false,
        signaturePolicy: "optional",
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_release_evidence_not_ready",
        }),
        expect.objectContaining({
          code: "production_fetch_cloud_signature_policy_not_required",
        }),
      ]),
    });
  });
});
