import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

import {
  buildContentFactorySignedReleaseGate,
  buildContentFactorySignedReleaseEvidenceTemplate,
  CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
  CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES,
  writeJsonFile,
} from "./plugin-content-factory-signed-release-gate-core.mjs";

const SIGNATURE_PROOF = {
  algorithm: "Ed25519",
  payloadHash:
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  publicKeyId: "content-factory-prod-root-2026",
  signature: "base64-signature",
  signedAt: "2026-07-03T00:00:00.000Z",
};
const READY_RELEASE_ID = "prod";
const READY_SIGNATURE_REF = `sigstore:content-factory-app@2.2.2:${READY_RELEASE_ID}`;
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

const READY_CATALOG = {
  apps: [
    {
      appId: "content-factory-app",
      appVersion: "2.2.2",
      identity: {
        manifestHash:
          "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        packageHash:
          "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        releaseId: READY_RELEASE_ID,
        signatureRef: READY_SIGNATURE_REF,
        sourceKind: "cloud_release",
        sourceUri:
          "https://updates.limeai.run/plugins/content-factory-app/prod/content-factory-app-2.2.2.lapp",
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
      notBefore: "2026-01-01T00:00:00.000Z",
      notAfter: "2027-01-01T00:00:00.000Z",
      publicKey: TEST_PUBLIC_KEY_PEM,
      publicKeyId: "content-factory-prod-root-2026",
      revoked: false,
    },
  ],
};

const READY_FETCH_CLOUD = {
  descriptor: {
    manifestHash:
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    packageHash:
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    sourceUri:
      "https://updates.limeai.run/plugins/content-factory-app/prod/content-factory-app-2.2.2.lapp",
  },
  manifestHashMatched: true,
  packageHashMatched: true,
  packageVerificationStatus: "verified",
  signatureProof: SIGNATURE_PROOF,
  signatureRef: READY_SIGNATURE_REF,
  signaturePolicy: "required",
  signatureVerificationStatus: "verified",
  sourceKind: "cloud_release",
  status: "ready",
};

const READY_PREFLIGHT = {
  schemaVersion: 1,
  appId: "content-factory-app",
  expectedVersion: "2.2.2",
  status: "ready",
  ready: true,
  package: {
    appId: "content-factory-app",
    exists: true,
    manifestHash:
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    missingEntries: [],
    packageHash:
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    validZip: true,
    version: "2.2.2",
  },
  appServerInspect: {
    manifestHash:
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    packageHash:
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    present: true,
    sourceKind: "local_folder",
  },
  signature: {
    algorithm: "Ed25519",
    appSignatureYamlPresent: true,
    payloadHash:
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    payloadHashValid: true,
    publicKeyId: "content-factory-prod-root-2026",
    schemaVersion: "plugin-cloud-release-signature/v1",
    signatureCryptographicVerificationStatus: "verified",
    signaturePayloadHashMatched: true,
    signaturePresent: true,
    signatureRef: READY_SIGNATURE_REF,
    signedAt: "2026-07-03T00:00:00.000Z",
    trustRootPresent: true,
    trustRootPublicKeyPresent: true,
    trustRootPublicKeyId: "content-factory-prod-root-2026",
  },
  catalog: {
    manifestHash:
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    packageHash:
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    packageUrl:
      "https://updates.limeai.run/plugins/content-factory-app/prod/content-factory-app-2.2.2.lapp",
    present: true,
    releaseId: READY_RELEASE_ID,
    signatureProofPresent: true,
    signatureRef: READY_SIGNATURE_REF,
    sourceKind: "cloud_release",
    version: "2.2.2",
  },
  bootstrap: {
    present: true,
    trustRootCount: 1,
  },
  fetchCloud: {
    manifestHashMatched: true,
    packageHashMatched: true,
    packageVerificationStatus: "verified",
    present: true,
    ready: true,
    signaturePolicy: "required",
    signatureVerificationStatus: "verified",
    sourceKind: "cloud_release",
    status: "ready",
  },
  publishReadiness: {
    configured: false,
    requirements: [
      {
        configured: false,
        env: ["PLUGIN_SIGNING_PRIVATE_KEY_PEM"],
        key: "signingPrivateKey",
      },
    ],
  },
  signingCommand:
    "npm run release:sign -- --package-url <https-url> --private-key-env PLUGIN_SIGNING_PRIVATE_KEY_PEM",
  missingRequirements: [],
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
      "/tmp/lime-runtime/events/sessions/session_prod/workflow-events.jsonl",
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
    manifestHash:
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    manifestHashMatched: true,
    packageHash:
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    packageHashMatched: true,
    packageVerificationStatus: "verified",
    releaseId: READY_RELEASE_ID,
    signaturePolicy: "required",
    signatureRef: READY_SIGNATURE_REF,
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

describe("content factory signed release gate", () => {
  it("缺少 production evidence 时 blocked", () => {
    const result = buildContentFactorySignedReleaseGate({
      expectedVersion: "2.2.2",
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({ code: "production_preflight_missing" }),
        expect.objectContaining({ code: "production_catalog_missing" }),
        expect.objectContaining({ code: "production_trust_roots_missing" }),
        expect.objectContaining({
          code: "production_fetch_cloud_evidence_missing",
        }),
        expect.objectContaining({ code: "production_gui_evidence_missing" }),
      ]),
    });
  });

  it("拒绝 fixture cloud release 和 localhost host generation evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: {
        apps: [
          {
            ...READY_CATALOG.apps[0],
            identity: {
              ...READY_CATALOG.apps[0].identity,
              signatureRef: "sigstore:content-factory-app@2.2.2:fixture",
              sourceUri:
                "https://updates.limeai.run/plugins/content-factory-app/fixture/content-factory-app-2.2.2.lapp",
            },
            releaseId: "content-factory-fixture-2.2.2",
          },
        ],
      },
      expectedVersion: "2.2.2",
      fetchCloud: {
        ...READY_FETCH_CLOUD,
        cloudReleaseFixture: { channel: "fixture" },
      },
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        hostGenerationFixture: {
          baseUrl: "http://127.0.0.1:48999",
          requestCount: 1,
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({ code: "production_package_url_not_https" }),
        expect.objectContaining({ code: "fixture_cloud_release_not_allowed" }),
        expect.objectContaining({
          code: "production_host_generation_not_live",
        }),
      ]),
    });
  });

  it("不会把 hostGenerationFixture: null 的 live GUI evidence 误判为 fixture", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        eventLogs: {
          ...READY_GUI_EVIDENCE.eventLogs,
          workflowJsonl:
            "/tmp/lime-runtime/events/sessions/session_prod/workflow-events.jsonl",
        },
        hostGenerationFixture: null,
        installedState: {
          sourceKind: "local_folder",
        },
        signatureVerificationStatus: null,
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result.guiEvidence).toMatchObject({
      fixtureLike: false,
      liveProviderUsed: true,
    });
    expect(result.missingRequirements).toEqual(
      expect.not.arrayContaining([
        expect.objectContaining({
          code: "production_host_generation_not_live",
        }),
      ]),
    );
    expect(result.missingRequirements).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "production_gui_not_cloud_release" }),
        expect.objectContaining({
          code: "production_gui_signature_not_verified",
        }),
      ]),
    );
  });

  it("拒绝只写 productionRoute 但缺 liveProviderUsed 的手写 GUI evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        assertions: {
          ...READY_GUI_EVIDENCE.assertions,
          liveProviderUsed: false,
        },
        providerEvidence: {
          productionRoute: true,
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        liveProviderUsed: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_host_generation_not_live",
        }),
      ]),
    });
  });

  it("拒绝 production evidence 模板占位值", () => {
    const placeholderProof = {
      ...SIGNATURE_PROOF,
      publicKeyId: "REPLACE_WITH_PRODUCTION_TRUST_ROOT_PUBLIC_KEY_ID",
      signature: "REPLACE_WITH_BASE64_SIGNATURE",
    };
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: {
        pluginSignatureTrustRoots: [
          {
            ...READY_BOOTSTRAP.pluginSignatureTrustRoots[0],
            publicKeyId: placeholderProof.publicKeyId,
            publicKeyPem:
              "-----BEGIN PUBLIC KEY-----\\nREPLACE_WITH_PUBLIC_KEY\\n-----END PUBLIC KEY-----",
          },
        ],
      },
      catalog: {
        apps: [
          {
            ...READY_CATALOG.apps[0],
            signatureProof: placeholderProof,
          },
        ],
      },
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_placeholder_values_present",
        }),
      ]),
      placeholders: {
        bootstrap: expect.arrayContaining([
          expect.stringContaining("REPLACE_WITH_PUBLIC_KEY"),
        ]),
        catalog: expect.arrayContaining([
          "REPLACE_WITH_PRODUCTION_TRUST_ROOT_PUBLIC_KEY_ID",
          "REPLACE_WITH_BASE64_SIGNATURE",
        ]),
      },
    });
  });

  it("拒绝 production evidence 中包含 Provider key 或 bearer token", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        providerEvidence: {
          authorization: "Bearer live-production-token-value",
          liveProviderUsed: true,
          productionRoute: true,
        },
        runtimeActionResponse: {
          ...READY_GUI_EVIDENCE.runtimeActionResponse,
          metadata: {
            ...READY_GUI_EVIDENCE.runtimeActionResponse.metadata,
            apiKey: "sk-production-secret-value",
          },
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_secret_values_present",
        }),
      ]),
      secretScan: {
        guiEvidence: expect.arrayContaining([
          expect.stringContaining("providerEvidence.authorization"),
          expect.stringContaining("runtimeActionResponse.metadata.apiKey"),
        ]),
      },
    });
    expect(JSON.stringify(result.missingRequirements)).not.toContain(
      "sk-production-secret-value",
    );
    expect(JSON.stringify(result.missingRequirements)).not.toContain(
      "live-production-token-value",
    );
  });

  it("接受 production signed release + trust root + live Provider GUI evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: true,
      status: "ready",
      bootstrap: {
        matchingTrustRoot: true,
      },
      fetchCloud: {
        ready: true,
      },
      guiEvidence: {
        liveProviderUsed: true,
        ready: true,
        workflowResumeLifecycle: {
          auditEventsPresent: true,
          actionMetadataPresent: true,
          workflowRunId: "turn_prod:content-article",
        },
      },
      missingRequirements: [],
    });
  });

  it("拒绝缺少 installed release identity 的 GUI production evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        installedState: {
          ...READY_GUI_EVIDENCE.installedState,
          manifestHash: undefined,
          packageHash: undefined,
          releaseId: undefined,
          signatureRef: undefined,
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        cloudReleaseRuntimeVerified: false,
        manifestHash: null,
        packageHash: null,
        releaseId: null,
        signatureRef: null,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_gui_package_hash_missing",
        }),
        expect.objectContaining({
          code: "production_gui_manifest_hash_missing",
        }),
        expect.objectContaining({
          code: "production_gui_release_id_missing",
        }),
        expect.objectContaining({
          code: "production_gui_signature_ref_missing",
        }),
      ]),
    });
  });

  it("拒绝 GUI installed release identity 与 catalog/preflight/fetchCloud 漂移", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        installedState: {
          ...READY_GUI_EVIDENCE.installedState,
          appVersion: "2.2.1",
          manifestHash:
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
          packageHash:
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
          releaseId: "stale-release",
          signatureRef: "sigstore:content-factory-app@2.2.2:stale-release",
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_gui_catalog_version_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_catalog_package_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_catalog_manifest_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_catalog_release_id_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_catalog_signature_ref_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_preflight_package_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_preflight_manifest_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_preflight_signature_ref_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_fetch_cloud_package_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_fetch_cloud_manifest_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_gui_fetch_cloud_signature_ref_mismatch",
        }),
      ]),
    });
  });

  it("拒绝 signaturePolicy optional 的 GUI production evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        installedState: {
          ...READY_GUI_EVIDENCE.installedState,
          cloudReleaseEvidenceStatus: "warning",
          manifestHashMatched: false,
          packageHashMatched: false,
          packageVerificationStatus: "not_configured",
          signaturePolicy: "optional",
          signatureVerificationStatus: "not_configured",
        },
        signatureVerificationStatus: "not_configured",
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        cloudReleaseRuntimeVerified: false,
        manifestHashMatched: false,
        packageHashMatched: false,
        packageVerificationStatus: "not_configured",
        signaturePolicy: "optional",
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_gui_signature_policy_not_required",
        }),
        expect.objectContaining({
          code: "production_gui_signature_not_verified",
        }),
        expect.objectContaining({
          code: "production_gui_release_evidence_not_ready",
        }),
        expect.objectContaining({
          code: "production_gui_package_verification_not_verified",
        }),
        expect.objectContaining({
          code: "production_gui_package_hash_not_matched",
        }),
        expect.objectContaining({
          code: "production_gui_manifest_hash_not_matched",
        }),
      ]),
    });
  });

  it("拒绝 remote sourceKind，production catalog 必须是 cloud_release", () => {
    const catalog = {
      apps: [
        {
          ...READY_CATALOG.apps[0],
          identity: {
            ...READY_CATALOG.apps[0].identity,
            sourceKind: "remote",
          },
        },
      ],
    };

    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      catalog: {
        sourceKind: "remote",
        sourceKindReady: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_catalog_not_cloud_release",
        }),
      ]),
    });
  });

  it("拒绝缺少 publicKey 的匹配 bootstrap trust root", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: {
        pluginSignatureTrustRoots: [
          {
            ...READY_BOOTSTRAP.pluginSignatureTrustRoots[0],
            publicKey: "",
          },
        ],
      },
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      bootstrap: {
        matchingTrustRoot: true,
        matchingTrustRootPublicKeyPresent: false,
        ready: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_signature_trust_root_public_key_missing",
        }),
      ]),
    });
  });

  it("拒绝 blocked production preflight 即使其他 production evidence ready", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: {
        ...READY_PREFLIGHT,
        missingRequirements: [
          {
            code: "production_app_signature_yaml_missing",
            detail: "app.signature.yaml is required.",
          },
        ],
        ready: false,
        status: "blocked",
      },
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({ code: "production_preflight_not_ready" }),
      ]),
      preflight: {
        missingRequirementCodes: ["production_app_signature_yaml_missing"],
        ready: false,
      },
    });
  });

  it("拒绝 preflight 与 catalog 的 packageHash / manifestHash 不一致", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: {
        ...READY_PREFLIGHT,
        package: {
          ...READY_PREFLIGHT.package,
          manifestHash:
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
          packageHash:
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        },
      },
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_preflight_catalog_package_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_preflight_catalog_manifest_hash_mismatch",
        }),
      ]),
    });
  });

  it("拒绝 ready preflight 与 production catalog 的签名 proof 不一致", () => {
    const driftedProof = {
      algorithm: "RSA-PSS-SHA256",
      payloadHash:
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      publicKeyId: "content-factory-prod-root-rotated",
      signature: "DRIFTED_SIGNATURE_SHOULD_NOT_LEAK",
      signedAt: "2026-07-04T00:00:00.000Z",
    };
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: {
        pluginSignatureTrustRoots: [
          {
            ...READY_BOOTSTRAP.pluginSignatureTrustRoots[0],
            algorithm: driftedProof.algorithm,
            publicKeyId: driftedProof.publicKeyId,
          },
        ],
      },
      catalog: {
        apps: [
          {
            ...READY_CATALOG.apps[0],
            identity: {
              ...READY_CATALOG.apps[0].identity,
              signatureRef: "sigstore:content-factory-app@2.2.2:rotated",
            },
            signatureProof: driftedProof,
          },
        ],
      },
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_signature_catalog_signature_ref_mismatch",
        }),
        expect.objectContaining({
          code: "production_signature_catalog_public_key_id_mismatch",
        }),
        expect.objectContaining({
          code: "production_signature_catalog_algorithm_mismatch",
        }),
        expect.objectContaining({
          code: "production_signature_catalog_payload_hash_mismatch",
        }),
        expect.objectContaining({
          code: "production_signature_catalog_signed_at_mismatch",
        }),
      ]),
    });
    expect(JSON.stringify(result)).not.toContain(
      "DRIFTED_SIGNATURE_SHOULD_NOT_LEAK",
    );
  });

  it("拒绝 production catalog 缺 releaseId 或 signatureRef 未绑定 releaseId", () => {
    const withoutReleaseId = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: {
        apps: [
          {
            ...READY_CATALOG.apps[0],
            identity: {
              ...READY_CATALOG.apps[0].identity,
              releaseId: undefined,
              signatureRef: "sigstore:content-factory-app@2.2.2",
            },
          },
        ],
      },
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(withoutReleaseId).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_release_id_missing",
        }),
        expect.objectContaining({
          code: "production_signature_catalog_signature_ref_mismatch",
        }),
      ]),
    });

    const mismatch = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: {
        apps: [
          {
            ...READY_CATALOG.apps[0],
            identity: {
              ...READY_CATALOG.apps[0].identity,
              releaseId: READY_RELEASE_ID,
              signatureRef: "sigstore:content-factory-app@2.2.2",
            },
          },
        ],
      },
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(mismatch.missingRequirements).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "production_signature_ref_release_id_mismatch",
        }),
      ]),
    );
  });

  it("拒绝没有 Electron IPC turn/start 的 GUI evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        assertions: {
          ...READY_GUI_EVIDENCE.assertions,
          turnStartViaElectronIpc: false,
        },
        trace: {
          ...READY_GUI_EVIDENCE.trace,
          appServerHandleJsonLinesSeen: false,
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_gui_turn_start_not_electron_ipc",
        }),
        expect.objectContaining({
          code: "production_gui_app_server_json_rpc_missing",
        }),
      ]),
    });
  });

  it("拒绝缺少 production collector provenance 的手写 GUI evidence", () => {
    const {
      cdp: _cdp,
      schemaVersion: _schemaVersion,
      ...handWrittenGuiEvidence
    } = {
      ...READY_GUI_EVIDENCE,
      eventLogs: {
        ...READY_GUI_EVIDENCE.eventLogs,
        workflowJsonlEventCount: undefined,
      },
      readModel: {
        articleDraftDocumentPresent: true,
        hostManagedGenerationStatus: "completed",
      },
      trace: {
        appServerHandleJsonLinesSeen: true,
        appServerMethodsSeen: ["turn/start"],
      },
    };

    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: handWrittenGuiEvidence,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_gui_collector_schema_missing",
        }),
        expect.objectContaining({
          code: "production_gui_cdp_evidence_missing",
        }),
        expect.objectContaining({
          code: "production_workflow_jsonl_missing",
        }),
        expect.objectContaining({
          code: "production_generated_article_marker_unclean",
        }),
        expect.objectContaining({
          code: "production_gui_turn_start_trace_missing",
        }),
        expect.objectContaining({
          code: "production_gui_current_app_server_methods_missing",
        }),
      ]),
    });
  });

  it("拒绝缺少 evidence/export workflow audit 摘要的 production GUI evidence", () => {
    const { evidenceExport: _evidenceExport, ...guiEvidenceWithoutExport } =
      READY_GUI_EVIDENCE;

    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: guiEvidenceWithoutExport,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        workflowAuditExportReady: false,
        workflowAuditEventCount: 0,
        workflowAuditMetadataOnly: false,
        workflowAuditRawContentExcluded: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_workflow_audit_export_missing",
        }),
        expect.objectContaining({
          code: "production_workflow_audit_export_empty",
        }),
        expect.objectContaining({
          code: "production_workflow_audit_not_metadata_only",
        }),
        expect.objectContaining({
          code: "production_workflow_audit_raw_content_included",
        }),
        expect.objectContaining({
          code: "production_workflow_audit_redaction_policy_missing",
        }),
      ]),
    });
  });

  it("拒绝缺少真实 resume lifecycle metadata 的 production GUI evidence", () => {
    const { runtimeActionResponse, ...guiEvidenceWithoutLifecycleMetadata } =
      READY_GUI_EVIDENCE;
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...guiEvidenceWithoutLifecycleMetadata,
        eventLogs: {
          workflowJsonl:
            "/tmp/lime-runtime/events/sessions/session_prod/workflow-events.jsonl",
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_workflow_resume_lifecycle_missing",
        }),
      ]),
      guiEvidence: {
        workflowResumeLifecycle: {
          auditEventsPresent: false,
          actionMetadataPresent: false,
        },
      },
    });
  });

  it("拒绝缺少右侧 workflow facts 隐藏显式证据的 production GUI evidence", () => {
    const {
      contentFactoryArticleWorkspaceWorkflowFactsHidden: _workflowFactsHidden,
      ...assertionsWithoutWorkflowFactsHidden
    } = READY_GUI_EVIDENCE.assertions;
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        assertions: assertionsWithoutWorkflowFactsHidden,
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        workflowFactsHidden: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_workflow_facts_visible",
        }),
      ]),
    });
  });

  it("拒绝右侧 workflow facts 可见的 production GUI evidence", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...READY_GUI_EVIDENCE,
        assertions: {
          ...READY_GUI_EVIDENCE.assertions,
          contentFactoryArticleWorkspaceWorkflowFactsHidden: false,
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        workflowFactsHidden: false,
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_workflow_facts_visible",
        }),
      ]),
    });
  });

  it("接受 runtime action response 的 workflowResume metadata 作为真实 lifecycle 证据", () => {
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: READY_GUI_EVIDENCE,
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: true,
      guiEvidence: {
        workflowResumeLifecycle: {
          auditEventsPresent: true,
          actionMetadataPresent: true,
          workflowRunId: "turn_prod:content-article",
        },
      },
      missingRequirements: [],
    });
  });

  it("拒绝把 legacy runtime resume contract 当作 action metadata", () => {
    const { runtimeActionResponse, ...guiEvidenceWithoutActionMetadata } =
      READY_GUI_EVIDENCE;
    const result = buildContentFactorySignedReleaseGate({
      bootstrap: READY_BOOTSTRAP,
      catalog: READY_CATALOG,
      expectedVersion: "2.2.2",
      fetchCloud: READY_FETCH_CLOUD,
      guiEvidence: {
        ...guiEvidenceWithoutActionMetadata,
        runtimeResumeContract: {
          decisions: [
            {
              actionId: "stale-review-action",
              decision: "approved",
              metadata: {
                workflowResume: {
                  stepId: "review",
                  workflowKey: "content_article_workflow",
                  workflowRunId: "turn_prod:stale",
                },
              },
            },
          ],
        },
      },
      preflight: READY_PREFLIGHT,
    });

    expect(result).toMatchObject({
      ready: false,
      status: "blocked",
      guiEvidence: {
        workflowResumeLifecycle: {
          actionMetadataPresent: false,
        },
      },
      missingRequirements: expect.arrayContaining([
        expect.objectContaining({
          code: "production_workflow_resume_lifecycle_missing",
        }),
      ]),
    });
  });

  it("生成 production evidence 模板时覆盖 gate 所需五类输入", () => {
    const template = buildContentFactorySignedReleaseEvidenceTemplate({
      expectedVersion: "2.2.2",
    });

    expect(template).toMatchObject({
      appId: "content-factory-app",
      expectedVersion: "2.2.2",
      catalog: {
        apps: [
          expect.objectContaining({
            appId: "content-factory-app",
            appVersion: "2.2.2",
            identity: expect.objectContaining({
              releaseId: "prod",
            }),
            signatureProof: expect.objectContaining({
              publicKeyId: expect.any(String),
            }),
          }),
        ],
      },
      fetchCloud: {
        packageHashMatched: true,
        signaturePolicy: "required",
        sourceKind: "cloud_release",
        signatureVerificationStatus: "verified",
      },
      preflight: {
        appServerInspect: {
          present: true,
        },
        package: {
          manifestHash: expect.any(String),
          packageHash: expect.any(String),
        },
        signature: {
          appSignatureYamlPresent: true,
          trustRootPresent: true,
        },
      },
      guiEvidence: {
        assertions: {
          contentFactoryArticleWorkspaceWorkflowFactsHidden: true,
          liveProviderUsed: true,
          turnStartViaElectronIpc: true,
          workflowResumeLifecyclePresent: true,
        },
        installedState: {
          manifestHashMatched: true,
          packageHashMatched: true,
          signaturePolicy: "required",
        },
        trace: {
          appServerHandleJsonLinesSeen: true,
        },
      },
    });
    expect(template.command).toContain("--evidence-dir .");
    expect(template.forbiddenMarkers).toContain("hostGenerationFixture");
  });

  it("CLI --check 写入 blocked gate JSON 并返回非零", () => {
    const outputDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-signed-release-gate-"),
    );
    const outputPath = path.join(outputDir, "gate.json");

    const result = spawnSync(
      process.execPath,
      [
        path.resolve("scripts/plugin/content-factory-signed-release-gate.mjs"),
        "--expected-version",
        "2.2.2",
        "--output",
        outputPath,
        "--check",
      ],
      { encoding: "utf8", env: { PATH: process.env.PATH } },
    );

    expect(result.status).toBe(1);
    expect(result.stdout).toContain("missingCodes=");
    expect(JSON.parse(fs.readFileSync(outputPath, "utf8"))).toMatchObject({
      ready: false,
      status: "blocked",
    });
  });

  it("CLI --write-template-dir 写入 production evidence 模板文件", () => {
    const outputDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-signed-release-template-"),
    );

    const result = spawnSync(
      process.execPath,
      [
        path.resolve("scripts/plugin/content-factory-signed-release-gate.mjs"),
        "--expected-version",
        "2.2.2",
        "--write-template-dir",
        outputDir,
      ],
      { encoding: "utf8", env: { PATH: process.env.PATH } },
    );

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("templateDir=");
    for (const fileName of [
      "content-factory-production-catalog.template.json",
      "content-factory-production-bootstrap.template.json",
      "content-factory-production-preflight.template.json",
      "content-factory-fetch-cloud-evidence.template.json",
      "content-factory-gui-evidence.template.json",
      "content-factory-signed-release-gate.template.json",
    ]) {
      expect(fs.existsSync(path.join(outputDir, fileName))).toBe(true);
    }
    const command = JSON.parse(
      fs.readFileSync(
        path.join(
          outputDir,
          "content-factory-signed-release-gate.template.json",
        ),
        "utf8",
      ),
    ).command;
    expect(command).toContain("--check");
    expect(command).toContain("--evidence-dir");
    expect(command).toContain(outputDir);
    expect(
      JSON.parse(
        fs.readFileSync(
          path.join(
            outputDir,
            "content-factory-signed-release-gate.template.json",
          ),
          "utf8",
        ),
      ).resultFile,
    ).toBe(CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME);
  });

  it("CLI --check 在全部 production evidence ready 时返回 0", () => {
    const outputDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "content-factory-signed-release-gate-ready-"),
    );
    const catalogPath = path.join(outputDir, "catalog.json");
    const bootstrapPath = path.join(outputDir, "bootstrap.json");
    const preflightPath = path.join(outputDir, "preflight.json");
    const fetchCloudPath = path.join(outputDir, "fetch-cloud.json");
    const guiEvidencePath = path.join(outputDir, "gui.json");
    const outputPath = path.join(outputDir, "gate.json");
    writeJsonFile(catalogPath, READY_CATALOG);
    writeJsonFile(bootstrapPath, READY_BOOTSTRAP);
    writeJsonFile(preflightPath, READY_PREFLIGHT);
    writeJsonFile(fetchCloudPath, READY_FETCH_CLOUD);
    writeJsonFile(guiEvidencePath, READY_GUI_EVIDENCE);

    const result = spawnSync(
      process.execPath,
      [
        path.resolve("scripts/plugin/content-factory-signed-release-gate.mjs"),
        "--catalog",
        catalogPath,
        "--bootstrap",
        bootstrapPath,
        "--fetch-cloud",
        fetchCloudPath,
        "--preflight",
        preflightPath,
        "--gui-evidence",
        guiEvidencePath,
        "--expected-version",
        "2.2.2",
        "--output",
        outputPath,
        "--check",
      ],
      { encoding: "utf8", env: { PATH: process.env.PATH } },
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(fs.readFileSync(outputPath, "utf8"))).toMatchObject({
      ready: true,
      status: "ready",
      missingRequirements: [],
    });
  });

  it("CLI --evidence-dir 读取模板目录并默认写入 result JSON", () => {
    const outputDir = fs.mkdtempSync(
      path.join(
        os.tmpdir(),
        "content-factory-signed-release-gate-evidence-dir-",
      ),
    );
    writeJsonFile(
      path.join(
        outputDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.catalog,
      ),
      READY_CATALOG,
    );
    writeJsonFile(
      path.join(
        outputDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.bootstrap,
      ),
      READY_BOOTSTRAP,
    );
    writeJsonFile(
      path.join(
        outputDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.fetchCloud,
      ),
      READY_FETCH_CLOUD,
    );
    writeJsonFile(
      path.join(
        outputDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.preflight,
      ),
      READY_PREFLIGHT,
    );
    writeJsonFile(
      path.join(
        outputDir,
        CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.guiEvidence,
      ),
      READY_GUI_EVIDENCE,
    );

    const result = spawnSync(
      process.execPath,
      [
        path.resolve("scripts/plugin/content-factory-signed-release-gate.mjs"),
        "--evidence-dir",
        outputDir,
        "--expected-version",
        "2.2.2",
        "--check",
      ],
      { encoding: "utf8", env: { PATH: process.env.PATH } },
    );
    const resultPath = path.join(
      outputDir,
      CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
    );

    expect(result.status).toBe(0);
    expect(result.stdout).toContain(`evidenceDir=${outputDir}`);
    expect(result.stdout).toContain(`output=${resultPath}`);
    expect(JSON.parse(fs.readFileSync(resultPath, "utf8"))).toMatchObject({
      ready: true,
      status: "ready",
      missingRequirements: [],
    });
  });
});
