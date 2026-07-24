import fs from "node:fs";
import path from "node:path";

import {
  APP_ID,
  CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
  CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES,
} from "./plugin-content-factory-signed-release-gate-constants.mjs";

export function buildContentFactorySignedReleaseEvidenceTemplate(input = {}) {
  const appId = input.appId || APP_ID;
  const expectedVersion = input.expectedVersion || "REPLACE_WITH_VERSION";
  const packageHash =
    input.packageHash || "sha256:REPLACE_WITH_64_HEX_PRODUCTION_PACKAGE_HASH";
  const manifestHash =
    input.manifestHash || "sha256:REPLACE_WITH_64_HEX_PRODUCTION_MANIFEST_HASH";
  const payloadHash =
    input.payloadHash || "sha256:REPLACE_WITH_64_HEX_CANONICAL_PAYLOAD_HASH";
  const publicKeyId =
    input.publicKeyId || "REPLACE_WITH_PRODUCTION_TRUST_ROOT_PUBLIC_KEY_ID";
  const algorithm = input.algorithm || "Ed25519";
  const signedAt = input.signedAt || "2026-07-03T00:00:00.000Z";
  const releaseId = input.releaseId || "prod";
  const signatureRef = `sigstore:${appId}@${expectedVersion}:${releaseId}`;
  const packageUrl =
    input.packageUrl ||
    `https://updates.limeai.run/plugins/${appId}/prod/${appId}-${expectedVersion}.lapp`;
  const workflowJsonl =
    input.workflowJsonl ||
    "/path/to/lime/runtime/events/sessions/session_<id>/workflow-events.jsonl";
  const catalog = {
    apps: [
      {
        appId,
        appVersion: expectedVersion,
        identity: {
          appId,
          appVersion: expectedVersion,
          manifestHash,
          packageHash,
          releaseId,
          signatureRef,
          sourceKind: "cloud_release",
          sourceUri: packageUrl,
        },
        packageUrl,
        signatureProof: {
          algorithm,
          payloadHash,
          publicKeyId,
          signature: "REPLACE_WITH_BASE64_SIGNATURE",
          signedAt,
        },
      },
    ],
  };
  const bootstrap = {
    pluginSignatureTrustRoots: [
      {
        algorithm,
        appIds: [appId],
        notAfter: "2027-01-01T00:00:00.000Z",
        notBefore: "2026-01-01T00:00:00.000Z",
        publicKeyId,
        publicKeyPem:
          "-----BEGIN PUBLIC KEY-----\\nREPLACE_WITH_PUBLIC_KEY\\n-----END PUBLIC KEY-----",
        revoked: false,
      },
    ],
  };
  const fetchCloud = {
    appId,
    descriptor: {
      appId,
      manifestHash,
      packageHash,
      packageUrl,
      releaseId,
      signatureRef,
      sourceUri: packageUrl,
      version: expectedVersion,
    },
    manifestHash,
    manifestHashMatched: true,
    packageHash,
    packageHashMatched: true,
    packageVerificationStatus: "verified",
    signatureProof: {
      algorithm,
      payloadHash,
      publicKeyId,
      signature: "REPLACE_WITH_BASE64_SIGNATURE",
      signedAt,
    },
    signatureRef,
    signaturePolicy: "required",
    signatureVerificationStatus: "verified",
    sourceKind: "cloud_release",
    status: "ready",
  };
  const preflight = {
    schemaVersion: 1,
    appId,
    status: "ready",
    ready: true,
    expectedVersion,
    package: {
      appId,
      exists: true,
      manifestHash,
      missingEntries: [],
      packageHash,
      validZip: true,
      version: expectedVersion,
    },
    appServerInspect: {
      manifestHash,
      packageHash,
      present: true,
      sourceKind: "local_folder",
    },
    signature: {
      algorithm,
      appSignatureYamlPresent: true,
      payloadHash,
      payloadHashValid: true,
      publicKeyId,
      signaturePresent: true,
      signatureRef,
      signedAt,
      trustRootPresent: true,
      trustRootPublicKeyId: publicKeyId,
    },
    catalog: {
      manifestHash,
      packageHash,
      packageUrl,
      present: true,
      releaseId,
      signatureProofPresent: true,
      signatureRef,
      sourceKind: "cloud_release",
      version: expectedVersion,
    },
    bootstrap: {
      present: true,
      trustRootCount: 1,
    },
    fetchCloud: {
      manifestHash,
      manifestHashMatched: true,
      packageHash,
      packageHashMatched: true,
      packageVerificationStatus: "verified",
      present: true,
      ready: true,
      signatureProofPresent: true,
      signatureRef,
      signaturePolicy: "required",
      signatureVerificationStatus: "verified",
      sourceKind: "cloud_release",
      status: "ready",
    },
    publishReadiness: {
      configured: true,
      note: "Non-sensitive operator readiness only. Values are never written to evidence and do not replace catalog/bootstrap/fetchCloud proof.",
      requirements: [
        {
          configured: true,
          env: [
            "PLUGIN_SIGNING_PRIVATE_KEY_PEM",
            "AGENT_APP_SIGNING_PRIVATE_KEY_PEM",
          ],
          key: "signingPrivateKey",
        },
        {
          configured: true,
          env: ["LIME_AGENT_APP_STUDIO_TOKEN"],
          key: "studioToken",
        },
        {
          configured: true,
          env: ["LIMECORE_TENANT_ID"],
          key: "tenantId",
        },
        {
          configured: true,
          env: ["LIME_AGENT_APP_STUDIO_API_BASE", "LIMECORE_API_BASE"],
          key: "apiBase",
        },
        {
          configured: true,
          env: ["CONTENT_FACTORY_PACKAGE_URL"],
          key: "packageUrl",
          remoteHttps: true,
        },
      ],
    },
    missingRequirements: [],
  };
  const guiEvidence = {
    schemaVersion: "content-factory-production-gui-evidence.v1",
    appId,
    assertions: {
      articleDraftDocumentPresent: true,
      contentFactoryArticleWorkspaceWorkflowFactsHidden: true,
      liveProviderUsed: true,
      turnStartViaElectronIpc: true,
      workflowResumeLifecyclePresent: true,
    },
    eventLogs: {
      workflowJsonl,
      workflowJsonlEventCount: 16,
      workflowResumeEvents: [
        {
          eventType: "workflow.step.resuming",
          payload: {
            actionId: "article-draft-review",
            decision: "approved",
            stepId: "draft",
            workflowKey: "content_article_workflow",
            workflowRunId: "turn_<id>:content-article",
          },
        },
        {
          eventType: "workflow.run.resuming",
          payload: {
            actionId: "article-draft-review",
            decision: "approved",
            stepId: "draft",
            workflowKey: "content_article_workflow",
            workflowRunId: "turn_<id>:content-article",
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
    runtimeActionResponse: {
      actionId: "article-draft-review",
      confirmed: true,
      method: "workflow/respond",
      metadata: {
        workflowResume: {
          stepId: "draft",
          workflowKey: "content_article_workflow",
          workflowRunId: "turn_<id>:content-article",
        },
      },
    },
    installedState: {
      appVersion: expectedVersion,
      cloudReleaseEvidenceStatus: "ready",
      manifestHash,
      manifestHashMatched: true,
      packageHash,
      packageHashMatched: true,
      packageVerificationStatus: "verified",
      releaseId,
      signaturePolicy: "required",
      signatureRef,
      signatureVerificationStatus: "verified",
      sourceKind: "cloud_release",
    },
    providerEvidence: {
      liveProviderUsed: true,
      productionRoute: true,
    },
    cdp: {
      attached: true,
      usedRealElectron: true,
    },
    readModel: {
      articleDraftDocumentLength: 3153,
      articleDraftDocumentPresent: true,
      generatedArticleMarkerClean: true,
      hostManagedGenerationOutputIds: ["article-draft-document"],
      hostManagedGenerationStatus: "completed",
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
      workflowResumeBindingCount: 1,
    },
  };
  return {
    schemaVersion: "content-factory-signed-release-evidence-template.v1",
    appId,
    expectedVersion,
    files: CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES,
    catalog,
    bootstrap,
    fetchCloud,
    preflight,
    guiEvidence,
    command: [
      "npm run plugin:content-factory-signed-release-gate --",
      "--evidence-dir .",
      `--expected-version ${expectedVersion}`,
      "--check",
    ].join(" "),
    forbiddenMarkers: [
      "fixture",
      "cloudReleaseFixture",
      "hostGenerationFixture",
      "localhost",
      "127.0.0.1",
      "signature_missing",
      "not_configured",
      "host_generation_unavailable",
    ],
    note: "Fill these files from production preflight, LimeCore catalog/bootstrap, App Server fetchCloud verification, and real Lime Desktop GUI evidence. Do not paste private signing keys or Provider API keys.",
  };
}

export function writeContentFactorySignedReleaseEvidenceTemplateDir(
  dirPath,
  input = {},
) {
  const template = buildContentFactorySignedReleaseEvidenceTemplate(input);
  const resolvedDir = path.resolve(process.cwd(), dirPath);
  fs.mkdirSync(resolvedDir, { recursive: true });
  const command = [
    "npm run plugin:content-factory-signed-release-gate --",
    `--evidence-dir ${resolvedDir}`,
    `--expected-version ${template.expectedVersion}`,
    "--check",
  ].join(" ");
  const outputs = {
    [CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.preflight]:
      template.preflight,
    [CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.catalog]:
      template.catalog,
    [CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.bootstrap]:
      template.bootstrap,
    [CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.fetchCloud]:
      template.fetchCloud,
    [CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.guiEvidence]:
      template.guiEvidence,
    [CONTENT_FACTORY_SIGNED_RELEASE_GATE_TEMPLATE_FILE_NAMES.readme]: {
      schemaVersion: template.schemaVersion,
      appId: template.appId,
      expectedVersion: template.expectedVersion,
      command,
      files: template.files,
      resultFile: CONTENT_FACTORY_SIGNED_RELEASE_GATE_RESULT_FILE_NAME,
      forbiddenMarkers: template.forbiddenMarkers,
      note: template.note,
    },
  };
  const written = [];
  for (const [fileName, value] of Object.entries(outputs)) {
    const filePath = path.join(resolvedDir, fileName);
    fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
    written.push(filePath);
  }
  return {
    dir: resolvedDir,
    files: written,
    template,
  };
}
