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
const RELEASE_ID = "prod";
const SIGNATURE_REF = `sigstore:content-factory-app@2.2.2:${RELEASE_ID}`;

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
      schemaVersion: "plugin-cloud-release-signature/v1",
      payloadHash:
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      payloadHashValid: true,
      publicKeyId: "content-factory-prod-root-2026",
      signatureCryptographicVerificationStatus: "verified",
      signaturePresent: true,
      signaturePayloadHashMatched: true,
      signatureRef: SIGNATURE_REF,
      signatureVerificationFailureCodes: [],
      signedAt: "2026-07-05T00:00:00.000Z",
      trustRootPresent: true,
      trustRootPublicKeyPresent: true,
      trustRootPublicKeyId: "content-factory-prod-root-2026",
    },
    publishReadiness: {
      configured: false,
      requirements: [
        {
          configured: false,
          env: ["LIME_AGENT_APP_STUDIO_TOKEN"],
          key: "studioToken",
        },
      ],
    },
    missingRequirements: [],
  };
}

function studioDryRun(blockers = []) {
  return {
    mode: "dry-run",
    plan: {
      appId: "content-factory-app",
      channel: "stable",
      version: "2.2.2",
    },
    releaseReadiness: {
      appId: "content-factory-app",
      blockers,
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
      ready: blockers.length === 0,
      warnings: [],
    },
  };
}

function fetchCloudEvidence() {
  return {
    schemaVersion: "content-factory-fetch-cloud-evidence.v1",
    appId: "content-factory-app",
    sourceKind: "cloud_release",
    status: "blocked",
    packageHashMatched: true,
    manifestHashMatched: true,
    packageVerificationStatus: "verified",
    signaturePolicy: "required",
    signatureVerificationStatus: "declared",
    descriptor: {
      appId: "content-factory-app",
      manifestHash: MANIFEST_HASH,
      packageHash: PACKAGE_HASH,
    },
  };
}

function readyFetchCloudEvidence() {
  return {
    schemaVersion: "content-factory-fetch-cloud-evidence.v1",
    appId: "content-factory-app",
    descriptor: {
      manifestHash: MANIFEST_HASH,
      packageHash: PACKAGE_HASH,
      sourceUri: "https://packages.example.com/content-factory-app-2.2.2.lapp",
    },
    manifestHash: MANIFEST_HASH,
    manifestHashMatched: true,
    packageHash: PACKAGE_HASH,
    packageHashMatched: true,
    packageVerificationStatus: "verified",
    signaturePolicy: "required",
    signatureProof: {
      algorithm: "Ed25519",
      payloadHash:
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      publicKeyId: "content-factory-prod-root-2026",
      signature: "base64-signature",
      signedAt: "2026-07-05T00:00:00.000Z",
    },
    signatureRef: SIGNATURE_REF,
    signatureVerificationStatus: "verified",
    sourceKind: "cloud_release",
    status: "ready",
  };
}

function bootstrapEvidence() {
  return {
    pluginSignatureTrustRoots: [
      {
        algorithm: "Ed25519",
        publicKey:
          "-----BEGIN PUBLIC KEY-----\nproduction-test-key\n-----END PUBLIC KEY-----",
        publicKeyId: "content-factory-prod-root-2026",
      },
    ],
  };
}

function guiEvidence() {
  return {
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
      workflowJsonl: "/tmp/content-factory-production/workflow-events.jsonl",
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
      sourceKind: "cloud_release",
      signatureVerificationStatus: "verified",
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
}

function catalogEvidence() {
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
          payloadHash:
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
          publicKeyId: "content-factory-prod-root-2026",
          signature: "base64-signature",
          signedAt: "2026-07-05T00:00:00.000Z",
        },
      },
    ],
  };
}

function commandValue(args, flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : "";
}

function createPipelineFixture() {
  const root = fs.mkdtempSync(
    path.join(os.tmpdir(), "content-factory-production-readiness-pipeline-"),
  );
  const contentFactoryDir = path.join(root, "content-factory-app");
  fs.mkdirSync(contentFactoryDir, { recursive: true });
  fs.mkdirSync(path.join(contentFactoryDir, "scripts"), { recursive: true });
  fs.writeFileSync(
    path.join(contentFactoryDir, "scripts", "sign-release.mjs"),
    "#!/usr/bin/env node\n",
    "utf8",
  );
  writeJson(path.join(contentFactoryDir, "package.json"), {
    name: "content-factory-app",
    version: "2.2.2",
  });
  return {
    catalogPath: path.join(root, "catalog.json"),
    contentFactoryDir,
    outputDir: path.join(root, "out"),
    preflightScript: path.join(root, "fake-preflight.mjs"),
    releaseEvidenceScript: path.join(root, "fake-release-evidence.mjs"),
    studioCli: path.join(root, "fake-studio-cli.mjs"),
    studioDir: path.join(root, "studio"),
  };
}

describe("content factory production readiness pipeline", () => {
  it("is exposed as an explicit npm production readiness pipeline", () => {
    const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
    expect(
      packageJson.scripts[
        "plugin:content-factory-production-readiness-pipeline"
      ],
    ).toBe(
      "node scripts/plugin/content-factory-production-readiness-pipeline.mjs",
    );
  });

  it("marks ready only when production preflight, Studio, bundle, and GUI evidence are all ready", () => {
    const fixture = createPipelineFixture();
    const bootstrapPath = path.join(
      fixture.contentFactoryDir,
      "bootstrap.json",
    );
    const fetchCloudPath = path.join(
      fixture.contentFactoryDir,
      "fetch-cloud.json",
    );
    const guiEvidencePath = path.join(
      fixture.contentFactoryDir,
      "gui-evidence.json",
    );
    writeJson(fixture.catalogPath, catalogEvidence());
    writeJson(bootstrapPath, bootstrapEvidence());
    writeJson(fetchCloudPath, readyFetchCloudEvidence());
    writeJson(guiEvidencePath, guiEvidence());
    const runner = (command, args) => {
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), {
          ...readyPreflight(),
          bootstrap: {
            present: true,
            trustRootCount: 1,
          },
          catalog: {
            manifestHash: MANIFEST_HASH,
            packageHash: PACKAGE_HASH,
            packageUrl:
              "https://packages.example.com/content-factory-app-2.2.2.lapp",
            present: true,
            signatureProofPresent: true,
            releaseId: RELEASE_ID,
            signatureRef: SIGNATURE_REF,
            sourceKind: "cloud_release",
            version: "2.2.2",
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
        });
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      appServerBin: "/tmp/app-server",
      bootstrapPath,
      catalogPath: fixture.catalogPath,
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      fetchCloudPath,
      guiEvidencePath,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(result.pipeline).toMatchObject({
      ready: true,
      status: "ready",
      blockers: [],
      steps: {
        evidenceBundle: {
          ready: true,
          status: "ready",
        },
        readinessReport: {
          ready: true,
          status: "ready",
        },
        studioDryRun: {
          ready: true,
        },
      },
    });
    expect(result.readinessReport).toMatchObject({
      ready: true,
      signedGate: {
        missingCount: 0,
        ready: true,
      },
    });
  });

  it("runs read-only preflight and Studio dry-run before bundling readiness evidence", () => {
    const fixture = createPipelineFixture();
    writeJson(fixture.catalogPath, catalogEvidence());
    const calls = [];
    const runner = (command, args) => {
      calls.push({ args, command });
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        writeJson(
          commandValue(args, "--fetch-cloud-output"),
          fetchCloudEvidence(),
        );
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(
            studioDryRun([{ code: "production_studio_token_missing" }]),
          ),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      appServerBin: "/tmp/app-server",
      catalogPath: fixture.catalogPath,
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      fetchCloudFromCatalog: true,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(result.pipeline.status).toBe("blocked");
    expect(result.pipeline.ready).toBe(false);
    expect(result.pipeline.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "production_trust_roots_missing",
        }),
        expect.objectContaining({
          code: "production_studio_token_missing",
        }),
      ]),
    );
    expect(result.pipeline.blockerPlan).toMatchObject({
      nextPhase: {
        id: "studio_publish_inputs",
        owner: "operator",
      },
    });
    expect(fs.existsSync(result.files.studioDryRun)).toBe(true);
    expect(fs.existsSync(result.files.bundle)).toBe(true);
    expect(fs.existsSync(result.files.readinessReport)).toBe(true);
    expect(fs.existsSync(result.files.pipeline)).toBe(true);

    const allArgs = calls.flatMap((call) => call.args);
    expect(calls.map((call) => call.args[0])).toEqual([
      fixture.studioCli,
      fixture.preflightScript,
    ]);
    expect(allArgs).toContain("--dry-run");
    expect(allArgs).toContain("--fetch-cloud-from-catalog");
    expect(allArgs).toContain("--fetch-cloud-output");
    expect(allArgs).not.toContain("--publish");
    expect(allArgs).not.toContain("--check");

    const pipelineJson = JSON.parse(
      fs.readFileSync(
        path.join(
          fixture.outputDir,
          CONTENT_FACTORY_PRODUCTION_READINESS_PIPELINE_FILE_NAME,
        ),
        "utf8",
      ),
    );
    expect(pipelineJson.steps.studioDryRun.present).toBe(true);
    expect(pipelineJson.fetchCloudFromCatalog).toMatchObject({
      executed: true,
      requested: true,
    });
    expect(pipelineJson.blockerPlan.nextPhase.id).toBe("studio_publish_inputs");
    expect(fs.existsSync(pipelineJson.files.fetchCloud)).toBe(true);
    const bundle = JSON.parse(
      fs.readFileSync(pipelineJson.files.bundle, "utf8"),
    );
    expect(bundle.sources.fetchCloud).toMatchObject({ present: true });
    expect(JSON.stringify(pipelineJson)).not.toContain("developer-token");
  });

  it("passes package-dir signature inputs to preflight and Studio dry-run by default", () => {
    const fixture = createPipelineFixture();
    const appSignaturePath = path.join(
      fixture.contentFactoryDir,
      "app.signature.yaml",
    );
    const trustRootPath = path.join(
      fixture.contentFactoryDir,
      "plugin-signature-trust-root.json",
    );
    fs.writeFileSync(appSignaturePath, "signature:\n  package: {}\n", "utf8");
    writeJson(trustRootPath, {
      algorithm: "Ed25519",
      publicKey: "public-key",
      publicKeyId: "content-factory-prod-root-2026",
    });
    const calls = [];
    const runner = (command, args) => {
      calls.push({ args, command });
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    const preflightArgs = calls.find(
      (call) => call.args[0] === fixture.preflightScript,
    )?.args;
    const studioArgs = calls.find(
      (call) => call.args[0] === fixture.studioCli,
    )?.args;
    expect(commandValue(preflightArgs, "--app-signature")).toBe(
      appSignaturePath,
    );
    expect(commandValue(preflightArgs, "--trust-root")).toBe(trustRootPath);
    expect(commandValue(studioArgs, "--app-signature")).toBe(appSignaturePath);
  });

  it("passes packageUrl through child env without writing the raw URL into pipeline evidence", () => {
    const fixture = createPipelineFixture();
    const packageUrl =
      "https://packages.example.com/content-factory-app-2.2.2.lapp";
    const observedPackageUrls = [];
    const runner = (command, args, options = {}) => {
      observedPackageUrls.push(options.env?.CONTENT_FACTORY_PACKAGE_URL || "");
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      outputDir: fixture.outputDir,
      packageUrl,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(observedPackageUrls).toEqual([packageUrl, packageUrl]);
    expect(result.pipeline.operatorReadiness.inputs.packageUrl).toEqual({
      configured: true,
      remoteHttps: true,
    });
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "packageUrl",
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "--package-url <https-url>",
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "requires local env: PLUGIN_SIGNING_PRIVATE_KEY_PEM, LIME_AGENT_APP_STUDIO_TOKEN",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "<private-key>",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "<token>",
    );
    expect(result.pipeline.operatorReadiness.signingCommandHint).toMatchObject({
      hasCurrentHashes: true,
      manifestHash: MANIFEST_HASH,
      packageHash: PACKAGE_HASH,
      scriptPresent: true,
    });
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain("--package-url <https-url>");
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain(PACKAGE_HASH);
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain(MANIFEST_HASH);
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain("--private-key-env PLUGIN_SIGNING_PRIVATE_KEY_PEM");
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).not.toContain("<private-key>");
    expect(JSON.stringify(result.pipeline)).not.toContain(packageUrl);
    expect(
      fs.readFileSync(
        path.join(
          fixture.outputDir,
          CONTENT_FACTORY_PRODUCTION_READINESS_PIPELINE_FILE_NAME,
        ),
        "utf8",
      ),
    ).not.toContain(packageUrl);
  });

  it("tracks release id and public key id as non-secret signing inputs", () => {
    const fixture = createPipelineFixture();
    const packageUrl =
      "https://packages.example.com/content-factory-app-2.2.2.lapp";
    const runner = (command, args) => {
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      outputDir: fixture.outputDir,
      packageUrl,
      preflightScript: fixture.preflightScript,
      publicKeyId: "content-factory-prod-root-2026",
      releaseId: "plugin-release-test",
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(result.pipeline.operatorReadiness.inputs).toMatchObject({
      publicKeyId: { configured: true },
      releaseId: { configured: true },
    });
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "publicKeyId",
    );
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "releaseId",
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "--release-id <release-id>",
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "--public-key-id <public-key-id>",
    );
    expect(result.pipeline.operatorReadiness.signingCommandHint).toMatchObject({
      publicKeyId: "content-factory-prod-root-2026",
      releaseId: "plugin-release-test",
    });
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain("--release-id plugin-release-test");
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain("--public-key-id content-factory-prod-root-2026");
    expect(
      result.pipeline.operatorReadiness.signingCommandHint.command,
    ).toContain("sigstore:content-factory-app@2.2.2:plugin-release-test");
    expect(JSON.stringify(result.pipeline)).not.toContain(packageUrl);
  });

  it("passes tenant, api base, and token env to child commands without leaking the token", () => {
    const fixture = createPipelineFixture();
    const secretToken = "developer-token-secret-value";
    const observedEnv = [];
    const runner = (command, args, options = {}) => {
      observedEnv.push({
        apiBase: options.env?.LIME_AGENT_APP_STUDIO_API_BASE || "",
        tenantId: options.env?.LIMECORE_TENANT_ID || "",
        token: options.env?.LIME_AGENT_APP_STUDIO_TOKEN || "",
      });
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      apiBase: "https://api.example.com",
      contentFactoryDir: fixture.contentFactoryDir,
      env: {
        PATH: process.env.PATH,
        STUDIO_TOKEN_FOR_TEST: secretToken,
      },
      expectedVersion: "2.2.2",
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
      studioTokenEnv: "STUDIO_TOKEN_FOR_TEST",
      tenantId: "tenant_prod_123",
    });

    expect(observedEnv).toEqual([
      {
        apiBase: "https://api.example.com",
        tenantId: "tenant_prod_123",
        token: secretToken,
      },
      {
        apiBase: "https://api.example.com",
        tenantId: "tenant_prod_123",
        token: secretToken,
      },
    ]);
    expect(result.pipeline.operatorReadiness.inputs).toMatchObject({
      apiBase: {
        configured: true,
        envName: null,
      },
      studioToken: {
        configured: true,
        envName: "STUDIO_TOKEN_FOR_TEST",
      },
      tenantId: {
        configured: true,
        envName: null,
      },
    });
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "apiBase",
    );
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "studioToken",
    );
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "tenantId",
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "--studio-token-env LIME_AGENT_APP_STUDIO_TOKEN",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "<token>",
    );
    expect(JSON.stringify(result.pipeline)).not.toContain(secretToken);
    expect(
      fs.readFileSync(
        path.join(
          fixture.outputDir,
          CONTENT_FACTORY_PRODUCTION_READINESS_PIPELINE_FILE_NAME,
        ),
        "utf8",
      ),
    ).not.toContain(secretToken);
  });

  it("skips fetchCloud-from-catalog without catalog so preflight evidence still exists", () => {
    const fixture = createPipelineFixture();
    const calls = [];
    const runner = (command, args) => {
      calls.push({ args, command });
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      fetchCloudFromCatalog: true,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    const allArgs = calls.flatMap((call) => call.args);
    expect(allArgs).not.toContain("--fetch-cloud-from-catalog");
    expect(allArgs).not.toContain("--fetch-cloud-output");
    expect(result.pipeline.fetchCloudFromCatalog).toMatchObject({
      executed: false,
      requested: true,
      skippedReason: "catalog_missing",
    });
    expect(result.pipeline.steps.preflight.present).toBe(true);
    expect(result.pipeline.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "production_catalog_missing" }),
      ]),
    );
    expect(result.pipeline.operatorReadiness).toMatchObject({
      ready: false,
      missingKeys: expect.arrayContaining([
        "appSignature",
        "signingPrivateKey",
        "releaseId",
        "publicKeyId",
        "trustRoot",
        "catalog",
        "bootstrap",
        "fetchCloudEvidence",
        "guiEvidence",
      ]),
    });
    expect(result.pipeline.operatorReadiness.missingActions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "appSignature",
          action: expect.stringContaining("--generate-signature-proof"),
        }),
        expect.objectContaining({
          key: "trustRoot",
          action: expect.stringContaining("--generate-signature-proof"),
        }),
        expect.objectContaining({
          key: "releaseId",
          action: expect.stringContaining("--release-id <release-id>"),
        }),
        expect.objectContaining({
          key: "publicKeyId",
          action: expect.stringContaining("--public-key-id <public-key-id>"),
        }),
        expect.objectContaining({
          key: "fetchCloudEvidence",
          action: expect.stringContaining("--fetch-cloud-from-catalog"),
        }),
        expect.objectContaining({
          key: "guiEvidence",
          action: expect.stringContaining("Electron CDP"),
        }),
      ]),
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "--generate-signature-proof",
    );
    expect(result.pipeline.operatorReadiness.commandHint).toContain(
      "--signing-private-key-env PLUGIN_SIGNING_PRIVATE_KEY_PEM",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "<private-key>",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "<token>",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "--app-signature <app.signature.yaml>",
    );
    expect(result.pipeline.operatorReadiness.commandHint).not.toContain(
      "--trust-root <plugin-signature-trust-root.json>",
    );
    expect(result.pipeline.blockers.map((item) => item.code)).not.toContain(
      "production_preflight_missing",
    );
  });

  it("fetches production catalog and bootstrap evidence before preflight when explicitly requested", () => {
    const fixture = createPipelineFixture();
    const packageUrl =
      "https://packages.example.com/content-factory-app-2.2.2.lapp";
    const secretToken = "developer-token-secret-value";
    const calls = [];
    const runner = (command, args, options = {}) => {
      calls.push({ args, command, env: options.env || {} });
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      if (args[0] === fixture.releaseEvidenceScript) {
        expect(commandValue(args, "--api-base")).toBe(
          "https://api.example.com",
        );
        expect(commandValue(args, "--tenant-id")).toBe("tenant_prod_123");
        expect(commandValue(args, "--studio-token-env")).toBe(
          "STUDIO_TOKEN_FOR_TEST",
        );
        expect(options.env.STUDIO_TOKEN_FOR_TEST).toBe(secretToken);
        writeJson(commandValue(args, "--catalog-output"), catalogEvidence());
        writeJson(
          commandValue(args, "--bootstrap-output"),
          bootstrapEvidence(),
        );
        writeJson(commandValue(args, "--output"), {
          bootstrap: {
            trustRootCount: 1,
          },
          catalog: {
            appFound: true,
            packageUrlPresent: true,
            signatureProofPresent: true,
          },
          ready: true,
          requested: true,
          status: "ready",
        });
        return {
          status: 0,
          stdout: "release evidence ok\n",
          stderr: "",
        };
      }
      if (args[0] === fixture.preflightScript) {
        const catalogPath = commandValue(args, "--catalog");
        const bootstrapPath = commandValue(args, "--bootstrap");
        expect(catalogPath).toMatch(
          /content-factory-production-catalog\.json$/,
        );
        expect(bootstrapPath).toMatch(
          /content-factory-production-bootstrap\.json$/,
        );
        expect(args).toContain("--fetch-cloud-from-catalog");
        writeJson(
          commandValue(args, "--fetch-cloud-output"),
          fetchCloudEvidence(),
        );
        writeJson(commandValue(args, "--output"), {
          ...readyPreflight(),
          bootstrap: {
            present: true,
            trustRootCount: 1,
          },
          catalog: {
            manifestHash: MANIFEST_HASH,
            packageHash: PACKAGE_HASH,
            packageUrl,
            present: true,
            releaseId: RELEASE_ID,
            signatureProofPresent: true,
            signatureRef: SIGNATURE_REF,
            sourceKind: "cloud_release",
            version: "2.2.2",
          },
          fetchCloud: {
            present: true,
            ready: false,
            status: "blocked",
          },
        });
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      apiBase: "https://api.example.com",
      contentFactoryDir: fixture.contentFactoryDir,
      env: {
        PATH: process.env.PATH,
        STUDIO_TOKEN_FOR_TEST: secretToken,
      },
      expectedVersion: "2.2.2",
      fetchCloudFromCatalog: true,
      fetchProductionReleaseEvidence: true,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      releaseEvidenceScript: fixture.releaseEvidenceScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
      studioTokenEnv: "STUDIO_TOKEN_FOR_TEST",
      tenantId: "tenant_prod_123",
    });

    expect(calls.map((call) => call.args[0])).toEqual([
      fixture.studioCli,
      fixture.releaseEvidenceScript,
      fixture.preflightScript,
    ]);
    expect(result.pipeline.steps.productionReleaseEvidence).toMatchObject({
      ready: true,
      requested: true,
      status: "ready",
    });
    expect(result.pipeline.fetchCloudFromCatalog).toMatchObject({
      executed: true,
      requested: true,
    });
    expect(result.pipeline.blockers.map((item) => item.code)).not.toContain(
      "production_release_evidence_inputs_missing",
    );
    expect(fs.existsSync(result.pipeline.files.productionCatalog)).toBe(true);
    expect(fs.existsSync(result.pipeline.files.productionBootstrap)).toBe(true);
    expect(JSON.stringify(result.pipeline)).not.toContain(secretToken);
    expect(JSON.stringify(result.pipeline)).not.toContain(packageUrl);
  });

  it("expands fetched release evidence missing requirements into phaseable blockers", () => {
    const fixture = createPipelineFixture();
    const secretToken = "developer-token-secret-value";
    const calls = [];
    const runner = (command, args, options = {}) => {
      calls.push({ args, command, env: options.env || {} });
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      if (args[0] === fixture.releaseEvidenceScript) {
        writeJson(commandValue(args, "--catalog-output"), catalogEvidence());
        writeJson(
          commandValue(args, "--bootstrap-output"),
          bootstrapEvidence(),
        );
        writeJson(commandValue(args, "--output"), {
          bootstrap: {
            matchingTrustRootPresent: false,
            trustRootCount: 1,
          },
          catalog: {
            appFound: true,
            signatureProofPublicKeyId: "content-factory-prod-root-2026",
          },
          missingRequirements: ["bootstrapMatchingTrustRoot"],
          ready: false,
          requested: true,
          status: "blocked",
        });
        return {
          status: 0,
          stdout: "release evidence blocked\n",
          stderr: "",
        };
      }
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      apiBase: "https://api.example.com",
      contentFactoryDir: fixture.contentFactoryDir,
      env: {
        PATH: process.env.PATH,
        STUDIO_TOKEN_FOR_TEST: secretToken,
      },
      expectedVersion: "2.2.2",
      fetchProductionReleaseEvidence: true,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      releaseEvidenceScript: fixture.releaseEvidenceScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
      studioTokenEnv: "STUDIO_TOKEN_FOR_TEST",
      tenantId: "tenant_prod_123",
    });

    expect(calls.map((call) => call.args[0])).toEqual([
      fixture.studioCli,
      fixture.releaseEvidenceScript,
      fixture.preflightScript,
    ]);
    expect(result.pipeline.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "production_release_evidence_bootstrap_matching_trust_root_missing",
        }),
      ]),
    );
    expect(result.pipeline.blockers.map((item) => item.code)).not.toContain(
      "production_release_evidence_not_ready",
    );
    expect(result.pipeline.blockerPlan.nextPhase).toMatchObject({
      id: "production_catalog_bootstrap",
    });
    expect(
      result.pipeline.blockerPlan.phases.find(
        (phase) => phase.id === "production_catalog_bootstrap",
      ),
    ).toMatchObject({
      blocked: true,
      codes: expect.arrayContaining([
        "production_release_evidence_bootstrap_matching_trust_root_missing",
      ]),
    });
    expect(JSON.stringify(result.pipeline)).not.toContain(secretToken);
  });

  it("does not call production release evidence fetch when required inputs are missing", () => {
    const fixture = createPipelineFixture();
    const calls = [];
    const runner = (command, args) => {
      calls.push({ args, command });
      if (args[0] === fixture.studioCli) {
        return {
          status: 0,
          stdout: JSON.stringify(studioDryRun([])),
          stderr: "",
        };
      }
      if (args[0] === fixture.releaseEvidenceScript) {
        throw new Error("release evidence script should not run");
      }
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      fetchProductionReleaseEvidence: true,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      releaseEvidenceScript: fixture.releaseEvidenceScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(calls.map((call) => call.args[0])).toEqual([
      fixture.studioCli,
      fixture.preflightScript,
    ]);
    expect(result.pipeline.steps.productionReleaseEvidence).toMatchObject({
      executable: false,
      missingKeys: ["tenantId", "studioToken"],
      outputs: {
        summary: {
          present: true,
        },
      },
      requested: true,
      skippedReason: "missing_inputs",
      status: "blocked",
    });
    expect(result.pipeline.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "production_release_evidence_inputs_missing",
        }),
      ]),
    );
    expect(fs.existsSync(result.pipeline.files.productionReleaseEvidence)).toBe(
      true,
    );
    expect(
      JSON.parse(
        fs.readFileSync(
          result.pipeline.files.productionReleaseEvidence,
          "utf8",
        ),
      ).outputs.summary.present,
    ).toBe(true);
  });

  it("uses the shared Studio/LimeCore default API base for release evidence input", () => {
    const fixture = createPipelineFixture();
    const runner = (command, args) => {
      if (args[0] === fixture.studioCli) {
        const dryRun = studioDryRun([]);
        return {
          status: 0,
          stdout: JSON.stringify({
            ...dryRun,
            releaseReadiness: {
              ...dryRun.releaseReadiness,
              checks: {
                ...dryRun.releaseReadiness.checks,
                auth: {
                  apiBaseConfigured: true,
                  tenantIdConfigured: false,
                  tokenConfigured: false,
                },
              },
            },
          }),
          stderr: "",
        };
      }
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      fetchProductionReleaseEvidence: true,
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      releaseEvidenceScript: fixture.releaseEvidenceScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(result.pipeline.operatorReadiness.inputs.apiBase).toMatchObject({
      configured: true,
      purpose: "release-evidence-fetch",
      source: "default",
      studioDryRunConfigured: true,
    });
    expect(result.pipeline.operatorReadiness.missingKeys).not.toContain(
      "apiBase",
    );
  });

  it("keeps the pipeline blocked when Studio dry-run does not produce JSON", () => {
    const fixture = createPipelineFixture();
    const runner = (_command, args) => {
      if (args[0] === fixture.preflightScript) {
        writeJson(commandValue(args, "--output"), readyPreflight());
        return { status: 0, stdout: "preflight ok\n", stderr: "" };
      }
      if (args[0] === fixture.studioCli) {
        return {
          status: 1,
          stdout: "",
          stderr: "missing studio inputs",
        };
      }
      throw new Error(`unexpected command: ${args.join(" ")}`);
    };

    const result = runContentFactoryProductionReadinessPipeline({
      contentFactoryDir: fixture.contentFactoryDir,
      expectedVersion: "2.2.2",
      outputDir: fixture.outputDir,
      preflightScript: fixture.preflightScript,
      runner,
      studioCli: fixture.studioCli,
      studioDir: fixture.studioDir,
    });

    expect(result.pipeline).toMatchObject({
      ready: false,
      status: "blocked",
      steps: {
        studioDryRun: {
          ok: false,
          present: false,
        },
      },
    });
    expect(result.pipeline.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "production_studio_dry_run_missing",
        }),
      ]),
    );
  });
});
