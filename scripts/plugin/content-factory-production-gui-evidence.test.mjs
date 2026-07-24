import fs from "node:fs";
import { describe, expect, it } from "vitest";

import {
  inferLiveProviderUsed,
  summarizeWorkflowFactsDom,
  summarizeInstalledState,
} from "./content-factory-production-gui-evidence.mjs";

const scriptPath = "scripts/plugin/content-factory-production-gui-evidence.mjs";

function readScript() {
  return fs.readFileSync(scriptPath, "utf8");
}

describe("content factory production GUI evidence collector", () => {
  it("is exposed as an explicit npm production evidence collector", () => {
    const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
    expect(
      packageJson.scripts["plugin:content-factory-production-gui-evidence"],
    ).toBe("node scripts/plugin/content-factory-production-gui-evidence.mjs");
  });

  it("collects only from real Electron CDP and current App Server methods", () => {
    const content = readScript();

    expect(content).toContain("chromium.connectOverCDP");
    expect(content).toContain("window.__LIME_ELECTRON__ === true");
    expect(content).toContain('"app_server_handle_json_lines"');
    expect(content).toContain('"pluginInstalled/list"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain('"evidence/export"');
    expect(content).not.toContain('"agentSession/action/respond"');
    expect(content).toContain('"workflow/respond"');
    expect(content).not.toContain('"thread/resume"');
    expect(content).toContain("--thread-id");
    expect(content).toContain("--turn-start-trace");
    expect(content).toContain("readProductionTurnStartTrace");
    expect(content).toContain("threadId,");
    expect(content).toContain("includeTurns: true");
    expect(content).toContain("readResult?.thread?.sessionId");
    expect(content).not.toContain(
      'callAppServer(page, "thread/read", {\n      sessionId',
    );
  });

  it("fails closed on production gate markers instead of inventing evidence", () => {
    const content = readScript();

    expect(content).toContain('sourceKind === "cloud_release"');
    expect(content).toContain('signaturePolicy === "required"');
    expect(content).toContain('signatureVerificationStatus === "verified"');
    expect(content).toContain('cloudReleaseEvidenceStatus === "ready"');
    expect(content).toContain('packageVerificationStatus === "verified"');
    expect(content).toContain("packageHashMatched === true");
    expect(content).toContain("manifestHashMatched === true");
    expect(content).toContain('hostManagedGenerationStatus === "completed"');
    expect(content).toContain("workflowJsonlEvents.length > 0");
    expect(content).toContain("workflowJsonlEventCount");
    expect(content).toContain("workflowJsonlEventTypes");
    expect(content).toContain("workflowAuditExported");
    expect(content).toContain("workflowAuditMetadataOnly");
    expect(content).toContain("workflowAuditRawContentExcluded");
    expect(content).toContain("workflowAuditRedactionPolicyPresent");
    expect(content).toContain("workflowResumeLifecyclePresent");
    expect(content).toContain(
      "contentFactoryArticleWorkspaceWorkflowFactsHidden",
    );
    expect(content).toContain("inspectArticleEditorWorkflowFacts");
    expect(content).toContain("generatedArticleMarkerClean");
    expect(content).toContain("tracedTurnStartViaElectronIpc");
    expect(content).toContain("statusFromAssertions");
    expect(content).toContain("missingAssertions");
  });

  it("requires real DOM evidence that Article Editor does not show workflow facts", () => {
    expect(
      summarizeWorkflowFactsDom({
        workflowDetailCount: 0,
        workflowStepCount: 0,
        sidePanelWorkflowFactMentioned: false,
      }),
    ).toMatchObject({
      hidden: true,
    });
    expect(
      summarizeWorkflowFactsDom({
        workflowDetailCount: 1,
        workflowStepCount: 0,
        sidePanelWorkflowFactMentioned: false,
      }),
    ).toMatchObject({
      hidden: false,
    });
    expect(
      summarizeWorkflowFactsDom({
        workflowDetailCount: 0,
        workflowStepCount: 0,
        sidePanelWorkflowFactMentioned: true,
      }),
    ).toMatchObject({
      hidden: false,
    });
  });

  it("recognizes workflowResume metadata from action/respond", () => {
    const content = readScript();

    expect(content).toContain(
      "../lib/content-factory-production-workflow-evidence.mjs",
    );
    expect(content).toContain("projectAppServerParamsForEvidence");
    expect(content).toContain("workflowResumeBindingsFromTrace");
    expect(content).toContain("workflowResumeEventBinding");
    expect(content).toContain("summarizeWorkflowResumeLifecycle");
    expect(content).toContain("actionMetadataPresent");
    expect(content).not.toContain("runtimeResumeContract");
    expect(content).toContain("summarizeEvidenceExport");
    expect(content).toContain("workflow_audit");
    expect(content).toContain("workflow_audit_metadata_only");
  });

  it("keeps secrets and local CDP URLs out of gate evidence", () => {
    const content = readScript();

    expect(content).not.toMatch(/sk-[A-Za-z0-9]{20,}/);
    expect(content).not.toContain("apiKey:");
    expect(content).toContain("Bearer [redacted]");
    expect(content).toContain("sanitizeText");
    expect(content).toContain("pageUrlKind");
    expect(content).toContain("endpointConfigured");
    expect(content).toContain("turnStartTrace");
    expect(content).toContain("--cdp-url or LIME_ELECTRON_CDP_URL is required");
    expect(content).toContain("sourceUriConfigured");
    expect(content).toContain("signaturePolicy");
    expect(content).toContain("packageHashMatched");
    expect(content).toContain("manifestHashMatched");
    expect(content).not.toContain("sourceUri: readString");
    expect(content).not.toContain("params: sanitizeJson");
    expect(content).not.toContain("url: options.cdpUrl");
    expect(content).not.toContain("pageUrl: runtime.href");
  });

  it("normalizes real App Server snake_case installed state without weakening the signed gate", () => {
    const installedState = summarizeInstalledState({
      app_id: "content-factory-app",
      app_version: "2.2.2",
      identity: {
        source_kind: "cloud_release",
        source_uri:
          "https://packages.example.com/content-factory-app-2.2.2.lapp",
        package_hash: "sha256:package-hash",
        manifest_hash: "sha256:manifest-hash",
        release_id: "release_2026_07_06",
        signature_ref: "sigstore:content-factory-app@2.2.2:release_2026_07_06",
      },
      setup: {
        cloud_release_evidence: {
          status: "ready",
          signature_policy: "required",
          signature_verification_status: "verified",
          package_verification_status: "verified",
          package_hash_matched: true,
          manifest_hash_matched: true,
        },
      },
    });

    expect(installedState).toEqual({
      appId: "content-factory-app",
      appVersion: "2.2.2",
      sourceKind: "cloud_release",
      sourceUriConfigured: true,
      packageHash: "sha256:package-hash",
      manifestHash: "sha256:manifest-hash",
      releaseId: "release_2026_07_06",
      signatureRef: "sigstore:content-factory-app@2.2.2:release_2026_07_06",
      signaturePolicy: "required",
      signatureVerificationStatus: "verified",
      packageVerificationStatus: "verified",
      packageHashMatched: true,
      manifestHashMatched: true,
      cloudReleaseEvidenceStatus: "ready",
    });
    expect(
      inferLiveProviderUsed({
        installedState,
        readModel: {
          articleDraftDocumentPresent: true,
          generatedArticleMarkerClean: true,
          hostManagedGenerationStatus: "completed",
        },
      }),
    ).toBe(true);
  });

  it("fails closed when signed release identity is incomplete", () => {
    const installedState = summarizeInstalledState({
      app_id: "content-factory-app",
      identity: {
        source_kind: "cloud_release",
        source_uri: "https://packages.example.com/content-factory-app.lapp",
        package_hash: "sha256:package-hash",
        manifest_hash: "sha256:manifest-hash",
      },
      setup: {
        cloud_release_evidence: {
          status: "ready",
          signature_policy: "required",
          signature_verification_status: "verified",
          package_verification_status: "verified",
          package_hash_matched: true,
          manifest_hash_matched: true,
        },
      },
    });

    expect(installedState.releaseId).toBe("");
    expect(installedState.signatureRef).toBe("");
    expect(
      inferLiveProviderUsed({
        installedState,
        readModel: {
          articleDraftDocumentPresent: true,
          generatedArticleMarkerClean: true,
          hostManagedGenerationStatus: "completed",
        },
      }),
    ).toBe(false);
  });
});
