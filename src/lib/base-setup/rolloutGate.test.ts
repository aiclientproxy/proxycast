import { describe, expect, it } from "vitest";
import type { BaseSetupPackage } from "./types";
import { evaluateBaseSetupRollout } from "./rolloutGate";

function createBaseSetupPackage(): BaseSetupPackage {
  return {
    id: "runtime-gated-pack",
    version: "0.1.0",
    title: "Runtime Gated Pack",
    summary: "验证 rollout gate 的最小包。",
    bundleRefs: [
      {
        id: "voice-bundle",
        source: "remote",
        pathOrUri: "lime://bundles/voice",
        kind: "skill_bundle",
      },
    ],
    catalogProjections: [
      {
        id: "voice-runtime",
        targetCatalog: "service_skill_catalog",
        entryKey: "voice-runtime",
        title: "配音",
        summary: "云端配音运行。",
        category: "媒体",
        outputHint: "配音结果",
        bundleRefId: "voice-bundle",
        slotProfileRef: "voice-input",
        bindingProfileRef: "cloud-voice",
        artifactProfileRef: "voice-artifact",
        scorecardProfileRef: "voice-scorecard",
        policyProfileRef: "voice-policy",
      },
    ],
    slotProfiles: [
      {
        id: "voice-input",
        slots: [
          {
            key: "script",
            label: "脚本",
            type: "textarea",
            required: true,
            placeholder: "输入要配音的脚本",
          },
        ],
      },
    ],
    bindingProfiles: [
      {
        id: "cloud-voice",
        bindingFamily: "agent_turn",
        runnerType: "managed",
      },
    ],
    artifactProfiles: [
      {
        id: "voice-artifact",
        deliveryContract: "artifact_bundle",
        requiredParts: ["audio.mp3"],
        viewerKind: "artifact_bundle",
      },
    ],
    scorecardProfiles: [
      {
        id: "voice-scorecard",
        metrics: ["acceptance_rate"],
      },
    ],
    policyProfiles: [
      {
        id: "voice-policy",
      },
    ],
    compatibility: {
      minAppVersion: "1.11.0",
      requiredKernelCapabilities: ["agent_turn"],
      seededFallback: true,
    },
  };
}

describe("evaluateBaseSetupRollout", () => {
  it("在宿主满足要求时返回 accept", () => {
    const result = evaluateBaseSetupRollout(createBaseSetupPackage(), {
      appVersion: "1.11.0",
      seededFallbackAvailable: true,
    });

    expect(result).toEqual({ decision: "accept" });
  });

  it("在结构错误且有兜底时返回 fallback_seeded", () => {
    const pkg = createBaseSetupPackage();
    pkg.catalogProjections[0].slotProfileRef = "missing-slot";

    const result = evaluateBaseSetupRollout(pkg, {
      appVersion: "1.11.0",
      seededFallbackAvailable: true,
    });

    expect(result.decision).toBe("fallback_seeded");
  });

  it("在版本不满足时返回 reject_upgrade_required", () => {
    const result = evaluateBaseSetupRollout(createBaseSetupPackage(), {
      appVersion: "1.10.0",
      seededFallbackAvailable: true,
    });

    expect(result.decision).toBe("reject_upgrade_required");
  });
});
