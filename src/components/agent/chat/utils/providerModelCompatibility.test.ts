import { describe, expect, it } from "vitest";
import {
  filterProviderModelsByCompatibility,
  getProviderModelCompatibilityIssue,
  resolveProviderModelCompatibility,
} from "./providerModelCompatibility";

describe("resolveProviderModelCompatibility", () => {
  it("应将本地 CLI Provider 的 gpt-5.3-codex 自动降级到 gpt-5.2-codex", () => {
    const result = resolveProviderModelCompatibility({
      providerType: "custom-123",
      configuredProviderType: "codex",
      model: "gpt-5.3-codex",
    });

    expect(result.changed).toBe(true);
    expect(result.model).toBe("gpt-5.2-codex");
    expect(result.reason).toContain("gpt-5.2-codex");
  });

  it("非本地 CLI Provider 不应调整模型", () => {
    const result = resolveProviderModelCompatibility({
      providerType: "anthropic",
      configuredProviderType: "anthropic",
      model: "gpt-5.3-codex",
    });

    expect(result.changed).toBe(false);
    expect(result.model).toBe("gpt-5.3-codex");
  });

  it("本地 CLI Provider 的其他模型不应调整", () => {
    const result = resolveProviderModelCompatibility({
      providerType: "codex",
      configuredProviderType: "codex",
      model: "gpt-5.2-codex",
    });

    expect(result.changed).toBe(false);
    expect(result.model).toBe("gpt-5.2-codex");
  });

  it("应返回不兼容模型的 UI 提示信息", () => {
    const issue = getProviderModelCompatibilityIssue({
      providerType: "custom-123",
      configuredProviderType: "codex",
      model: "gpt-5.3-codex",
    });

    expect(issue).toEqual({
      code: "local_cli_account_model_unsupported",
      message: "当前本地 CLI 登录态不支持该模型",
      suggestedModel: "gpt-5.2-codex",
    });
  });

  it("模型选择器应拒绝缺少权威能力来源的实时目录项", () => {
    expect(
      getProviderModelCompatibilityIssue({
        providerType: "custom-agnes",
        configuredProviderType: "openai",
        model: "agnes-video-v2.0",
        capabilityProvenance: "inferred_hint",
        enforceExecutableCapability: true,
      }),
    ).toEqual({
      code: "runtime_capability_not_authoritative",
      message: "模型缺少可执行的权威能力声明",
    });

    expect(
      getProviderModelCompatibilityIssue({
        providerType: "custom-agnes",
        configuredProviderType: "openai",
        model: "agnes-2.5-flash",
        capabilityProvenance: "canonical",
        enforceExecutableCapability: true,
      }),
    ).toBeNull();
  });

  it("Provider 显式声明的模型应保留可执行路由，不依赖能力猜测授权", () => {
    expect(
      getProviderModelCompatibilityIssue({
        providerType: "deepseek",
        configuredProviderType: "openai",
        model: "deepseek-v4-pro",
        capabilityProvenance: "inferred_hint",
        providerDeclaredModel: true,
        enforceExecutableCapability: true,
      }),
    ).toBeNull();
  });

  it("应过滤不兼容模型并保留兼容模型", () => {
    const result = filterProviderModelsByCompatibility(
      {
        providerType: "custom-123",
        configuredProviderType: "codex",
      },
      ["gpt-5.3-codex", "gpt-5.2-codex", "gpt-5.2"],
    );

    expect(result.compatibleModels).toEqual(["gpt-5.2-codex", "gpt-5.2"]);
    expect(result.incompatibleModels).toEqual([
      {
        model: "gpt-5.3-codex",
        issue: {
          code: "local_cli_account_model_unsupported",
          message: "当前本地 CLI 登录态不支持该模型",
          suggestedModel: "gpt-5.2-codex",
        },
      },
    ]);
  });
});
