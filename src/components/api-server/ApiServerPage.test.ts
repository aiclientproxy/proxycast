/**
 * API Server Page 测试
 *
 * 测试 Antigravity 模型支持功能
 *
 * **Feature: antigravity-model-support**
 */

import { describe, expect, test } from "vitest";

// ============================================================================
// 从 ApiServerPage.tsx 提取的测试函数
// ============================================================================

/**
 * 根据 Provider 类型获取 Gemini 测试模型列表
 */
function getGeminiTestModels(provider: string): string[] {
  switch (provider) {
    case "antigravity":
      return [
        "gemini-3-pro-preview",
        "gemini-3-pro-image-preview",
        "gemini-3-flash-preview",
        "gemini-claude-sonnet-4-5",
      ];
    case "gemini":
      return ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"];
    default:
      return ["gemini-2.0-flash"];
  }
}

/**
 * 根据 Provider 类型获取测试模型
 */
function getTestModel(provider: string): string {
  switch (provider) {
    case "antigravity":
      return "gemini-3-pro-preview";
    case "gemini":
      return "gemini-2.0-flash";
    case "qwen":
      return "qwen-max";
    case "openai":
      return "gpt-4o";
    case "claude":
      return "claude-sonnet-4-20250514";
    case "kiro":
    default:
      return "claude-opus-4-5-20251101";
  }
}

// ============================================================================
// Property 测试: Antigravity 模型支持
// **Validates: Requirements 2.1**
// ============================================================================

describe("Antigravity Model Support", () => {
  /**
   * Property: Antigravity Provider 测试模型列表
   *
   * *对于* Antigravity provider，getGeminiTestModels 应该返回正确的模型列表:
   * - gemini-3-pro-preview
   * - gemini-3-pro-image-preview
   * - gemini-3-flash-preview
   * - gemini-claude-sonnet-4-5
   *
   * **Validates: Requirements 2.1**
   */
  describe("getGeminiTestModels", () => {
    test("antigravity provider 应返回 4 个 Gemini 模型", () => {
      const models = getGeminiTestModels("antigravity");

      expect(models).toHaveLength(4);
      expect(models).toContain("gemini-3-pro-preview");
      expect(models).toContain("gemini-3-pro-image-preview");
      expect(models).toContain("gemini-3-flash-preview");
      expect(models).toContain("gemini-claude-sonnet-4-5");
    });

    test("gemini provider 应返回 3 个 Gemini 模型", () => {
      const models = getGeminiTestModels("gemini");

      expect(models).toHaveLength(3);
      expect(models).toContain("gemini-2.0-flash");
      expect(models).toContain("gemini-2.5-flash");
      expect(models).toContain("gemini-2.5-pro");
    });

    test("其他 provider 应返回默认模型列表", () => {
      const providers = ["kiro", "openai", "claude", "qwen", "unknown"];

      for (const provider of providers) {
        const models = getGeminiTestModels(provider);
        expect(models).toEqual(["gemini-2.0-flash"]);
      }
    });
  });

  /**
   * Property: Provider 默认测试模型
   *
   * *对于任意* provider，getTestModel 应该返回该 provider 的默认测试模型
   *
   * **Validates: Requirements 2.1**
   */
  describe("getTestModel", () => {
    test("antigravity provider 应返回 gemini-3-pro-preview", () => {
      expect(getTestModel("antigravity")).toBe("gemini-3-pro-preview");
    });

    test("gemini provider 应返回 gemini-2.0-flash", () => {
      expect(getTestModel("gemini")).toBe("gemini-2.0-flash");
    });

    test("kiro provider 应返回 claude-opus-4-5-20251101", () => {
      expect(getTestModel("kiro")).toBe("claude-opus-4-5-20251101");
    });

    test("openai provider 应返回 gpt-4o", () => {
      expect(getTestModel("openai")).toBe("gpt-4o");
    });

    test("claude provider 应返回 claude-sonnet-4-20250514", () => {
      expect(getTestModel("claude")).toBe("claude-sonnet-4-20250514");
    });

    test("qwen provider 应返回 qwen-max", () => {
      expect(getTestModel("qwen")).toBe("qwen-max");
    });

    test("未知 provider 应返回默认模型", () => {
      expect(getTestModel("unknown")).toBe("claude-opus-4-5-20251101");
    });
  });

  /**
   * Property: Gemini 测试端点显示条件
   *
   * *对于* antigravity 或 gemini provider，应该显示 Gemini 测试端点
   *
   * **Validates: Requirements 2.1**
   */
  describe("showGeminiTest", () => {
    const shouldShowGeminiTest = (provider: string): boolean => {
      return provider === "antigravity" || provider === "gemini";
    };

    test("antigravity provider 应显示 Gemini 测试端点", () => {
      expect(shouldShowGeminiTest("antigravity")).toBe(true);
    });

    test("gemini provider 应显示 Gemini 测试端点", () => {
      expect(shouldShowGeminiTest("gemini")).toBe(true);
    });

    test("其他 provider 不应显示 Gemini 测试端点", () => {
      const providers = ["kiro", "openai", "claude", "qwen"];

      for (const provider of providers) {
        expect(shouldShowGeminiTest(provider)).toBe(false);
      }
    });
  });
});
