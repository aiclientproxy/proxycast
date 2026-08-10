import { describe, expect, it, vi } from "vitest";
import {
  listCollaborationModes,
  resolveCollaborationModeMask,
} from "./collaborationModes";

describe("collaborationModes", () => {
  it("通过 exact App Server method 读取并规范化 Desktop mode catalog", async () => {
    const request = vi.fn(async () => ({
      id: 1,
      result: {
        data: [
          {
            name: " Plan ",
            mode: "plan" as const,
            model: null,
            reasoning_effort: " medium ",
          },
          {
            name: "Default",
            mode: "default" as const,
            model: null,
            reasoning_effort: null,
          },
        ],
      },
      configWarnings: [],
    }));

    await expect(listCollaborationModes({ request })).resolves.toEqual([
      {
        name: "Plan",
        mode: "plan",
        model: null,
        reasoning_effort: "medium",
      },
      {
        name: "Default",
        mode: "default",
        model: null,
        reasoning_effort: null,
      },
    ]);
    expect(request).toHaveBeenCalledWith("collaborationMode/list", {});
  });

  it("按 mode 解析唯一 preset，重复或缺失时 fail closed", async () => {
    const validClient = {
      request: vi.fn(async () => ({
        id: 1,
        result: {
          data: [
            {
              name: "Plan",
              mode: "plan" as const,
              model: null,
              reasoning_effort: "medium",
            },
          ],
        },
        configWarnings: [],
      })),
    };
    await expect(
      resolveCollaborationModeMask("plan", validClient),
    ).resolves.toMatchObject({ mode: "plan", reasoning_effort: "medium" });

    const duplicateClient = {
      request: vi.fn(async () => ({
        id: 1,
        result: {
          data: [
            {
              name: "Plan A",
              mode: "plan" as const,
              model: null,
              reasoning_effort: "medium",
            },
            {
              name: "Plan B",
              mode: "plan" as const,
              model: null,
              reasoning_effort: "high",
            },
          ],
        },
        configWarnings: [],
      })),
    };
    await expect(
      resolveCollaborationModeMask("plan", duplicateClient),
    ).rejects.toThrow("must return exactly one plan preset");
  });
});
