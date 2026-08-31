import { describe, expect, it, vi } from "vitest";
import {
  listPermissionProfiles,
  resolveAllowedPermissionProfile,
} from "./permissionProfiles";

describe("permissionProfiles", () => {
  it("通过 exact App Server method 读取并规范化 Desktop profile catalog", async () => {
    const request = vi.fn(async () => ({
      id: 1,
      result: {
        data: [
          { id: " :read-only ", description: null, allowed: true },
          { id: ":workspace", description: " Coding ", allowed: true },
        ],
      },
      configWarnings: [],
    }));

    await expect(listPermissionProfiles({}, { request })).resolves.toEqual([
      { id: ":read-only", description: undefined, allowed: true },
      { id: ":workspace", description: "Coding", allowed: true },
    ]);
    expect(request).toHaveBeenCalledWith("permissionProfile/list", {});
  });

  it("只解析唯一且 allowed 的 profile", async () => {
    const allowedClient = {
      request: vi.fn(async () => ({
        id: 1,
        result: {
          data: [{ id: ":workspace", description: null, allowed: true }],
        },
        configWarnings: [],
      })),
    };
    await expect(
      resolveAllowedPermissionProfile(
        ":workspace",
        " /workspace/project ",
        allowedClient,
      ),
    ).resolves.toMatchObject({ id: ":workspace", allowed: true });
    expect(allowedClient.request).toHaveBeenCalledWith(
      "permissionProfile/list",
      { cwd: "/workspace/project" },
    );

    const deniedClient = {
      request: vi.fn(async () => ({
        id: 1,
        result: {
          data: [{ id: ":workspace", description: null, allowed: false }],
        },
        configWarnings: [],
      })),
    };
    await expect(
      resolveAllowedPermissionProfile(":workspace", undefined, deniedClient),
    ).rejects.toThrow("is not allowed");
  });
});
