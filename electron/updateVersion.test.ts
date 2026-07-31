import { describe, expect, it } from "vitest";
import { assessUpdateCandidate, normalizeUpdateVersion } from "./updateVersion";

describe("updateVersion", () => {
  it.each([
    ["1.117.0", "1.117.0"],
    ["v1.117.0", "1.117.0"],
    ["1.116.0", "1.117.0"],
  ])("候选版本 %s 相对当前版本 %s 不应视为更新", (candidate, current) => {
    expect(assessUpdateCandidate(candidate, current)).toEqual({
      status: "not_newer",
      version: candidate.replace(/^v/i, ""),
    });
  });

  it("只应把严格更高版本视为可安装更新", () => {
    expect(assessUpdateCandidate("1.118.0", "1.117.0")).toEqual({
      status: "newer",
      version: "1.118.0",
    });
  });

  it.each([undefined, "", "latest", "1.117", "1.117.0-01"])(
    "无效候选版本 %s 应 fail closed",
    (candidate) => {
      expect(assessUpdateCandidate(candidate, "1.117.0")).toEqual({
        status: "invalid",
        version: null,
      });
    },
  );

  it("应遵循 SemVer 预发布版本优先级并忽略构建元数据", () => {
    expect(assessUpdateCandidate("1.118.0-rc.1", "1.118.0-beta.9").status).toBe(
      "newer",
    );
    expect(assessUpdateCandidate("1.118.0", "1.118.0-rc.1").status).toBe(
      "newer",
    );
    expect(
      assessUpdateCandidate("1.118.0+build.2", "1.118.0+build.1").status,
    ).toBe("not_newer");
    expect(normalizeUpdateVersion("V1.118.0-rc.1+build.2")).toBe(
      "1.118.0-rc.1+build.2",
    );
  });
});
