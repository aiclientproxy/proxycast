import { describe, expect, it } from "vitest";

import {
  normalizeCandidateRunId,
  normalizeCandidateSha,
} from "./release-candidate-identity.mjs";

describe("release candidate identity", () => {
  it("normalizes a full Git commit SHA", () => {
    expect(
      normalizeCandidateSha(" AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "),
    ).toBe("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  });

  it("rejects missing, short, or non-hex candidate SHAs", () => {
    expect(() => normalizeCandidateSha("")).toThrow("candidate SHA");
    expect(() => normalizeCandidateSha("abc123")).toThrow("candidate SHA");
    expect(() => normalizeCandidateSha("z".repeat(40))).toThrow(
      "candidate SHA",
    );
  });

  it("accepts bounded run ids and rejects unsafe values", () => {
    expect(normalizeCandidateRunId(" release-42_win32-x64 ")).toBe(
      "release-42_win32-x64",
    );
    expect(() => normalizeCandidateRunId("release/42")).toThrow(
      "candidate run id",
    );
    expect(() => normalizeCandidateRunId("x".repeat(129))).toThrow(
      "candidate run id",
    );
  });
});
