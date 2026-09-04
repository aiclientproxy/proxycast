import { describe, expect, it } from "vitest";

import { checkCliBoundary } from "./cli-boundary.mjs";

describe("CLI current owner boundary", () => {
  it("keeps CLI/TUI on the App Server owner and retired task CLI absent", () => {
    expect(checkCliBoundary()).toEqual([]);
  });
});
