import { describe, expect, it } from "vitest";
import {
  formatProjectFileMentionPath,
  replaceProjectFileMentionToken,
} from "./projectFileMention";

describe("project file mention text projection", () => {
  it("只替换当前 @token 并保留其余输入", () => {
    expect(
      replaceProjectFileMentionToken({
        value: "compare @app with @other",
        tokenStart: 8,
        tokenEnd: 12,
        path: "src/app.rs",
      }),
    ).toEqual({
      value: "compare src/app.rs with @other",
      cursorPos: 19,
    });
  });

  it("空格路径使用双引号并在末尾保留输入分隔符", () => {
    expect(formatProjectFileMentionPath("docs/product brief.md")).toBe(
      '"docs/product brief.md"',
    );
    expect(
      replaceProjectFileMentionToken({
        value: "@brief",
        tokenStart: 0,
        tokenEnd: 6,
        path: "docs/product brief.md",
      }),
    ).toEqual({
      value: '"docs/product brief.md" ',
      cursorPos: 24,
    });
  });

  it("相邻内容之间补分隔符但不改写路径", () => {
    expect(
      replaceProjectFileMentionToken({
        value: "before@app@next",
        tokenStart: 6,
        tokenEnd: 10,
        path: "src/app.rs",
      }).value,
    ).toBe("before src/app.rs @next");
  });
});
