import { describe, expect, it, vi } from "vitest";
import {
  searchProjectFiles,
  type FuzzyFileSearchClient,
} from "./fuzzyFileSearch";

describe("searchProjectFiles", () => {
  it("通过 typed fuzzyFileSearch 网关查询单一项目根目录", async () => {
    const searchFiles = vi.fn().mockResolvedValue({
      result: {
        files: [
          {
            root: "/workspace",
            path: "src/app.rs",
            match_type: "file",
            file_name: "app.rs",
            score: 84,
            indices: [4, 5, 6],
          },
        ],
      },
    });
    const signal = new AbortController().signal;

    const files = await searchProjectFiles(
      {
        query: "app",
        rootPath: "/workspace",
        cancellationToken: "composer",
      },
      { signal },
      { searchFiles } as unknown as FuzzyFileSearchClient,
    );

    expect(searchFiles).toHaveBeenCalledWith(
      {
        query: "app",
        roots: ["/workspace"],
        cancellationToken: "composer",
      },
      { signal },
    );
    expect(files).toEqual([
      expect.objectContaining({ path: "src/app.rs", match_type: "file" }),
    ]);
  });
});
