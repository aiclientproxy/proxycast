import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const repoRoot = process.cwd();
const srcRoot = join(repoRoot, "src");
const retiredSource = join(srcRoot, "lib/siteAdapterCatalogBootstrap.ts");

function listProductionSources(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolutePath = join(directory, entry.name);
    if (entry.isDirectory()) {
      return listProductionSources(absolutePath);
    }
    if (
      !entry.isFile() ||
      !/\.(?:ts|tsx)$/u.test(entry.name) ||
      /(?:\.test|\.testFixtures)\.(?:ts|tsx)$/u.test(entry.name)
    ) {
      return [];
    }
    return [absolutePath];
  });
}

describe("site adapter catalog bootstrap negative guard", () => {
  it("已退役的 Renderer bootstrap owner 不得恢复", () => {
    expect(existsSync(retiredSource)).toBe(false);
  });

  it("生产源码不得重新引用已退役 bootstrap owner", () => {
    const references = listProductionSources(srcRoot)
      .filter((absolutePath) =>
        readFileSync(absolutePath, "utf8").includes(
          "siteAdapterCatalogBootstrap",
        ),
      )
      .map((absolutePath) => relative(repoRoot, absolutePath))
      .sort();

    expect(references).toEqual([]);
  });
});
