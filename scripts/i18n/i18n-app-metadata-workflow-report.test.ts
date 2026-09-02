import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  analyzeAppMetadataWorkflowReport,
  formatAppMetadataWorkflowReport,
  runCli,
} from "./i18n-app-metadata-workflow-report";

const tempDirs: string[] = [];

function createTempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lime-i18n-app-metadata-"));
  tempDirs.push(dir);
  return dir;
}

function writeFile(root: string, relativePath: string, content: string): void {
  const filePath = path.join(root, relativePath);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
}

function forgeConfigFixture(): string {
  return [
    'const PRODUCT_NAME = "Lime";',
    'const APP_ID = "com.limecloud.lime";',
    'const RELEASE_OUTPUT_DIR = "release-electron";',
    "",
    "export default {",
    "  packagerConfig: {",
    "    protocols: [",
    "      {",
    '        schemes: ["lime"],',
    "      },",
    "    ],",
    "  },",
    "};",
  ].join("\n");
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true });
  }
});

describe("i18n app metadata workflow report", () => {
  it("应识别 app / installer 元数据仍是单语事实源", () => {
    const root = createTempDir();

    writeFile(
      root,
      "package.json",
      JSON.stringify(
        {
          name: "lime",
          version: "1.47.0",
          description: "AI content workspace for Chinese creators.",
          keywords: ["ai", "lime"],
        },
        null,
        2,
      ),
    );
    writeFile(
      root,
      "lime-rs/Cargo.toml",
      [
        "[workspace.package]",
        'version = "1.47.0"',
        "",
        "[package]",
        'name = "lime"',
        'version = "1.47.0"',
        'description = "AI API Proxy Desktop App"',
        'homepage = "https://github.com/aiclientproxy/lime"',
      ].join("\n"),
    );
    writeFile(root, "forge.config.mjs", forgeConfigFixture());
    writeFile(
      root,
      "internal/roadmap/i18n/evidence/app-metadata-locale-build-manifest.json",
      JSON.stringify(
        {
          schemaVersion: "lime.i18n.appMetadataLocaleBuildManifest.v1",
          scope: {
            generatedConfigEmissionAllowed: false,
            manifestGenerationAllowed: true,
          },
          summary: {
            generatedConfigEmissionAllowed: false,
            manifestGenerationAllowed: true,
            workflowStatus: "ready",
          },
        },
        null,
        2,
      ),
    );
    writeFile(
      root,
      "internal/roadmap/i18n/app-metadata-translation-scope.json",
      JSON.stringify(
        {
          schemaVersion: "lime.i18n.appMetadataTranslationScope.v1",
          sourceLocale: "zh-CN",
          targetLocales: ["en-US"],
          workflowStatus: "ready",
          owner: "release",
          manifestGenerationAllowed: true,
          generatedMetadataAllowed: false,
          items: [
            {
              path: "package.json",
              field: "description",
              localization: "translatable",
              priority: "required-before-multilingual-release",
            },
            {
              path: "package.json",
              field: "keywords",
              localization: "source-only",
              priority: "source-only",
            },
            {
              path: "forge.config.mjs",
              field: "productName",
              localization: "stable-brand",
              priority: "stable",
            },
            {
              path: "forge.config.mjs",
              field: "appId",
              localization: "stable-identifier",
              priority: "stable",
            },
            {
              path: "forge.config.mjs",
              field: "protocols[0].schemes",
              localization: "stable-identifier",
              priority: "stable",
            },
            {
              path: "forge.config.mjs",
              field: "mac.icon",
              localization: "source-only",
              priority: "source-only",
            },
            {
              path: "forge.config.mjs",
              field: "mac.target",
              localization: "source-only",
              priority: "source-only",
            },
            {
              path: "forge.config.mjs",
              field: "win.icon",
              localization: "source-only",
              priority: "source-only",
            },
            {
              path: "forge.config.mjs",
              field: "win.target",
              localization: "source-only",
              priority: "source-only",
            },
          ],
        },
        null,
        2,
      ),
    );

    const report = analyzeAppMetadataWorkflowReport({ repoRoot: root });

    expect(report.schemaVersion).toBe("lime.i18n.appMetadataWorkflowReport.v2");
    expect(report.summary.hasInstallerLocalizationWorkflow).toBe(true);
    expect(report.summary.hasAppMetadataLocaleBuildManifest).toBe(true);
    expect(report.summary.appMetadataLocaleBuildManifestReady).toBe(true);
    expect(report.summary.hasMetadataTranslationScope).toBe(true);
    expect(report.summary.hasLocalizedAppMetadataArtifacts).toBe(true);
    expect(report.summary.hasLocaleAwareMetadataSources).toBe(true);
    expect(report.summary.metadataMissingScopedFieldCount).toBe(0);
    expect(report.summary.metadataReviewedFieldCount).toBe(9);
    expect(report.summary.metadataScopeItemCount).toBe(9);
    expect(report.summary.metadataTranslatableFieldCount).toBe(1);
    expect(report.summary.metadataUnscopedFieldCount).toBe(0);
    expect(report.appMetadataTranslationScope).toEqual(
      expect.objectContaining({
        generatedMetadataAllowed: false,
        manifestGenerationAllowed: true,
        itemCount: 9,
        owner: "release",
        requiredBeforeMultilingualReleaseCount: 1,
        schemaVersion: "lime.i18n.appMetadataTranslationScope.v1",
        sourceLocale: "zh-CN",
        sourceOnlyFieldCount: 5,
        stableFieldCount: 3,
        targetLocales: ["en-US"],
        translatableFieldCount: 1,
        workflowStatus: "ready",
      }),
    );
    expect(report.appMetadataLocaleBuildManifest).toEqual(
      expect.objectContaining({
        exists: true,
        generatedConfigEmissionAllowed: false,
        manifestGenerationAllowed: true,
        schemaVersion: "lime.i18n.appMetadataLocaleBuildManifest.v1",
        workflowStatus: "ready",
      }),
    );
    expect(report.metadataFieldCoverage).toEqual(
      expect.objectContaining({
        missingScopedFields: [],
        unscopedMetadataFields: [],
      }),
    );
    expect(report.electronForgeConfig.productName).toBe("Lime");
    expect(report.electronForgeConfig.appId).toBe("com.limecloud.lime");
    expect(report.electronForgeConfig.deepLinkSchemes).toEqual(["lime"]);
    expect(report.electronForgeConfig.macTargets).toEqual(["dmg", "zip"]);
    expect(report.electronForgeConfig.winTargets).toEqual(["squirrel"]);
    expect(formatAppMetadataWorkflowReport(report, "text")).toContain(
      "[i18n:app-metadata] workflow inventory",
    );
    expect(JSON.parse(formatAppMetadataWorkflowReport(report, "json"))).toEqual(
      expect.objectContaining({
        schemaVersion: "lime.i18n.appMetadataWorkflowReport.v2",
      }),
    );
  });

  it("应识别 metadata scope 漏管和引用失效字段", () => {
    const root = createTempDir();

    writeFile(
      root,
      "package.json",
      JSON.stringify(
        {
          name: "lime",
          version: "1.47.0",
          description: "AI content workspace for Chinese creators.",
        },
        null,
        2,
      ),
    );
    writeFile(
      root,
      "lime-rs/Cargo.toml",
      '[package]\nname = "lime"\nversion = "1.47.0"\n',
    );
    writeFile(root, "forge.config.mjs", forgeConfigFixture());
    writeFile(
      root,
      "internal/roadmap/i18n/app-metadata-translation-scope.json",
      JSON.stringify(
        {
          schemaVersion: "lime.i18n.appMetadataTranslationScope.v1",
          sourceLocale: "zh-CN",
          targetLocales: ["en-US"],
          workflowStatus: "not-started",
          owner: "release",
          generatedMetadataAllowed: false,
          items: [
            {
              path: "package.json",
              field: "description",
              localization: "translatable",
              priority: "required-before-multilingual-release",
            },
            {
              path: "forge.config.mjs",
              field: "protocols[1].schemes",
              localization: "translatable",
              priority: "required-before-multilingual-release",
            },
          ],
        },
        null,
        2,
      ),
    );

    const report = analyzeAppMetadataWorkflowReport({ repoRoot: root });

    expect(report.summary.hasInstallerLocalizationWorkflow).toBe(false);
    expect(report.summary.hasAppMetadataLocaleBuildManifest).toBe(false);
    expect(report.summary.appMetadataLocaleBuildManifestReady).toBe(false);
    expect(report.summary.metadataMissingScopedFieldCount).toBe(1);
    expect(report.summary.metadataUnscopedFieldCount).toBe(7);
    expect(report.metadataFieldCoverage.missingScopedFields).toEqual([
      "forge.config.mjs#protocols[1].schemes",
    ]);
    expect(report.metadataFieldCoverage.unscopedMetadataFields).toEqual([
      "forge.config.mjs#appId",
      "forge.config.mjs#mac.icon",
      "forge.config.mjs#mac.target",
      "forge.config.mjs#productName",
      "forge.config.mjs#protocols[0].schemes",
      "forge.config.mjs#win.icon",
      "forge.config.mjs#win.target",
    ]);
  });

  it("应支持 CLI 写出 JSON", () => {
    const root = createTempDir();
    writeFile(
      root,
      "package.json",
      JSON.stringify({ name: "lime", version: "1.47.0" }, null, 2),
    );
    writeFile(
      root,
      "lime-rs/Cargo.toml",
      '[package]\nname = "lime"\nversion = "1.47.0"\n',
    );
    writeFile(root, "forge.config.mjs", forgeConfigFixture());

    const outFile = path.join(root, "report.json");
    const writeSpy = vi
      .spyOn(process.stdout, "write")
      .mockImplementation(() => true);

    const exitCode = runCli([
      "--format",
      "json",
      "--repo-root",
      root,
      "--output",
      outFile,
    ]);

    expect(exitCode).toBe(0);
    expect(writeSpy).not.toHaveBeenCalled();
    expect(JSON.parse(fs.readFileSync(outFile, "utf8"))).toEqual(
      expect.objectContaining({
        schemaVersion: "lime.i18n.appMetadataWorkflowReport.v2",
      }),
    );
  });
});
