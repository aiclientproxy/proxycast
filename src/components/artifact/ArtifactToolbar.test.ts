/**
 * @file Artifact 工具栏属性测试
 * @description 使用 fast-check 进行属性测试，验证工具栏文件扩展名的正确性
 * @module components/artifact/ArtifactToolbar.test
 * @requirements 13.2
 */

import { describe, test, expect } from "vitest";
import * as fc from "fast-check";
import { fastCheckRuns } from "../../test/fastCheckRuns";
import { artifactRegistry, ArtifactRegistry } from "@/lib/artifact/registry";
import {
  ALL_ARTIFACT_TYPES,
  LIGHTWEIGHT_ARTIFACT_TYPES,
  CANVAS_ARTIFACT_TYPES,
  DEFAULT_FILE_EXTENSIONS,
  type ArtifactType,
  type Artifact,
} from "@/lib/artifact/types";

// ============================================================================
// 辅助函数（从 ArtifactToolbar.tsx 复制，用于测试）
// ============================================================================

/**
 * 根据语言获取文件扩展名
 * @param language - 编程语言
 * @returns 文件扩展名
 */
function getLanguageExtension(language: string): string {
  const langExtMap: Record<string, string> = {
    javascript: "js",
    typescript: "ts",
    python: "py",
    rust: "rs",
    go: "go",
    java: "java",
    cpp: "cpp",
    c: "c",
    csharp: "cs",
    ruby: "rb",
    php: "php",
    swift: "swift",
    kotlin: "kt",
    scala: "scala",
    html: "html",
    css: "css",
    scss: "scss",
    less: "less",
    json: "json",
    yaml: "yaml",
    yml: "yml",
    xml: "xml",
    markdown: "md",
    sql: "sql",
    shell: "sh",
    bash: "sh",
    powershell: "ps1",
    dockerfile: "dockerfile",
    tsx: "tsx",
    jsx: "jsx",
    vue: "vue",
    svelte: "svelte",
  };

  const lower = language.toLowerCase();
  // 使用 hasOwnProperty 检查，避免原型链上的属性污染
  return Object.prototype.hasOwnProperty.call(langExtMap, lower)
    ? langExtMap[lower]
    : "txt";
}

/**
 * 清理文件名，移除非法字符
 * @param name - 原始文件名
 * @returns 清理后的文件名
 */
function sanitizeFilename(name: string): string {
  const sanitized = name
    .replace(/[<>:"/\\|?*]/g, "_")
    .replace(/\s+/g, "_")
    .trim();
  return sanitized || "artifact";
}

/**
 * 根据 Artifact 类型和元数据生成文件名
 * @param artifact - Artifact 对象
 * @param registry - 注册表实例
 * @returns 文件名
 */
function generateFilename(
  artifact: Artifact,
  registry: ArtifactRegistry,
): string {
  if (artifact.meta.filename) {
    return artifact.meta.filename;
  }

  const ext = registry.getFileExtension(artifact.type);

  if (artifact.type === "code" && artifact.meta.language) {
    const langExt = getLanguageExtension(artifact.meta.language);
    return `${sanitizeFilename(artifact.title)}.${langExt}`;
  }

  return `${sanitizeFilename(artifact.title)}.${ext}`;
}

// ============================================================================
// 自定义生成器 (Arbitraries)
// ============================================================================

/** Artifact 类型生成器 */
const artifactTypeArb = fc.constantFrom(...ALL_ARTIFACT_TYPES);

/** 轻量类型生成器 */
const lightweightTypeArb = fc.constantFrom(...LIGHTWEIGHT_ARTIFACT_TYPES);

/** Canvas 类型生成器 */
const canvasTypeArb = fc.constantFrom(...CANVAS_ARTIFACT_TYPES);

/** 支持的编程语言生成器 */
const supportedLanguageArb = fc.constantFrom(
  "javascript",
  "typescript",
  "python",
  "rust",
  "go",
  "java",
  "cpp",
  "c",
  "csharp",
  "ruby",
  "php",
  "swift",
  "kotlin",
  "scala",
  "html",
  "css",
  "scss",
  "less",
  "json",
  "yaml",
  "yml",
  "xml",
  "markdown",
  "sql",
  "shell",
  "bash",
  "powershell",
  "dockerfile",
  "tsx",
  "jsx",
  "vue",
  "svelte",
);

/** 未知语言生成器 */
const unknownLanguageArb = fc
  .string({ minLength: 1, maxLength: 20 })
  .filter(
    (s) =>
      ![
        "javascript",
        "typescript",
        "python",
        "rust",
        "go",
        "java",
        "cpp",
        "c",
        "csharp",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "html",
        "css",
        "scss",
        "less",
        "json",
        "yaml",
        "yml",
        "xml",
        "markdown",
        "sql",
        "shell",
        "bash",
        "powershell",
        "dockerfile",
        "tsx",
        "jsx",
        "vue",
        "svelte",
      ].includes(s.toLowerCase()),
  );

/** 安全标题生成器 */
const safeTitleArb = fc
  .array(
    fc.constantFrom(
      ..."abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-".split(
        "",
      ),
    ),
    { minLength: 1, maxLength: 20 },
  )
  .map((chars) => chars.join(""));

/** 安全内容生成器 */
const safeContentArb = fc.string({ minLength: 0, maxLength: 100 });

/** 创建 Artifact 对象的辅助函数 */
function createArtifact(
  type: ArtifactType,
  title: string,
  content: string,
  meta: Record<string, unknown> = {},
): Artifact {
  return {
    id: crypto.randomUUID(),
    type,
    title,
    content,
    status: "complete",
    meta,
    position: { start: 0, end: 0 },
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

/** Artifact 生成器 */
const artifactArb = fc
  .record({
    type: artifactTypeArb,
    title: safeTitleArb,
    content: safeContentArb,
    language: fc.option(supportedLanguageArb, { nil: undefined }),
  })
  .map(({ type, title, content, language }) =>
    createArtifact(type, title, content, language ? { language } : {}),
  );

/** 带语言的代码 Artifact 生成器 */
const codeArtifactWithLanguageArb = fc
  .record({
    title: safeTitleArb,
    content: safeContentArb,
    language: supportedLanguageArb,
  })
  .map(({ title, content, language }) =>
    createArtifact("code", title, content, { language }),
  );

// ============================================================================
// 有效扩展名定义
// ============================================================================

/** 各类型的有效文件扩展名 */
const VALID_EXTENSIONS: Record<ArtifactType, string[]> = {
  document: ["md"],
  code: [
    "txt",
    "js",
    "ts",
    "py",
    "rs",
    "go",
    "java",
    "cpp",
    "c",
    "cs",
    "rb",
    "php",
    "swift",
    "kt",
    "scala",
    "html",
    "css",
    "scss",
    "less",
    "json",
    "yaml",
    "yml",
    "xml",
    "md",
    "sql",
    "sh",
    "ps1",
    "dockerfile",
    "tsx",
    "jsx",
    "vue",
    "svelte",
  ],
  html: ["html"],
  svg: ["svg"],
  mermaid: ["mmd"],
  react: ["jsx"],
  "canvas:document": ["md"],
  "canvas:video": ["txt"],
  "canvas:design": ["json"],
};

/** 语言到扩展名的映射 */
const LANGUAGE_EXTENSION_MAP: Record<string, string> = {
  javascript: "js",
  typescript: "ts",
  python: "py",
  rust: "rs",
  go: "go",
  java: "java",
  cpp: "cpp",
  c: "c",
  csharp: "cs",
  ruby: "rb",
  php: "php",
  swift: "swift",
  kotlin: "kt",
  scala: "scala",
  html: "html",
  css: "css",
  scss: "scss",
  less: "less",
  json: "json",
  yaml: "yaml",
  yml: "yml",
  xml: "xml",
  markdown: "md",
  sql: "sql",
  shell: "sh",
  bash: "sh",
  powershell: "ps1",
  dockerfile: "dockerfile",
  tsx: "tsx",
  jsx: "jsx",
  vue: "vue",
  svelte: "svelte",
};

// ============================================================================
// Property 10: 工具栏文件扩展名正确性
// **Validates: Requirements 13.2**
// ============================================================================

describe("Property 10: 工具栏文件扩展名正确性", () => {
  /**
   * **Validates: Requirements 13.2**
   * 对于任何 Artifact 类型，getFileExtension 应返回该类型的有效扩展名
   */
  test("getFileExtension 应返回有效的文件扩展名", () => {
    fc.assert(
      fc.property(artifactTypeArb, (type) => {
        const ext = artifactRegistry.getFileExtension(type);

        // 扩展名应该是非空字符串
        expect(typeof ext).toBe("string");
        expect(ext.length).toBeGreaterThan(0);

        // 扩展名应该在该类型的有效扩展名列表中
        expect(VALID_EXTENSIONS[type]).toContain(ext);
      }),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 对于代码类型，根据语言应返回正确的扩展名
   */
  test("代码类型应根据语言返回正确的扩展名", () => {
    fc.assert(
      fc.property(supportedLanguageArb, (language) => {
        const ext = getLanguageExtension(language);

        // 扩展名应该是非空字符串
        expect(typeof ext).toBe("string");
        expect(ext.length).toBeGreaterThan(0);

        // 扩展名应该与预期映射一致
        const expectedExt = LANGUAGE_EXTENSION_MAP[language.toLowerCase()];
        expect(ext).toBe(expectedExt);
      }),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 对于未知语言，应返回默认扩展名 'txt'
   */
  test("未知语言应返回默认扩展名 txt", () => {
    fc.assert(
      fc.property(unknownLanguageArb, (language) => {
        const ext = getLanguageExtension(language);
        expect(ext).toBe("txt");
      }),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 生成的文件名应包含正确的扩展名
   */
  test("生成的文件名应包含正确的扩展名", () => {
    fc.assert(
      fc.property(artifactArb, (artifact) => {
        const registry = new ArtifactRegistry();
        const filename = generateFilename(artifact, registry);

        // 文件名应该是非空字符串
        expect(typeof filename).toBe("string");
        expect(filename.length).toBeGreaterThan(0);

        // 文件名应该包含扩展名
        expect(filename).toContain(".");

        // 提取扩展名
        const ext = filename.split(".").pop() || "";

        // 扩展名应该在该类型的有效扩展名列表中
        expect(VALID_EXTENSIONS[artifact.type]).toContain(ext);
      }),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 带语言的代码 Artifact 应使用语言对应的扩展名
   */
  test("带语言的代码 Artifact 应使用语言对应的扩展名", () => {
    fc.assert(
      fc.property(codeArtifactWithLanguageArb, (artifact) => {
        const registry = new ArtifactRegistry();
        const filename = generateFilename(artifact, registry);

        // 提取扩展名
        const ext = filename.split(".").pop() || "";

        // 扩展名应该与语言对应
        const language = artifact.meta.language as string;
        const expectedExt = LANGUAGE_EXTENSION_MAP[language.toLowerCase()];
        expect(ext).toBe(expectedExt);
      }),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 轻量类型应返回预定义的默认扩展名
   */
  test("轻量类型应返回预定义的默认扩展名", () => {
    fc.assert(
      fc.property(lightweightTypeArb, (type) => {
        const ext = artifactRegistry.getFileExtension(type);
        const expectedExt = DEFAULT_FILE_EXTENSIONS[type];
        expect(ext).toBe(expectedExt);
      }),
      { numRuns: fastCheckRuns(50) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * Canvas 类型应返回预定义的默认扩展名
   */
  test("Canvas 类型应返回预定义的默认扩展名", () => {
    fc.assert(
      fc.property(canvasTypeArb, (type) => {
        const ext = artifactRegistry.getFileExtension(type);
        const expectedExt = DEFAULT_FILE_EXTENSIONS[type];
        expect(ext).toBe(expectedExt);
      }),
      { numRuns: fastCheckRuns(50) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 语言扩展名映射应该是大小写不敏感的
   */
  test("语言扩展名映射应该是大小写不敏感的", () => {
    fc.assert(
      fc.property(
        supportedLanguageArb,
        fc.constantFrom("lower", "upper", "mixed"),
        (language, caseType) => {
          let transformedLang: string;
          switch (caseType) {
            case "lower":
              transformedLang = language.toLowerCase();
              break;
            case "upper":
              transformedLang = language.toUpperCase();
              break;
            case "mixed":
              transformedLang = language
                .split("")
                .map((c, i) =>
                  i % 2 === 0 ? c.toUpperCase() : c.toLowerCase(),
                )
                .join("");
              break;
            default:
              transformedLang = language;
          }

          const ext = getLanguageExtension(transformedLang);
          const expectedExt = LANGUAGE_EXTENSION_MAP[language.toLowerCase()];
          expect(ext).toBe(expectedExt);
        },
      ),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 如果 Artifact 有自定义文件名，应直接使用
   */
  test("自定义文件名应被直接使用", () => {
    fc.assert(
      fc.property(
        artifactTypeArb,
        safeTitleArb,
        safeContentArb,
        fc.string({ minLength: 1, maxLength: 20 }).map((s) => `${s}.custom`),
        (type, title, content, customFilename) => {
          const artifact = createArtifact(type, title, content, {
            filename: customFilename,
          });
          const registry = new ArtifactRegistry();
          const filename = generateFilename(artifact, registry);

          expect(filename).toBe(customFilename);
        },
      ),
      { numRuns: fastCheckRuns(100) },
    );
  });

  /**
   * **Validates: Requirements 13.2**
   * 文件名应该被正确清理（移除非法字符）
   */
  test("文件名应该被正确清理", () => {
    fc.assert(
      fc.property(
        artifactTypeArb,
        fc.string({ minLength: 1, maxLength: 30 }),
        safeContentArb,
        (type, rawTitle, content) => {
          const artifact = createArtifact(type, rawTitle, content);
          const registry = new ArtifactRegistry();
          const filename = generateFilename(artifact, registry);

          // 文件名不应包含非法字符
          const illegalChars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"];
          for (const char of illegalChars) {
            // 扩展名部分可能包含这些字符，所以只检查文件名部分
            const nameWithoutExt = filename.substring(
              0,
              filename.lastIndexOf("."),
            );
            expect(nameWithoutExt).not.toContain(char);
          }
        },
      ),
      { numRuns: fastCheckRuns(100) },
    );
  });
});
