import { save as saveDialog } from "@/lib/desktop-host/dialog";
import { saveExportedDocument } from "@/lib/api/document-export";
import type { ExportFormat } from "../types";
import { markdownToHtml } from "../editor/utils/markdown";

type FileExportFormat = Exclude<ExportFormat, "clipboard">;

interface ExportConfig {
  defaultFilename: string;
  extensions: string[];
  dialogTitle: string;
  successMessage: string;
  contentFactory: (content: string) => string;
}

const EXPORT_CONFIGS: Record<FileExportFormat, ExportConfig> = {
  markdown: {
    defaultFilename: "document.md",
    extensions: ["md"],
    dialogTitle: "导出 Markdown",
    successMessage: "📄 已导出 Markdown 文件",
    contentFactory: (content) => content,
  },
  word: {
    defaultFilename: "document.doc",
    extensions: ["doc"],
    dialogTitle: "导出 Word",
    successMessage: "🧾 已导出 Word 文件",
    contentFactory: (content) => `\ufeff${buildWordHtml(content)}`,
  },
  text: {
    defaultFilename: "document.txt",
    extensions: ["txt"],
    dialogTitle: "导出纯文本",
    successMessage: "📝 已导出纯文本文件",
    contentFactory: (content) => stripMarkdown(content),
  },
};

function stripMarkdown(content: string) {
  return content
    .replace(/#{1,6}\s/g, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/`(.+?)`/g, "$1")
    .replace(/\[(.+?)\]\(.+?\)/g, "$1");
}

function buildWordHtml(content: string) {
  const body = markdownToHtml(content);

  return `<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:w="urn:schemas-microsoft-com:office:word"
      xmlns="http://www.w3.org/TR/REC-html40">
  <head>
    <meta charset="utf-8" />
    <title>document</title>
    <!--[if gte mso 9]>
    <xml>
      <w:WordDocument>
        <w:View>Print</w:View>
        <w:Zoom>100</w:Zoom>
      </w:WordDocument>
    </xml>
    <![endif]-->
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.7;
        color: #111827;
        margin: 32px;
      }

      img {
        max-width: 100%;
        height: auto;
      }

      pre, code {
        white-space: pre-wrap;
        word-break: break-word;
      }
    </style>
  </head>
  <body>${body}</body>
</html>`;
}

function ensureFileExtension(filePath: string, extension: string) {
  const normalizedExtension = extension.startsWith(".")
    ? extension.toLowerCase()
    : `.${extension.toLowerCase()}`;

  if (filePath.toLowerCase().endsWith(normalizedExtension)) {
    return filePath;
  }

  return `${filePath}${normalizedExtension}`;
}

async function exportFileContent(content: string, format: FileExportFormat) {
  const config = EXPORT_CONFIGS[format];
  const selectedPath = await saveDialog({
    title: config.dialogTitle,
    defaultPath: config.defaultFilename,
    filters: [
      {
        name: config.dialogTitle,
        extensions: config.extensions,
      },
    ],
  });

  if (!selectedPath) {
    return null;
  }

  const targetPath = ensureFileExtension(selectedPath, config.extensions[0]);
  await saveExportedDocument(targetPath, config.contentFactory(content));
  return config.successMessage;
}

export async function exportDocumentContent(
  content: string,
  format: ExportFormat,
) {
  switch (format) {
    case "markdown": {
      return exportFileContent(content, format);
    }
    case "word": {
      return exportFileContent(content, format);
    }
    case "text": {
      return exportFileContent(content, format);
    }
    case "clipboard": {
      await navigator.clipboard.writeText(content);
      return "📋 已复制到剪贴板";
    }
  }
}
