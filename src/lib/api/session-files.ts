/**
 * 会话文件存储 API
 *
 * 提供与后端 session_files 模块的通信接口。
 * 参考 claude-code-open 的 ~/.claude/sessions 设计。
 *
 * @module lib/api/session-files
 */

import {
  openPathWithDefaultApp,
  revealPathInFinder,
} from "@/lib/api/fileSystem";
import { createAppServerClient } from "@/lib/api/appServer";
import { readFilePreview } from "@/lib/api/fileBrowser";

// ============================================================================
// 类型定义
// ============================================================================

/** 会话元数据 */
export interface SessionMeta {
  /** 会话 ID */
  sessionId: string;
  /** 会话标题 */
  title?: string;
  /** 主题类型 */
  theme?: string;
  /** 创建模式 */
  creationMode?: string;
  /** 创建时间（Unix 时间戳，毫秒） */
  createdAt: number;
  /** 更新时间 */
  updatedAt: number;
  /** 文件数量 */
  fileCount: number;
  /** 总文件大小（字节） */
  totalSize: number;
}

/** 会话文件信息 */
export interface SessionFile {
  /** 文件名 */
  name: string;
  /** 文件类型 */
  fileType: string;
  /** 文件元数据 */
  metadata?: Record<string, unknown>;
  /** 文件大小（字节） */
  size: number;
  /** 创建时间 */
  createdAt: number;
  /** 更新时间 */
  updatedAt: number;
}

async function readDocumentTextFromAppServer(
  filePath: string,
): Promise<string> {
  const preview = await readFilePreview(filePath, 2 * 1024 * 1024);
  if (preview.error) {
    throw new Error(preview.error);
  }
  if (preview.isBinary) {
    throw new Error("当前文稿导入只支持文本文件");
  }
  if (typeof preview.content !== "string") {
    throw new Error("fs/readFile did not return document text");
  }
  return preview.content;
}

// ============================================================================
// 会话管理 API
// ============================================================================

/**
 * 获取或创建会话
 */
export async function getOrCreateSession(
  sessionId: string,
): Promise<SessionMeta> {
  const response = await createAppServerClient().getOrCreateSessionFile({
    sessionId,
  });
  return response.result.meta;
}

/**
 * 更新会话元数据
 */
export async function updateSessionMeta(
  sessionId: string,
  updates: {
    title?: string;
    theme?: string;
    creationMode?: string;
  },
): Promise<SessionMeta> {
  const response = await createAppServerClient().updateSessionFileMeta({
    sessionId,
    ...updates,
  });
  return response.result.meta;
}

// ============================================================================
// 文件管理 API
// ============================================================================

/**
 * 保存文件到会话
 */
export async function saveFile(
  sessionId: string,
  fileName: string,
  content: string,
  metadata?: Record<string, unknown>,
): Promise<SessionFile> {
  const response = await createAppServerClient().saveSessionFile({
    sessionId,
    fileName,
    content,
    metadata,
  });
  return response.result.file as SessionFile;
}

/**
 * 读取会话文件
 */
export async function readFile(
  sessionId: string,
  fileName: string,
): Promise<string> {
  const response = await createAppServerClient().readSessionFile({
    sessionId,
    fileName,
  });
  return response.result.content;
}

/**
 * 解析会话文件绝对路径
 */
export async function resolveFilePath(
  sessionId: string,
  fileName: string,
): Promise<string> {
  const response = await createAppServerClient().resolveSessionFilePath({
    sessionId,
    fileName,
  });
  return response.result.path;
}

/**
 * 在 Finder/文件管理器中定位会话文件
 */
export async function revealFileInFinder(
  sessionId: string,
  fileName: string,
): Promise<void> {
  const path = await resolveFilePath(sessionId, fileName);
  await revealPathInFinder(path);
}

/**
 * 使用系统默认应用打开会话文件
 */
export async function openFileWithDefaultApp(
  sessionId: string,
  fileName: string,
): Promise<void> {
  const path = await resolveFilePath(sessionId, fileName);
  await openPathWithDefaultApp(path);
}

/**
 * 删除会话文件
 */
export async function deleteFile(
  sessionId: string,
  fileName: string,
): Promise<void> {
  await createAppServerClient().deleteSessionFile({
    sessionId,
    fileName,
  });
}

/**
 * 列出会话中的文件
 */
export async function listFiles(sessionId: string): Promise<SessionFile[]> {
  const response = await createAppServerClient().listSessionFiles({
    sessionId,
  });
  return response.result.files as SessionFile[];
}

// ============================================================================
// 文档导入 API
// ============================================================================

/**
 * 导入文档内容
 * @param filePath 本地文档文件路径
 * @returns 文档的文本内容
 */
export async function importDocument(filePath: string): Promise<string> {
  return readDocumentTextFromAppServer(filePath);
}
