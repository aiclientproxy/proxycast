/**
 * @file Artifact 类型定义
 * @description Artifact 系统的核心类型定义，包括类型枚举、状态、元数据和渲染器接口
 * @module lib/artifact/types
 * @requirements 1.1, 1.2, 1.4, 1.5
 */

import type React from "react";

/**
 * Artifact 类型枚举
 * 定义系统支持的所有 Artifact 类型
 *
 * 轻量类型：code, html, svg, mermaid, react
 * Canvas 类型：canvas:document, canvas:video, canvas:design
 *
 * @requirements 1.1, 1.2
 */
export type ArtifactType =
  // 轻量类型
  | "document"
  | "code"
  | "html"
  | "svg"
  | "mermaid"
  | "react"
  // Canvas 类型
  | "canvas:document"
  | "canvas:video"
  | "canvas:design";

/**
 * Artifact 状态枚举
 * 定义 Artifact 在生命周期中的各种状态
 *
 * - pending: 等待处理
 * - streaming: 流式生成中
 * - complete: 生成完成
 * - error: 发生错误
 *
 * @requirements 1.5
 */
export type ArtifactStatus = "pending" | "streaming" | "complete" | "error";

/**
 * Artifact 元数据接口
 * 存储 Artifact 的附加信息
 *
 * @requirements 1.4
 */
export interface ArtifactMeta {
  /** 代码语言 (code 类型) */
  language?: string;
  /** 文件名 */
  filename?: string;
  /** Canvas 平台 (canvas 类型) */
  platform?: string;
  /** 自定义数据 - 允许扩展任意字段 */
  [key: string]: unknown;
}

/**
 * Artifact 实例接口
 * 表示一个完整的 Artifact 对象
 *
 * @requirements 1.4
 */
export interface Artifact {
  /** 唯一标识 */
  id: string;
  /** 类型 */
  type: ArtifactType;
  /** 标题 */
  title: string;
  /** 内容 */
  content: string;
  /** 状态 */
  status: ArtifactStatus;
  /** 元数据 */
  meta: ArtifactMeta;
  /** 在原始响应中的位置 */
  position: { start: number; end: number };
  /** 创建时间戳 */
  createdAt: number;
  /** 更新时间戳 */
  updatedAt: number;
  /** 错误信息（仅在 status 为 error 时存在） */
  error?: string;
}

/**
 * 渲染器组件 Props 接口
 * 所有 Artifact 渲染器组件的通用 Props 定义
 */
export interface ArtifactRendererProps {
  /** 要渲染的 Artifact 对象 */
  artifact: Artifact;
  /** 是否处于流式生成状态 */
  isStreaming?: boolean;
  /** 渲染色调 */
  tone?: "dark" | "light";
  /** 内容变更回调（用于可编辑的渲染器） */
  onContentChange?: (content: string) => void;
  /** 是否隐藏内部工具栏（当外部已有工具栏时使用） */
  hideToolbar?: boolean;
  /** 视图模式（用于代码预览） */
  viewMode?: "source" | "preview";
  /** 视图模式变更回调 */
  onViewModeChange?: (mode: "source" | "preview") => void;
  /** 预览尺寸 */
  previewSize?: "mobile" | "tablet" | "desktop";
  /** 预览尺寸变更回调 */
  onPreviewSizeChange?: (size: "mobile" | "tablet" | "desktop") => void;
}

/**
 * 渲染器注册项接口
 * 用于在注册表中注册渲染器
 */
export interface RendererEntry {
  /** Artifact 类型 */
  type: ArtifactType;
  /** 显示名称 */
  displayName: string;
  /** 图标标识 */
  icon: string;
  /** 渲染器组件，允许普通组件或懒加载组件 */
  component:
    | React.ComponentType<ArtifactRendererProps>
    | React.ExoticComponent<ArtifactRendererProps>;
  /** 是否支持编辑 */
  canEdit?: boolean;
  /** 下载时的文件扩展名 */
  fileExtension?: string;
}

/**
 * 轻量 Artifact 类型列表
 * 用于类型检查和分类
 */
export const LIGHTWEIGHT_ARTIFACT_TYPES: ArtifactType[] = [
  "document",
  "code",
  "html",
  "svg",
  "mermaid",
  "react",
];

/**
 * Canvas Artifact 类型列表
 * 用于类型检查和分类
 */
export const CANVAS_ARTIFACT_TYPES: ArtifactType[] = [
  "canvas:document",
  "canvas:video",
  "canvas:design",
];

/**
 * 所有 Artifact 类型列表
 */
export const ALL_ARTIFACT_TYPES: ArtifactType[] = [
  ...LIGHTWEIGHT_ARTIFACT_TYPES,
  ...CANVAS_ARTIFACT_TYPES,
];

export function normalizeArtifactType(type: string): ArtifactType | null {
  const normalized = type.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  return ALL_ARTIFACT_TYPES.includes(normalized as ArtifactType)
    ? (normalized as ArtifactType)
    : null;
}

/**
 * 检查是否为 Canvas 类型
 * @param type - Artifact 类型
 * @returns 是否为 Canvas 类型
 */
export function isCanvasType(type: ArtifactType): boolean {
  return type.startsWith("canvas:");
}

/**
 * 检查是否为轻量类型
 * @param type - Artifact 类型
 * @returns 是否为轻量类型
 */
export function isLightweightType(type: ArtifactType): boolean {
  return LIGHTWEIGHT_ARTIFACT_TYPES.includes(type);
}

/**
 * 默认文件扩展名映射
 */
export const DEFAULT_FILE_EXTENSIONS: Record<ArtifactType, string> = {
  document: "md",
  code: "txt",
  html: "html",
  svg: "svg",
  mermaid: "mmd",
  react: "jsx",
  "canvas:document": "md",
  "canvas:video": "txt",
  "canvas:design": "json",
};
