/**
 * @file Artifact 渲染器导出入口
 * @description 对外仅暴露轻量渲染器注册边界，具体渲染器实现保持文件级入口
 * @module components/artifact/renderers
 * @requirements 3.1
 */

import { lazy } from "react";
import { artifactRegistry } from "@/lib/artifact/registry";
import type { RendererEntry, ArtifactType } from "@/lib/artifact/types";
import { DocumentRenderer } from "./DocumentRenderer";

// ============================================================================
// 懒加载渲染器组件
// ============================================================================

/**
 * 懒加载的代码渲染器
 */
const LazyCodeRenderer = lazy(() => import("./CodeRenderer"));

/**
 * 懒加载的 HTML 渲染器
 */
const LazyHtmlRenderer = lazy(() => import("./HtmlRenderer"));

/**
 * 懒加载的 SVG 渲染器
 */
const LazySvgRenderer = lazy(() => import("./SvgRenderer"));

/**
 * 懒加载的 Mermaid 渲染器
 */
const LazyMermaidRenderer = lazy(() => import("./MermaidRenderer"));

/**
 * 懒加载的 React 渲染器
 */
const LazyReactRenderer = lazy(() => import("./ReactRenderer"));

// ============================================================================
// 渲染器注册项定义
// ============================================================================

/**
 * 轻量渲染器注册项列表
 * 定义所有轻量类型 Artifact 的渲染器配置
 */
const LIGHTWEIGHT_RENDERER_ENTRIES: RendererEntry[] = [
  {
    type: "document" as ArtifactType,
    displayName: "文档",
    icon: "file-text",
    component: DocumentRenderer,
    canEdit: false,
    fileExtension: "md",
  },
  {
    type: "code" as ArtifactType,
    displayName: "代码",
    icon: "code",
    component: LazyCodeRenderer,
    canEdit: false,
    fileExtension: "txt",
  },
  {
    type: "html" as ArtifactType,
    displayName: "HTML",
    icon: "html",
    component: LazyHtmlRenderer,
    canEdit: false,
    fileExtension: "html",
  },
  {
    type: "svg" as ArtifactType,
    displayName: "SVG",
    icon: "image",
    component: LazySvgRenderer,
    canEdit: false,
    fileExtension: "svg",
  },
  {
    type: "mermaid" as ArtifactType,
    displayName: "Mermaid 图表",
    icon: "diagram",
    component: LazyMermaidRenderer,
    canEdit: false,
    fileExtension: "mmd",
  },
  {
    type: "react" as ArtifactType,
    displayName: "React 组件",
    icon: "react",
    component: LazyReactRenderer,
    canEdit: false,
    fileExtension: "jsx",
  },
];

// ============================================================================
// 注册函数
// ============================================================================

/**
 * 注册所有轻量渲染器到 ArtifactRegistry
 *
 * 此函数应在应用初始化时调用，将所有轻量类型的渲染器注册到全局注册表中。
 * 注册后，ArtifactRenderer 组件可以根据 Artifact 类型自动选择对应的渲染器。
 *
 * @requirements 3.1
 *
 * @example
 * ```typescript
 * // 在应用入口处调用
 * import { registerLightweightRenderers } from '@/components/artifact/renderers';
 *
 * // 注册所有轻量渲染器
 * registerLightweightRenderers();
 * ```
 */
export function registerLightweightRenderers(): void {
  for (const entry of LIGHTWEIGHT_RENDERER_ENTRIES) {
    artifactRegistry.register(entry);
  }
}

/**
 * 检查轻量渲染器是否已注册
 *
 * @returns 是否所有轻量渲染器都已注册
 */
export function areLightweightRenderersRegistered(): boolean {
  return LIGHTWEIGHT_RENDERER_ENTRIES.every((entry) =>
    artifactRegistry.has(entry.type),
  );
}
