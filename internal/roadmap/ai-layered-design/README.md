# Lime AI 图层化设计路线图

> 状态：proposal  
> 更新时间：2026-05-05  
> 目标：把 AI 图片生成升级为可编辑的设计工程输出，让 Lime 能生成、调整、重生成和导出多图层海报 / 封面 / 商品图，而不是只返回一张扁平图片。

配套研究：

- [../../research/ai-layered-design/README.md](../../research/ai-layered-design/README.md)
- [../../research/ai-layered-design/architecture-breakdown.md](../../research/ai-layered-design/architecture-breakdown.md)
- [../../research/ai-layered-design/model-and-tooling-map.md](../../research/ai-layered-design/model-and-tooling-map.md)
- [../../research/ai-layered-design/lime-gap-analysis.md](../../research/ai-layered-design/lime-gap-analysis.md)

配套图纸：

- [./architecture-diagrams.md](./architecture-diagrams.md)：系统分层、前端工作台、后端服务、存储、provider 与 evidence 边界图
- [./prototype.md](./prototype.md)：生成结果卡、图层设计工作台、属性栏、拆层确认页、导出弹窗低保真原型
- [./sequences.md](./sequences.md)：原生分层生成、打开编辑、单层重生成、扁平图拆层、导出时序图
- [./flowcharts.md](./flowcharts.md)：入口分流、Layer Planner、mask 质量、clean plate、Canvas 状态机、导出决策流程图
- [./diagrams.md](./diagrams.md)：早期总览图纸，后续具体实现优先引用上述拆分图纸
- [./model-slot-http-json-executor-contract.md](./model-slot-http-json-executor-contract.md)：真实 matting / inpaint / OCR 服务接入 standard JSON executor 的 HTTP contract、样例与验收命令
- 完整目标外部事实索引由 `scripts/verify-layered-design-completion-evidence.mjs` 校验；输入位于本地忽略的 `evidence/completion-evidence.json`，不作为仓库导航目标
- Photoshop / Photopea / Figma 互操作记录由 `scripts/verify-layered-design-photopea-open.mjs` 生成到本地 evidence 目录，不作为仓库内固定文档

相关 Lime 事实源：

- [../../aiprompts/command-runtime.md](../../aiprompts/command-runtime.md)
- [../warp/artifact-graph.md](../warp/artifact-graph.md)
- [../task/model-routing.md](../task/model-routing.md)

## 1. 先给结论

Lime 不应该先自训图层模型。

Lime 应该先做的是：

**用现成图像模型生成和编辑资产，用 Lime 自己的 `LayeredDesignDocument` 承载图层工程。**

一句话北极星：

**Lime 的图片生成从“输出图片”升级为“输出可编辑设计工程”。**

## 2. 固定主链

后续所有实现必须收敛到下面这条主链：

```text
用户目标 / @海报 / @配图
  -> Layer Planner 拆成图层计划
  -> Asset Generator 调用 gpt-image-2 / Gemini / provider seam 分层生成
  -> Matting / Mask / Inpaint 生成透明图层和 clean plate
  -> LayeredDesignDocument 保存图层、资产、预览和历史
  -> Canvas Editor 提供移动、缩放、隐藏、重排、单层重生成
  -> Exporter 导出 PNG / JSON / 后续 PSD
  -> Artifact / Evidence 记录设计工程和生成过程
```

这条主链意味着：

1. `gpt-image-2` 是资产生成与编辑器，不是图层事实源。
2. `LayeredDesignDocument` 是 current 设计工程事实源。
3. PNG 是预览或导出结果，不是唯一保存对象。
4. 单层重生成必须保留其他图层 transform 和 zIndex。
5. 后续 PSD 导出消费同一份图层文档，不另开协议。

## 3. 非目标

本路线图明确不做：

1. 不自训 image-to-PSD 大模型。
2. 不承诺任意图片完美拆成 Photoshop 原始图层。
3. 不新增平行图片设计 runtime。
4. 不绕过现有 provider / model routing / media task / Workspace 主链。
5. 不把完整 Photoshop / Figma 替代作为首期目标。
6. 不默认把艺术 Logo 矢量化或字体完全识别。
7. 不把普通文案继续烘焙成不可编辑图片层。

## 4. 产品对象分层

### 4.1 LayeredDesignDocument

`LayeredDesignDocument` 是 Lime AI 设计项目的唯一 current 事实源。

它负责保存：

1. 画布尺寸和背景。
2. 图层列表。
3. 图层 transform。
4. 资产引用。
5. 预览图。
6. 单层生成来源。
7. 编辑历史。

固定边界：

**任何图层编辑都必须回写文档，不能只停留在前端 Canvas 状态。**

### 4.2 Layer Planner

Layer Planner 负责把用户目标拆成可编辑图层计划：

1. 背景层。
2. 主体层。
3. 特效层。
4. Logo 层。
5. TextLayer。
6. ShapeLayer。
7. GroupLayer。

固定边界：

**Planner 不生成最终图片，只生成可执行图层计划。**

### 4.3 Asset Generator

Asset Generator 负责调用 provider 生成 bitmap 资产：

1. 背景。
2. 主体。
3. 特效。
4. Logo。
5. clean plate。
6. 单层替换图。

固定边界：

**Asset Generator 的输出必须注册为 asset，并绑定到某个 layer 或 edit history。**

### 4.4 Canvas Editor

Canvas Editor 是用户可见编辑面：

1. 画布预览。
2. 图层栏。
3. 属性面板。
4. 拖拽缩放。
5. 显示/隐藏/锁定。
6. 重排和分组。
7. 单层重生成。

固定边界：

**Canvas Editor 不直接定义模型调用协议，只消费 `LayeredDesignDocument`。**

### 4.5 Exporter

Exporter 负责把同一设计文档输出成：

1. PNG。
2. 项目 JSON。
3. 后续 PSD。
4. 后续打包资源目录。

固定边界：

**导出格式是投影，不是新的设计事实源。**

## 5. 分阶段路线

| 阶段 | 目标 | 主产物 |
| --- | --- | --- |
| P0 | 文档与边界落盘 | research + roadmap + current/compat/deprecated/dead 分类 |
| P1 | 原生分层生成 | `LayeredDesignDocument` draft + 分层资产 + 合成预览 |
| P2 | Canvas 可编辑 | 图层栏、transform、隐藏/锁定、单层重生成 |
| P3 | 扁平图拆层 | 主要对象 mask、RGBA 图层、clean plate、OCR TextLayer |
| P4 | 专业交付 | PNG / JSON 稳定导出，PSD-like 导出试点 |

## 6. 验收总线

每个阶段都必须回答：

1. 是否仍只有一个设计工程事实源。
2. 是否能重新打开继续编辑。
3. 是否能证明单层修改不破坏其他图层。
4. 是否能导出和当前 Canvas 一致的 PNG。
5. 是否能在 artifact / evidence 中追踪模型、prompt、资产和编辑历史。

一句话：

**可交付标准不是“图片好看”，而是“设计项目能被继续编辑”。**
