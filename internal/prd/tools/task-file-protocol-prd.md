# Lime 统一 Task File 协议 PRD

更新时间：2026-04-03
状态：Archived / historical evidence

> 本文保留早期 task-file 方案用于历史追溯，不再定义 current 产品或 CLI 接口。当前业务主链是 `Desktop / CLI / TUI -> App Server JSON-RPC -> RuntimeCore -> canonical Thread/Turn/Item`；文中的 `lime task`、Tauri 业务命令、CLI 直读 `.lime/tasks` 与“Task File 是产品唯一真相源”等表述均已废弃，禁止据此恢复实现。

## 1. 背景

Lime 当前已经具备一套轻量 task file 底座，能够覆盖：

- `image_generate`
- `cover_generate`
- `video_generate`
- `broadcast_generate`
- `url_parse`
- `typesetting`
- `modal_resource_search`

现有协议已经解决了“任务创建、状态读取、简单重试”的基础问题，但当 Lime 开始把图片生成、图片编辑、视频生成、视频编辑、自动化工作流、聊天区动态渲染统一收口到同一条主线时，现有 task file 还存在四类缺口：

1. **顶层字段不够稳定**
   - 现在更像“任务记录壳”，还不是“统一任务协议信封”
2. **任务结果缺少结构化语义**
   - `result` 还是宽泛 `Value`，对前端、CLI、worker、调试都不够友好
3. **重试与执行历史太轻**
   - 目前只有 `retry_count` 与 `source_task_id`，不够支撑一任务多尝试
4. **缺少关系图能力**
   - 无法自然表达父子任务、依赖任务、来源素材、正文占位槽位

这会直接影响后续几条主线：

- 聊天区中的动态占位图与结果替换
- 图片生成 / 图片编辑 / 视频生成 / 视频编辑的统一观察面
- 队列、重试、幂等、恢复、诊断能力
- 类似竞品“统一任务面板”的可交付体验

因此，task file 需要从“轻量任务记录”升级为“统一任务协议”。

---

## 2. 设计目标

本次协议设计只解决六个问题：

1. **所有异步任务共用一个顶层信封**
2. **不同任务类型拥有各自的强类型 payload / result 子协议**
3. **重试不再默认创建新任务文件，而是进入同一任务的多次尝试**
4. **首期即支持父子任务、依赖关系、来源关系**
5. **前端可以只观察 task file，就完成占位、进度、结果替换、失败重试**
6. **CLI、worker、前端、测试都围绕同一事实源，而不是各自维护状态**

非目标：

- 不把所有任务字段塞成一个超大万能 schema
- 不让每种任务各写一份完全独立的 JSON 协议
- 不把大日志、大二进制结果直接塞进主 task file

---

## 3. 核心结论

### 3.1 统一信封 + 类型化子协议

统一顶层任务信封，稳定承载：

- 身份
- 生命周期
- 执行尝试
- 关系图
- 进度
- UI 提示
- 审计信息

不同任务类型的差异全部进入：

- `payload`
- `result`

### 3.2 一任务多尝试

每个业务任务保持稳定 `task_id`，重试不会默认生成新的 task file。

统一使用：

- `task_id`
- `current_attempt_id`
- `attempts[]`

这样前端能稳定盯住一条任务卡，CLI 也能查看历史尝试。

### 3.3 首期支持父子任务与依赖关系

任务关系不是“以后再说”的增强项，而是首期协议能力。统一支持：

- 父任务
- 根任务
- 前置依赖
- 子任务
- 来源素材
- 来源任务尝试
- 正文占位槽位

### 3.4 Task File 仍是唯一真相源

未来即使加数据库索引或任务看板缓存层，task file 仍然是当前唯一事实源。

- 前端观察它
- CLI 读取它
- worker 更新它
- 测试回放它

---

## 4. 顶层统一信封

推荐统一结构如下：

```json
{
  "task_id": "task_01HQ...",
  "task_type": "image_generate",
  "task_family": "image",
  "title": "正文配图：多模态实验室",
  "summary": "生成一张用于正文中的配图",
  "status": "running",
  "normalized_status": "running",
  "priority": "normal",
  "created_at": "2026-04-03T09:00:00Z",
  "updated_at": "2026-04-03T09:00:12Z",
  "submitted_at": "2026-04-03T09:00:02Z",
  "started_at": "2026-04-03T09:00:05Z",
  "completed_at": null,
  "cancelled_at": null,
  "idempotency_key": "article-123-slot-4-image-generate-v1",
  "workspace_id": "workspace_demo",
  "project_id": "project_demo",
  "session_id": "session_demo",
  "content_id": "content_demo",
  "requested_by": {
    "kind": "skill",
    "name": "image_generate"
  },
  "entrypoint": "claw_chat",
  "tags": ["article-inline", "multimodal", "image"],
  "payload_schema_version": "1.0",
  "result_schema_version": "1.0",
  "payload": {},
  "result": null,
  "last_error": null,
  "current_attempt_id": "attempt_01HQ...",
  "attempts": [],
  "relationships": {},
  "artifacts": [],
  "progress": {},
  "ui_hints": {},
  "audit": {}
}
```

### 4.1 顶层字段职责

#### 身份字段

- `task_id`
- `task_type`
- `task_family`

要求：

- `task_type` 表示具体任务
- `task_family` 表示 UI / 队列 / 筛选的聚合维度

#### 生命周期字段

- `status`
- `normalized_status`
- `created_at`
- `updated_at`
- `submitted_at`
- `started_at`
- `completed_at`
- `cancelled_at`

#### 路由与归属字段

- `workspace_id`
- `project_id`
- `session_id`
- `content_id`
- `requested_by`
- `entrypoint`

#### 幂等与审计字段

- `idempotency_key`
- `tags`
- `audit`

#### 协议演进字段

- `payload_schema_version`
- `result_schema_version`

---

## 5. 任务类型分层

不要只依赖 `task_type` 承载全部语义，统一拆成：

### 5.1 `task_family`

建议首批 family：

- `image`
- `video`
- `audio`
- `document`
- `resource`
- `automation`

### 5.2 `task_type`

建议首批 type：

- `image_generate`
- `image_edit`
- `image_variation`
- `cover_generate`
- `video_generate`
- `video_edit`
- `video_upscale`
- `url_parse`
- `typesetting`
- `modal_resource_search`

### 5.3 为什么必须有 family

因为以下需求都更适合按 family 而不是按 type 聚合：

- 统一任务面板分组
- 队列资源配额
- 前端工作台默认图标与布局
- 未来权限与能力开关

例如：

- `image_generate / image_edit / image_variation / cover_generate`
  都属于 `image`
- `video_generate / video_edit / video_upscale`
  都属于 `video`

---

## 6. 状态机

统一状态如下：

- `draft`
- `pending_submit`
- `queued`
- `running`
- `partial`
- `succeeded`
- `failed`
- `cancelled`

### 6.1 状态语义

- `draft`
  - 本地构建态，尚未正式提交到任务系统
- `pending_submit`
  - task file 已创建，但尚未真正进入执行队列
- `queued`
  - 已进入执行队列，等待 worker 消费
- `running`
  - worker 已开始执行
- `partial`
  - 已返回部分结果，可供前端先渲染中间态
- `succeeded`
  - 当前任务成功完成
- `failed`
  - 当前任务失败
- `cancelled`
  - 任务被取消

### 6.2 顶层状态与 attempt 状态关系

- 顶层 `status` 表示当前任务总状态
- `attempts[].status` 表示单次尝试状态
- 历史尝试失败不代表顶层失败
- 只要当前 attempt 成功，顶层状态就是 `succeeded`

---

## 7. 一任务多尝试

### 7.1 为什么不用“每次重试一个新任务”

如果每次重试都新建任务文件，会带来这些问题：

- 前端需要在旧卡片和新卡片之间重新绑定
- 聊天区动态替换更复杂
- 统一任务列表会出现大量碎片任务
- “这其实还是同一个任务”的语义丢失

因此默认采用：

- 一个稳定 `task_id`
- 多次 `attempts[]`

### 7.2 Attempt 结构

```json
{
  "attempt_id": "attempt_01HQ...",
  "attempt_index": 2,
  "status": "running",
  "queued_at": "2026-04-03T09:01:00Z",
  "started_at": "2026-04-03T09:01:03Z",
  "completed_at": null,
  "provider": "fal",
  "model": "fal-ai/nano-banana-pro",
  "worker_id": "worker_local_1",
  "input_snapshot": {},
  "result_snapshot": null,
  "error": null,
  "metrics": {
    "queue_ms": 3000,
    "run_ms": 0
  },
  "logs_ref": ".lime/tasks-logs/task_01HQ.../attempt_2.jsonl"
}
```

### 7.3 规则

- `retry` 只追加新 attempt
- `current_attempt_id` 指向当前生效 attempt
- `input_snapshot` 保留当次执行输入
- `result_snapshot` 保留当次执行结果快照
- `logs_ref` 指向大日志文件，不直接写在主 task file 中

---

## 8. 父子任务与依赖关系

统一在 `relationships` 中定义任务图结构：

```json
{
  "parent_task_id": "task_parent_01",
  "root_task_id": "task_root_01",
  "depends_on_task_ids": ["task_dep_01"],
  "child_task_ids": ["task_child_01", "task_child_02"],
  "source_asset_ids": ["asset_01"],
  "derived_from_attempt_id": "attempt_01",
  "triggered_by_skill": "image_generate",
  "triggered_by_message_id": "message_01",
  "slot_id": "article-image-slot-4"
}
```

### 8.1 适用场景

#### 图片编辑

- `image_edit` 依赖一张原图任务
- 使用 `source_asset_ids`
- 可选 `parent_task_id`

#### 视频生成

- `video_generate` 依赖图片任务或素材任务
- 使用 `depends_on_task_ids`

#### 正文多图

- 一篇文章下多张图片属于同一父任务或同一根任务
- 使用 `parent_task_id / root_task_id`

#### 正文占位替换

- 用 `slot_id` 把正文中的占位块与任务绑定

### 8.2 为什么首期必须支持关系图

因为一旦只保留扁平任务列表，后续以下能力都会变复杂：

- 图片编辑链
- 视频生成链
- 多图正文插槽替换
- 工作流调试
- 任务树面板

---

## 9. Payload / Result 子协议

顶层信封统一，差异全部进入子协议。

## 9.1 图片生成

### `image_generate.payload`

```json
{
  "prompt": "一个充满未来感的实验室，中心是一个发光的大脑",
  "negative_prompt": "模糊，低清晰度",
  "style": "cinematic",
  "size": "1280x720",
  "aspect_ratio": "16:9",
  "count": 1,
  "usage": "article-inline",
  "reference_assets": []
}
```

### `image_generate.result`

```json
{
  "prompt": "一个充满未来感的实验室，中心是一个发光的大脑",
  "provider": "fal",
  "model": "fal-ai/nano-banana-pro",
  "seed": 12345,
  "images": [
    {
      "asset_id": "asset_img_01",
      "kind": "image",
      "url": "https://...",
      "thumbnail_url": "https://...",
      "mime_type": "image/png",
      "width": 1280,
      "height": 720,
      "size_bytes": 734002,
      "storage_ref": "s3://...",
      "preview_ref": ".lime/previews/asset_img_01.json"
    }
  ],
  "billing": {
    "provider_cost": 0.04
  }
}
```

## 9.2 图片编辑

### `image_edit.payload`

```json
{
  "prompt": "去掉图中的文字，保留主体和整体构图",
  "edit_mode": "inpaint",
  "source_asset": "asset_img_01",
  "mask_asset": null,
  "strength": 0.65,
  "preserve_regions": ["subject"]
}
```

### `image_edit.result`

```json
{
  "provider": "fal",
  "model": "fal-ai/...",
  "images": [
    {
      "asset_id": "asset_img_edit_01",
      "kind": "image",
      "url": "https://...",
      "thumbnail_url": "https://...",
      "mime_type": "image/png",
      "width": 1280,
      "height": 720
    }
  ],
  "diff_summary": "已移除文字，主体保持不变"
}
```

## 9.3 视频生成

### `video_generate.payload`

```json
{
  "prompt": "未来实验室镜头缓慢推进，发光大脑悬浮在中央",
  "duration": 5,
  "resolution": "1080p",
  "aspect_ratio": "16:9",
  "fps": 24,
  "camera_motion": "slow_push_in",
  "start_frame_asset": "asset_img_01",
  "end_frame_asset": null,
  "audio_mode": "mute"
}
```

### `video_generate.result`

```json
{
  "provider": "runway",
  "model": "gen4",
  "videos": [
    {
      "asset_id": "asset_video_01",
      "kind": "video",
      "url": "https://...",
      "thumbnail_url": "https://...",
      "mime_type": "video/mp4",
      "width": 1920,
      "height": 1080,
      "duration_ms": 5000
    }
  ],
  "poster_frames": ["asset_img_frame_01"]
}
```

## 9.4 视频编辑

### `video_edit.payload`

```json
{
  "source_asset": "asset_video_01",
  "edit_mode": "trim_and_overlay",
  "trim": {
    "start_ms": 1000,
    "end_ms": 4500
  },
  "overlay_assets": ["asset_logo_01"],
  "subtitle_spec": null,
  "target_format": "mp4"
}
```

### `video_edit.result`

```json
{
  "provider": "local_ffmpeg",
  "model": null,
  "videos": [
    {
      "asset_id": "asset_video_edit_01",
      "kind": "video",
      "url": "file:///...",
      "thumbnail_url": "file:///...",
      "mime_type": "video/mp4",
      "width": 1920,
      "height": 1080,
      "duration_ms": 3500
    }
  ],
  "timeline_summary": "裁剪并叠加品牌角标"
}
```

---

## 10. 进度、错误与 UI 提示

为了支撑聊天区动态占位与结果替换，task file 不能只提供最终状态，还必须提供“可渲染的运行时信息”。

## 10.1 `progress`

```json
{
  "phase": "rendering_preview",
  "percent": 42,
  "message": "正在生成预览图",
  "preview_slots": [
    {
      "slot_id": "article-image-slot-4",
      "label": "正文配图 1",
      "status": "running"
    }
  ]
}
```

要求：

- `phase` 适合前端直接渲染
- `percent` 可选，但若提供必须可信
- `preview_slots` 用于正文占位块与消息卡绑定

## 10.2 `last_error`

错误结构不再只是一段字符串：

```json
{
  "code": "provider_timeout",
  "message": "FAL 请求超时",
  "retryable": true,
  "stage": "provider_call",
  "provider_code": "TIMEOUT",
  "occurred_at": "2026-04-03T09:05:00Z"
}
```

要求：

- 前端可以直接根据 `retryable` 决定是否展示重试按钮
- CLI 可以根据 `code` 给出更稳定的退出提示

## 10.3 `ui_hints`

```json
{
  "render_mode": "media_placeholder_card",
  "placeholder_text": "[img:multimodal:一个充满未来感的实验室]",
  "preferred_surface": "claw_chat",
  "open_action": "open_image_workbench"
}
```

要求：

- 前端不必猜该如何渲染
- 但 `ui_hints` 只能是提示，不能覆盖真实任务状态

---

## 11. Artifacts 与大文件策略

主 task file 不直接存放大日志、大二进制结果。

统一使用 `artifacts[]` 指向外部资源：

```json
[
  {
    "artifact_id": "artifact_preview_01",
    "kind": "preview_manifest",
    "path": ".lime/task-artifacts/task_01/preview.json"
  },
  {
    "artifact_id": "artifact_log_01",
    "kind": "attempt_log",
    "path": ".lime/task-logs/task_01/attempt_2.jsonl"
  }
]
```

规则：

- 主文件只保留快速读所需信息
- 详细日志使用外部引用
- 结果媒体本体使用 `url / storage_ref`

---

## 12. 存储路径建议

当前根目录继续保持：

- `.lime/tasks`

但建议按 family / type 继续细分：

- `.lime/tasks/image/image_generate/<task_id>.json`
- `.lime/tasks/image/image_edit/<task_id>.json`
- `.lime/tasks/video/video_generate/<task_id>.json`
- `.lime/tasks/video/video_edit/<task_id>.json`

相关外置目录：

- `.lime/task-logs/<task_id>/attempt_<n>.jsonl`
- `.lime/task-artifacts/<task_id>/...`
- `.lime/previews/<asset_id>.json`

这样做的好处：

- 目录语义更清晰
- family/type 过滤更便宜
- worker 和 CLI 更容易做局部扫描

---

## 13. CLI 与前端接口影响

## 13.1 CLI

现有 CLI 需要在协议升级后扩展到：

- `lime task create <type>`
- `lime task status <task-ref>`
- `lime task result <task-ref>`
- `lime task attempts <task-ref>`
- `lime task retry <task-ref>`
- `lime task cancel <task-ref>`
- `lime task list --family image --type image_edit --status running`
- 可选：`lime task graph <task-ref>`

## 13.2 前端 / Tauri

前端统一通过 API 网关读取任务：

- `lime_task_get_status`
- `lime_task_get_result`
- `lime_task_get_attempts`
- `lime_task_list`
- `lime_task_retry`
- `lime_task_cancel`

前端不允许在页面和组件里自行解析零散任务文件路径。

---

## 14. 典型场景

## 14.1 正文插图生成

流程：

1. 用户在聊天区中触发 `image_generate`
2. 创建 `task_family=image / task_type=image_generate`
3. `relationships.slot_id` 绑定正文占位块
4. 前端先显示占位图
5. worker 完成后写回 `result.images[]`
6. 前端原位替换为真实图片

## 14.2 基于图片生成视频

流程：

1. 用户选中一张已生成图片
2. 创建 `video_generate`
3. `depends_on_task_ids` 指向图像任务
4. `payload.start_frame_asset` 指向图像资产
5. 任务面板中可看到“视频任务依赖图片任务”

## 14.3 图片失败后重试

流程：

1. 顶层 `task_id` 不变
2. 新增 `attempts[2]`
3. `current_attempt_id` 切换到 attempt 2
4. 前端仍更新同一张任务卡

---

## 15. 测试与验收

## 15.1 协议测试

- 顶层信封反序列化稳定
- `image_generate / image_edit / video_generate / video_edit` 子协议通过 schema 校验
- 顶层状态与 attempt 状态语义一致
- 重试只新增 attempt，不新增 task_id
- 关系字段能正确表达父子任务与依赖关系

## 15.2 CLI 测试

- `create/status/result/list/retry/cancel/attempts` 输出稳定
- `--family / --type / --status` 过滤有效
- `idempotency_key` 生效
- 大日志不塞进主文件

## 15.3 前端测试

- 同一 `task_id` 在聊天区原位更新
- 占位图到真实图替换成功
- 失败后展示结构化错误与重试入口
- `slot_id` 绑定的正文占位块替换正确
- 刷新或恢复会话后能从 task file 恢复显示

## 15.4 可交付验收

这套协议至少要支撑以下三个已验证任务族：

1. 图片生成
2. 图片编辑
3. 视频生成

并且保证：

- 聊天区、CLI、worker 读到的是同一份状态
- 可以支撑统一任务面板，而不需要第二套状态系统

---

## 16. 分阶段落地建议

## Phase 1：统一信封升级

- 升级 `TaskArtifactRecord`
- 补充 `task_family / progress / relationships / ui_hints`
- 把 `last_error` 从字符串升级为对象

## Phase 2：attempts 模型

- 引入 `attempts[]`
- 引入 `current_attempt_id`
- 调整 retry 语义

## Phase 3：图片与视频子协议

- 先补齐：
  - `image_generate`
  - `image_edit`
  - `video_generate`
- 让 worker 能稳定写回结果

## Phase 4：前端与任务面板

- 前端基于统一协议渲染聊天区任务卡
- 正文占位块改为 `slot_id + task_id` 绑定
- 准备统一任务面板

---

## 17. 最终结论

task file 应该被设计成 Lime 的统一任务协议，而不是某个业务临时写出的 JSON 记录。

正确方向不是：

- 每种任务各自定义一份完全不同的文件
- 或者把所有任务字段全塞进一份万能顶层 schema

而是：

- **统一顶层信封**
- **类型化 payload/result**
- **一任务多尝试**
- **父子任务与依赖关系**
- **结构化进度、错误、UI 提示**

这样 Lime 才能把图片生成、图片编辑、视频生成、视频编辑、聊天区动态渲染、CLI、worker、统一任务面板全部收敛到同一条主链。
