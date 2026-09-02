# lime-websocket

<!-- 一旦我所属的文件夹有所变化，请更新我 -->

## 架构说明

WebSocket API 支持模块，提供持久连接的 API 请求能力。
不依赖桌面宿主，可独立使用。

## 文件索引

- `src/lib.rs` - 模块入口、WsConnectionManager、类型重导出
- `src/handler.rs` - WebSocket 升级处理器、消息路由
- `src/lifecycle.rs` - 心跳管理、连接生命周期、优雅关闭
- `src/processor.rs` - 消息解析、请求验证、响应构建
- `src/stream.rs` - SSE 到 WebSocket 流式转换、背压控制
- `src/tests.rs` - 单元测试和属性测试（51 个）

## 更新提醒

任何文件变更后，请更新此文档和相关的上级文档。
