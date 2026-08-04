## Lime v1.120.1

### 新功能

### 修复

- 修复普通运行时 Thread Item 的内部 provenance metadata 穿过 v2 投影边界，避免 `contextCompaction` 等公开 item 泄漏 `sourceEventType` 等私有字段；历史导入项仍保留受控的 typed metadata。

### 优化与重构

### 测试与质量

- 补充 projection library 单测与 fork-compaction JSON-RPC 集成回归，确认 metadata 边界和重启后的替换/尾部恢复行为。

### 文档

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.120.1`。

**完整变更**: `v1.120.0` -> `v1.120.1`
