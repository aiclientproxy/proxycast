# 生产运维与上线就绪清单

本文档用于生产环境部署、运行、备份与恢复的最小操作规范，避免上线后缺少可执行流程。

## 部署前检查

- 确认 API Key 已更换（禁止使用默认值 `proxy_cast`）。
- 确认监听地址：
  - 本机使用 `127.0.0.1`/`localhost`。
- 当前版本仅支持本地监听，不支持对外服务。
- 若需要 HTTPS，请使用反向代理终止 TLS；当前服务端未启用内置 TLS。
- 确认磁盘权限可写：`~/.proxycast/`、`~/.proxycast/request_logs/`、应用数据目录（macOS: `~/Library/Application Support/proxycast/`，Linux: `~/.local/share/proxycast/`，Windows: `%APPDATA%\\proxycast\\`）。

## 配置路径与加载顺序

- YAML 配置（优先）：
  - macOS: `~/Library/Application Support/proxycast/config.yaml`
  - Linux: `~/.config/proxycast/config.yaml`
  - Windows: `%APPDATA%\\proxycast\\config.yaml`
- JSON 配置（兼容）：macOS `~/Library/Application Support/proxycast/config.json`，Linux `~/.config/proxycast/config.json`，Windows `%APPDATA%\\proxycast\\config.json`
- 旧版遗留路径：`~/.proxycast/config.json`（检测到会提示手动迁移）
- 两者都不存在时使用默认配置。

## 数据与日志位置

- SQLite 数据库：`~/.proxycast/proxycast.db`
- 凭证池副本目录：macOS `~/Library/Application Support/proxycast/credentials/`，Linux `~/.local/share/proxycast/credentials/`，Windows `%APPDATA%\\proxycast\\credentials\\`
- OAuth/Token 目录（默认）：`~/.proxycast/auth/`
- 日志目录：`~/.proxycast/logs/`
- 请求日志目录：`~/.proxycast/request_logs/`

## 备份与恢复

### 备份

1. 停止 ProxyCast 服务。
2. 复制以下路径：
   - 配置文件（macOS: `~/Library/Application Support/proxycast/config.yaml`，Linux: `~/.config/proxycast/config.yaml`，Windows: `%APPDATA%\\proxycast\\config.yaml`）
   - 凭证池副本目录（macOS: `~/Library/Application Support/proxycast/credentials/`，Linux: `~/.local/share/proxycast/credentials/`，Windows: `%APPDATA%\\proxycast\\credentials\\`）
   - `~/.proxycast/proxycast.db`
   - `~/.proxycast/auth/`（如需要保留 OAuth/Token）
   - `~/.proxycast/logs/`、`~/.proxycast/request_logs/`（如需保留日志）
3. 将备份文件存入受控存储（加密磁盘或安全存储）。

### 恢复

1. 停止 ProxyCast 服务。
2. 恢复上述文件到原路径。
3. 启动服务并检查 `/health`。

## 升级与回滚

- 升级前执行备份流程。
- 升级后若出现异常：
  - 恢复备份文件。
  - 回滚到上一个稳定版本的安装包。

## 运行与排障

- 健康检查：`GET /health`
- 常见问题排查：
  - 端口占用：修改配置端口或释放占用端口。
  - 配置解析失败：检查 YAML/JSON 语法，确认缩进正确。
  - 数据库初始化失败：检查 `~/.proxycast/` 权限与磁盘空间。

## 安全基线

- 禁止默认 API key。
- 当前版本未实现内置 TLS，远程管理必须保持关闭且仅本地访问。

## 管理 API 基线

- 管理 API 启用后会对失败认证进行短期限制，避免暴力尝试。
- 建议仅在内网使用，并配合独立强密钥。
