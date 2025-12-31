//! Bash 工具模块
//!
//! 提供 Shell 命令执行功能，支持 bash/zsh/powershell
//! 符合 Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
//!
//! ## 功能
//! - Shell 配置检测（bash/zsh/powershell）
//! - 命令执行（捕获 stdout/stderr）
//! - 超时控制
//! - 防止交互的环境变量设置

use super::registry::Tool;
use super::security::SecurityManager;
use super::types::{JsonSchema, PropertySchema, ToolDefinition, ToolError, ToolResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// 默认超时时间（秒）
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// 最大输出大小（字节）
const MAX_OUTPUT_SIZE: usize = 1024 * 1024; // 1MB

/// Shell 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellType {
    /// Bash shell
    Bash,
    /// Zsh shell
    Zsh,
    /// PowerShell (Windows)
    PowerShell,
    /// Cmd (Windows fallback)
    Cmd,
    /// Sh (Unix fallback)
    Sh,
}

impl ShellType {
    /// 获取 shell 可执行文件路径
    pub fn executable(&self) -> &'static str {
        match self {
            ShellType::Bash => "bash",
            ShellType::Zsh => "zsh",
            ShellType::PowerShell => "powershell",
            ShellType::Cmd => "cmd",
            ShellType::Sh => "sh",
        }
    }

    /// 获取执行命令的参数
    pub fn command_args(&self) -> Vec<&'static str> {
        match self {
            ShellType::Bash | ShellType::Zsh | ShellType::Sh => vec!["-c"],
            ShellType::PowerShell => vec!["-Command"],
            ShellType::Cmd => vec!["/C"],
        }
    }
}

/// Bash 执行结果
#[derive(Debug, Clone)]
pub struct BashExecutionResult {
    /// 标准输出
    pub stdout: String,
    /// 标准错误
    pub stderr: String,
    /// 退出码
    pub exit_code: Option<i32>,
    /// 是否超时
    pub timed_out: bool,
}

impl BashExecutionResult {
    /// 创建成功结果
    pub fn success(stdout: String, stderr: String, exit_code: i32) -> Self {
        Self {
            stdout,
            stderr,
            exit_code: Some(exit_code),
            timed_out: false,
        }
    }

    /// 创建超时结果
    pub fn timeout(stdout: String, stderr: String) -> Self {
        Self {
            stdout,
            stderr,
            exit_code: None,
            timed_out: true,
        }
    }

    /// 检查命令是否成功执行
    pub fn is_success(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out
    }

    /// 获取合并的输出
    pub fn combined_output(&self) -> String {
        let mut output = String::new();
        if !self.stdout.is_empty() {
            output.push_str(&self.stdout);
        }
        if !self.stderr.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&self.stderr);
        }
        output
    }
}

/// Bash 工具
///
/// 执行 shell 命令并返回结果
/// Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
pub struct BashTool {
    /// 安全管理器
    security: Arc<SecurityManager>,
    /// 默认工作目录
    working_dir: PathBuf,
    /// 超时时间（秒）
    timeout_secs: u64,
    /// Shell 类型
    shell_type: ShellType,
}

impl BashTool {
    /// 创建新的 Bash 工具
    pub fn new(security: Arc<SecurityManager>) -> Self {
        let working_dir = security.base_dir().to_path_buf();
        let shell_type = Self::detect_shell();

        Self {
            security,
            working_dir,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            shell_type,
        }
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// 设置工作目录
    pub fn with_working_dir(mut self, working_dir: PathBuf) -> Self {
        self.working_dir = working_dir;
        self
    }

    /// 设置 shell 类型
    pub fn with_shell_type(mut self, shell_type: ShellType) -> Self {
        self.shell_type = shell_type;
        self
    }

    /// 检测用户默认 shell
    ///
    /// Requirements: 3.1 - THE Bash_Executor SHALL execute it in the user's default shell
    pub fn detect_shell() -> ShellType {
        #[cfg(windows)]
        {
            // Windows 优先使用 PowerShell
            if Self::is_shell_available("powershell") {
                return ShellType::PowerShell;
            }
            return ShellType::Cmd;
        }

        #[cfg(not(windows))]
        {
            // Unix 系统检查 SHELL 环境变量
            if let Ok(shell) = env::var("SHELL") {
                if shell.contains("zsh") {
                    return ShellType::Zsh;
                }
                if shell.contains("bash") {
                    return ShellType::Bash;
                }
            }

            // 检查可用的 shell
            if Self::is_shell_available("zsh") {
                return ShellType::Zsh;
            }
            if Self::is_shell_available("bash") {
                return ShellType::Bash;
            }
            ShellType::Sh
        }
    }

    /// 检查 shell 是否可用
    fn is_shell_available(shell: &str) -> bool {
        std::process::Command::new("which")
            .arg(shell)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// 获取防止交互的环境变量
    ///
    /// Requirements: 3.4 - THE Bash_Executor SHALL set appropriate environment variables to prevent interactive prompts
    /// Requirements: 8.4 - THE Bash_Executor SHALL set environment variables to disable interactive editors and prompts
    pub fn get_non_interactive_env() -> HashMap<String, String> {
        let mut env = HashMap::new();

        // 标记为非交互终端
        env.insert("GOOSE_TERMINAL".to_string(), "1".to_string());

        // 禁用 Git 交互提示
        env.insert("GIT_TERMINAL_PROMPT".to_string(), "0".to_string());

        // 设置非交互编辑器
        env.insert("EDITOR".to_string(), "cat".to_string());
        env.insert("VISUAL".to_string(), "cat".to_string());
        env.insert("GIT_EDITOR".to_string(), "cat".to_string());

        // 禁用 SSH 交互
        env.insert(
            "GIT_SSH_COMMAND".to_string(),
            "ssh -o BatchMode=yes".to_string(),
        );

        // 禁用 GPG 交互
        env.insert("GPG_TTY".to_string(), "".to_string());

        // 设置 CI 环境变量（许多工具会检查这个）
        env.insert("CI".to_string(), "true".to_string());

        // 禁用颜色输出（避免 ANSI 转义序列）
        env.insert("NO_COLOR".to_string(), "1".to_string());
        env.insert("TERM".to_string(), "dumb".to_string());

        // npm/yarn 非交互模式
        env.insert("npm_config_yes".to_string(), "true".to_string());

        // 禁用 pager
        env.insert("PAGER".to_string(), "cat".to_string());
        env.insert("GIT_PAGER".to_string(), "cat".to_string());

        env
    }

    /// 执行命令
    ///
    /// Requirements: 3.1, 3.2, 3.3, 3.5
    pub async fn execute_command(
        &self,
        command: &str,
        working_dir: Option<&PathBuf>,
        timeout_secs: Option<u64>,
    ) -> Result<BashExecutionResult, ToolError> {
        let work_dir = working_dir.unwrap_or(&self.working_dir);
        let timeout_duration = Duration::from_secs(timeout_secs.unwrap_or(self.timeout_secs));

        info!(
            "[BashTool] 执行命令: {} (工作目录: {:?}, 超时: {:?})",
            command, work_dir, timeout_duration
        );

        // 构建命令
        let mut cmd = Command::new(self.shell_type.executable());

        // 添加命令参数
        for arg in self.shell_type.command_args() {
            cmd.arg(arg);
        }
        cmd.arg(command);

        // 设置工作目录
        // Requirements: 3.6 - THE Bash_Executor SHALL support a configurable working directory
        cmd.current_dir(work_dir);

        // 设置环境变量
        // Requirements: 3.4 - prevent interactive prompts
        for (key, value) in Self::get_non_interactive_env() {
            cmd.env(key, value);
        }

        // 配置标准输入输出
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // 启动进程
        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(format!("无法启动 shell 进程: {}", e)))?;

        // 获取输出流
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| ToolError::ExecutionFailed("无法获取 stdout".to_string()))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| ToolError::ExecutionFailed("无法获取 stderr".to_string()))?;

        // 异步读取输出
        let stdout_handle = tokio::spawn(async move {
            let mut buffer = Vec::new();
            let _ = stdout.read_to_end(&mut buffer).await;
            buffer
        });

        let stderr_handle = tokio::spawn(async move {
            let mut buffer = Vec::new();
            let _ = stderr.read_to_end(&mut buffer).await;
            buffer
        });

        // 等待进程完成（带超时）
        // Requirements: 3.3 - WHEN a command exceeds the timeout limit, THE Bash_Executor SHALL terminate it
        let result = timeout(timeout_duration, child.wait()).await;

        match result {
            Ok(Ok(status)) => {
                // 进程正常完成
                let stdout_bytes = stdout_handle.await.unwrap_or_default();
                let stderr_bytes = stderr_handle.await.unwrap_or_default();

                let stdout_str =
                    Self::truncate_output(String::from_utf8_lossy(&stdout_bytes).to_string());
                let stderr_str =
                    Self::truncate_output(String::from_utf8_lossy(&stderr_bytes).to_string());

                let exit_code = status.code().unwrap_or(-1);

                debug!(
                    "[BashTool] 命令完成: exit_code={}, stdout_len={}, stderr_len={}",
                    exit_code,
                    stdout_str.len(),
                    stderr_str.len()
                );

                Ok(BashExecutionResult::success(
                    stdout_str, stderr_str, exit_code,
                ))
            }
            Ok(Err(e)) => {
                // 进程等待失败
                Err(ToolError::ExecutionFailed(format!("等待进程失败: {}", e)))
            }
            Err(_) => {
                // 超时
                warn!("[BashTool] 命令超时，正在终止进程");

                // 尝试终止进程
                let _ = child.kill().await;

                // 获取已有的输出
                let stdout_bytes = stdout_handle.await.unwrap_or_default();
                let stderr_bytes = stderr_handle.await.unwrap_or_default();

                let stdout_str =
                    Self::truncate_output(String::from_utf8_lossy(&stdout_bytes).to_string());
                let stderr_str =
                    Self::truncate_output(String::from_utf8_lossy(&stderr_bytes).to_string());

                Ok(BashExecutionResult::timeout(stdout_str, stderr_str))
            }
        }
    }

    /// 截断过长的输出
    fn truncate_output(output: String) -> String {
        if output.len() > MAX_OUTPUT_SIZE {
            let truncated = &output[..MAX_OUTPUT_SIZE];
            format!(
                "{}\n\n[输出已截断，原始大小: {} 字节]",
                truncated,
                output.len()
            )
        } else {
            output
        }
    }

    /// 获取当前 shell 类型
    pub fn shell_type(&self) -> ShellType {
        self.shell_type
    }

    /// 获取当前工作目录
    pub fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// 获取超时时间
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }
}

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "bash",
            "Execute a bash command in the shell. Use this for running system commands, \
             scripts, or any command-line operations. The command will be executed in the \
             configured working directory with a timeout limit.",
        )
        .with_parameters(
            JsonSchema::new()
                .add_property(
                    "command",
                    PropertySchema::string(
                        "The bash command to execute. Can be any valid shell command.",
                    ),
                    true,
                )
                .add_property(
                    "timeout",
                    PropertySchema::integer(
                        "Optional timeout in seconds. Defaults to 120 seconds.",
                    )
                    .with_default(serde_json::json!(120)),
                    false,
                ),
        )
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        // 解析参数
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 command 参数".to_string()))?;

        let timeout_secs = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);

        // 执行命令
        let result = self
            .execute_command(command, None, Some(timeout_secs))
            .await?;

        // 构建输出
        // Requirements: 3.2 - THE Bash_Executor SHALL capture both stdout and stderr
        // Requirements: 3.5 - IF a command fails, THEN THE Bash_Executor SHALL return the exit code and error output
        if result.timed_out {
            let output = format!(
                "命令执行超时（{}秒）\n\n已捕获的输出:\n{}",
                timeout_secs,
                result.combined_output()
            );
            return Err(ToolError::Timeout);
        }

        let output = if result.is_success() {
            result.combined_output()
        } else {
            format!(
                "命令执行失败 (退出码: {})\n\n{}",
                result.exit_code.unwrap_or(-1),
                result.combined_output()
            )
        };

        if result.is_success() {
            Ok(ToolResult::success(output))
        } else {
            Ok(ToolResult::failure_with_output(
                output.clone(),
                format!("命令退出码: {}", result.exit_code.unwrap_or(-1)),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_tool() -> (BashTool, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = BashTool::new(security);
        (tool, temp_dir)
    }

    #[test]
    fn test_shell_detection() {
        let shell_type = BashTool::detect_shell();
        // 应该检测到某种 shell
        assert!(matches!(
            shell_type,
            ShellType::Bash
                | ShellType::Zsh
                | ShellType::Sh
                | ShellType::PowerShell
                | ShellType::Cmd
        ));
    }

    #[test]
    fn test_non_interactive_env() {
        let env = BashTool::get_non_interactive_env();

        // 检查关键环境变量
        assert_eq!(env.get("GOOSE_TERMINAL"), Some(&"1".to_string()));
        assert_eq!(env.get("GIT_TERMINAL_PROMPT"), Some(&"0".to_string()));
        assert_eq!(env.get("CI"), Some(&"true".to_string()));
    }

    #[test]
    fn test_tool_definition() {
        let (tool, _temp_dir) = setup_test_tool();
        let def = tool.definition();

        assert_eq!(def.name, "bash");
        assert!(!def.description.is_empty());
        assert!(def.parameters.required.contains(&"command".to_string()));
    }

    #[tokio::test]
    async fn test_execute_simple_command() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool
            .execute_command("echo 'Hello, World!'", None, None)
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_success());
        assert!(result.stdout.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_execute_command_with_stderr() {
        let (tool, _temp_dir) = setup_test_tool();

        // 使用一个会产生 stderr 的命令
        let result = tool
            .execute_command("echo 'stdout' && echo 'stderr' >&2", None, None)
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_success());
        assert!(result.stdout.contains("stdout"));
        assert!(result.stderr.contains("stderr"));
    }

    #[tokio::test]
    async fn test_execute_failing_command() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool.execute_command("exit 1", None, None).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.is_success());
        assert_eq!(result.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_execute_with_timeout() {
        let (tool, _temp_dir) = setup_test_tool();

        // 使用一个会超时的命令（1秒超时）
        let result = tool.execute_command("sleep 10", None, Some(1)).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.timed_out);
        assert!(result.exit_code.is_none());
    }

    #[tokio::test]
    async fn test_tool_execute() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool
            .execute(serde_json::json!({
                "command": "echo 'test'"
            }))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.output.contains("test"));
    }

    #[tokio::test]
    async fn test_tool_execute_missing_command() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool.execute(serde_json::json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[test]
    fn test_shell_type_executable() {
        assert_eq!(ShellType::Bash.executable(), "bash");
        assert_eq!(ShellType::Zsh.executable(), "zsh");
        assert_eq!(ShellType::PowerShell.executable(), "powershell");
    }

    #[test]
    fn test_shell_type_command_args() {
        assert_eq!(ShellType::Bash.command_args(), vec!["-c"]);
        assert_eq!(ShellType::Zsh.command_args(), vec!["-c"]);
        assert_eq!(ShellType::PowerShell.command_args(), vec!["-Command"]);
        assert_eq!(ShellType::Cmd.command_args(), vec!["/C"]);
    }

    #[test]
    fn test_truncate_output() {
        // 短输出不截断
        let short = "Hello".to_string();
        assert_eq!(BashTool::truncate_output(short.clone()), short);

        // 长输出截断
        let long = "x".repeat(MAX_OUTPUT_SIZE + 100);
        let truncated = BashTool::truncate_output(long.clone());
        assert!(truncated.len() < long.len());
        assert!(truncated.contains("[输出已截断"));
    }

    #[test]
    fn test_bash_execution_result() {
        let success = BashExecutionResult::success("out".to_string(), "err".to_string(), 0);
        assert!(success.is_success());
        assert_eq!(success.exit_code, Some(0));

        let failure = BashExecutionResult::success("out".to_string(), "err".to_string(), 1);
        assert!(!failure.is_success());

        let timeout = BashExecutionResult::timeout("out".to_string(), "err".to_string());
        assert!(!timeout.is_success());
        assert!(timeout.timed_out);
    }

    #[test]
    fn test_combined_output() {
        let result = BashExecutionResult::success(
            "stdout content".to_string(),
            "stderr content".to_string(),
            0,
        );
        let combined = result.combined_output();
        assert!(combined.contains("stdout content"));
        assert!(combined.contains("stderr content"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::TempDir;

    /// 生成有效的简单命令（echo 命令）
    /// 避免以 '-' 开头（会被 echo 解释为选项）
    fn arb_echo_content() -> impl Strategy<Value = String> {
        // 生成安全的字符串内容（以字母开头，避免特殊 shell 字符）
        "[a-zA-Z][a-zA-Z0-9 _]{0,49}".prop_map(|s| s)
    }

    /// 生成有效的退出码
    fn arb_exit_code() -> impl Strategy<Value = i32> {
        0..=255i32
    }

    /// 生成有效的超时时间（秒）
    fn arb_timeout_secs() -> impl Strategy<Value = u64> {
        1..=10u64
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 4: Bash 命令执行捕获**
        /// **Validates: Requirements 3.1, 3.2, 3.5**
        ///
        /// *For any* Bash 命令执行，返回的结果应该包含命令的 stdout 和 stderr 输出，
        /// 以及正确的退出码。
        #[test]
        fn prop_bash_captures_stdout(content in arb_echo_content()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 执行 echo 命令
                let command = format!("echo '{}'", content);
                let result = tool.execute_command(&command, None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                // 验证 stdout 包含输出内容
                prop_assert!(
                    result.stdout.contains(&content),
                    "stdout 应该包含 echo 的内容: expected '{}', got '{}'",
                    content,
                    result.stdout
                );

                // 验证退出码为 0
                prop_assert_eq!(
                    result.exit_code,
                    Some(0),
                    "成功命令的退出码应该为 0"
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 4: Bash 命令执行捕获 - stderr**
        /// **Validates: Requirements 3.1, 3.2, 3.5**
        ///
        /// *For any* 输出到 stderr 的命令，结果应该捕获 stderr 内容。
        #[test]
        fn prop_bash_captures_stderr(content in arb_echo_content()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 执行输出到 stderr 的命令
                let command = format!("echo '{}' >&2", content);
                let result = tool.execute_command(&command, None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                // 验证 stderr 包含输出内容
                prop_assert!(
                    result.stderr.contains(&content),
                    "stderr 应该包含 echo 的内容: expected '{}', got '{}'",
                    content,
                    result.stderr
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 4: Bash 命令执行捕获 - 退出码**
        /// **Validates: Requirements 3.1, 3.2, 3.5**
        ///
        /// *For any* 指定退出码的命令，结果应该返回正确的退出码。
        #[test]
        fn prop_bash_captures_exit_code(exit_code in arb_exit_code()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 执行带指定退出码的命令
                let command = format!("exit {}", exit_code);
                let result = tool.execute_command(&command, None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该完成（即使退出码非零）");
                let result = result.unwrap();

                // 验证退出码正确
                prop_assert_eq!(
                    result.exit_code,
                    Some(exit_code),
                    "退出码应该匹配: expected {}, got {:?}",
                    exit_code,
                    result.exit_code
                );

                // 验证 is_success 方法正确
                prop_assert_eq!(
                    result.is_success(),
                    exit_code == 0,
                    "is_success() 应该在退出码为 0 时返回 true"
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 4: Bash 命令执行捕获 - stdout 和 stderr 同时**
        /// **Validates: Requirements 3.1, 3.2, 3.5**
        ///
        /// *For any* 同时输出到 stdout 和 stderr 的命令，结果应该分别捕获两者。
        #[test]
        fn prop_bash_captures_both_streams(
            stdout_content in arb_echo_content(),
            stderr_content in arb_echo_content()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 执行同时输出到 stdout 和 stderr 的命令
                let command = format!("echo '{}' && echo '{}' >&2", stdout_content, stderr_content);
                let result = tool.execute_command(&command, None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                // 验证 stdout 包含内容
                prop_assert!(
                    result.stdout.contains(&stdout_content),
                    "stdout 应该包含内容: expected '{}', got '{}'",
                    stdout_content,
                    result.stdout
                );

                // 验证 stderr 包含内容
                prop_assert!(
                    result.stderr.contains(&stderr_content),
                    "stderr 应该包含内容: expected '{}', got '{}'",
                    stderr_content,
                    result.stderr
                );

                // 验证 combined_output 包含两者
                let combined = result.combined_output();
                prop_assert!(
                    combined.contains(&stdout_content) && combined.contains(&stderr_content),
                    "combined_output 应该包含 stdout 和 stderr"
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 4: Bash 命令执行捕获 - 超时**
        /// **Validates: Requirements 3.3**
        ///
        /// *For any* 超时的命令，结果应该标记为超时。
        #[test]
        fn prop_bash_timeout_detection(timeout_secs in 1u64..=2u64) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 执行一个会超时的命令（sleep 时间大于超时时间）
                let sleep_time = timeout_secs + 5;
                let command = format!("sleep {}", sleep_time);
                let result = tool.execute_command(&command, None, Some(timeout_secs)).await;

                prop_assert!(result.is_ok(), "超时命令应该返回结果而不是错误");
                let result = result.unwrap();

                // 验证超时标记
                prop_assert!(
                    result.timed_out,
                    "命令应该被标记为超时"
                );

                // 验证退出码为 None（因为进程被终止）
                prop_assert!(
                    result.exit_code.is_none(),
                    "超时命令的退出码应该为 None"
                );

                // 验证 is_success 返回 false
                prop_assert!(
                    !result.is_success(),
                    "超时命令的 is_success() 应该返回 false"
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 5: Bash 环境变量设置**
        /// **Validates: Requirements 3.4, 8.4**
        ///
        /// *For any* Bash 命令执行，执行环境应该包含 GOOSE_TERMINAL、GIT_TERMINAL_PROMPT=0 等
        /// 防止交互的环境变量。
        #[test]
        fn prop_bash_env_goose_terminal_set(_dummy in 0..100u32) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 检查 GOOSE_TERMINAL 环境变量
                let result = tool.execute_command("echo $GOOSE_TERMINAL", None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                prop_assert!(
                    result.stdout.trim() == "1",
                    "GOOSE_TERMINAL 应该设置为 1，实际值: '{}'",
                    result.stdout.trim()
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 5: Bash 环境变量设置 - GIT_TERMINAL_PROMPT**
        /// **Validates: Requirements 3.4, 8.4**
        ///
        /// *For any* Bash 命令执行，GIT_TERMINAL_PROMPT 应该设置为 0 以禁用 Git 交互提示。
        #[test]
        fn prop_bash_env_git_terminal_prompt_disabled(_dummy in 0..100u32) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 检查 GIT_TERMINAL_PROMPT 环境变量
                let result = tool.execute_command("echo $GIT_TERMINAL_PROMPT", None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                prop_assert!(
                    result.stdout.trim() == "0",
                    "GIT_TERMINAL_PROMPT 应该设置为 0，实际值: '{}'",
                    result.stdout.trim()
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 5: Bash 环境变量设置 - CI**
        /// **Validates: Requirements 3.4, 8.4**
        ///
        /// *For any* Bash 命令执行，CI 环境变量应该设置为 true。
        #[test]
        fn prop_bash_env_ci_set(_dummy in 0..100u32) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 检查 CI 环境变量
                let result = tool.execute_command("echo $CI", None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                prop_assert!(
                    result.stdout.trim() == "true",
                    "CI 应该设置为 true，实际值: '{}'",
                    result.stdout.trim()
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 5: Bash 环境变量设置 - EDITOR**
        /// **Validates: Requirements 3.4, 8.4**
        ///
        /// *For any* Bash 命令执行，EDITOR 应该设置为非交互式编辑器（cat）。
        #[test]
        fn prop_bash_env_editor_non_interactive(_dummy in 0..100u32) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 检查 EDITOR 环境变量
                let result = tool.execute_command("echo $EDITOR", None, None).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();

                prop_assert!(
                    result.stdout.trim() == "cat",
                    "EDITOR 应该设置为 cat，实际值: '{}'",
                    result.stdout.trim()
                );

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 5: Bash 环境变量设置 - 所有关键变量**
        /// **Validates: Requirements 3.4, 8.4**
        ///
        /// *For any* Bash 命令执行，所有防止交互的关键环境变量都应该正确设置。
        #[test]
        fn prop_bash_env_all_non_interactive_vars(_dummy in 0..100u32) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let security = Arc::new(SecurityManager::new(temp_dir.path()));
                let tool = BashTool::new(security);

                // 检查多个环境变量
                let result = tool.execute_command(
                    "echo \"GOOSE=$GOOSE_TERMINAL,GIT=$GIT_TERMINAL_PROMPT,CI=$CI,PAGER=$PAGER\"",
                    None,
                    None
                ).await;

                prop_assert!(result.is_ok(), "命令执行应该成功");
                let result = result.unwrap();
                let output = result.stdout.trim();

                // 验证所有关键变量
                prop_assert!(
                    output.contains("GOOSE=1"),
                    "GOOSE_TERMINAL 应该为 1，输出: '{}'",
                    output
                );
                prop_assert!(
                    output.contains("GIT=0"),
                    "GIT_TERMINAL_PROMPT 应该为 0，输出: '{}'",
                    output
                );
                prop_assert!(
                    output.contains("CI=true"),
                    "CI 应该为 true，输出: '{}'",
                    output
                );
                prop_assert!(
                    output.contains("PAGER=cat"),
                    "PAGER 应该为 cat，输出: '{}'",
                    output
                );

                Ok(())
            })?;
        }
    }
}
