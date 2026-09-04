use app_server_protocol::JsonRpcMessage;
use async_trait::async_trait;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// App Server JSON-RPC 的 transport-neutral 会话边界。
///
/// 当前生产实现是本地 stdio；未来 Cloud transport 应在实现内部完成认证、
/// 租户隔离、协议版本协商和断线恢复，并继续复用 `ClientSession` 的请求与事件语义。
#[async_trait]
pub trait SessionTransport: Send + 'static {
    async fn send(&mut self, message: JsonRpcMessage) -> io::Result<()>;
    async fn receive(&mut self) -> io::Result<Option<JsonRpcMessage>>;
    async fn close(&mut self) -> io::Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StdioTransportConfig {
    pub app_server_bin: PathBuf,
    pub args: Vec<OsString>,
}

impl StdioTransportConfig {
    pub fn runtime(app_server_bin: impl Into<PathBuf>) -> Self {
        Self {
            app_server_bin: app_server_bin.into(),
            args: vec!["--stdio".into(), "--backend".into(), "runtime".into()],
        }
    }

    pub fn with_arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }
}

pub struct StdioTransport {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl StdioTransport {
    pub async fn spawn(config: StdioTransportConfig) -> io::Result<Self> {
        let mut command = Command::new(&config.app_server_bin);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = command.spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "app-server stdin is unavailable")
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "app-server stdout is unavailable",
            )
        })?;

        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout).lines(),
        })
    }
}

#[async_trait]
impl SessionTransport for StdioTransport {
    async fn send(&mut self, message: JsonRpcMessage) -> io::Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "app-server stdin is closed")
        })?;
        let line = app_server_transport::encode_message(&message)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await
    }

    async fn receive(&mut self) -> io::Result<Option<JsonRpcMessage>> {
        let Some(line) = self.stdout.next_line().await? else {
            return Ok(None);
        };
        app_server_transport::decode_message(&line)
            .map(Some)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    async fn close(&mut self) -> io::Result<()> {
        if let Some(mut stdin) = self.stdin.take() {
            stdin.shutdown().await?;
        }

        match tokio::time::timeout(DEFAULT_SHUTDOWN_TIMEOUT, self.child.wait()).await {
            Ok(result) => result.map(|_| ()),
            Err(_) => {
                self.child.start_kill()?;
                self.child.wait().await.map(|_| ())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_config_uses_current_app_server_backend() {
        let config = StdioTransportConfig::runtime("app-server");

        assert_eq!(config.app_server_bin, PathBuf::from("app-server"));
        assert_eq!(
            config.args,
            vec![
                OsString::from("--stdio"),
                OsString::from("--backend"),
                OsString::from("runtime")
            ]
        );
    }
}
