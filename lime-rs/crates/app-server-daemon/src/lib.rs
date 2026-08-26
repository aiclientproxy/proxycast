use app_server_protocol::PROTOCOL_VERSION;
use app_server_transport::DEFAULT_LISTEN_URL;
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;
use std::time::Instant;

mod backend;
mod client;
mod lifecycle;
mod managed_sidecar;
mod settings;
mod update_policy;

pub use backend::BackendKind;
pub use backend::BackendPaths;
pub use backend::BackendReadinessProbe;
pub use backend::BackendStartDecision;
pub use backend::SidecarBackendMode;
pub use client::initialize_probe_request;
pub use client::initialized_notification;
pub use client::parse_version_from_user_agent;
pub use client::probe_info_from_initialize_response;
pub use client::ProbeInfo;
pub use client::DAEMON_CLIENT_NAME;
pub use client::INITIALIZE_REQUEST_ID;
pub use lifecycle::acquire_operation_lock;
pub use lifecycle::unsupported_lifecycle_output;
pub use lifecycle::DaemonStatePaths;
pub use lifecycle::LifecycleCommand;
pub use lifecycle::LifecycleOutput;
pub use lifecycle::LifecycleStatus;
pub use lifecycle::OperationLock;
pub use lifecycle::OperationLockError;
pub use managed_sidecar::executable_identity;
pub use managed_sidecar::executable_identity_from_bytes;
pub use managed_sidecar::managed_sidecar_binary_path;
pub use managed_sidecar::ExecutableIdentity;
pub use settings::DaemonSettings;
pub use update_policy::update_modes_for_identities;
pub use update_policy::RestartMode;
pub use update_policy::UpdateLoopControl;
pub use update_policy::UpdaterRefreshMode;

#[cfg(windows)]
pub const SIDECAR_BINARY_NAME: &str = "app-server.exe";

#[cfg(not(windows))]
pub const SIDECAR_BINARY_NAME: &str = "app-server";

pub const DEFAULT_SIDECAR_ENV_VAR: &str = "APP_SERVER_BIN";
pub const PID_FILE_NAME: &str = "app-server.pid";
pub const UPDATE_PID_FILE_NAME: &str = "app-server-updater.pid";
pub const OPERATION_LOCK_FILE_NAME: &str = "daemon.lock";
pub const SETTINGS_FILE_NAME: &str = "settings.json";
pub const STDERR_LOG_FILE_NAME: &str = "app-server.stderr.log";
pub const STATE_DIR_NAME: &str = "app-server-daemon";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarLaunchConfig {
    pub binary_path: PathBuf,
    pub listen_url: String,
    pub expected_sha256: Option<String>,
    pub backend_mode: SidecarBackendMode,
    pub backend_command: Option<String>,
    pub backend_args: Vec<String>,
    pub backend_timeout_ms: Option<u64>,
    pub app_policy_path: Option<PathBuf>,
}

impl SidecarLaunchConfig {
    pub fn stdio(binary_path: impl Into<PathBuf>) -> Self {
        Self {
            binary_path: binary_path.into(),
            listen_url: DEFAULT_LISTEN_URL.to_string(),
            expected_sha256: None,
            backend_mode: SidecarBackendMode::Unavailable,
            backend_command: None,
            backend_args: Vec::new(),
            backend_timeout_ms: None,
            app_policy_path: None,
        }
    }

    pub fn args(&self) -> Vec<String> {
        let mut args = if self.listen_url == DEFAULT_LISTEN_URL {
            vec!["--stdio".to_string()]
        } else {
            vec!["--listen".to_string(), self.listen_url.clone()]
        };
        args.push("--backend".to_string());
        args.push(self.backend_mode.as_str().to_string());
        if self.backend_mode == SidecarBackendMode::External {
            if let Some(command) = self.backend_command.as_deref() {
                args.push("--backend-command".to_string());
                args.push(command.to_string());
            }
            for backend_arg in &self.backend_args {
                args.push("--backend-arg".to_string());
                args.push(backend_arg.clone());
            }
            if let Some(timeout_ms) = self.backend_timeout_ms {
                args.push("--backend-timeout-ms".to_string());
                args.push(timeout_ms.to_string());
            }
        }
        if let Some(app_policy_path) = &self.app_policy_path {
            args.push("--app-policy".to_string());
            args.push(app_policy_path.to_string_lossy().to_string());
        }
        args
    }

    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.binary_path);
        command
            .args(self.args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    pub fn spawn(&self) -> std::io::Result<Child> {
        self.command().spawn()
    }

    pub fn spawn_verified(&self) -> Result<SidecarProcess, SidecarProcessError> {
        SidecarProcess::spawn(self)
    }

    pub fn probe_readiness(
        &self,
        stderr_log_file: impl Into<PathBuf>,
        client_version: impl Into<String>,
    ) -> Result<SidecarReadinessReport, SidecarReadinessError> {
        probe_sidecar_readiness(self, stderr_log_file, client_version)
    }

    pub fn verify_expected_sha256(&self) -> Result<(), String> {
        verify_sidecar_file_sha256(self)
    }

    pub fn requires_sha256_verification(&self) -> bool {
        self.expected_sha256
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
    }
}

#[derive(Debug)]
pub struct SidecarProcess {
    child: Child,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarReadinessReport {
    pub pid: u32,
    pub probe_info: ProbeInfo,
    pub stderr_log_file: PathBuf,
    pub stderr_bytes: usize,
}

impl SidecarProcess {
    pub fn spawn(config: &SidecarLaunchConfig) -> Result<Self, SidecarProcessError> {
        config
            .verify_expected_sha256()
            .map_err(SidecarProcessError::Sha256)?;
        let child = config.spawn().map_err(SidecarProcessError::Spawn)?;
        Ok(Self { child })
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn child(&mut self) -> &mut Child {
        &mut self.child
    }

    pub fn into_child(self) -> Child {
        self.child
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        self.child.wait()
    }

    pub fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }
}

#[derive(Debug)]
pub enum SidecarProcessError {
    Sha256(String),
    Spawn(io::Error),
}

impl fmt::Display for SidecarProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sha256(error) => write!(formatter, "{error}"),
            Self::Spawn(error) => write!(formatter, "failed to spawn app-server sidecar: {error}"),
        }
    }
}

impl StdError for SidecarProcessError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Sha256(_) => None,
            Self::Spawn(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum SidecarReadinessError {
    Sha256(String),
    Spawn(io::Error),
    MissingPipe(&'static str),
    Io(io::Error),
    Protocol(String),
    StderrLog(io::Error),
    Exit(ExitStatus),
    Timeout,
}

impl fmt::Display for SidecarReadinessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sha256(error) => write!(formatter, "{error}"),
            Self::Spawn(error) => write!(formatter, "failed to spawn app-server sidecar: {error}"),
            Self::MissingPipe(pipe) => write!(formatter, "app-server sidecar {pipe} is not piped"),
            Self::Io(error) => write!(formatter, "app-server sidecar I/O failed: {error}"),
            Self::Protocol(error) => write!(formatter, "{error}"),
            Self::StderrLog(error) => {
                write!(formatter, "failed to write sidecar stderr log: {error}")
            }
            Self::Exit(status) => {
                write!(formatter, "app-server sidecar exited with status {status}")
            }
            Self::Timeout => write!(
                formatter,
                "app-server sidecar did not exit after readiness probe"
            ),
        }
    }
}

impl StdError for SidecarReadinessError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Spawn(error) | Self::Io(error) | Self::StderrLog(error) => Some(error),
            Self::Sha256(_)
            | Self::MissingPipe(_)
            | Self::Protocol(_)
            | Self::Exit(_)
            | Self::Timeout => None,
        }
    }
}

pub fn probe_sidecar_readiness(
    config: &SidecarLaunchConfig,
    stderr_log_file: impl Into<PathBuf>,
    client_version: impl Into<String>,
) -> Result<SidecarReadinessReport, SidecarReadinessError> {
    config
        .verify_expected_sha256()
        .map_err(SidecarReadinessError::Sha256)?;
    let stderr_log_file = stderr_log_file.into();
    let mut child = config.spawn().map_err(SidecarReadinessError::Spawn)?;
    let result = probe_sidecar_readiness_with_child(
        &mut child,
        stderr_log_file.clone(),
        initialize_probe_request(client_version),
        initialized_notification(),
    );
    cleanup_sidecar_child(&mut child);
    if result.is_err() {
        let _ = drain_child_stderr(&mut child, &stderr_log_file)
            .map_err(SidecarReadinessError::StderrLog)?;
    }
    result
}

fn probe_sidecar_readiness_with_child(
    child: &mut Child,
    stderr_log_file: PathBuf,
    initialize: app_server_protocol::JsonRpcMessage,
    initialized: app_server_protocol::JsonRpcMessage,
) -> Result<SidecarReadinessReport, SidecarReadinessError> {
    let pid = child.id();
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or(SidecarReadinessError::MissingPipe("stdin"))?;
        write_jsonrpc_line(stdin, &initialize).map_err(SidecarReadinessError::Io)?;
    }

    let probe_info = {
        let stdout = child
            .stdout
            .as_mut()
            .ok_or(SidecarReadinessError::MissingPipe("stdout"))?;
        let mut stdout = BufReader::new(stdout);
        let mut line = String::new();
        stdout
            .read_line(&mut line)
            .map_err(SidecarReadinessError::Io)?;
        if line.is_empty() {
            return Err(SidecarReadinessError::Protocol(
                "app-server closed stdout before initialize response".to_string(),
            ));
        }
        let message = app_server_transport::decode_message(&line)
            .map_err(|error| SidecarReadinessError::Protocol(error.to_string()))?;
        probe_info_from_initialize_response(&message)
            .map_err(SidecarReadinessError::Protocol)?
            .ok_or_else(|| {
                SidecarReadinessError::Protocol(
                    "app-server did not return daemon initialize response".to_string(),
                )
            })?
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or(SidecarReadinessError::MissingPipe("stdin"))?;
        write_jsonrpc_line(stdin, &initialized).map_err(SidecarReadinessError::Io)?;
    }
    drop(child.stdin.take());

    wait_for_sidecar_exit(child, Duration::from_secs(2))?;
    let stderr_bytes =
        drain_child_stderr(child, &stderr_log_file).map_err(SidecarReadinessError::StderrLog)?;

    Ok(SidecarReadinessReport {
        pid,
        probe_info,
        stderr_log_file,
        stderr_bytes,
    })
}

fn write_jsonrpc_line(
    writer: &mut impl Write,
    message: &app_server_protocol::JsonRpcMessage,
) -> io::Result<()> {
    let line = app_server_transport::encode_message(message)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(line.as_bytes())?;
    writer.flush()
}

fn drain_child_stderr(child: &mut Child, stderr_log_file: &PathBuf) -> io::Result<usize> {
    let Some(stderr) = child.stderr.as_mut() else {
        return Ok(0);
    };
    let mut buffer = Vec::new();
    stderr.read_to_end(&mut buffer)?;
    if buffer.is_empty() {
        return Ok(0);
    }
    if let Some(parent) = stderr_log_file.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(stderr_log_file, &buffer)?;
    Ok(buffer.len())
}

fn wait_for_sidecar_exit(
    child: &mut Child,
    timeout: Duration,
) -> Result<(), SidecarReadinessError> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait().map_err(SidecarReadinessError::Io)? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(SidecarReadinessError::Exit(status)),
            None if Instant::now() >= deadline => return Err(SidecarReadinessError::Timeout),
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

fn cleanup_sidecar_child(child: &mut Child) {
    if child.try_wait().ok().flatten().is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppServerReleaseManifest {
    pub version: String,
    pub protocol_version: String,
    pub artifacts: Vec<AppServerReleaseArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppServerReleaseArtifact {
    pub platform: String,
    pub url: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windows_sandbox_setup_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windows_sandbox_runner_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSidecarLaunchConfig {
    pub config: SidecarLaunchConfig,
    pub artifact: AppServerReleaseArtifact,
    pub binary_path_source: SidecarBinaryPathSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarBinaryPathSource {
    Env,
    Resources,
    Dev,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarBinaryPathResolution {
    pub binary_path: PathBuf,
    pub source: SidecarBinaryPathSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarBinaryPathOptions {
    pub env_var_name: String,
    pub env_binary_path: Option<PathBuf>,
    pub allow_env_override: bool,
    pub resources_path: Option<PathBuf>,
    pub resource_relative_path: Option<PathBuf>,
    pub dev_binary_path: Option<PathBuf>,
    pub backend_mode: SidecarBackendMode,
    pub backend_command: Option<String>,
    pub backend_args: Vec<String>,
    pub backend_timeout_ms: Option<u64>,
    pub app_policy_path: Option<PathBuf>,
    pub platform: String,
    pub arch: String,
}

impl Default for SidecarBinaryPathOptions {
    fn default() -> Self {
        let env_var_name = DEFAULT_SIDECAR_ENV_VAR.to_string();
        Self {
            env_binary_path: env::var_os(&env_var_name).map(PathBuf::from),
            env_var_name,
            allow_env_override: true,
            resources_path: None,
            resource_relative_path: None,
            dev_binary_path: None,
            backend_mode: SidecarBackendMode::Unavailable,
            backend_command: None,
            backend_args: Vec::new(),
            backend_timeout_ms: None,
            app_policy_path: None,
            platform: env::consts::OS.to_string(),
            arch: env::consts::ARCH.to_string(),
        }
    }
}

impl SidecarBinaryPathOptions {
    pub fn apply_daemon_settings(&mut self, settings: &DaemonSettings) -> Result<(), String> {
        self.allow_env_override = settings.allow_env_override;
        self.resource_relative_path = settings.resource_relative_path.clone();
        if let Some(backend_mode) = settings.backend_mode.as_deref() {
            self.backend_mode = SidecarBackendMode::parse(backend_mode.trim())?;
        }
        self.backend_command = settings.backend_command.clone();
        self.backend_args = settings.backend_args.clone();
        self.backend_timeout_ms = settings.backend_timeout_ms;
        self.app_policy_path = settings.app_policy_path.clone();
        Ok(())
    }

    pub fn with_daemon_settings(mut self, settings: &DaemonSettings) -> Result<Self, String> {
        self.apply_daemon_settings(settings)?;
        Ok(self)
    }

    pub fn with_daemon_state_paths(self, state_paths: &DaemonStatePaths) -> Result<Self, String> {
        let settings = DaemonSettings::load(&state_paths.settings_file)?;
        self.with_daemon_settings(&settings)
    }
}

pub fn sidecar_binary_name_for_platform(platform: &str) -> &'static str {
    if platform == "windows" || platform == "win32" {
        "app-server.exe"
    } else {
        "app-server"
    }
}

pub fn platform_key(platform: &str, arch: &str) -> String {
    match (platform, arch) {
        ("macos" | "darwin", "aarch64" | "arm64") => "darwin-arm64".to_string(),
        ("macos" | "darwin", _) => "darwin-x64".to_string(),
        ("windows" | "win32", _) => "win32-x64".to_string(),
        ("linux", _) => "linux-x64".to_string(),
        _ => format!("{platform}-{arch}"),
    }
}

pub fn find_release_artifact<'a>(
    manifest: &'a AppServerReleaseManifest,
    platform: &str,
) -> Option<&'a AppServerReleaseArtifact> {
    manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.platform == platform)
}

pub fn default_packaged_sidecar_relative_path(platform: &str, arch: &str) -> PathBuf {
    PathBuf::from("app-server")
        .join(platform_key(platform, arch))
        .join(sidecar_binary_name_for_platform(platform))
}

pub fn resolve_sidecar_binary_path(
    options: &SidecarBinaryPathOptions,
) -> Option<SidecarBinaryPathResolution> {
    if options.allow_env_override {
        if let Some(binary_path) = non_empty_path(options.env_binary_path.as_ref()) {
            return Some(SidecarBinaryPathResolution {
                binary_path: binary_path.to_path_buf(),
                source: SidecarBinaryPathSource::Env,
            });
        }
    }

    if let Some(resources_path) = non_empty_path(options.resources_path.as_ref()) {
        let relative_path = options.resource_relative_path.clone().unwrap_or_else(|| {
            default_packaged_sidecar_relative_path(&options.platform, &options.arch)
        });
        return Some(SidecarBinaryPathResolution {
            binary_path: resources_path.join(relative_path),
            source: SidecarBinaryPathSource::Resources,
        });
    }

    non_empty_path(options.dev_binary_path.as_ref()).map(|binary_path| {
        SidecarBinaryPathResolution {
            binary_path: binary_path.to_path_buf(),
            source: SidecarBinaryPathSource::Dev,
        }
    })
}

pub fn resolve_sidecar_from_release_manifest(
    manifest: &AppServerReleaseManifest,
    options: &SidecarBinaryPathOptions,
    listen_url: Option<String>,
) -> Result<Option<ResolvedSidecarLaunchConfig>, String> {
    if manifest.protocol_version != PROTOCOL_VERSION {
        return Err(format!(
            "unsupported app-server protocol: expected {}, got {}",
            PROTOCOL_VERSION, manifest.protocol_version
        ));
    }

    let platform = platform_key(&options.platform, &options.arch);
    let Some(artifact) = find_release_artifact(manifest, &platform) else {
        return Ok(None);
    };
    let Some(binary_path) = resolve_sidecar_binary_path(options) else {
        return Ok(None);
    };

    let expected_sha256 = match binary_path.source {
        SidecarBinaryPathSource::Resources => Some(artifact.sha256.clone()),
        SidecarBinaryPathSource::Env | SidecarBinaryPathSource::Dev => None,
    };

    Ok(Some(ResolvedSidecarLaunchConfig {
        config: SidecarLaunchConfig {
            binary_path: binary_path.binary_path,
            listen_url: listen_url.unwrap_or_else(|| DEFAULT_LISTEN_URL.to_string()),
            expected_sha256,
            backend_mode: options.backend_mode,
            backend_command: options.backend_command.clone(),
            backend_args: options.backend_args.clone(),
            backend_timeout_ms: options.backend_timeout_ms,
            app_policy_path: options.app_policy_path.clone(),
        },
        artifact: artifact.clone(),
        binary_path_source: binary_path.source,
    }))
}

pub fn parse_release_manifest(content: &str) -> Result<AppServerReleaseManifest, String> {
    serde_json::from_str(content)
        .map_err(|error| format!("failed to parse app-server release manifest: {error}"))
}

pub fn read_release_manifest(path: impl Into<PathBuf>) -> Result<AppServerReleaseManifest, String> {
    let path = path.into();
    let content = fs::read_to_string(&path).map_err(|error| {
        format!(
            "failed to read app-server release manifest {}: {}",
            path.display(),
            error
        )
    })?;
    parse_release_manifest(&content)
}

pub fn resolve_sidecar_from_release_manifest_path(
    manifest_path: impl Into<PathBuf>,
    options: &SidecarBinaryPathOptions,
    listen_url: Option<String>,
) -> Result<Option<ResolvedSidecarLaunchConfig>, String> {
    let manifest = read_release_manifest(manifest_path)?;
    resolve_sidecar_from_release_manifest(&manifest, options, listen_url)
}

pub fn resolve_sidecar_from_release_manifest_path_with_daemon_state(
    manifest_path: impl Into<PathBuf>,
    state_paths: &DaemonStatePaths,
    options: SidecarBinaryPathOptions,
    listen_url: Option<String>,
) -> Result<Option<ResolvedSidecarLaunchConfig>, String> {
    let options = options.with_daemon_state_paths(state_paths)?;
    resolve_sidecar_from_release_manifest_path(manifest_path, &options, listen_url)
}

fn non_empty_path(path: Option<&PathBuf>) -> Option<&PathBuf> {
    path.filter(|value| !value.as_os_str().to_string_lossy().trim().is_empty())
}

pub fn sha256_hex(content: impl AsRef<[u8]>) -> String {
    use sha2::Digest;

    let mut hasher = sha2::Sha256::new();
    hasher.update(content.as_ref());
    hex::encode(hasher.finalize())
}

pub fn sha256_file(path: impl Into<PathBuf>) -> std::io::Result<String> {
    fs::read(path.into()).map(sha256_hex)
}

pub fn assert_sha256(actual_sha256: &str, expected_sha256: &str) -> Result<(), String> {
    if actual_sha256
        .trim()
        .eq_ignore_ascii_case(expected_sha256.trim())
    {
        Ok(())
    } else {
        Err("app-server sha256 mismatch".to_string())
    }
}

pub fn verify_sidecar_file_sha256(config: &SidecarLaunchConfig) -> Result<(), String> {
    let Some(expected_sha256) = config
        .expected_sha256
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let actual_sha256 = sha256_file(config.binary_path.clone()).map_err(|error| {
        format!(
            "failed to read app-server sidecar for sha256 verification: {}",
            error
        )
    })?;
    assert_sha256(&actual_sha256, expected_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn stdio_sidecar_uses_codex_style_stdio_flag() {
        let config = SidecarLaunchConfig::stdio(PathBuf::from(SIDECAR_BINARY_NAME));

        assert_eq!(
            config.args(),
            vec![
                "--stdio".to_string(),
                "--backend".to_string(),
                "unavailable".to_string()
            ]
        );
    }

    #[test]
    fn sidecar_command_uses_launch_args() {
        let mut config = SidecarLaunchConfig::stdio("app-server");
        config.listen_url = "local://test".to_string();
        let command = config.command();

        assert_eq!(command.get_program().to_string_lossy(), "app-server");
        assert_eq!(
            command
                .get_args()
                .map(|arg| arg.to_string_lossy().to_string())
                .collect::<Vec<_>>(),
            vec![
                "--listen".to_string(),
                "local://test".to_string(),
                "--backend".to_string(),
                "unavailable".to_string()
            ]
        );
    }

    #[test]
    fn sidecar_args_follow_standalone_backend_cli() {
        let mut config = SidecarLaunchConfig::stdio("app-server");
        assert_eq!(
            config.args(),
            vec![
                "--stdio".to_string(),
                "--backend".to_string(),
                "unavailable".to_string()
            ]
        );

        config.backend_mode = SidecarBackendMode::Mock;
        config.app_policy_path = Some(PathBuf::from("/tmp/content-studio.policy.json"));
        assert_eq!(
            config.args(),
            vec![
                "--stdio".to_string(),
                "--backend".to_string(),
                "mock".to_string(),
                "--app-policy".to_string(),
                "/tmp/content-studio.policy.json".to_string()
            ]
        );

        config.backend_mode = SidecarBackendMode::External;
        config.backend_command = Some("/usr/local/bin/content-backend".to_string());
        config.backend_args = vec!["--workspace".to_string(), "/tmp/content".to_string()];
        config.backend_timeout_ms = Some(30_000);
        assert_eq!(
            config.args(),
            vec![
                "--stdio".to_string(),
                "--backend".to_string(),
                "external".to_string(),
                "--backend-command".to_string(),
                "/usr/local/bin/content-backend".to_string(),
                "--backend-arg".to_string(),
                "--workspace".to_string(),
                "--backend-arg".to_string(),
                "/tmp/content".to_string(),
                "--backend-timeout-ms".to_string(),
                "30000".to_string(),
                "--app-policy".to_string(),
                "/tmp/content-studio.policy.json".to_string()
            ]
        );
    }

    #[test]
    fn sha256_verification_is_required_only_for_non_empty_expected_hash() {
        let mut config = SidecarLaunchConfig::stdio(PathBuf::from("app-server"));
        assert!(!config.requires_sha256_verification());

        config.expected_sha256 = Some("  ".to_string());
        assert!(!config.requires_sha256_verification());

        config.expected_sha256 = Some("abc".to_string());
        assert!(config.requires_sha256_verification());
    }

    #[test]
    fn verifies_sidecar_file_sha256_when_expected_hash_is_present() {
        let path = temp_sidecar_path("sha256-ok");
        fs::write(&path, b"sidecar-binary").expect("write sidecar");
        let expected_sha256 = sha256_hex(b"sidecar-binary");
        let mut config = SidecarLaunchConfig::stdio(path.clone());
        config.expected_sha256 = Some(expected_sha256.to_uppercase());

        assert!(config.verify_expected_sha256().is_ok());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_sidecar_file_sha256_mismatch() {
        let path = temp_sidecar_path("sha256-bad");
        fs::write(&path, b"sidecar-binary").expect("write sidecar");
        let mut config = SidecarLaunchConfig::stdio(path.clone());
        config.expected_sha256 = Some("bad".to_string());

        let error = config
            .verify_expected_sha256()
            .expect_err("sha mismatch should fail");
        assert!(error.contains("sha256 mismatch"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn spawn_verified_rejects_sha256_mismatch_before_process_start() {
        let path = temp_sidecar_path("sha256-spawn-bad");
        fs::write(&path, b"sidecar-binary").expect("write sidecar");
        let mut config = SidecarLaunchConfig::stdio(path.clone());
        config.expected_sha256 = Some("bad".to_string());

        let error = config
            .spawn_verified()
            .expect_err("sha mismatch should stop spawn");

        assert!(matches!(error, SidecarProcessError::Sha256(_)));
        assert!(error.to_string().contains("sha256 mismatch"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn readiness_report_records_probe_info_and_stderr_target() {
        let report = SidecarReadinessReport {
            pid: 42,
            probe_info: ProbeInfo {
                app_server_version: "1.59.0".to_string(),
                protocol_version: PROTOCOL_VERSION.to_string(),
            },
            stderr_log_file: PathBuf::from("/state/app-server.stderr.log"),
            stderr_bytes: 0,
        };

        assert_eq!(report.pid, 42);
        assert_eq!(report.probe_info.protocol_version, PROTOCOL_VERSION);
        assert_eq!(
            report.stderr_log_file,
            PathBuf::from("/state/app-server.stderr.log")
        );
    }

    #[test]
    fn readiness_probe_rejects_sha256_mismatch_before_process_start() {
        let path = temp_sidecar_path("readiness-sha256-bad");
        fs::write(&path, b"sidecar-binary").expect("write sidecar");
        let mut config = SidecarLaunchConfig::stdio(path.clone());
        config.expected_sha256 = Some("bad".to_string());

        let error = config
            .probe_readiness(temp_sidecar_path("readiness-stderr"), "1.59.0")
            .expect_err("sha mismatch should stop readiness probe");

        assert!(matches!(error, SidecarReadinessError::Sha256(_)));
        assert!(error.to_string().contains("sha256 mismatch"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn readiness_probe_smokes_real_app_server_when_env_is_set() {
        let Some(binary_path) = env::var_os("APP_SERVER_DAEMON_TEST_APP_SERVER_BIN") else {
            return;
        };
        let stderr_log_file = temp_sidecar_path("readiness-stderr");
        let config = SidecarLaunchConfig::stdio(PathBuf::from(binary_path));

        let report = config
            .probe_readiness(stderr_log_file.clone(), "1.59.0")
            .expect("readiness probe");

        assert!(report.pid > 0);
        assert_eq!(report.probe_info.protocol_version, PROTOCOL_VERSION);
        assert_eq!(report.stderr_log_file, stderr_log_file);
        if report.stderr_bytes > 0 {
            assert!(report.stderr_log_file.exists());
        }
        let _ = fs::remove_file(stderr_log_file);
    }

    #[test]
    fn readiness_probe_drains_stderr_when_sidecar_exits_before_initialize_response() {
        let Some(binary_path) = env::var_os("APP_SERVER_DAEMON_TEST_APP_SERVER_BIN") else {
            return;
        };
        let stderr_log_file = temp_sidecar_path("readiness-failed-stderr");
        let mut config = SidecarLaunchConfig::stdio(PathBuf::from(binary_path));
        config.backend_mode = SidecarBackendMode::External;

        let error = config
            .probe_readiness(stderr_log_file.clone(), "1.59.0")
            .expect_err("external backend without command should fail readiness");

        assert!(matches!(
            error,
            SidecarReadinessError::Protocol(_) | SidecarReadinessError::Io(_)
        ));
        let stderr = fs::read_to_string(&stderr_log_file).expect("stderr log");
        assert!(
            stderr.contains("--backend-command is required when --backend external"),
            "unexpected stderr: {stderr}"
        );
        let _ = fs::remove_file(stderr_log_file);
    }

    #[test]
    fn resolves_packaged_sidecar_relative_paths() {
        assert_eq!(
            default_packaged_sidecar_relative_path("darwin", "arm64"),
            PathBuf::from("app-server")
                .join("darwin-arm64")
                .join("app-server")
        );
        assert_eq!(
            default_packaged_sidecar_relative_path("windows", "x86_64"),
            PathBuf::from("app-server")
                .join("win32-x64")
                .join("app-server.exe")
        );
    }

    #[test]
    fn resolves_sidecar_binary_path_priority() {
        let options = SidecarBinaryPathOptions {
            env_binary_path: Some(PathBuf::from("/custom/app-server")),
            resources_path: Some(PathBuf::from("/app/resources")),
            dev_binary_path: Some(PathBuf::from("/dev/app-server")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };
        assert_eq!(
            resolve_sidecar_binary_path(&options),
            Some(SidecarBinaryPathResolution {
                binary_path: PathBuf::from("/custom/app-server"),
                source: SidecarBinaryPathSource::Env,
            })
        );

        let options = SidecarBinaryPathOptions {
            allow_env_override: false,
            ..options
        };
        assert_eq!(
            resolve_sidecar_binary_path(&options),
            Some(SidecarBinaryPathResolution {
                binary_path: PathBuf::from("/app/resources")
                    .join("app-server")
                    .join("darwin-arm64")
                    .join("app-server"),
                source: SidecarBinaryPathSource::Resources,
            })
        );

        let options = SidecarBinaryPathOptions {
            env_binary_path: None,
            resources_path: None,
            dev_binary_path: Some(PathBuf::from("/dev/app-server")),
            ..SidecarBinaryPathOptions::default()
        };
        assert_eq!(
            resolve_sidecar_binary_path(&options),
            Some(SidecarBinaryPathResolution {
                binary_path: PathBuf::from("/dev/app-server"),
                source: SidecarBinaryPathSource::Dev,
            })
        );

        let options = SidecarBinaryPathOptions {
            env_binary_path: Some(PathBuf::from("  ")),
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };
        assert_eq!(
            resolve_sidecar_binary_path(&options),
            Some(SidecarBinaryPathResolution {
                binary_path: PathBuf::from("/app/resources")
                    .join("app-server")
                    .join("darwin-arm64")
                    .join("app-server"),
                source: SidecarBinaryPathSource::Resources,
            })
        );
    }

    #[test]
    fn resolves_sidecar_from_release_manifest() {
        let manifest = AppServerReleaseManifest {
            version: "1.58.0".to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            artifacts: vec![AppServerReleaseArtifact {
                platform: "darwin-arm64".to_string(),
                url: "https://example/app-server-darwin-arm64.tar.gz".to_string(),
                sha256: "abc".to_string(),
                windows_sandbox_setup_sha256: None,
                windows_sandbox_runner_sha256: None,
            }],
        };
        let options = SidecarBinaryPathOptions {
            env_binary_path: None,
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };

        assert_eq!(
            resolve_sidecar_from_release_manifest(&manifest, &options, None).expect("manifest"),
            Some(ResolvedSidecarLaunchConfig {
                artifact: manifest.artifacts[0].clone(),
                binary_path_source: SidecarBinaryPathSource::Resources,
                config: SidecarLaunchConfig {
                    binary_path: PathBuf::from("/app/resources")
                        .join("app-server")
                        .join("darwin-arm64")
                        .join("app-server"),
                    listen_url: DEFAULT_LISTEN_URL.to_string(),
                    expected_sha256: Some("abc".to_string()),
                    backend_mode: SidecarBackendMode::Unavailable,
                    backend_command: None,
                    backend_args: Vec::new(),
                    backend_timeout_ms: None,
                    app_policy_path: None,
                },
            })
        );

        let options = SidecarBinaryPathOptions {
            env_binary_path: Some(PathBuf::from("/dev/app-server")),
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };
        let resolved =
            resolve_sidecar_from_release_manifest(&manifest, &options, None).expect("manifest");
        assert_eq!(
            resolved.expect("resolved").config,
            SidecarLaunchConfig {
                binary_path: PathBuf::from("/dev/app-server"),
                listen_url: DEFAULT_LISTEN_URL.to_string(),
                expected_sha256: None,
                backend_mode: SidecarBackendMode::Unavailable,
                backend_command: None,
                backend_args: Vec::new(),
                backend_timeout_ms: None,
                app_policy_path: None,
            }
        );
    }

    #[test]
    fn release_manifest_resolution_preserves_backend_launch_options() {
        let manifest = AppServerReleaseManifest {
            version: "1.58.0".to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            artifacts: vec![AppServerReleaseArtifact {
                platform: "darwin-arm64".to_string(),
                url: "https://example/app-server-darwin-arm64.tar.gz".to_string(),
                sha256: "abc".to_string(),
                windows_sandbox_setup_sha256: None,
                windows_sandbox_runner_sha256: None,
            }],
        };
        let options = SidecarBinaryPathOptions {
            env_binary_path: None,
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            backend_mode: SidecarBackendMode::External,
            backend_command: Some("/app/content-backend".to_string()),
            backend_args: vec!["--workspace".to_string(), "/app/workspace".to_string()],
            backend_timeout_ms: Some(45_000),
            app_policy_path: Some(PathBuf::from("/app/content-studio.policy.json")),
            ..SidecarBinaryPathOptions::default()
        };

        let resolved =
            resolve_sidecar_from_release_manifest(&manifest, &options, None).expect("manifest");
        let config = resolved.expect("resolved").config;

        assert_eq!(config.backend_mode, SidecarBackendMode::External);
        assert_eq!(
            config.backend_command.as_deref(),
            Some("/app/content-backend")
        );
        assert_eq!(
            config.backend_args,
            vec!["--workspace".to_string(), "/app/workspace".to_string()]
        );
        assert_eq!(config.backend_timeout_ms, Some(45_000));
        assert_eq!(
            config.app_policy_path,
            Some(PathBuf::from("/app/content-studio.policy.json"))
        );
    }

    #[test]
    fn daemon_settings_apply_to_sidecar_binary_options_and_manifest_resolution() {
        let manifest = AppServerReleaseManifest {
            version: "1.58.0".to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            artifacts: vec![AppServerReleaseArtifact {
                platform: "darwin-arm64".to_string(),
                url: "https://example/app-server-darwin-arm64.tar.gz".to_string(),
                sha256: "abc".to_string(),
                windows_sandbox_setup_sha256: None,
                windows_sandbox_runner_sha256: None,
            }],
        };
        let settings = DaemonSettings {
            allow_env_override: false,
            resource_relative_path: Some(
                PathBuf::from("custom")
                    .join("app-server")
                    .join("darwin-arm64")
                    .join("app-server"),
            ),
            backend_mode: Some("external".to_string()),
            backend_command: Some("node".to_string()),
            backend_args: vec![
                "resources/app-server/backend/content-backend.mjs".to_string(),
                "--workspace".to_string(),
                "/app/workspace".to_string(),
            ],
            backend_timeout_ms: Some(45_000),
            app_policy_path: Some(PathBuf::from("/app/content-studio.policy.json")),
        };
        let options = SidecarBinaryPathOptions {
            env_binary_path: Some(PathBuf::from("/dev/app-server")),
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        }
        .with_daemon_settings(&settings)
        .expect("settings");

        let resolved =
            resolve_sidecar_from_release_manifest(&manifest, &options, None).expect("manifest");
        let config = resolved.expect("resolved").config;

        assert_eq!(
            config.binary_path,
            PathBuf::from("/app/resources")
                .join("custom")
                .join("app-server")
                .join("darwin-arm64")
                .join("app-server")
        );
        assert_eq!(config.expected_sha256, Some("abc".to_string()));
        assert_eq!(config.backend_mode, SidecarBackendMode::External);
        assert_eq!(config.backend_command.as_deref(), Some("node"));
        assert_eq!(
            config.backend_args,
            vec![
                "resources/app-server/backend/content-backend.mjs".to_string(),
                "--workspace".to_string(),
                "/app/workspace".to_string(),
            ]
        );
        assert_eq!(config.backend_timeout_ms, Some(45_000));
        assert_eq!(
            config.app_policy_path,
            Some(PathBuf::from("/app/content-studio.policy.json"))
        );
    }

    #[test]
    fn with_daemon_state_paths_loads_settings_file_into_manifest_resolution() {
        let temp_root = temp_sidecar_path("daemon-state-settings");
        let state_paths = DaemonStatePaths::new(&temp_root);
        let manifest_path = temp_root.join("app-server.release.json");
        fs::create_dir_all(&temp_root).expect("create temp root");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
                    "version": "1.58.0",
                    "protocolVersion": "{}",
                    "artifacts": [
                        {{
                            "platform": "darwin-arm64",
                            "url": "https://example/app-server-darwin-arm64.tar.gz",
                            "sha256": "abc"
                        }}
                    ]
                }}"#,
                PROTOCOL_VERSION
            ),
        )
        .expect("write manifest");
        DaemonSettings {
            allow_env_override: false,
            resource_relative_path: Some(
                PathBuf::from("custom")
                    .join("app-server")
                    .join("darwin-arm64")
                    .join("app-server"),
            ),
            backend_mode: Some("external".to_string()),
            backend_command: Some("node".to_string()),
            backend_args: vec![
                "resources/app-server/backend/content-backend.mjs".to_string(),
                "--workspace".to_string(),
                "/app/workspace".to_string(),
            ],
            backend_timeout_ms: Some(45_000),
            app_policy_path: Some(PathBuf::from("/app/content-studio.policy.json")),
        }
        .save(&state_paths.settings_file)
        .expect("save settings");
        let options = SidecarBinaryPathOptions {
            env_binary_path: Some(PathBuf::from("/dev/app-server")),
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };

        let resolved = resolve_sidecar_from_release_manifest_path_with_daemon_state(
            &manifest_path,
            &state_paths,
            options,
            None,
        )
        .expect("manifest")
        .expect("resolved");
        let config = resolved.config;

        assert_eq!(
            config.binary_path,
            PathBuf::from("/app/resources")
                .join("custom")
                .join("app-server")
                .join("darwin-arm64")
                .join("app-server")
        );
        assert_eq!(config.expected_sha256, Some("abc".to_string()));
        assert_eq!(config.backend_mode, SidecarBackendMode::External);
        assert_eq!(config.backend_command.as_deref(), Some("node"));
        assert_eq!(
            config.backend_args,
            vec![
                "resources/app-server/backend/content-backend.mjs".to_string(),
                "--workspace".to_string(),
                "/app/workspace".to_string(),
            ]
        );
        assert_eq!(config.backend_timeout_ms, Some(45_000));
        assert_eq!(
            config.app_policy_path,
            Some(PathBuf::from("/app/content-studio.policy.json"))
        );
        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn with_daemon_state_paths_missing_settings_uses_default_resolution() {
        let temp_root = temp_sidecar_path("daemon-state-missing-settings");
        let state_paths = DaemonStatePaths::new(&temp_root);
        let manifest_path = temp_root.join("app-server.release.json");
        fs::create_dir_all(&temp_root).expect("create temp root");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
                    "version": "1.58.0",
                    "protocolVersion": "{}",
                    "artifacts": [
                        {{
                            "platform": "darwin-arm64",
                            "url": "https://example/app-server-darwin-arm64.tar.gz",
                            "sha256": "abc"
                        }}
                    ]
                }}"#,
                PROTOCOL_VERSION
            ),
        )
        .expect("write manifest");
        let options = SidecarBinaryPathOptions {
            env_binary_path: None,
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };

        let resolved = resolve_sidecar_from_release_manifest_path_with_daemon_state(
            &manifest_path,
            &state_paths,
            options,
            None,
        )
        .expect("missing settings should use defaults")
        .expect("resolved");

        assert_eq!(
            resolved.config.binary_path,
            PathBuf::from("/app/resources")
                .join("app-server")
                .join("darwin-arm64")
                .join("app-server")
        );
        assert_eq!(resolved.config.expected_sha256, Some("abc".to_string()));
        assert_eq!(
            resolved.config.backend_mode,
            SidecarBackendMode::Unavailable
        );
        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn daemon_settings_reject_unsupported_backend_mode_before_launch_resolution() {
        let settings = DaemonSettings {
            backend_mode: Some("agent".to_string()),
            ..DaemonSettings::default()
        };
        let error = SidecarBinaryPathOptions::default()
            .with_daemon_settings(&settings)
            .expect_err("unsupported backend");

        assert!(error.contains("unsupported app-server backend mode: agent"));
    }

    #[test]
    fn resolves_sidecar_from_release_manifest_file() {
        let manifest_path = temp_sidecar_path("manifest-json");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
                    "version": "1.58.0",
                    "protocolVersion": "{}",
                    "artifacts": [
                        {{
                            "platform": "darwin-arm64",
                            "url": "https://example/app-server-darwin-arm64.tar.gz",
                            "sha256": "abc"
                        }}
                    ]
                }}"#,
                PROTOCOL_VERSION
            ),
        )
        .expect("write manifest");
        let options = SidecarBinaryPathOptions {
            env_binary_path: None,
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "darwin".to_string(),
            arch: "arm64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };

        let resolved = resolve_sidecar_from_release_manifest_path(&manifest_path, &options, None)
            .expect("manifest")
            .expect("resolved");

        assert_eq!(
            resolved.binary_path_source,
            SidecarBinaryPathSource::Resources
        );
        assert_eq!(
            resolved.config.binary_path,
            PathBuf::from("/app/resources")
                .join("app-server")
                .join("darwin-arm64")
                .join("app-server")
        );
        assert_eq!(resolved.config.expected_sha256, Some("abc".to_string()));
        let _ = fs::remove_file(manifest_path);
    }

    #[test]
    fn release_manifest_rejects_protocol_drift_and_missing_artifact() {
        let manifest = AppServerReleaseManifest {
            version: "1.58.0".to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            artifacts: Vec::new(),
        };
        let options = SidecarBinaryPathOptions {
            resources_path: Some(PathBuf::from("/app/resources")),
            platform: "linux".to_string(),
            arch: "x64".to_string(),
            ..SidecarBinaryPathOptions::default()
        };

        assert_eq!(
            resolve_sidecar_from_release_manifest(&manifest, &options, None).expect("manifest"),
            None
        );
        let error = resolve_sidecar_from_release_manifest(
            &AppServerReleaseManifest {
                protocol_version: "appserver.v9".to_string(),
                ..manifest
            },
            &options,
            None,
        )
        .expect_err("protocol drift should fail");
        assert!(error.contains("unsupported app-server protocol"));
    }

    fn temp_sidecar_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        env::temp_dir().join(format!("app-server-daemon-{name}-{nanos}"))
    }
}
