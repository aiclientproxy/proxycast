use crate::types::{McpServerConfig, DEFAULT_MCP_SERVER_ENVIRONMENT_ID};
use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::OsStr;
use std::ffi::OsString;
#[cfg(unix)]
use std::path::Path;

/// MCP 服务器环境的运行时注册表。
///
/// 当前只有本机执行器。远程执行器接入前，显式引用未知环境必须拒绝，
/// 不能把配置身份降级成本机进程。
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct McpEnvironmentRegistry {
    available: HashMap<String, McpEnvironment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum McpEnvironment {
    Local,
}

impl Default for McpEnvironmentRegistry {
    fn default() -> Self {
        Self::local()
    }
}

impl McpEnvironmentRegistry {
    pub(crate) fn local() -> Self {
        Self {
            available: HashMap::from([(
                DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
                McpEnvironment::Local,
            )]),
        }
    }

    pub(crate) fn resolve_server_environment(
        &self,
        server_name: &str,
        config: &McpServerConfig,
    ) -> Result<McpEnvironment, String> {
        let environment_id = config.environment_id.trim();
        if environment_id.is_empty() {
            return Err(format!(
                "MCP server `{server_name}` requires a non-empty environment id"
            ));
        }
        if let Some(environment) = self.available.get(environment_id) {
            return Ok(*environment);
        }
        Err(format!(
            "MCP server `{server_name}` references unknown environment id `{environment_id}`"
        ))
    }
}

/// 构造本地 MCP 子进程的最小环境，避免把宿主凭证和无关变量隐式传入。
pub(crate) fn process_environment(
    overrides: &HashMap<String, String>,
) -> HashMap<OsString, OsString> {
    let mut environment = HashMap::new();
    for key in DEFAULT_MCP_ENV_VARS {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(OsString::from(key), value);
        }
    }
    #[cfg(unix)]
    if !overrides.contains_key("PATH") {
        augment_unix_path(&mut environment);
    }
    for (key, value) in overrides {
        environment.insert(OsString::from(key), OsString::from(value));
    }
    environment
}

#[cfg(unix)]
fn augment_unix_path(environment: &mut HashMap<OsString, OsString>) {
    let current_path = environment
        .get(OsStr::new("PATH"))
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    let home = environment
        .get(OsStr::new("HOME"))
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "/Users/unknown".to_string());
    let extra_paths = [
        format!("{home}/.nvm/versions/node/*/bin"),
        format!("{home}/.local/bin"),
        format!("{home}/.cargo/bin"),
        format!("{home}/Library/pnpm"),
        format!("{home}/.bun/bin"),
        "/usr/local/bin".to_string(),
        "/opt/homebrew/bin".to_string(),
        "/opt/homebrew/sbin".to_string(),
    ];
    let mut resolved_paths = Vec::new();
    for path in extra_paths {
        if path.contains('*') {
            if let Ok(entries) = glob::glob(&path) {
                let mut matches: Vec<String> = entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.to_string_lossy().into_owned())
                    .collect();
                matches.sort();
                if let Some(path) = matches.last() {
                    resolved_paths.push(path.clone());
                }
            }
        } else if Path::new(&path).exists() {
            resolved_paths.push(path);
        }
    }
    if !resolved_paths.is_empty() {
        let merged = if current_path.is_empty() {
            resolved_paths.join(":")
        } else {
            format!("{}:{current_path}", resolved_paths.join(":"))
        };
        environment.insert(OsString::from("PATH"), OsString::from(merged));
    }
}

#[cfg(unix)]
const DEFAULT_MCP_ENV_VARS: &[&str] = &[
    "HOME",
    "LOGNAME",
    "PATH",
    "SHELL",
    "USER",
    "__CF_USER_TEXT_ENCODING",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "TZ",
];

#[cfg(windows)]
const DEFAULT_MCP_ENV_VARS: &[&str] = &[
    "PATH",
    "PATHEXT",
    "SHELL",
    "COMSPEC",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "USERNAME",
    "USERDOMAIN",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PROGRAMDATA",
    "LOCALAPPDATA",
    "APPDATA",
    "TEMP",
    "TMP",
    "TMPDIR",
    "POWERSHELL",
    "PWSH",
];

#[cfg(not(any(unix, windows)))]
const DEFAULT_MCP_ENV_VARS: &[&str] = &[];

#[cfg(test)]
mod tests {
    use super::{process_environment, McpEnvironment, McpEnvironmentRegistry};
    use crate::types::{McpServerConfig, McpServerTransport};
    use std::collections::HashMap;
    use std::ffi::{OsStr, OsString};

    #[test]
    fn process_environment_applies_explicit_overrides() {
        let environment = process_environment(&HashMap::from([(
            "LIME_MCP_EXPLICIT_ENV".to_string(),
            "value".to_string(),
        )]));

        assert_eq!(
            environment
                .get(OsStr::new("LIME_MCP_EXPLICIT_ENV"))
                .map(OsString::as_os_str),
            Some(OsStr::new("value"))
        );
    }

    #[test]
    fn process_environment_preserves_explicit_path() {
        let environment = process_environment(&HashMap::from([(
            "PATH".to_string(),
            "/explicit/path".to_string(),
        )]));

        assert_eq!(
            environment.get(OsStr::new("PATH")).map(OsString::as_os_str),
            Some(OsStr::new("/explicit/path"))
        );
    }

    #[test]
    fn registry_resolves_local_to_typed_placement() {
        let registry = McpEnvironmentRegistry::local();
        let config = McpServerConfig {
            transport: McpServerTransport::Stdio {
                command: "server".to_string(),
                args: Vec::new(),
                env: HashMap::new(),
                cwd: None,
            },
            environment_id: "local".to_string(),
            enabled: true,
            startup_timeout: 30,
            tool_timeout: None,
            enabled_tools: None,
            disabled_tools: Vec::new(),
            required: false,
            supports_parallel_tool_calls: false,
            scopes: None,
            oauth: None,
            oauth_resource: None,
        };

        assert_eq!(
            registry
                .resolve_server_environment("server", &config)
                .expect("local environment"),
            McpEnvironment::Local
        );
    }
}
