//! Agent Plugins v1 `mcp.json` parser and runtime lowering.
//!
//! The portable package format is deliberately parsed separately from the
//! internal `McpServerConfig` shape. This keeps client-only runtime fields out
//! of the package contract and makes path/security checks explicit.

use crate::types::{McpServerConfig, McpServerTransport, DEFAULT_MCP_SERVER_ENVIRONMENT_ID};
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use url::Host;

pub const AGENT_PLUGIN_MCP_SCHEMA_URI: &str =
    "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json";
const PLUGIN_ROOT_VARIABLE: &str = "PLUGIN_ROOT";
const PLUGIN_DATA_VARIABLE: &str = "PLUGIN_DATA";
const CLIENT_OWNED_HTTP_HEADERS: &[&str] = &[
    "accept",
    "authorization",
    "connection",
    "content-encoding",
    "content-length",
    "content-type",
    "host",
    "last-event-id",
    "mcp-protocol-version",
    "mcp-session-id",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "user-agent",
];

#[derive(Debug, Clone)]
pub struct AgentPluginMcpServerParseError {
    pub name: String,
    pub message: String,
}

#[cfg(test)]
#[path = "agent_plugin_config_tests.rs"]
mod agent_plugin_config_tests;

#[derive(Debug, Default)]
pub struct AgentPluginMcpConfigParseOutcome {
    pub servers: BTreeMap<String, McpServerConfig>,
    pub errors: Vec<AgentPluginMcpServerParseError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AgentPluginMcpFile {
    #[serde(rename = "$schema")]
    schema: String,
    mcp_servers: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum AgentPluginMcpServer {
    #[serde(rename = "stdio")]
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        cwd: Option<String>,
    },
    #[serde(rename = "streamable-http")]
    StreamableHttp {
        url: String,
        headers: Option<BTreeMap<String, String>>,
    },
    #[serde(rename = "sse")]
    Sse {
        #[serde(rename = "url")]
        _url: String,
        #[serde(rename = "headers")]
        _headers: Option<BTreeMap<String, String>>,
    },
}

/// Parse a standard Agent Plugins v1 `mcp.json` and lower valid servers into
/// Lime's internal MCP configuration. A malformed file rejects the component;
/// a malformed sibling server is isolated in `errors`.
pub fn parse_agent_plugin_mcp_config(
    plugin_root: &Path,
    plugin_data_root: &Path,
    contents: &str,
) -> Result<AgentPluginMcpConfigParseOutcome, String> {
    let AgentPluginMcpFile {
        schema,
        mcp_servers,
    } = serde_json::from_str(contents)
        .map_err(|error| format!("解析 Agent Plugins MCP 配置失败: {error}"))?;
    if schema != AGENT_PLUGIN_MCP_SCHEMA_URI {
        return Err(format!(
            "不支持的 Agent Plugins MCP schema `{schema}`，仅支持 `{AGENT_PLUGIN_MCP_SCHEMA_URI}`。"
        ));
    }

    let root = absolute_plugin_path(plugin_root)?;
    let data = absolute_plugin_path(plugin_data_root)?;

    let mut outcome = AgentPluginMcpConfigParseOutcome::default();
    for (name, value) in mcp_servers {
        match normalize_server(value, &root, &data) {
            Ok(config) => {
                outcome.servers.insert(name, config);
            }
            Err(message) => outcome
                .errors
                .push(AgentPluginMcpServerParseError { name, message }),
        }
    }
    Ok(outcome)
}

fn normalize_server(
    value: JsonValue,
    plugin_root: &Path,
    plugin_data_root: &Path,
) -> Result<McpServerConfig, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Agent Plugins MCP server 必须是 object。".to_string())?;
    match object.get("type").and_then(JsonValue::as_str) {
        Some("stdio") => reject_explicit_null(object, "cwd")?,
        Some("streamable-http" | "sse") => reject_explicit_null(object, "headers")?,
        _ => {}
    }
    let server: AgentPluginMcpServer = serde_json::from_value(value)
        .map_err(|error| format!("MCP server schema 无效: {error}"))?;
    match server {
        AgentPluginMcpServer::Stdio {
            command,
            args,
            env,
            cwd,
        } => normalize_stdio(command, args, env, cwd, plugin_root, plugin_data_root),
        AgentPluginMcpServer::StreamableHttp { url, headers } => normalize_http(url, headers),
        AgentPluginMcpServer::Sse { .. } => {
            Err("Agent Plugins legacy SSE transport 不受支持。".to_string())
        }
    }
}

fn normalize_stdio(
    mut command: String,
    mut args: Vec<String>,
    mut env: BTreeMap<String, String>,
    cwd: Option<String>,
    plugin_root: &Path,
    plugin_data_root: &Path,
) -> Result<McpServerConfig, String> {
    #[cfg(windows)]
    let has_windows_prefix = matches!(
        Path::new(&command).components().next(),
        Some(std::path::Component::Prefix(_))
    );
    #[cfg(not(windows))]
    let has_windows_prefix = false;
    let bare = !command.is_empty()
        && !command.contains('/')
        && !command.contains('\\')
        && !has_windows_prefix;
    let relative = command.starts_with("./") && is_portable_relative_path(&command);
    if !bare && !relative {
        return Err(
            "Agent Plugins stdio command 必须是裸 executable 或包内 `./` 路径。".to_string(),
        );
    }
    for reserved in [PLUGIN_ROOT_VARIABLE, PLUGIN_DATA_VARIABLE] {
        if env
            .keys()
            .any(|name| environment_variable_names_match(name, reserved))
        {
            return Err(format!(
                "Agent Plugins stdio env 不得覆盖保留变量 `{reserved}`。"
            ));
        }
    }
    #[cfg(windows)]
    {
        let mut normalized = BTreeMap::new();
        for (name, value) in env {
            let name = name.to_ascii_uppercase();
            if normalized.insert(name.clone(), value).is_some() {
                return Err(format!("重复的大小写不敏感环境变量 `{name}`。"));
            }
        }
        env = normalized;
    }

    let root = host_path_string(plugin_root);
    let data = host_path_string(plugin_data_root);
    if relative {
        command = host_path_string(&resolve_contained_host_path(
            &command,
            plugin_root,
            plugin_root,
        )?);
    }
    for value in &mut args {
        *value = expand_placeholders(value, &root, &data);
    }
    for value in env.values_mut() {
        *value = expand_placeholders(value, &root, &data);
    }
    let configured_cwd = cwd.as_deref().unwrap_or("${PLUGIN_ROOT}");
    let cwd_root = parse_cwd_root(configured_cwd).ok_or_else(|| {
        "Agent Plugins stdio cwd 必须是包内 `./`、`${PLUGIN_ROOT}` 或 `${PLUGIN_DATA}` 路径。"
            .to_string()
    })?;
    let expanded_cwd = expand_placeholders(configured_cwd, &root, &data);
    let cwd_base = match cwd_root {
        CwdRoot::Package => plugin_root,
        CwdRoot::Data => plugin_data_root,
    };
    let cwd = host_path_string(&resolve_contained_host_path(
        &expanded_cwd,
        cwd_base,
        cwd_base,
    )?);
    env.insert(PLUGIN_ROOT_VARIABLE.to_string(), root);
    env.insert(PLUGIN_DATA_VARIABLE.to_string(), data);

    Ok(McpServerConfig {
        transport: McpServerTransport::Stdio {
            command,
            args,
            env: env.into_iter().collect::<HashMap<_, _>>(),
            cwd: Some(cwd),
        },
        environment_id: DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
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
    })
}

fn normalize_http(
    url: String,
    mut headers: Option<BTreeMap<String, String>>,
) -> Result<McpServerConfig, String> {
    validate_url(&url)?;
    if let Some(configured) = headers.as_mut() {
        validate_headers(configured)?;
        configured.retain(|name, _| {
            !CLIENT_OWNED_HTTP_HEADERS
                .iter()
                .any(|owned| name.eq_ignore_ascii_case(owned))
        });
    }
    Ok(McpServerConfig {
        transport: McpServerTransport::StreamableHttp {
            url,
            bearer_token_env_var: None,
            http_headers: headers
                .filter(|values| !values.is_empty())
                .map(|values| values.into_iter().collect()),
            env_http_headers: None,
        },
        environment_id: DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
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
    })
}

fn reject_explicit_null(object: &JsonMap<String, JsonValue>, field: &str) -> Result<(), String> {
    if object.get(field).is_some_and(JsonValue::is_null) {
        return Err(format!("Agent Plugins MCP `{field}` 必须使用声明的类型。"));
    }
    Ok(())
}

fn environment_variable_names_match(left: &str, right: &str) -> bool {
    if cfg!(windows) {
        left.eq_ignore_ascii_case(right)
    } else {
        left == right
    }
}

fn validate_url(raw: &str) -> Result<(), String> {
    if raw.is_empty() {
        return Err("Agent Plugins HTTP server requires a non-empty url。".to_string());
    }
    let parsed = url::Url::parse(raw).map_err(|error| format!("MCP URL 无效 `{raw}`: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err("Agent Plugins MCP URL 必须是绝对 HTTP/HTTPS 地址。".to_string());
    }
    if !parsed.username().is_empty() || parsed.password().is_some() || parsed.fragment().is_some() {
        return Err("Agent Plugins MCP URL 不得包含 userinfo 或 fragment。".to_string());
    }
    let loopback = match parsed.host() {
        Some(Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(address)) => address.is_loopback(),
        Some(Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    };
    if parsed.scheme() == "http" && !loopback {
        return Err("非 loopback Agent Plugins MCP endpoint 必须使用 HTTPS。".to_string());
    }
    Ok(())
}

fn validate_headers(headers: &BTreeMap<String, String>) -> Result<(), String> {
    let mut seen = HashSet::new();
    for (name, value) in headers {
        if !seen.insert(name.to_ascii_lowercase()) {
            return Err(format!("重复的 HTTP header `{name}`。"));
        }
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
        {
            return Err(format!("HTTP header 名称无效 `{name}`。"));
        }
        if value
            .bytes()
            .any(|byte| (byte < 32 && byte != b'\t') || byte == 127)
        {
            return Err(format!("HTTP header `{name}` 的值无效。"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CwdRoot {
    Package,
    Data,
}

fn parse_cwd_root(value: &str) -> Option<CwdRoot> {
    if value == "./"
        || value
            .strip_prefix("./")
            .is_some_and(is_portable_path_suffix)
    {
        return Some(CwdRoot::Package);
    }
    for (placeholder, root) in [
        ("${PLUGIN_ROOT}", CwdRoot::Package),
        ("${PLUGIN_DATA}", CwdRoot::Data),
    ] {
        if value == placeholder
            || value
                .strip_prefix(&format!("{placeholder}/"))
                .is_some_and(|suffix| suffix.is_empty() || is_portable_path_suffix(suffix))
        {
            return Some(root);
        }
    }
    None
}

fn expand_placeholders(value: &str, plugin_root: &str, plugin_data: &str) -> String {
    const ROOT: &str = "${PLUGIN_ROOT}";
    const DATA: &str = "${PLUGIN_DATA}";
    let mut output = String::with_capacity(value.len());
    let mut remaining = value;
    loop {
        let next = match (remaining.find(ROOT), remaining.find(DATA)) {
            (Some(root), Some(data)) if root <= data => Some((root, ROOT, plugin_root)),
            (Some(_), Some(data)) => Some((data, DATA, plugin_data)),
            (Some(root), None) => Some((root, ROOT, plugin_root)),
            (None, Some(data)) => Some((data, DATA, plugin_data)),
            (None, None) => None,
        };
        let Some((index, placeholder, replacement)) = next else {
            output.push_str(remaining);
            break;
        };
        output.push_str(&remaining[..index]);
        output.push_str(replacement);
        remaining = &remaining[index + placeholder.len()..];
    }
    output
}

fn absolute_plugin_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|error| format!("解析 Plugin 路径失败: {error}"))?
    };
    resolve_existing_path_prefix(&absolute)
}

fn resolve_contained_host_path(
    value: &str,
    root: &Path,
    allowed_root: &Path,
) -> Result<PathBuf, String> {
    let value_path = Path::new(value);
    let path = if value_path.is_absolute() {
        value_path.to_path_buf()
    } else {
        root.join(value_path)
    };
    let path = resolve_existing_path_prefix(&path)?;
    if !path.starts_with(allowed_root) {
        return Err(format!(
            "展开路径 `{}` 超出允许目录 `{}`。",
            value,
            allowed_root.display()
        ));
    }
    Ok(path)
}

fn resolve_existing_path_prefix(path: &Path) -> Result<PathBuf, String> {
    let mut existing = path.to_path_buf();
    let mut missing = Vec::<OsString>::new();
    loop {
        match std::fs::canonicalize(&existing) {
            Ok(mut resolved) => {
                for component in missing.iter().rev() {
                    resolved.push(component);
                }
                return Ok(lexical_normalize(&resolved));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if std::fs::symlink_metadata(&existing)
                    .is_ok_and(|metadata| metadata.file_type().is_symlink())
                {
                    return Err(format!(
                        "不允许通过 symlink 解析路径 `{}`。",
                        path.display()
                    ));
                }
                let Some(component) = existing.components().next_back() else {
                    return Err(format!("解析路径失败 `{}`: {error}", path.display()));
                };
                if matches!(
                    component,
                    std::path::Component::Prefix(_) | std::path::Component::RootDir
                ) {
                    return Err(format!("解析路径失败 `{}`: {error}", path.display()));
                }
                let component = component.as_os_str().to_os_string();
                if !existing.pop() {
                    return Err(format!("解析路径失败 `{}`: {error}", path.display()));
                }
                missing.push(component);
            }
            Err(error) => return Err(format!("解析路径失败 `{}`: {error}", path.display())),
        }
    }
}

fn host_path_string(path: &Path) -> String {
    let rendered = path.to_string_lossy();
    #[cfg(windows)]
    if let Some(path) = rendered.strip_prefix(r"\\?\") {
        return path
            .strip_prefix(r"UNC\")
            .map(|path| format!(r"\\{path}"))
            .unwrap_or_else(|| path.to_string());
    }
    rendered.into_owned()
}

fn is_portable_relative_path(value: &str) -> bool {
    value
        .strip_prefix("./")
        .is_some_and(is_portable_path_suffix)
}

fn is_portable_path_suffix(value: &str) -> bool {
    !value.is_empty() && !value.contains('\\')
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn roots() -> (TempDir, PathBuf, PathBuf) {
        let temp = TempDir::new().unwrap();
        let plugin_root = temp.path().join("plugin");
        let data_root = temp.path().join("data");
        std::fs::create_dir_all(&plugin_root).unwrap();
        (temp, plugin_root, data_root)
    }

    #[test]
    fn lowers_stdio_placeholders_and_injects_reserved_paths() {
        let (_temp, root, data) = roots();
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        let outcome = parse_agent_plugin_mcp_config(
            &root,
            &data,
            &format!(
                r#"{{"$schema":"{AGENT_PLUGIN_MCP_SCHEMA_URI}","mcpServers":{{"demo":{{"type":"stdio","command":"./bin/server","args":["${{PLUGIN_ROOT}}/x","${{PLUGIN_DATA}}/state"],"env":{{"VALUE":"${{PLUGIN_DATA}}/cache"}},"cwd":"${{PLUGIN_ROOT}}/bin"}}}}}}"#
            ),
        )
        .unwrap();
        let config = outcome.servers.get("demo").unwrap();
        assert!(outcome.errors.is_empty());
        assert_eq!(
            config.env().get("PLUGIN_ROOT"),
            Some(
                &std::fs::canonicalize(&root)
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            )
        );
        assert_eq!(
            config.env().get("PLUGIN_DATA"),
            Some(
                &std::fs::canonicalize(&data)
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            )
        );
        assert!(config.command().ends_with("/bin/server"));
        assert!(config.sanitized_cwd().unwrap().ends_with("plugin/bin"));
    }

    #[test]
    fn isolates_bad_siblings_and_rejects_reserved_env_and_http_insecure() {
        let (_temp, root, data) = roots();
        let outcome = parse_agent_plugin_mcp_config(
            &root,
            &data,
            &format!(
                r#"{{"$schema":"{AGENT_PLUGIN_MCP_SCHEMA_URI}","mcpServers":{{"good":{{"type":"stdio","command":"python"}},"reserved":{{"type":"stdio","command":"python","env":{{"PLUGIN_ROOT":"bad"}}}},"remote":{{"type":"streamable-http","url":"http://example.com"}}}}}}"#
            ),
        )
        .unwrap();
        assert!(outcome.servers.contains_key("good"));
        assert_eq!(outcome.errors.len(), 2);
    }

    #[test]
    fn rejects_legacy_schema_and_cwd_escape() {
        let (_temp, root, data) = roots();
        let error = parse_agent_plugin_mcp_config(
            &root,
            &data,
            r#"{"$schema":"https://agent-plugins.org/schemas/2.0.0/mcp.schema.json","mcpServers":{}}"#,
        )
        .unwrap_err();
        assert!(error.contains("不支持"));

        let outcome = parse_agent_plugin_mcp_config(
            &root,
            &data,
            &format!(
                r#"{{"$schema":"{AGENT_PLUGIN_MCP_SCHEMA_URI}","mcpServers":{{"escape":{{"type":"stdio","command":"python","cwd":"${{PLUGIN_ROOT}}/../outside"}}}}}}"#
            ),
        )
        .unwrap();
        assert!(outcome.servers.is_empty());
        assert_eq!(outcome.errors.len(), 1);
    }
}
