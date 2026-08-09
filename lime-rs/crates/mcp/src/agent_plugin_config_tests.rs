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
fn expands_placeholders_once_and_keeps_args_env_opaque() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("${PLUGIN_DATA}");
    let data = temp.path().join("data");
    std::fs::create_dir_all(&root).unwrap();
    let root = std::fs::canonicalize(&root).unwrap();
    let data = std::fs::canonicalize(&data).unwrap_or_else(|_| {
        std::fs::create_dir_all(&data).unwrap();
        std::fs::canonicalize(&data).unwrap()
    });
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "demo": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["${PLUGIN_ROOT}:${PLUGIN_DATA}", "${PLUGIN_ROOT}/../opaque"],
                    "env": {"OPAQUE": "${PLUGIN_DATA}/../opaque"}
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let config = outcome.servers.get("demo").unwrap();
    let root = root.to_string_lossy();
    let data = data.to_string_lossy();
    assert_eq!(
        config.args(),
        &[format!("{root}:{data}"), format!("{root}/../opaque"),]
    );
    assert_eq!(
        config.env().get("OPAQUE"),
        Some(&format!("{data}/../opaque"))
    );
}

#[test]
fn lowers_http_headers_and_rejects_invalid_http_edges() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "valid": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {
                        "Authorization": "package-value",
                        "HOST": "package-host",
                        "X-Plugin": "demo",
                        "X-Plugin-Name": "cafe"
                    }
                },
                "loopback": {
                    "type": "streamable-http",
                    "url": "http://127.0.0.1:8787/mcp"
                },
                "client-owned-only": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {
                        "Authorization": "package-value",
                        "Host": "package-host"
                    }
                },
                "userinfo": {
                    "type": "streamable-http",
                    "url": "https://user:secret@example.com/mcp"
                },
                "fragment": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp#fragment"
                },
                "insecure": {
                    "type": "streamable-http",
                    "url": "http://example.com/mcp"
                },
                "duplicate-header": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X-Demo": "one", "x-demo": "two"}
                },
                "invalid-header-name": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"Bad Name": "value"}
                },
                "invalid-header-value": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X-Demo": "line\nvalue"}
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    assert!(outcome.servers.contains_key("valid"));
    assert!(outcome.servers.contains_key("loopback"));
    assert_eq!(outcome.servers.len(), 3);
    for name in [
        "userinfo",
        "fragment",
        "insecure",
        "duplicate-header",
        "invalid-header-name",
        "invalid-header-value",
    ] {
        assert!(
            outcome.errors.iter().any(|error| error.name == name),
            "missing parser error for {name}"
        );
    }

    let McpServerTransport::StreamableHttp {
        url, http_headers, ..
    } = &outcome.servers["valid"].transport
    else {
        panic!("expected streamable HTTP transport");
    };
    assert_eq!(url, "https://example.com/mcp");
    let headers = http_headers.as_ref().unwrap();
    assert_eq!(headers.get("X-Plugin"), Some(&"demo".to_string()));
    assert_eq!(headers.get("X-Plugin-Name"), Some(&"cafe".to_string()));
    assert!(!headers
        .keys()
        .any(|name| name.eq_ignore_ascii_case("authorization")));
    assert!(!headers.keys().any(|name| name.eq_ignore_ascii_case("host")));

    let McpServerTransport::StreamableHttp { http_headers, .. } =
        &outcome.servers["client-owned-only"].transport
    else {
        panic!("expected streamable HTTP transport");
    };
    assert!(http_headers.is_none());
}

#[test]
fn rejects_explicit_null_and_unknown_server_shapes_without_disabling_siblings() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "good": {"type": "stdio", "command": "python"},
                "null-cwd": {"type": "stdio", "command": "python", "cwd": null},
                "null-headers": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": null
                },
                "unknown-type": {"type": "websocket", "url": "https://example.com/mcp"},
                "unknown-field": {"type": "stdio", "command": "python", "future": true}
            }
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(outcome.servers.keys().collect::<Vec<_>>(), vec!["good"]);
    assert_eq!(outcome.errors.len(), 4);
    for name in ["null-cwd", "null-headers", "unknown-type", "unknown-field"] {
        assert!(outcome.errors.iter().any(|error| error.name == name));
    }
}

#[test]
fn enforces_contained_portable_paths() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "valid": {"type": "stdio", "command": "python"},
                "contained": {"type": "stdio", "command": "./bin/../server"},
                "redundant": {"type": "stdio", "command": ".//bin/server"},
                "command-escape": {"type": "stdio", "command": "./../server"},
                "cwd-escape": {
                    "type": "stdio",
                    "command": "python",
                    "cwd": "${PLUGIN_ROOT}/../outside"
                },
                "backslash": {
                    "type": "stdio",
                    "command": "./scripts\\..\\outside"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(
        outcome.servers.keys().collect::<Vec<_>>(),
        vec!["contained", "redundant", "valid"]
    );
    assert_eq!(outcome.errors.len(), 3);
    for name in ["command-escape", "cwd-escape", "backslash"] {
        assert!(outcome.errors.iter().any(|error| error.name == name));
    }
}

#[cfg(unix)]
#[test]
fn rejects_missing_descendant_below_escaping_symlink() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().unwrap();
    let root = temp.path().join("plugin");
    let data = temp.path().join("data");
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    symlink(&outside, root.join("link")).unwrap();

    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "escape": {"type": "stdio", "command": "./link/missing"}
            }
        })
        .to_string(),
    )
    .unwrap();

    assert!(outcome.servers.is_empty());
    assert_eq!(outcome.errors.len(), 1);
    assert_eq!(outcome.errors[0].name, "escape");
    assert!(outcome.errors[0].message.contains("超出允许目录"));
}

#[cfg(windows)]
#[test]
fn rejects_reserved_environment_aliases_case_insensitively() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "reserved": {
                    "type": "stdio",
                    "command": "python",
                    "env": {"plugin_root": "bad"}
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    assert!(outcome.servers.is_empty());
    assert_eq!(outcome.errors.len(), 1);
}

#[cfg(windows)]
#[test]
fn normalizes_windows_environment_keys_and_rejects_case_duplicates() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "configured": {
                    "type": "stdio",
                    "command": "python",
                    "env": {"Path": "configured"}
                },
                "duplicate": {
                    "type": "stdio",
                    "command": "python",
                    "env": {"PATH": "one", "Path": "two"}
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(
        outcome.servers.keys().collect::<Vec<_>>(),
        vec!["configured"]
    );
    assert_eq!(outcome.errors.len(), 1);
    assert_eq!(outcome.errors[0].name, "duplicate");
    assert_eq!(
        outcome.servers["configured"].env().get("PATH"),
        Some(&"configured".to_string())
    );
    assert!(!outcome.servers["configured"].env().contains_key("Path"));
}

#[cfg(windows)]
#[test]
fn rejects_drive_relative_windows_command() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "drive-relative": {"type": "stdio", "command": "C:server.exe"}
            }
        })
        .to_string(),
    )
    .unwrap();

    assert!(outcome.servers.is_empty());
    assert_eq!(outcome.errors.len(), 1);
}

#[cfg(windows)]
#[test]
fn rejects_absolute_unc_and_extended_windows_commands() {
    let (_temp, root, data) = roots();
    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "absolute": {"type": "stdio", "command": r"C:\server.exe"},
                "drive-relative": {"type": "stdio", "command": "C:server.exe"},
                "unc": {"type": "stdio", "command": r"\\server\share\server.exe"},
                "extended": {"type": "stdio", "command": r"\\?\C:\server.exe"}
            }
        })
        .to_string(),
    )
    .unwrap();

    assert!(outcome.servers.is_empty());
    assert_eq!(outcome.errors.len(), 4);
}

#[cfg(windows)]
#[test]
fn preserves_windows_plugin_data_and_emits_host_paths() {
    let (_temp, root, data) = roots();
    let contents = serde_json::json!({
        "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
        "mcpServers": {
            "demo": {
                "type": "stdio",
                "command": "python",
                "args": ["${PLUGIN_ROOT}", "${PLUGIN_DATA}"],
                "cwd": "${PLUGIN_DATA}/state"
            }
        }
    })
    .to_string();

    let first = parse_agent_plugin_mcp_config(&root, &data, &contents).unwrap();
    std::fs::create_dir_all(data.join("state")).unwrap();
    std::fs::write(data.join("state").join("marker.txt"), "persisted").unwrap();
    let second = parse_agent_plugin_mcp_config(&root, &data, &contents).unwrap();

    let config = &second.servers["demo"];
    let expected_root = host_path_string(&std::fs::canonicalize(&root).unwrap());
    let expected_data = host_path_string(&std::fs::canonicalize(&data).unwrap());
    let expected_cwd = host_path_string(&data.join("state"));
    assert_eq!(
        config.args(),
        &[expected_root.clone(), expected_data.clone()]
    );
    assert_eq!(config.env().get(PLUGIN_ROOT_VARIABLE), Some(&expected_root));
    assert_eq!(config.env().get(PLUGIN_DATA_VARIABLE), Some(&expected_data));
    assert_eq!(
        std::fs::read_to_string(data.join("state").join("marker.txt")).unwrap(),
        "persisted"
    );
    assert_eq!(first.servers.len(), second.servers.len());
    let McpServerTransport::Stdio { cwd, .. } = &config.transport else {
        panic!("expected stdio transport");
    };
    assert_eq!(cwd.as_deref(), Some(expected_cwd.as_str()));
}

#[cfg(windows)]
#[test]
fn rejects_missing_descendant_below_escaping_junction() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("plugin");
    let data = temp.path().join("data");
    let outside = temp.path().join("outside");
    let junction = root.join("link");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("server.exe"), "fixture").unwrap();
    let output = std::process::Command::new("cmd.exe")
        .args(["/C", "mklink", "/J"])
        .arg(&junction)
        .arg(&outside)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "create junction failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let outcome = parse_agent_plugin_mcp_config(
        &root,
        &data,
        &serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_URI,
            "mcpServers": {
                "existing-escape": {"type": "stdio", "command": "./link/server.exe"},
                "missing-escape": {"type": "stdio", "command": "./link/missing"}
            }
        })
        .to_string(),
    )
    .unwrap();

    assert!(outcome.servers.is_empty());
    assert_eq!(outcome.errors.len(), 2);
    for name in ["existing-escape", "missing-escape"] {
        let error = outcome
            .errors
            .iter()
            .find(|error| error.name == name)
            .unwrap();
        assert!(error.message.contains("超出允许目录"));
    }
}
