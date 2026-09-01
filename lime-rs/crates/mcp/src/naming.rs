const MCP_RUNTIME_TOOL_PREFIX: &str = "mcp__";

/// Codex-compatible MCP server name grammar.
///
/// Package-style names such as `npm:@modelcontextprotocol/server-sequential.thinking`
/// are valid, while whitespace and path separators outside the package grammar are not.
pub fn is_valid_server_name(server_name: &str) -> bool {
    let name = server_name.trim();
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | ':' | '@' | '/' | '.'))
}

pub fn validate_server_name(server_name: &str) -> Result<&str, String> {
    let name = server_name.trim();
    if is_valid_server_name(name) {
        Ok(name)
    } else {
        Err(format!(
            "invalid MCP server name `{server_name}` (use letters, numbers, '-', '_', ':', '@', '/', '.')"
        ))
    }
}

/// Returns a TOML table key suitable for recovery hints.
pub fn config_key_for_server_name(server_name: &str) -> String {
    if server_name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    {
        server_name.to_string()
    } else {
        serde_json::Value::String(server_name.to_string()).to_string()
    }
}

pub fn build_runtime_tool_name(server_name: &str, tool_name: &str) -> String {
    format!("{MCP_RUNTIME_TOOL_PREFIX}{server_name}__{tool_name}")
}

pub fn extract_runtime_inner_tool_name<'a>(
    server_name: &str,
    tool_name: &'a str,
) -> Option<&'a str> {
    let prefix = build_runtime_tool_name(server_name, "");
    tool_name
        .strip_prefix(prefix.as_str())
        .filter(|inner_name| !inner_name.is_empty())
}

pub fn parse_runtime_tool_name<'a>(
    tool_name: &str,
    server_names: impl Iterator<Item = &'a String>,
) -> Option<(String, String)> {
    server_names
        .filter_map(|server_name| {
            extract_runtime_inner_tool_name(server_name, tool_name)
                .map(|inner_name| (server_name.clone(), inner_name.to_string()))
        })
        .max_by_key(|(server_name, _)| server_name.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_package_style_server_names() {
        assert!(is_valid_server_name(
            "npm:@modelcontextprotocol/server-sequential.thinking"
        ));
        assert!(!is_valid_server_name("docs server"));
        assert!(!is_valid_server_name("docs\0server"));
    }

    #[test]
    fn quotes_non_bare_config_keys() {
        assert_eq!(config_key_for_server_name("docs"), "docs");
        assert_eq!(
            config_key_for_server_name("npm:@scope/server.tools"),
            "\"npm:@scope/server.tools\""
        );
    }

    #[test]
    fn parse_uses_longest_server_name_match() {
        let servers = ["docs".to_string(), "docs__admin".to_string()];
        let parsed = parse_runtime_tool_name("mcp__docs__admin__search", servers.iter()).unwrap();

        assert_eq!(parsed, ("docs__admin".to_string(), "search".to_string()));
    }

    #[test]
    fn extract_rejects_empty_inner_tool_name() {
        assert_eq!(extract_runtime_inner_tool_name("docs", "mcp__docs__"), None);
    }
}
