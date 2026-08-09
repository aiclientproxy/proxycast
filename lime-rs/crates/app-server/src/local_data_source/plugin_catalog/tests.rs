use super::*;
use tempfile::TempDir;

fn fixture(root: &Path) {
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(root.join("skills/example")).unwrap();
    fs::write(
        root.join("skills/example/SKILL.md"),
        "---\nname: example\n---\n",
    )
    .unwrap();
    fs::write(
        root.join("plugin.json"),
        r#"{
          "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name": "example-plugin",
          "version": "1.2.3",
          "description": "Example"
        }"#,
    )
    .unwrap();
}

fn install_params(source: &Path) -> PluginCatalogInstallParams {
    PluginCatalogInstallParams {
        source_path: source.to_string_lossy().into_owned(),
        marketplace_id: Some("test-marketplace".to_string()),
        source: Some("repo".to_string()),
        expected_digest: None,
    }
}

#[test]
fn installs_standard_manifest_into_v3_store() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    let data_root = temp.path().join("data");
    let response = install(&data_root, install_params(&source)).unwrap();
    assert_eq!(response.plugin.id, "example-plugin");
    assert_eq!(response.plugin.marketplace_id, "test-marketplace");
    assert!(response.plugin.content_digest.starts_with("sha256:"));
    let detail = read(
        &data_root,
        PluginCatalogReadParams {
            plugin_id: "example-plugin".to_string(),
        },
    )
    .unwrap();
    assert_eq!(detail.plugin.skills.len(), 1);
    let snapshots = enabled_plugin_turn_snapshots(&data_root).unwrap();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].id, "example-plugin");
    assert_eq!(snapshots[0].config_name, "example-plugin@test-marketplace");
    assert_eq!(snapshots[0].display_name, "example-plugin");
    assert_eq!(snapshots[0].skill_names, vec!["example"]);
    assert_eq!(
        installed(&data_root, PluginCatalogInstalledParams {})
            .unwrap()
            .plugins
            .len(),
        1
    );
    let disabled = set_enabled(
        &data_root,
        app_server_protocol::protocol::v2::PluginCatalogEnabledSetParams {
            plugin_id: "example-plugin".to_string(),
            enabled: false,
        },
    )
    .unwrap();
    assert!(!disabled.plugin.enabled);
    assert!(enabled_plugin_turn_snapshots(&data_root)
        .unwrap()
        .is_empty());
    assert!(
        uninstall(
            &data_root,
            PluginCatalogUninstallParams {
                plugin_id: "example-plugin".to_string()
            }
        )
        .unwrap()
        .uninstalled
    );
}

#[test]
fn repeated_install_is_idempotent_but_same_version_digest_conflict_fails() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    let data_root = temp.path().join("data");
    let first = install(&data_root, install_params(&source)).unwrap();
    let second = install(&data_root, install_params(&source)).unwrap();
    assert_eq!(first.plugin.content_digest, second.plugin.content_digest);

    fs::write(source.join("skills/example/SKILL.md"), "changed").unwrap();
    let error = install(&data_root, install_params(&source)).unwrap_err();
    assert!(error.contains("content digest 不同"));
}

#[test]
fn expected_digest_and_optional_version_are_valid() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    let data_root = temp.path().join("data");
    let mut params = install_params(&source);
    params.expected_digest = Some("sha256:wrong".to_string());
    assert!(install(&data_root, params)
        .unwrap_err()
        .contains("digest 不一致"));

    fs::write(
        source.join("plugin.json"),
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"example-plugin","version":"1.2"}"#,
    )
    .unwrap();
    assert_eq!(
        manifest_version(&read_manifest(&source).unwrap()).unwrap(),
        "1.2"
    );
}

#[test]
fn rejects_manifest_without_standard_schema_and_parent_paths() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("plugin.json"),
        r#"{"name":"example-plugin","version":"1.0.0","schemaVersion":"lime.plugin.package.v1"}"#,
    )
    .unwrap();
    assert!(read_manifest(&root).is_err());
    assert!(resource_path(&root, "../outside").is_err());
}

#[test]
fn requires_root_standard_manifest_and_direct_child_skills() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(root.join(".codex-plugin")).unwrap();
    fs::write(
        root.join(".codex-plugin/plugin.json"),
        r#"{"name":"legacy","version":"1.0.0"}"#,
    )
    .unwrap();
    assert!(discover_package_roots(&root).unwrap().is_empty());

    fixture(&root);
    fs::create_dir_all(root.join("skills/group/nested")).unwrap();
    fs::write(root.join("skills/group/nested/SKILL.md"), "nested").unwrap();
    let capabilities = skill_capabilities(&root).unwrap();
    assert_eq!(capabilities.len(), 1);
    assert_eq!(capabilities[0].id, "example");
}

#[test]
fn manifest_matches_codex_agent_plugin_validation_and_extension_precedence() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(root.join(".codex-plugin")).unwrap();
    fs::write(
        root.join(".codex-plugin/plugin.json"),
        r#"{"interface":{"displayName":"Legacy Codex"}}"#,
    )
    .unwrap();
    fs::write(
        root.join("plugin.json"),
        r#"{
          "$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name":"acme.tools",
          "futureField":true,
          "extensions":{
            "com.openai":{"interface":{"displayName":"Inline Codex"}},
            "com.example.unimplemented":{"future":true}
          }
        }"#,
    )
    .unwrap();

    let manifest = read_manifest(&root).unwrap();
    assert_eq!(manifest_name(&manifest).unwrap(), "acme.tools");
    assert!(manifest.get("futureField").is_none());
    assert_eq!(
        interface_string(&manifest, "displayName").as_deref(),
        Some("Inline Codex")
    );

    fs::write(
        root.join("plugin.json"),
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"acme.tools","extensions":false}"#,
    )
    .unwrap();
    let manifest = read_manifest(&root).unwrap();
    assert_eq!(
        interface_string(&manifest, "displayName").as_deref(),
        Some("Legacy Codex")
    );
}

#[test]
fn codex_apps_extension_uses_config_path_and_isolates_invalid_config() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(root.join(".codex-plugin")).unwrap();
    fs::write(
        root.join("plugin.json"),
        r#"{
          "$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name":"acme.tools",
          "extensions":{
            "com.openai":{"apps":"./apps.json"}
          }
        }"#,
    )
    .unwrap();
    fs::write(
        root.join(".codex-plugin/plugin.json"),
        r#"{"apps":"./legacy-apps.json"}"#,
    )
    .unwrap();
    fs::write(
        root.join("apps.json"),
        r#"{"apps":{"Calendar":{"id":"calendar","category":"productivity"}}}"#,
    )
    .unwrap();
    fs::write(
        root.join("legacy-apps.json"),
        r#"{"apps":{"Legacy":{"id":"legacy"}}}"#,
    )
    .unwrap();

    let manifest = read_manifest(&root).unwrap();
    let detail = build_capability_detail(&root, &manifest).unwrap();
    assert_eq!(detail.apps.len(), 1);
    assert_eq!(detail.apps[0].id, "calendar");
    assert_eq!(detail.apps[0].name, "Calendar");

    fs::write(root.join("apps.json"), r#"{"apps":{"Broken":{}}}"#).unwrap();
    let manifest = read_manifest(&root).unwrap();
    assert!(build_capability_detail(&root, &manifest)
        .unwrap()
        .apps
        .is_empty());

    fs::write(
        root.join("plugin.json"),
        r#"{
          "$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name":"acme.tools",
          "extensions":{"com.openai":{"apps":"apps.json"}}
        }"#,
    )
    .unwrap();
    let manifest = read_manifest(&root).unwrap();
    assert!(build_capability_detail(&root, &manifest)
        .unwrap()
        .apps
        .is_empty());

    fs::write(
        root.join("plugin.json"),
        r#"{
          "$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name":"acme.tools",
          "extensions":{"com.openai":{"apps":{"inline":true}}}
        }"#,
    )
    .unwrap();
    assert!(read_manifest(&root)
        .unwrap_err()
        .contains("apps 必须是包内相对路径"));
}

#[test]
fn manifest_rejects_invalid_declared_metadata_types() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(&root).unwrap();

    for invalid in [
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo","version":null}"#,
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo","homepage":42}"#,
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo","author":{"name":null}}"#,
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo","author":{"future":"value"}}"#,
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo","keywords":["valid",1]}"#,
    ] {
        fs::write(root.join("plugin.json"), invalid).unwrap();
        assert!(read_manifest(&root).is_err(), "accepted {invalid}");
    }
}

#[cfg(unix)]
#[test]
fn manifest_rejects_symlinked_root_file() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(&root).unwrap();
    let target = temp.path().join("plugin.json");
    fs::write(
        &target,
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo"}"#,
    )
    .unwrap();
    symlink(target, root.join("plugin.json")).unwrap();

    assert!(read_manifest(&root).is_err());
}

#[test]
fn resolves_codex_marketplace_sources_from_marketplace_root() {
    let temp = TempDir::new().unwrap();
    let marketplace_root = temp.path().join("openai-bundled");
    let package_root = marketplace_root.join("plugins/browser");
    fixture(&package_root);
    let marketplace_path = marketplace_root.join(".agents/plugins/marketplace.json");
    fs::create_dir_all(marketplace_path.parent().unwrap()).unwrap();
    fs::write(
        &marketplace_path,
        r#"{
          "name": "openai-bundled",
          "plugins": [{
            "name": "example-plugin",
            "source": {"source": "local", "path": "./plugins/browser"}
          }]
        }"#,
    )
    .unwrap();

    let response = list(
        &temp.path().join("data"),
        PluginCatalogListParams {
            marketplace_paths: vec![marketplace_path.to_string_lossy().into_owned()],
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(response.plugins.len(), 1);
    assert_eq!(response.plugins[0].id, "example-plugin");
    assert_eq!(response.plugins[0].marketplace_id, "openai-bundled");
}

#[test]
fn rejects_marketplace_local_source_outside_marketplace_root() {
    let temp = TempDir::new().unwrap();
    let marketplace_root = temp.path().join("marketplace");
    let outside = temp.path().join("outside");
    fixture(&outside);
    let marketplace_path = marketplace_root.join(".agents/plugins/marketplace.json");
    fs::create_dir_all(marketplace_path.parent().unwrap()).unwrap();
    fs::write(
        &marketplace_path,
        r#"{
          "name": "invalid-local-source",
          "plugins": [{
            "name": "example-plugin",
            "source": {"source": "local", "path": "../outside"}
          }]
        }"#,
    )
    .unwrap();

    let error = discover_package_roots(&marketplace_path).unwrap_err();
    assert!(error.contains("local source 越界"), "{error}");
}

#[test]
fn loads_default_mcp_file_and_resolves_cwd_to_installed_package_root() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    fs::create_dir_all(source.join("scripts")).unwrap();
    fs::write(
        source.join("mcp.json"),
        r#"{
          "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
          "mcpServers": {
            "demo": {"type": "stdio", "command": "demo-mcp", "cwd": "./scripts"}
          }
        }"#,
    )
    .unwrap();
    let data_root = temp.path().join("data");
    install(&data_root, install_params(&source)).unwrap();

    let specs = list_plugin_mcp_runtime_server_specs(&data_root).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "plugin__example-plugin__demo");
    assert_eq!(specs[0].plugin_id.as_deref(), Some("example-plugin"));
    assert!(data_root.join("data/example-plugin").is_dir());
    assert_eq!(
        specs[0].config.sanitized_cwd(),
        Some(fs::canonicalize(data_root.join("v3/packages/example-plugin/1.2.3/scripts")).unwrap())
    );
    let snapshots = enabled_plugin_turn_snapshots(&data_root).unwrap();
    assert_eq!(
        snapshots[0].mcp_server_names,
        vec!["plugin__example-plugin__demo"]
    );
}

#[test]
fn does_not_create_plugin_data_for_http_only_servers() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    fs::write(
        source.join("mcp.json"),
        r#"{
          "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
          "mcpServers": {
            "remote": {"type": "streamable-http", "url": "https://example.com/mcp"}
          }
        }"#,
    )
    .unwrap();
    let data_root = temp.path().join("data");
    install(&data_root, install_params(&source)).unwrap();

    let specs = list_plugin_mcp_runtime_server_specs(&data_root).unwrap();

    assert_eq!(specs.len(), 1);
    assert!(!data_root.join("data/example-plugin").exists());
}

#[test]
fn ignores_manifest_mcp_declarations_and_reads_only_root_file() {
    let temp = TempDir::new().unwrap();
    let inline_source = temp.path().join("inline");
    fixture(&inline_source);
    fs::write(
        inline_source.join("plugin.json"),
        r#"{
          "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name": "inline-plugin",
          "version": "1.0.0"
        }"#,
    )
    .unwrap();

    let file_source = temp.path().join("file");
    fixture(&file_source);
    fs::write(
        file_source.join("plugin.json"),
        r#"{
          "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
          "name": "file-plugin",
          "version": "1.0.0"
        }"#,
    )
    .unwrap();
    fs::create_dir_all(file_source.join("config")).unwrap();
    fs::write(
        file_source.join("mcp.json"),
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/mcp.schema.json","mcpServers":{"file":{"type":"stdio","command":"file-mcp"}}}"#,
    )
    .unwrap();

    let data_root = temp.path().join("data");
    install(&data_root, install_params(&inline_source)).unwrap();
    install(&data_root, install_params(&file_source)).unwrap();
    let specs = list_plugin_mcp_runtime_server_specs(&data_root).unwrap();
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        vec!["plugin__file-plugin__file"]
    );
}

#[test]
fn disabled_plugin_and_invalid_mcp_siblings_are_fail_closed() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    fs::create_dir_all(temp.path().join("outside")).unwrap();
    fs::write(
        source.join("mcp.json"),
        r#"{
          "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
          "mcpServers": {
            "valid": {"type": "stdio", "command": "valid-mcp"},
            "invalid": {"type": "streamable-http", "url": "ftp://not-supported"},
            "escape": {"type": "stdio", "command": "escape-mcp", "cwd": "./../outside"}
          }
        }"#,
    )
    .unwrap();
    let data_root = temp.path().join("data");
    install(&data_root, install_params(&source)).unwrap();

    let specs = list_plugin_mcp_runtime_server_specs(&data_root).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "plugin__example-plugin__valid");

    set_enabled(
        &data_root,
        app_server_protocol::protocol::v2::PluginCatalogEnabledSetParams {
            plugin_id: "example-plugin".to_string(),
            enabled: false,
        },
    )
    .unwrap();
    assert!(list_plugin_mcp_runtime_server_specs(&data_root)
        .unwrap()
        .is_empty());
}
