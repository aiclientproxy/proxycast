use super::*;
use tempfile::TempDir;

fn fixture(root: &Path) {
    fs::create_dir_all(root.join(".codex-plugin")).unwrap();
    fs::create_dir_all(root.join("skills/example")).unwrap();
    fs::write(
        root.join("skills/example/SKILL.md"),
        "---\nname: example\n---\n",
    )
    .unwrap();
    fs::write(
        root.join(".codex-plugin/plugin.json"),
        r#"{
          "name": "example-plugin",
          "version": "1.2.3",
          "description": "Example",
          "skills": "./skills/",
          "interface": {"displayName": "Example Plugin"}
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
fn installs_codex_manifest_into_v2_store() {
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
    let activations = enabled_activation_descriptors(&data_root).unwrap();
    assert_eq!(activations.len(), 1);
    assert_eq!(activations[0]["pluginId"], "example-plugin");
    assert_eq!(
        activations[0]["runtimeCapabilities"]["skills"][0]["id"],
        "example"
    );
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
    assert!(enabled_activation_descriptors(&data_root)
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
fn expected_digest_and_semver_fail_closed() {
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
        source.join(".codex-plugin/plugin.json"),
        r#"{"name":"example-plugin","version":"1.2"}"#,
    )
    .unwrap();
    assert!(read_manifest(&source).unwrap_err().contains("semver"));
}

#[test]
fn rejects_legacy_manifest_and_parent_paths() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("source");
    fs::create_dir_all(root.join(".codex-plugin")).unwrap();
    fs::write(
        root.join(".codex-plugin/plugin.json"),
        r#"{"name":"example-plugin","version":"1.0.0","schemaVersion":"lime.plugin.package.v1"}"#,
    )
    .unwrap();
    assert!(read_manifest(&root).is_err());
    assert!(resource_path(&root, "../outside").is_err());
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
        source.join(".mcp.json"),
        r#"{
          "mcpServers": {
            "demo": {"command": "demo-mcp", "cwd": "scripts"}
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
    assert_eq!(
        specs[0].config.sanitized_cwd(),
        Some(fs::canonicalize(data_root.join("v2/packages/example-plugin/1.2.3/scripts")).unwrap())
    );
    let activation = enabled_activation_descriptors(&data_root).unwrap();
    assert_eq!(
        activation[0]["runtimeCapabilities"]["mcpServers"][0]["runtimeName"],
        "plugin__example-plugin__demo"
    );
}

#[test]
fn loads_inline_and_manifest_path_mcp_declarations() {
    let temp = TempDir::new().unwrap();
    let inline_source = temp.path().join("inline");
    fixture(&inline_source);
    fs::write(
        inline_source.join(".codex-plugin/plugin.json"),
        r#"{
          "name": "inline-plugin",
          "version": "1.0.0",
          "mcpServers": {"inline": {"command": "inline-mcp"}}
        }"#,
    )
    .unwrap();

    let file_source = temp.path().join("file");
    fixture(&file_source);
    fs::write(
        file_source.join(".codex-plugin/plugin.json"),
        r#"{
          "name": "file-plugin",
          "version": "1.0.0",
          "mcpServers": "./config/mcp.json"
        }"#,
    )
    .unwrap();
    fs::create_dir_all(file_source.join("config")).unwrap();
    fs::write(
        file_source.join("config/mcp.json"),
        r#"{"mcpServers":{"file":{"command":"file-mcp"}}}"#,
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
        vec!["plugin__file-plugin__file", "plugin__inline-plugin__inline"]
    );
}

#[test]
fn disabled_plugin_and_invalid_mcp_siblings_are_fail_closed() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("source");
    fixture(&source);
    fs::create_dir_all(temp.path().join("outside")).unwrap();
    fs::write(
        source.join(".mcp.json"),
        r#"{
          "mcpServers": {
            "valid": {"command": "valid-mcp"},
            "invalid": {"url": "ftp://not-supported"},
            "escape": {"command": "escape-mcp", "cwd": "../outside"}
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
