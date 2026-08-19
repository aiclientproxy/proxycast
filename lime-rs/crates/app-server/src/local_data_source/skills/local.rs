use super::common::parse_skill_package_app;
use super::common::resolve_skill_package_dir;
use super::common::scan_installed_skill_directories_from_roots;
use super::common::skill_package_app_lookup_roots;
use super::common::skill_package_app_root;
use super::common::validate_skill_package_directory;
use super::common::SkillPackageApp;
use app_server_protocol::SkillInstalledDirectoriesListResponse;
use app_server_protocol::SkillLocalImportParams;
use app_server_protocol::SkillLocalImportResponse;
use app_server_protocol::SkillLocalInspectParams;
use app_server_protocol::SkillLocalInspectResponse;
use app_server_protocol::SkillLocalRenameParams;
use app_server_protocol::SkillLocalRenameResponse;
use app_server_protocol::SkillRemoteInspectParams;
use app_server_protocol::SkillRemoteInspectResponse;
use app_server_protocol::SkillScaffoldCreateParams;
use app_server_protocol::SkillScaffoldCreateResponse;
use chrono::SecondsFormat;
use chrono::Utc;
use lime_core::app_paths;
use lime_services::skill_service::SkillService;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

fn validate_skill_rename_directory(old_directory: &str, new_directory: &str) -> Result<(), String> {
    validate_skill_package_directory(old_directory)?;
    validate_skill_package_directory(new_directory)?;
    if old_directory == new_directory {
        return Err("New skill directory must be different".to_string());
    }
    Ok(())
}

pub(crate) fn rename_local_skill(
    params: SkillLocalRenameParams,
) -> Result<SkillLocalRenameResponse, String> {
    let app = parse_skill_package_app(&params.app)?;
    validate_skill_rename_directory(&params.directory, &params.new_directory)?;
    let skills_root = skill_package_app_root(app)?;
    let source_dir = resolve_skill_package_dir(&[skills_root.clone()], &params.directory)?;
    let target_dir = skills_root.join(&params.new_directory);

    if target_dir.exists() {
        return Err(format!(
            "Skill directory already exists: {}",
            params.new_directory
        ));
    }

    fs::rename(&source_dir, &target_dir).map_err(|error| {
        format!(
            "Failed to rename skill directory {} -> {}: {error}",
            source_dir.display(),
            target_dir.display()
        )
    })?;

    Ok(SkillLocalRenameResponse {
        directory: params.new_directory,
    })
}

pub(crate) fn list_installed_skill_directories(
) -> Result<SkillInstalledDirectoriesListResponse, String> {
    Ok(SkillInstalledDirectoriesListResponse {
        directories: scan_installed_skill_directories_from_roots(&skill_package_app_lookup_roots(
            SkillPackageApp::Lime,
        )?),
    })
}

pub(crate) fn inspect_local_skill(
    params: SkillLocalInspectParams,
) -> Result<SkillLocalInspectResponse, String> {
    let app = parse_skill_package_app(&params.app)?;
    let skill_roots = skill_package_app_lookup_roots(app)?;
    let skill_dir = resolve_skill_package_dir(&skill_roots, &params.directory)?;
    let inspection = SkillService::inspect_skill_dir(&skill_dir)
        .map_err(|error| format!("Skill failed inspection: {error}"))?;
    Ok(SkillLocalInspectResponse {
        inspection: serde_json::to_value(inspection)
            .map_err(|error| format!("Failed to serialize skill inspection: {error}"))?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillScaffoldTarget {
    Project,
    User,
}

impl SkillScaffoldTarget {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "project" => Ok(Self::Project),
            "user" => Ok(Self::User),
            _ => Err(format!("Unsupported scaffold target: {value}")),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateSkillScaffoldRequest {
    target: String,
    directory: String,
    name: String,
    description: String,
    #[serde(default)]
    when_to_use: Vec<String>,
    #[serde(default)]
    inputs: Vec<String>,
    #[serde(default)]
    outputs: Vec<String>,
    #[serde(default)]
    steps: Vec<String>,
    #[serde(default)]
    fallback_strategy: Vec<String>,
}

#[derive(Serialize)]
struct SkillScaffoldFrontmatter<'a> {
    name: &'a str,
    description: &'a str,
}

struct SkillScaffoldSections {
    when_to_use: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    steps: Vec<String>,
    fallback_strategy: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillScaffoldRegistration {
    None,
    Workspace,
}

fn normalize_scaffold_items(items: &[String], fallback: &[&str]) -> Vec<String> {
    let normalized: Vec<String> = items
        .iter()
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect();

    if normalized.is_empty() {
        return fallback.iter().map(|item| (*item).to_string()).collect();
    }

    normalized
}

fn build_skill_scaffold_sections(request: &CreateSkillScaffoldRequest) -> SkillScaffoldSections {
    SkillScaffoldSections {
        when_to_use: normalize_scaffold_items(
            &request.when_to_use,
            &[
                "当你需要重复完成这类任务时使用。",
                "适合把一次成功结果沉淀成稳定可复用的工作流。",
            ],
        ),
        inputs: normalize_scaffold_items(
            &request.inputs,
            &[
                "用户目标、主题与成功标准。",
                "受众、风格、篇幅、平台或交付格式等约束。",
                "如有参考资料、示例或素材，请一并提供。",
            ],
        ),
        outputs: normalize_scaffold_items(
            &request.outputs,
            &[
                "交付一份可直接使用的完整结果。",
                "保留清晰的结构层级、重点信息与必要说明。",
            ],
        ),
        steps: normalize_scaffold_items(
            &request.steps,
            &[
                "先确认目标、边界与交付格式。",
                "提炼可复用的结构骨架，再补齐关键信息。",
                "输出可直接交付的首版结果，并为后续迭代留好锚点。",
            ],
        ),
        fallback_strategy: normalize_scaffold_items(
            &request.fallback_strategy,
            &[
                "信息不足时，先补问最关键的约束，不要自行假设事实。",
                "原结果不可直接复用时，先提炼最小骨架，再继续展开。",
            ],
        ),
    }
}

fn render_bullet_list(items: &[String]) -> String {
    items.iter().map(|item| format!("- {item}\n")).collect()
}

fn render_ordered_list(items: &[String]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(index, item)| format!("{}. {item}\n", index + 1))
        .collect()
}

fn resolve_skill_scaffold_root(
    app: SkillPackageApp,
    target: SkillScaffoldTarget,
) -> Result<PathBuf, String> {
    match target {
        SkillScaffoldTarget::User => skill_package_app_root(app),
        SkillScaffoldTarget::Project => match app {
            SkillPackageApp::Lime => app_paths::resolve_project_skills_dir()
                .ok_or_else(|| "Failed to resolve project skills directory".to_string()),
            _ => Err("Project skill scaffold is only supported for lime".to_string()),
        },
    }
}

fn build_skill_scaffold_content(request: &CreateSkillScaffoldRequest) -> Result<String, String> {
    let name = request.name.trim();
    let description = request.description.trim();
    let sections = build_skill_scaffold_sections(request);
    let frontmatter = serde_yaml::to_string(&SkillScaffoldFrontmatter { name, description })
        .map_err(|error| format!("Failed to build skill frontmatter: {error}"))?;
    let frontmatter = frontmatter.strip_prefix("---\n").unwrap_or(&frontmatter);

    Ok(format!(
        "---\n{frontmatter}---\n\n# {name}\n\n## 何时使用\n{when_to_use}\n## 输入\n{inputs}\n## 执行步骤\n{steps}\n## 输出\n{outputs}\n## 失败回退\n{fallback_strategy}\n## 维护提示\n- 如需引用资料，请将文件放到 `references/` 目录。\n- 如需脚本或素材，请分别放到 `scripts/` 与 `assets/` 目录。\n- 如需长期沉淀模板或示例，优先放到相邻目录，不要把所有细节都塞进主文件。\n",
        when_to_use = render_bullet_list(&sections.when_to_use),
        inputs = render_bullet_list(&sections.inputs),
        steps = render_ordered_list(&sections.steps),
        outputs = render_bullet_list(&sections.outputs),
        fallback_strategy = render_bullet_list(&sections.fallback_strategy),
    ))
}

fn create_skill_scaffold_in_root(
    skills_root: &Path,
    request: &CreateSkillScaffoldRequest,
    registration: SkillScaffoldRegistration,
) -> Result<lime_core::models::skill_model::SkillPackageInspection, String> {
    let directory = request.directory.trim();
    validate_skill_package_directory(directory)?;

    let name = request.name.trim();
    if name.is_empty() {
        return Err("Skill name is required".to_string());
    }

    let description = request.description.trim();
    if description.is_empty() {
        return Err("Skill description is required".to_string());
    }

    fs::create_dir_all(skills_root).map_err(|error| {
        format!(
            "Failed to create skills root {}: {error}",
            skills_root.display()
        )
    })?;

    let skill_dir = skills_root.join(directory);
    if skill_dir.exists() {
        return Err(format!("Skill directory already exists: {directory}"));
    }

    fs::create_dir_all(&skill_dir).map_err(|error| {
        format!(
            "Failed to create skill directory {}: {error}",
            skill_dir.display()
        )
    })?;

    let skill_md_content = build_skill_scaffold_content(request)?;
    let skill_md_path = skill_dir.join("SKILL.md");
    if let Err(error) = fs::write(&skill_md_path, skill_md_content) {
        let _ = fs::remove_dir_all(&skill_dir);
        return Err(format!(
            "Failed to write scaffold file {}: {error}",
            skill_md_path.display()
        ));
    }

    match SkillService::inspect_skill_dir(&skill_dir) {
        Ok(inspection) => {
            if matches!(registration, SkillScaffoldRegistration::Workspace) {
                if let Err(error) = write_skill_scaffold_registration(&skill_dir, directory) {
                    let _ = fs::remove_dir_all(&skill_dir);
                    return Err(error);
                }
            }
            Ok(inspection)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&skill_dir);
            Err(format!("Created scaffold failed inspection: {error}"))
        }
    }
}

fn write_skill_scaffold_registration(skill_dir: &Path, directory: &str) -> Result<(), String> {
    let registration_dir = skill_dir.join(".lime");
    fs::create_dir_all(&registration_dir).map_err(|error| {
        format!(
            "Failed to create skill registration directory {}: {error}",
            registration_dir.display()
        )
    })?;

    let registration = json!({
        "registrationId": format!("skill-scaffold-{directory}"),
        "registeredAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        "skillDirectory": directory,
        "registeredSkillDirectory": skill_dir.to_string_lossy().to_string(),
        "sourceDraftId": "skill-scaffold",
        "sourceVerificationReportId": "skill-scaffold-create",
        "generatedFileCount": 1,
        "permissionSummary": [],
    });
    let content = serde_json::to_string_pretty(&registration)
        .map_err(|error| format!("Failed to serialize skill registration: {error}"))?;
    let registration_file = registration_dir.join("registration.json");
    fs::write(&registration_file, format!("{content}\n")).map_err(|error| {
        format!(
            "Failed to write skill registration {}: {error}",
            registration_file.display()
        )
    })
}

fn copy_skill_directory_recursive(source: &Path, destination: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(source)
        .map_err(|error| format!("Failed to read skill source {}: {error}", source.display()))?;

    if metadata.file_type().is_symlink() {
        return Err(format!(
            "Skill package contains unsupported symlink: {}",
            source.display()
        ));
    }

    if metadata.is_dir() {
        fs::create_dir_all(destination).map_err(|error| {
            format!(
                "Failed to create target directory {}: {error}",
                destination.display()
            )
        })?;

        for entry in fs::read_dir(source).map_err(|error| {
            format!(
                "Failed to read skill directory {}: {error}",
                source.display()
            )
        })? {
            let entry = entry.map_err(|error| format!("Failed to read skill entry: {error}"))?;
            copy_skill_directory_recursive(&entry.path(), &destination.join(entry.file_name()))?;
        }
        return Ok(());
    }

    if metadata.is_file() {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "Failed to create target parent directory {}: {error}",
                    parent.display()
                )
            })?;
        }
        fs::copy(source, destination).map_err(|error| {
            format!(
                "Failed to copy skill file {} -> {}: {error}",
                source.display(),
                destination.display()
            )
        })?;
        return Ok(());
    }

    Err(format!(
        "Unsupported skill package entry: {}",
        source.display()
    ))
}

fn import_local_skill_into_root(skills_root: &Path, source_path: &Path) -> Result<String, String> {
    let canonical_source = source_path.canonicalize().map_err(|error| {
        format!(
            "Failed to resolve skill source {}: {error}",
            source_path.display()
        )
    })?;

    if !canonical_source.is_dir() {
        return Err("Skill source path must be a directory".to_string());
    }

    let inspection = SkillService::inspect_skill_dir(&canonical_source).map_err(|error| {
        format!(
            "Skill source is invalid {}: {error}",
            canonical_source.display()
        )
    })?;

    if !inspection.standard_compliance.validation_errors.is_empty() {
        return Err(format!(
            "Skill package is not Agent Skills compliant: {}",
            inspection.standard_compliance.validation_errors.join("; ")
        ));
    }

    let directory = canonical_source
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Failed to resolve skill directory name".to_string())?;
    validate_skill_package_directory(directory)?;

    fs::create_dir_all(skills_root).map_err(|error| {
        format!(
            "Failed to create skills root {}: {error}",
            skills_root.display()
        )
    })?;

    let target_dir = skills_root.join(directory);
    if target_dir.exists() {
        return Err(format!("Skill directory already exists: {directory}"));
    }

    if let Err(error) = copy_skill_directory_recursive(&canonical_source, &target_dir) {
        let _ = fs::remove_dir_all(&target_dir);
        return Err(error);
    }

    match SkillService::inspect_skill_dir(&target_dir) {
        Ok(imported_inspection)
            if imported_inspection
                .standard_compliance
                .validation_errors
                .is_empty() =>
        {
            Ok(directory.to_string())
        }
        Ok(imported_inspection) => {
            let _ = fs::remove_dir_all(&target_dir);
            Err(format!(
                "Imported skill is not Agent Skills compliant: {}",
                imported_inspection
                    .standard_compliance
                    .validation_errors
                    .join("; ")
            ))
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&target_dir);
            Err(format!("Imported skill failed inspection: {error}"))
        }
    }
}

fn validate_remote_skill_directory(directory: &str) -> Result<(), String> {
    let directory = directory.trim();
    if directory.is_empty() {
        return Err("Skill directory is required".to_string());
    }

    if directory.contains("..")
        || directory.contains('\\')
        || directory.starts_with('/')
        || directory.starts_with("./")
        || directory.ends_with("/.")
        || directory.contains("/./")
        || directory.split('/').any(str::is_empty)
    {
        return Err("Invalid skill directory".to_string());
    }

    let mut has_component = false;
    for component in Path::new(directory).components() {
        match component {
            Component::Normal(_) => has_component = true,
            _ => return Err("Invalid skill directory".to_string()),
        }
    }

    if has_component {
        Ok(())
    } else {
        Err("Skill directory is required".to_string())
    }
}

pub(crate) fn create_skill_scaffold(
    params: SkillScaffoldCreateParams,
) -> Result<SkillScaffoldCreateResponse, String> {
    let app = parse_skill_package_app(&params.app)?;
    let request: CreateSkillScaffoldRequest = serde_json::from_value(params.request)
        .map_err(|error| format!("Invalid skill scaffold request: {error}"))?;
    let target = SkillScaffoldTarget::parse(&request.target)?;
    let skills_root = resolve_skill_scaffold_root(app, target)?;
    let registration = if matches!(target, SkillScaffoldTarget::Project) {
        SkillScaffoldRegistration::Workspace
    } else {
        SkillScaffoldRegistration::None
    };
    let inspection = create_skill_scaffold_in_root(&skills_root, &request, registration)?;
    if matches!(app, SkillPackageApp::Lime) {
        crate::skill_registry::reload_lime_skill_registry();
    }
    Ok(SkillScaffoldCreateResponse {
        inspection: serde_json::to_value(inspection)
            .map_err(|error| format!("Failed to serialize skill inspection: {error}"))?,
    })
}

pub(crate) fn import_local_skill(
    params: SkillLocalImportParams,
) -> Result<SkillLocalImportResponse, String> {
    let app = parse_skill_package_app(&params.app)?;
    let skills_root = skill_package_app_root(app)?;
    let directory = import_local_skill_into_root(&skills_root, Path::new(&params.source_path))?;
    if matches!(app, SkillPackageApp::Lime) {
        crate::skill_registry::reload_lime_skill_registry();
    }
    Ok(SkillLocalImportResponse { directory })
}

pub(crate) async fn inspect_remote_skill(
    params: SkillRemoteInspectParams,
) -> Result<SkillRemoteInspectResponse, String> {
    validate_remote_skill_directory(&params.directory)?;
    let service = SkillService::new().map_err(|error| error.to_string())?;
    let inspection = service
        .inspect_remote_skill(
            &params.owner,
            &params.name,
            &params.branch,
            &params.directory,
        )
        .await
        .map_err(|error| error.to_string())?;
    Ok(SkillRemoteInspectResponse {
        inspection: serde_json::to_value(inspection)
            .map_err(|error| format!("Failed to serialize skill inspection: {error}"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::WorkspaceSkillBindingsListParams;
    use tempfile::TempDir;

    fn scaffold_request(target: &str, directory: &str) -> CreateSkillScaffoldRequest {
        CreateSkillScaffoldRequest {
            target: target.to_string(),
            directory: directory.to_string(),
            name: "专家复盘".to_string(),
            description: "把专家对话沉淀成可复用流程。".to_string(),
            when_to_use: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            steps: Vec::new(),
            fallback_strategy: Vec::new(),
        }
    }

    #[test]
    fn project_scaffold_registration_is_visible_as_workspace_binding() {
        let temp = TempDir::new().expect("temp dir");
        let skills_root = temp.path().join(".agents").join("skills");
        let request = scaffold_request("project", "expert-review");

        create_skill_scaffold_in_root(&skills_root, &request, SkillScaffoldRegistration::Workspace)
            .expect("create project scaffold");

        let registration_file = skills_root
            .join("expert-review")
            .join(".lime")
            .join("registration.json");
        let registration: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(registration_file).expect("read registration"),
        )
        .expect("parse registration");
        assert_eq!(
            registration["sourceVerificationReportId"],
            json!("skill-scaffold-create")
        );

        let bindings =
            crate::local_data_source::skills::workspace::list_workspace_skill_bindings_value(
                WorkspaceSkillBindingsListParams {
                    workspace_root: temp.path().to_string_lossy().to_string(),
                    caller: Some("assistant".to_string()),
                    workbench: true,
                },
            )
            .expect("list workspace bindings");

        assert_eq!(
            bindings.pointer("/counts/ready_for_manual_enable_total"),
            Some(&json!(1))
        );
        assert_eq!(
            bindings.pointer("/bindings/0/binding_status"),
            Some(&json!("ready_for_manual_enable"))
        );
    }

    #[test]
    fn user_scaffold_does_not_write_workspace_registration() {
        let temp = TempDir::new().expect("temp dir");
        let request = scaffold_request("user", "personal-review");

        create_skill_scaffold_in_root(temp.path(), &request, SkillScaffoldRegistration::None)
            .expect("create user scaffold");

        assert!(!temp
            .path()
            .join("personal-review")
            .join(".lime")
            .join("registration.json")
            .exists());
    }
}
