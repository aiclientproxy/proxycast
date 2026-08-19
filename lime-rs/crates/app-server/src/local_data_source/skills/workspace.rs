use app_server_protocol::WorkspaceRegisteredSkillsListParams;
use app_server_protocol::WorkspaceSkillBindingsListParams;
use lime_skills::load_skill_from_file;
use serde_json::json;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

pub(crate) fn list_workspace_skill_bindings_value(
    params: WorkspaceSkillBindingsListParams,
) -> Result<Value, String> {
    let caller = lime_core::tool_calling::normalize_tool_caller(params.caller.as_deref())
        .unwrap_or_else(|| "assistant".to_string());
    let workspace_root = workspace_root_path(&params.workspace_root)?;
    let registered_skills =
        list_workspace_registered_skills_value(WorkspaceRegisteredSkillsListParams {
            workspace_root: params.workspace_root,
        })?;
    let mut bindings = registered_skills
        .into_iter()
        .map(workspace_registered_skill_to_binding_value)
        .collect::<Vec<_>>();

    bindings.sort_by(|left, right| {
        let left_key = left
            .get("directory")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let right_key = right
            .get("directory")
            .and_then(Value::as_str)
            .unwrap_or_default();
        left_key.cmp(right_key)
    });
    let ready_total = bindings
        .iter()
        .filter(|binding| {
            binding.get("binding_status").and_then(Value::as_str) == Some("ready_for_manual_enable")
        })
        .count();
    let blocked_total = bindings.len().saturating_sub(ready_total);
    Ok(json!({
        "request": {
            "workspace_root": workspace_root.to_string_lossy().to_string(),
            "caller": caller,
            "surface": {
                "workbench": params.workbench,
            },
        },
        "warnings": [
            "当前只返回 workspace 本地注册 Skill 的只读 readiness；不会 reload Skill，也不会注入默认工具面。"
        ],
        "counts": {
            "registered_total": bindings.len(),
            "ready_for_manual_enable_total": ready_total,
            "blocked_total": blocked_total,
            "query_loop_visible_total": 0,
            "tool_runtime_visible_total": 0,
            "launch_enabled_total": 0,
        },
        "bindings": bindings,
    }))
}

pub(crate) fn list_workspace_registered_skills_value(
    params: WorkspaceRegisteredSkillsListParams,
) -> Result<Vec<Value>, String> {
    let workspace_root = workspace_root_path(&params.workspace_root)?;
    let skills_root = workspace_root.join(".agents").join("skills");
    let mut skills = Vec::new();
    if !skills_root.exists() {
        return Ok(skills);
    }
    let skills_root_metadata = fs::symlink_metadata(&skills_root)
        .map_err(|error| format!("read workspace skills root failed: {error}"))?;
    if skills_root_metadata.file_type().is_symlink() {
        return Err(format!(
            "workspace skills root must not be a symlink: {}",
            skills_root.display()
        ));
    }
    if !skills_root_metadata.is_dir() {
        return Err(format!(
            "workspace skills root must be a directory: {}",
            skills_root.display()
        ));
    }
    let canonical_skills_root = fs::canonicalize(&skills_root)
        .map_err(|error| format!("canonicalize workspace skills root failed: {error}"))?;

    let mut entries = fs::read_dir(&skills_root)
        .map_err(|error| format!("read workspace skills failed: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read workspace skill entry failed: {error}"))?;
    entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_string());
    for entry in entries {
        let skill_dir = entry.path();
        let skill_dir_metadata = fs::symlink_metadata(&skill_dir)
            .map_err(|error| format!("read workspace skill metadata failed: {error}"))?;
        if skill_dir_metadata.file_type().is_symlink() {
            return Err(format!(
                "workspace registered skill must not be a symlink: {}",
                skill_dir.display()
            ));
        }
        if !skill_dir_metadata.is_dir() {
            continue;
        }
        let skill_file = skill_dir.join("SKILL.md");
        let registration_file = skill_dir.join(".lime").join("registration.json");
        let skill_file_metadata = match fs::symlink_metadata(&skill_file) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        let registration_file_metadata = match fs::symlink_metadata(&registration_file) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if skill_file_metadata.file_type().is_symlink()
            || registration_file_metadata.file_type().is_symlink()
        {
            return Err(format!(
                "workspace registered skill files must not be symlinks: {}",
                skill_dir.display()
            ));
        }
        if !skill_file_metadata.is_file() || !registration_file_metadata.is_file() {
            continue;
        }
        let canonical_skill_dir = fs::canonicalize(&skill_dir)
            .map_err(|error| format!("canonicalize workspace skill directory failed: {error}"))?;
        let canonical_skill_file = fs::canonicalize(&skill_file)
            .map_err(|error| format!("canonicalize workspace skill file failed: {error}"))?;
        let canonical_registration_file =
            fs::canonicalize(&registration_file).map_err(|error| {
                format!("canonicalize workspace skill registration failed: {error}")
            })?;
        if !canonical_skill_dir.starts_with(&canonical_skills_root)
            || !canonical_skill_file.starts_with(&canonical_skill_dir)
            || !canonical_registration_file.starts_with(&canonical_skill_dir)
        {
            return Err(format!(
                "workspace registered skill path escaped workspace root: {}",
                skill_dir.display()
            ));
        }
        let directory = entry.file_name().to_string_lossy().to_string();
        let skill = load_skill_from_file(&directory, &skill_file)?;
        let registration: Value = fs::read_to_string(&registration_file)
            .map_err(|error| format!("read skill registration failed: {error}"))
            .and_then(|content| {
                serde_json::from_str(&content)
                    .map_err(|error| format!("parse skill registration failed: {error}"))
            })?;
        let standard_compliance = serde_json::to_value(&skill.standard_compliance)
            .map_err(|error| format!("serialize skill standard compliance failed: {error}"))?;
        let permission_summary = registration_permission_summary(&registration);
        let allowed_tools = skill.allowed_tools.clone().unwrap_or_default();
        skills.push(json!({
            "key": format!("workspace:{directory}"),
            "name": skill.display_name,
            "description": skill.description,
            "directory": directory,
            "registered_skill_directory": skill_dir.to_string_lossy().to_string(),
            "registration": registration,
            "permission_summary": permission_summary,
            "metadata": skill.metadata,
            "allowed_tools": allowed_tools,
            "resource_summary": skill_resource_summary(&skill_dir)?,
            "standard_compliance": standard_compliance,
            "launch_enabled": false,
            "runtime_gate": "已注册为 Workspace 本地 Skill 包；进入运行前还需要 P3C runtime binding 与 tool_runtime 授权。",
        }));
    }

    Ok(skills)
}

fn workspace_registered_skill_to_binding_value(skill: Value) -> Value {
    let directory = skill
        .get("directory")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let registration = skill.get("registration").cloned().unwrap_or(Value::Null);
    let source_verification_report_id = registration
        .get("sourceVerificationReportId")
        .or_else(|| registration.get("source_verification_report_id"))
        .and_then(Value::as_str);
    let validation_errors = skill
        .pointer("/standard_compliance/validation_errors")
        .or_else(|| skill.pointer("/standard_compliance/validationErrors"))
        .or_else(|| skill.pointer("/standardCompliance/validationErrors"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let binding_status = if validation_errors == 0 && source_verification_report_id.is_some() {
        "ready_for_manual_enable"
    } else {
        "blocked"
    };
    let binding_status_reason = if binding_status == "ready_for_manual_enable" {
        "已具备 workspace skill runtime binding 候选资格；当前仍未注入默认工具面。"
    } else if validation_errors > 0 {
        "Agent Skills 标准检查仍有问题，不能进入 runtime binding。"
    } else {
        "缺少来源 verification report，不能证明该 Skill 通过注册前验证。"
    };

    json!({
        "key": format!("workspace_skill:{directory}"),
        "name": skill.get("name").cloned().unwrap_or(Value::Null),
        "description": skill.get("description").cloned().unwrap_or(Value::Null),
        "directory": skill.get("directory").cloned().unwrap_or(Value::Null),
        "registered_skill_directory": skill
            .get("registered_skill_directory")
            .cloned()
            .unwrap_or(Value::Null),
        "registration": registration,
        "permission_summary": skill
            .get("permission_summary")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "metadata": skill.get("metadata").cloned().unwrap_or_else(|| json!({})),
        "allowed_tools": skill
            .get("allowed_tools")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "resource_summary": skill
            .get("resource_summary")
            .cloned()
            .unwrap_or_else(|| json!({})),
        "standard_compliance": skill
            .get("standard_compliance")
            .cloned()
            .unwrap_or_else(|| json!({})),
        "runtime_binding_target": "workspace_skill",
        "binding_status": binding_status,
        "binding_status_reason": binding_status_reason,
        "next_gate": if binding_status == "ready_for_manual_enable" {
            "manual_runtime_enable"
        } else {
            "restore_verification_provenance"
        },
        "query_loop_visible": false,
        "tool_runtime_visible": false,
        "launch_enabled": false,
        "runtime_gate": "等待显式 session enable 与 tool_runtime 授权裁剪。",
    })
}

fn registration_permission_summary(registration: &Value) -> Vec<String> {
    registration
        .get("permissionSummary")
        .or_else(|| registration.get("permission_summary"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn workspace_root_path(workspace_root: &str) -> Result<PathBuf, String> {
    let workspace_root = PathBuf::from(workspace_root.trim());
    if !workspace_root.is_absolute() {
        return Err(format!(
            "workspaceRoot must be absolute: {}",
            workspace_root.display()
        ));
    }
    Ok(workspace_root)
}

fn skill_resource_summary(skill_dir: &Path) -> Result<Value, String> {
    let references = skill_dir.join("references");
    let scripts = skill_dir.join("scripts");
    let assets = skill_dir.join("assets");
    Ok(json!({
        "hasScripts": scripts.is_dir(),
        "hasReferences": references.is_dir(),
        "hasAssets": assets.is_dir(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_registered_skill(workspace_root: &Path, directory: &str) -> PathBuf {
        let skill_dir = workspace_root
            .join(".agents")
            .join("skills")
            .join(directory);
        fs::create_dir_all(skill_dir.join(".lime")).expect("create registered skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            [
                "---",
                "name: 只读报告",
                "description: 读取本地数据并生成报告。",
                "allowed-tools: Read",
                "---",
                "",
                "# 只读报告",
            ]
            .join("\n"),
        )
        .expect("write skill");
        fs::write(
            skill_dir.join(".lime").join("registration.json"),
            json!({
                "registrationId": "capreg-1",
                "registeredAt": "2026-06-06T00:00:00.000Z",
                "skillDirectory": directory,
                "registeredSkillDirectory": skill_dir.to_string_lossy(),
                "sourceDraftId": "capdraft-1",
                "sourceVerificationReportId": "capver-1",
                "generatedFileCount": 2,
                "permissionSummary": ["Level 0 只读发现"]
            })
            .to_string(),
        )
        .expect("write registration");
        skill_dir
    }

    #[test]
    fn list_workspace_registered_skills_value_discovers_registered_skill() {
        let temp = TempDir::new().expect("temp dir");
        let skill_dir = write_registered_skill(temp.path(), "readonly-report");

        let skills = list_workspace_registered_skills_value(WorkspaceRegisteredSkillsListParams {
            workspace_root: temp.path().to_string_lossy().to_string(),
        })
        .expect("list registered skills");

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0]["key"], json!("workspace:readonly-report"));
        assert_eq!(skills[0]["name"], json!("只读报告"));
        assert_eq!(
            skills[0]["registered_skill_directory"],
            json!(skill_dir.to_string_lossy().to_string())
        );
        assert_eq!(
            skills[0]["registration"]["sourceVerificationReportId"],
            json!("capver-1")
        );
        assert_eq!(skills[0]["permission_summary"], json!(["Level 0 只读发现"]));
        assert_eq!(skills[0]["launch_enabled"], json!(false));
    }

    #[test]
    fn list_workspace_skill_bindings_value_projects_readiness_without_launch() {
        let temp = TempDir::new().expect("temp dir");
        let skill_dir = write_registered_skill(temp.path(), "readonly-report");

        let bindings = list_workspace_skill_bindings_value(WorkspaceSkillBindingsListParams {
            workspace_root: temp.path().to_string_lossy().to_string(),
            caller: Some("assistant".to_string()),
            workbench: true,
        })
        .expect("list workspace skill bindings");

        assert_eq!(
            bindings.pointer("/counts/registered_total"),
            Some(&json!(1))
        );
        assert_eq!(
            bindings.pointer("/counts/ready_for_manual_enable_total"),
            Some(&json!(1))
        );
        assert_eq!(
            bindings.pointer("/counts/query_loop_visible_total"),
            Some(&json!(0))
        );
        assert_eq!(
            bindings.pointer("/counts/tool_runtime_visible_total"),
            Some(&json!(0))
        );
        assert_eq!(
            bindings.pointer("/counts/launch_enabled_total"),
            Some(&json!(0))
        );
        let binding = bindings
            .pointer("/bindings/0")
            .expect("binding should be present");
        assert_eq!(
            binding.get("registered_skill_directory"),
            Some(&json!(skill_dir.to_string_lossy().to_string()))
        );
        assert_eq!(
            binding.get("binding_status"),
            Some(&json!("ready_for_manual_enable"))
        );
        assert_eq!(
            binding.get("next_gate"),
            Some(&json!("manual_runtime_enable"))
        );
        assert_eq!(binding.get("query_loop_visible"), Some(&json!(false)));
        assert_eq!(binding.get("tool_runtime_visible"), Some(&json!(false)));
        assert_eq!(binding.get("launch_enabled"), Some(&json!(false)));
    }

    #[test]
    fn list_workspace_registered_skills_value_ignores_standard_skill_without_registration() {
        let temp = TempDir::new().expect("temp dir");
        let skill_dir = temp
            .path()
            .join(".agents")
            .join("skills")
            .join("manual-standard-skill");
        fs::create_dir_all(&skill_dir).expect("create standard skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            [
                "---",
                "name: 手工标准 Skill",
                "description: 没有 P3A provenance。",
                "---",
                "",
                "# 手工标准 Skill",
            ]
            .join("\n"),
        )
        .expect("write skill");

        let skills = list_workspace_registered_skills_value(WorkspaceRegisteredSkillsListParams {
            workspace_root: temp.path().to_string_lossy().to_string(),
        })
        .expect("list registered skills");

        assert!(skills.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn list_workspace_registered_skills_value_rejects_symlink_skill_directory() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().expect("temp dir");
        let skills_root = temp.path().join(".agents").join("skills");
        let outside = temp.path().join("outside-skill");
        fs::create_dir_all(&outside).expect("create outside skill");
        fs::create_dir_all(&skills_root).expect("create skills root");
        fs::write(
            outside.join("SKILL.md"),
            [
                "---",
                "name: 外部 Skill",
                "description: 不应通过 symlink 暴露。",
                "---",
                "",
                "# 外部 Skill",
            ]
            .join("\n"),
        )
        .expect("write outside skill");
        symlink(&outside, skills_root.join("outside-skill")).expect("symlink skill");

        let error = list_workspace_registered_skills_value(WorkspaceRegisteredSkillsListParams {
            workspace_root: temp.path().to_string_lossy().to_string(),
        })
        .expect_err("reject symlink skill directory");

        assert!(error.contains("must not be a symlink"));
    }
}
