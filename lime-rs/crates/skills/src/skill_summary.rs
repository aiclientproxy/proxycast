//! Skill 列表 / selector 使用的轻量摘要加载。
//!
//! 参考 Codex 的 snapshot 思路：列表阶段只读取 metadata，正文和完整包 inspection
//! 保持在详情读取路径按需执行。

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

use lime_core::models::SkillStandardCompliance;

use crate::skill_loader::{
    parse_allowed_tools, parse_boolean, parse_skill_frontmatter, SkillFrontmatter,
};

#[derive(Debug, Clone)]
pub struct LoadedSkillSummary {
    pub skill_name: String,
    pub display_name: String,
    pub description: String,
    pub local_directory_path: PathBuf,
    pub allowed_tools: Option<Vec<String>>,
    pub argument_hint: Option<String>,
    pub when_to_use: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub disable_model_invocation: bool,
    pub execution_mode: String,
    pub standard_compliance: SkillStandardCompliance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillSummaryLoadError {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct SkillSummaryLoadReport {
    pub summaries: Vec<LoadedSkillSummary>,
    pub errors: Vec<SkillSummaryLoadError>,
}

#[derive(Debug, Clone)]
struct SkillSummaryCacheEntry {
    signature: String,
    loaded_at: Instant,
    report: SkillSummaryLoadReport,
}

const SKILL_SUMMARY_CACHE_TTL: Duration = Duration::from_secs(5);

pub fn load_skill_summary_from_file(
    skill_name: &str,
    file_path: &Path,
) -> Result<LoadedSkillSummary, String> {
    let content = read_skill_frontmatter_for_summary(file_path)?;

    let (mut frontmatter, _markdown_content) = parse_skill_frontmatter(&content);
    let base_dir = file_path
        .parent()
        .ok_or_else(|| "Skill 文件缺少父目录".to_string())?;
    validate_workflow_ref_for_summary(base_dir, &mut frontmatter);

    let display_name = frontmatter
        .name
        .clone()
        .unwrap_or_else(|| skill_name.to_string());
    let description = frontmatter.description.clone().unwrap_or_default();
    let local_directory_path = base_dir
        .canonicalize()
        .unwrap_or_else(|_| base_dir.to_path_buf());
    let allowed_tools = frontmatter.allowed_tools.clone().or_else(|| {
        parse_allowed_tools(
            frontmatter
                .metadata
                .get("allowed_tools")
                .map(|value| value.as_str()),
        )
    });
    let disable_model_invocation =
        parse_boolean(frontmatter.disable_model_invocation.as_deref(), false);
    let execution_mode = frontmatter.execution_mode.clone().unwrap_or_else(|| {
        if frontmatter
            .workflow_ref
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
            || frontmatter
                .steps_json
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
        {
            "workflow".to_string()
        } else {
            "prompt".to_string()
        }
    });
    let mut standard_compliance = SkillStandardCompliance {
        is_standard: frontmatter.validation_errors.is_empty(),
        validation_errors: frontmatter.validation_errors,
        deprecated_fields: frontmatter.deprecated_fields,
    };
    standard_compliance.validation_errors.sort();
    standard_compliance.validation_errors.dedup();
    standard_compliance.is_standard = standard_compliance.validation_errors.is_empty();

    Ok(LoadedSkillSummary {
        skill_name: skill_name.to_string(),
        display_name,
        description,
        local_directory_path,
        allowed_tools,
        argument_hint: frontmatter.argument_hint,
        when_to_use: frontmatter.when_to_use,
        model: frontmatter.model,
        provider: frontmatter.provider,
        disable_model_invocation,
        execution_mode,
        standard_compliance,
    })
}

fn read_skill_frontmatter_for_summary(file_path: &Path) -> Result<String, String> {
    let file = File::open(file_path).map_err(|e| format!("读取 Skill 文件失败: {}", e))?;
    let mut reader = BufReader::new(file);

    let mut content = String::new();
    let mut line = String::new();
    let bytes = reader
        .read_line(&mut line)
        .map_err(|e| format!("读取 Skill 文件失败: {}", e))?;
    if bytes == 0 || line.trim() != "---" {
        return Err("缺少以 --- 包裹的 YAML frontmatter".to_string());
    }
    content.push_str(&line);

    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| format!("读取 Skill 文件失败: {}", e))?;
        if bytes == 0 {
            return Err("缺少结束的 YAML frontmatter 分隔符 ---".to_string());
        }
        let is_end = line.trim() == "---";
        content.push_str(&line);
        if is_end {
            break;
        }
    }

    Ok(content)
}

pub fn load_skill_summaries_from_directory(dir_path: &Path) -> Vec<LoadedSkillSummary> {
    load_skill_summary_report_from_directory(dir_path).summaries
}

pub fn load_skill_summary_report_from_directory(dir_path: &Path) -> SkillSummaryLoadReport {
    let normalized_dir = dir_path
        .canonicalize()
        .unwrap_or_else(|_| dir_path.to_path_buf());
    let signature = skill_summary_root_signature(&normalized_dir);
    if let Some(cached) = read_cached_skill_summary_report(&normalized_dir, &signature) {
        return cached;
    }

    let report = load_skill_summary_report_from_directory_uncached(&normalized_dir);
    store_cached_skill_summary_report(normalized_dir, signature, &report);
    report
}

fn load_skill_summary_report_from_directory_uncached(dir_path: &Path) -> SkillSummaryLoadReport {
    let mut report = SkillSummaryLoadReport::default();

    if !dir_path.exists() {
        return report;
    }

    let entries = match std::fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(error) => {
            report.errors.push(SkillSummaryLoadError {
                path: dir_path.to_path_buf(),
                message: format!("读取 Skill 目录失败: {error}"),
            });
            return report;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.errors.push(SkillSummaryLoadError {
                    path: dir_path.to_path_buf(),
                    message: format!("读取 Skill 目录项失败: {error}"),
                });
                continue;
            }
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let skill_file = path.join("SKILL.md");
        if !skill_file.is_file() {
            continue;
        }

        let skill_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        match load_skill_summary_from_file(&skill_name, &skill_file) {
            Ok(skill) if skill.standard_compliance.validation_errors.is_empty() => {
                report.summaries.push(skill);
            }
            Ok(skill) => report.errors.push(SkillSummaryLoadError {
                path: skill_file,
                message: skill.standard_compliance.validation_errors.join("; "),
            }),
            Err(message) => report.errors.push(SkillSummaryLoadError {
                path: skill_file,
                message,
            }),
        }
    }

    report
}

fn validate_workflow_ref_for_summary(base_dir: &Path, frontmatter: &mut SkillFrontmatter) {
    let Some(workflow_ref) = frontmatter.workflow_ref.as_deref() else {
        return;
    };

    let workflow_ref = workflow_ref.trim();
    if workflow_ref.is_empty() {
        frontmatter
            .validation_errors
            .push("字段 `metadata.lime_workflow_ref` 不能为空".to_string());
        return;
    }

    match read_skill_relative_file_for_summary(base_dir, workflow_ref) {
        Ok(content) => {
            if let Err(error) = validate_workflow_content_for_summary(workflow_ref, &content) {
                frontmatter.validation_errors.push(format!(
                    "字段 `metadata.lime_workflow_ref` 校验失败: {error}"
                ));
            }
        }
        Err(error) => {
            frontmatter.validation_errors.push(format!(
                "字段 `metadata.lime_workflow_ref` 校验失败: {error}"
            ));
        }
    }
}

fn read_skill_relative_file_for_summary(
    skill_dir: &Path,
    relative_path: &str,
) -> Result<String, String> {
    let workflow_path = resolve_skill_relative_file_for_summary(skill_dir, relative_path)?;
    std::fs::read_to_string(&workflow_path).map_err(|error| {
        format!(
            "无法读取 workflow 引用文件 `{}`: {error}",
            workflow_path.display()
        )
    })
}

fn resolve_skill_relative_file_for_summary(
    skill_dir: &Path,
    relative_path: &str,
) -> Result<PathBuf, String> {
    let normalized = normalize_skill_relative_path_for_summary(relative_path)?;
    let candidate = skill_dir.join(&normalized);
    if !candidate.is_file() {
        return Err(format!("引用文件不存在: {}", normalized.display()));
    }

    let canonical_skill_dir = skill_dir
        .canonicalize()
        .map_err(|error| format!("无法解析 skill 包目录: {error}"))?;
    let canonical_candidate = candidate
        .canonicalize()
        .map_err(|error| format!("无法解析引用文件: {}: {error}", candidate.display()))?;
    if !canonical_candidate.starts_with(&canonical_skill_dir) {
        return Err("不能引用 skill 包外路径".to_string());
    }

    Ok(canonical_candidate)
}

fn normalize_skill_relative_path_for_summary(relative_path: &str) -> Result<PathBuf, String> {
    let relative_path = Path::new(relative_path);
    if relative_path.is_absolute() {
        return Err("必须引用 skill 包内的相对路径".to_string());
    }

    let mut normalized = PathBuf::new();
    for component in relative_path.components() {
        match component {
            Component::Normal(value) => normalized.push(value),
            Component::CurDir => {}
            _ => return Err("不能引用 skill 包外路径".to_string()),
        }
    }

    if normalized.as_os_str().is_empty() {
        return Err("引用文件路径不能为空".to_string());
    }

    Ok(normalized)
}

fn validate_workflow_content_for_summary(workflow_ref: &str, content: &str) -> Result<(), String> {
    let parsed = serde_yaml::from_str::<serde_yaml::Value>(content)
        .map_err(|error| format!("`{workflow_ref}` 无法解析为 JSON/YAML: {error}"))?;

    let is_valid = parsed.is_sequence()
        || parsed
            .as_mapping()
            .and_then(|mapping| {
                mapping
                    .get(serde_yaml::Value::String("steps".to_string()))
                    .and_then(|value| value.as_sequence())
            })
            .is_some();

    if !is_valid {
        return Err(format!(
            "`{workflow_ref}` 必须是数组，或包含数组字段 `steps` 的对象"
        ));
    }

    Ok(())
}

fn skill_summary_cache() -> &'static Mutex<HashMap<PathBuf, SkillSummaryCacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, SkillSummaryCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn read_cached_skill_summary_report(
    dir_path: &Path,
    signature: &str,
) -> Option<SkillSummaryLoadReport> {
    let guard = skill_summary_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let entry = guard.get(dir_path)?;
    (entry.signature == signature && entry.loaded_at.elapsed() < SKILL_SUMMARY_CACHE_TTL)
        .then(|| entry.report.clone())
}

fn store_cached_skill_summary_report(
    dir_path: PathBuf,
    signature: String,
    report: &SkillSummaryLoadReport,
) {
    let mut guard = skill_summary_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    guard.insert(
        dir_path,
        SkillSummaryCacheEntry {
            signature,
            loaded_at: Instant::now(),
            report: report.clone(),
        },
    );
}

pub(crate) fn invalidate_skill_summary_cache() {
    let mut guard = skill_summary_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    guard.clear();
}

fn skill_summary_root_signature(root: &Path) -> String {
    if !root.exists() {
        return "missing".to_string();
    }

    let mut parts = vec![format!("root={}", path_mtime(root).unwrap_or_default())];
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut child_signatures = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .filter_map(|path| {
                let skill_file = path.join("SKILL.md");
                let name = path.file_name()?.to_string_lossy().to_string();
                Some(format!(
                    "{name}:{}",
                    path_mtime(&skill_file).unwrap_or_default()
                ))
            })
            .collect::<Vec<_>>();
        child_signatures.sort();
        parts.extend(child_signatures);
    }
    parts.join(",")
}

fn path_mtime(path: &Path) -> Option<u128> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{load_skill_summaries_from_directory, load_skill_summary_report_from_directory};
    use tempfile::TempDir;

    #[test]
    fn summaries_skip_invalid_workflow_reference() {
        let temp_dir = TempDir::new().unwrap();
        let skills_dir = temp_dir.path();

        let valid_dir = skills_dir.join("summary-valid");
        std::fs::create_dir(&valid_dir).unwrap();
        std::fs::create_dir(valid_dir.join("references")).unwrap();
        std::fs::write(
            valid_dir.join("references/workflow.yaml"),
            r#"
steps:
  - id: draft
    name: Draft
    prompt: Write draft
"#,
        )
        .unwrap();
        std::fs::write(
            valid_dir.join("SKILL.md"),
            r#"---
name: summary-valid
description: Valid workflow skill
metadata:
  lime_workflow_ref: references/workflow.yaml
---

# Valid

Full body should not be needed for list summaries.
"#,
        )
        .unwrap();

        let invalid_dir = skills_dir.join("summary-invalid");
        std::fs::create_dir(&invalid_dir).unwrap();
        std::fs::write(
            invalid_dir.join("SKILL.md"),
            r#"---
name: summary-invalid
description: Invalid workflow skill
metadata:
  lime_workflow_ref: references/missing.yaml
---

# Invalid
"#,
        )
        .unwrap();

        let summaries = load_skill_summaries_from_directory(skills_dir);

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].skill_name, "summary-valid");
        assert_eq!(summaries[0].execution_mode, "workflow");
    }

    #[test]
    fn summaries_skip_skill_without_frontmatter() {
        let temp_dir = TempDir::new().unwrap();
        let skill_dir = temp_dir.path().join("broken");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "# Broken\n").unwrap();

        let summaries = load_skill_summaries_from_directory(temp_dir.path());

        assert!(summaries.is_empty());
    }

    #[test]
    fn report_preserves_invalid_skill_path_and_error() {
        let temp_dir = TempDir::new().unwrap();
        let skill_dir = temp_dir.path().join("broken");
        std::fs::create_dir(&skill_dir).unwrap();
        let skill_file = skill_dir.join("SKILL.md");
        std::fs::write(&skill_file, "# Broken\n").unwrap();

        let report = load_skill_summary_report_from_directory(temp_dir.path());

        assert!(report.summaries.is_empty());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(
            report.errors[0].path,
            skill_file.canonicalize().expect("canonical skill path")
        );
        assert!(report.errors[0].message.contains("frontmatter"));
    }
}
