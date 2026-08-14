use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
    RuntimeToolPolicyErrorKind, RuntimeToolTurnContext,
};
use lime_skills::{
    build_agent_skill_snapshot_from_workspace, search_agent_skills, AgentSkillMetadata,
    AgentSkillSearchOptions, AgentSkillSearchResult, AgentSkillSnapshot,
    DEFAULT_AGENT_SKILL_SEARCH_LIMIT, SKILL_SNAPSHOT_TURN_METADATA_KEY,
};
use serde::Deserialize;
use serde_json::{json, Map as JsonMap, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub const SKILL_SEARCH_TOOL_NAME: &str = "skill_search";
pub const MAX_SKILL_SEARCH_LIMIT: usize = 20;

#[derive(Debug, Default)]
pub struct RuntimeSkillSearchExecutor;

impl RuntimeSkillSearchExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn handle() -> RuntimeToolExecutorHandle {
        RuntimeToolExecutorHandle::new(Arc::new(Self::new()))
    }

    async fn execute_skill_search(
        &self,
        request: RuntimeToolExecutionRequest<'_>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        if request.tool_name != SKILL_SEARCH_TOOL_NAME {
            return Err(runtime_skill_search_error(format!(
                "skill_search executor cannot run tool '{}'",
                request.tool_name
            )));
        }
        if request
            .context
            .cancel_token()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(runtime_skill_search_error("skill_search cancelled"));
        }

        let input = parse_input(request.params.clone())?;
        let workspace = resolve_skill_search_workspace(
            request.context.working_directory(),
            request.turn_context,
        );
        let snapshot =
            skill_snapshot_from_turn_context(request.turn_context).unwrap_or_else(|| {
                build_agent_skill_snapshot_from_workspace(
                    workspace.working_dir.as_deref(),
                    workspace.project_root.as_deref(),
                )
            });
        let results = search_agent_skills(
            &snapshot,
            &input.query,
            AgentSkillSearchOptions {
                limit: input.limit,
                ..AgentSkillSearchOptions::default()
            },
        );
        let output = skill_search_output(&input.query, &workspace, snapshot.skills.len(), &results);
        let output_text = serde_json::to_string_pretty(&output).map_err(|error| {
            runtime_skill_search_error(format!("序列化 skill_search 结果失败: {error}"))
        })?;

        Ok(RuntimeToolExecutionResult::new(
            true,
            output_text,
            None,
            skill_search_metadata(&input.query, snapshot.skills.len(), results.len()),
        ))
    }
}

fn skill_snapshot_from_turn_context(
    turn_context: Option<&RuntimeToolTurnContext>,
) -> Option<AgentSkillSnapshot> {
    let value = turn_context?
        .metadata
        .get(SKILL_SNAPSHOT_TURN_METADATA_KEY)?;
    match serde_json::from_value(value.clone()) {
        Ok(snapshot) => Some(snapshot),
        Err(error) => {
            tracing::warn!(error = %error, "忽略无法解析的 skill_search turn snapshot");
            None
        }
    }
}

impl RuntimeToolExecutor for RuntimeSkillSearchExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move { self.execute_skill_search(request).await })
    }
}

pub fn runtime_skill_search_executor_handle() -> RuntimeToolExecutorHandle {
    static HANDLE: OnceLock<RuntimeToolExecutorHandle> = OnceLock::new();
    HANDLE
        .get_or_init(RuntimeSkillSearchExecutor::handle)
        .clone()
}

pub fn skill_search_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        SKILL_SEARCH_TOOL_NAME,
        "Search available Agent Skills by lightweight metadata. For expert-bound or workspace-enabled skill candidates, call this before Skill so the selector evidence is recorded. Returns matching skill names, scopes, locators, and reasons only; it does not read SKILL.md bodies, enable SkillTool, or expand tool permissions.",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language task or keywords to match against Agent Skill metadata."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_SKILL_SEARCH_LIMIT,
                    "description": "Maximum number of metadata matches to return. Defaults to 8."
                }
            },
            "required": ["query"]
        }),
    )
}

pub fn check_runtime_skill_search_permissions(
    params: &Value,
) -> Result<(), RuntimeToolExecutionError> {
    parse_input(params.clone()).map(|_| ())
}

#[derive(Debug, Deserialize)]
struct SkillSearchInput {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Clone)]
struct ParsedSkillSearchInput {
    query: String,
    limit: usize,
}

#[derive(Debug, Clone, Default)]
struct SkillSearchWorkspace {
    working_dir: Option<PathBuf>,
    project_root: Option<PathBuf>,
}

fn parse_input(params: Value) -> Result<ParsedSkillSearchInput, RuntimeToolExecutionError> {
    let input: SkillSearchInput = serde_json::from_value(params)
        .map_err(|error| runtime_skill_search_error(format!("skill_search 参数无效: {error}")))?;
    let query = input.query.trim().to_string();
    if query.chars().count() < 2 {
        return Err(runtime_skill_search_error("query 至少需要 2 个非空字符"));
    }
    let limit = input
        .limit
        .unwrap_or(DEFAULT_AGENT_SKILL_SEARCH_LIMIT)
        .clamp(1, MAX_SKILL_SEARCH_LIMIT);

    Ok(ParsedSkillSearchInput { query, limit })
}

fn resolve_skill_search_workspace(
    working_directory: &Path,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> SkillSearchWorkspace {
    let metadata = turn_context.map(|turn_context| &turn_context.metadata);
    let project_root = turn_context
        .and_then(|turn_context| {
            turn_context_metadata_path(&turn_context.metadata, PROJECT_ROOT_KEYS)
        })
        .or_else(|| metadata.and_then(|metadata| metadata_path(metadata, PROJECT_ROOT_POINTERS)));
    let working_dir = turn_context
        .and_then(|turn_context| turn_context.cwd.clone())
        .or_else(|| {
            turn_context.and_then(|turn_context| {
                turn_context_metadata_path(&turn_context.metadata, WORKING_DIR_KEYS)
            })
        })
        .or_else(|| metadata.and_then(|metadata| metadata_path(metadata, WORKING_DIR_POINTERS)))
        .or_else(|| Some(working_directory.to_path_buf()));

    SkillSearchWorkspace {
        working_dir: normalize_existing_path(working_dir),
        project_root: normalize_existing_path(project_root),
    }
}

const PROJECT_ROOT_KEYS: &[&str] = &["project_root", "projectRoot"];
const WORKING_DIR_KEYS: &[&str] = &["working_directory", "workingDirectory", "cwd"];
const PROJECT_ROOT_POINTERS: &[&str] = &[
    "/harness/project_root",
    "/harness/projectRoot",
    "/project_root",
    "/projectRoot",
];
const WORKING_DIR_POINTERS: &[&str] = &[
    "/harness/working_directory",
    "/harness/workingDirectory",
    "/harness/cwd",
    "/working_directory",
    "/workingDirectory",
    "/cwd",
];

fn turn_context_metadata_path(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<PathBuf> {
    keys.iter()
        .filter_map(|key| metadata.get(*key))
        .find_map(value_to_path)
}

fn metadata_path(metadata: &HashMap<String, Value>, pointers: &[&str]) -> Option<PathBuf> {
    let value = metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<JsonMap<String, Value>>();
    let value = Value::Object(value);
    pointers
        .iter()
        .filter_map(|pointer| value.pointer(pointer))
        .find_map(value_to_path)
}

fn value_to_path(value: &Value) -> Option<PathBuf> {
    value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn normalize_existing_path(path: Option<PathBuf>) -> Option<PathBuf> {
    let path = path?;
    Some(path.canonicalize().unwrap_or(path))
}

fn skill_search_output(
    query: &str,
    workspace: &SkillSearchWorkspace,
    snapshot_skill_count: usize,
    results: &[AgentSkillSearchResult],
) -> Value {
    json!({
        "query": query,
        "resultCount": results.len(),
        "snapshotSkillCount": snapshot_skill_count,
        "workingDirectory": workspace.working_dir.as_ref().map(|path| path.display().to_string()),
        "projectRoot": workspace.project_root.as_ref().map(|path| path.display().to_string()),
        "results": results
            .iter()
            .map(skill_search_result_value)
            .collect::<Vec<_>>(),
        "usage": "Search results are metadata only. Local skills use SKILL.md; scope=orchestrator skill:// locators must be read with read_mcp_resource from codex_apps. Do not treat this search as execution approval."
    })
}

fn skill_search_result_value(result: &AgentSkillSearchResult) -> Value {
    let skill = &result.skill;
    json!({
        "skillId": skill.stable_id(),
        "name": skill.name,
        "displayName": skill.interface.display_name,
        "description": skill.description,
        "scope": skill.scope.as_label(),
        "source": skill.source.as_label(),
        "authority": skill.authority.as_label(),
        "enabled": skill.enabled,
        "directory": path_string(&skill.directory),
        "skillFilePath": path_string(&skill.skill_file_path),
        "locator": skill_locator_value(skill),
        "declaredTools": skill.capabilities,
        "dependencies": skill.dependencies,
        "executionMode": skill.interface.execution_mode,
        "argumentHint": skill.interface.argument_hint,
        "whenToUse": skill.policy.when_to_use,
        "score": result.score,
        "matchedTerms": result.matched_terms,
        "reason": result.reason,
    })
}

fn skill_locator_value(skill: &AgentSkillMetadata) -> Value {
    json!({
        "skillId": skill.stable_id(),
        "source": skill.source.as_label(),
        "authority": skill.authority.as_label(),
        "name": skill.name,
        "directory": path_string(&skill.directory),
        "skillFilePath": path_string(&skill.skill_file_path),
    })
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}

fn skill_search_metadata(
    query: &str,
    snapshot_skill_count: usize,
    result_count: usize,
) -> HashMap<String, Value> {
    HashMap::from([
        ("tool_family".to_string(), json!("skill_search")),
        ("skill_search_query".to_string(), json!(query)),
        (
            "skill_search_snapshot_skill_count".to_string(),
            json!(snapshot_skill_count),
        ),
        ("skill_search_result_count".to_string(), json!(result_count)),
    ])
}

fn runtime_skill_search_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    let message = message.into();
    RuntimeToolExecutionError::new(
        message.clone(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(message)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use agent_protocol::turn_context::TurnContextOverride;
    use serde_json::json;
    use tempfile::TempDir;

    fn context(path: PathBuf) -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: path,
            session_id: "test-session".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
    }

    #[tokio::test]
    async fn search_returns_metadata_without_body_or_gate_metadata() {
        let root = TempDir::new().expect("root");
        let skills_root = root.path().join(".agents").join("skills");
        write_skill(
            &skills_root,
            "research",
            "Research",
            "uniquefactsprobe 事实核验与联网调研",
        );
        write_skill(&skills_root, "writer", "Writer", "写作润色");
        let context = context(root.path().to_path_buf());
        let params = json!({ "query": "uniquefactsprobe", "limit": 3 });

        let result = runtime_skill_search_executor_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: SKILL_SEARCH_TOOL_NAME,
                params: &params,
                context: &context,
                turn_context: None,
            })
            .await
            .expect("search should work");

        assert!(result.success);
        let output: Value = serde_json::from_str(&result.output).expect("json output");
        let results = output["results"].as_array().expect("results");
        let research = results
            .iter()
            .find(|result| result["name"] == "research")
            .expect("research result should exist");
        assert_eq!(research["skillId"], "project:research");
        assert_eq!(research["locator"]["skillId"], "project:research");
        assert_eq!(research["displayName"], "Research");
        let expected_skill_file = skills_root
            .join("research")
            .join("SKILL.md")
            .canonicalize()
            .unwrap();
        assert_eq!(
            research["locator"]["skillFilePath"],
            json!(expected_skill_file.display().to_string())
        );
        assert!(result.metadata.contains_key("tool_family"));
        assert!(!result.metadata.contains_key("skill_name"));
        assert!(!result
            .metadata
            .contains_key("workspace_skill_runtime_enable"));
        assert!(!output.to_string().contains("SECRET BODY"));
    }

    #[tokio::test]
    async fn search_uses_turn_context_project_root_when_present() {
        let project = TempDir::new().expect("project");
        let nested = project.path().join("nested");
        std::fs::create_dir_all(&nested).expect("nested");
        let skills_root = project.path().join(".agents").join("skills");
        write_skill(
            &skills_root,
            "translate",
            "Translate",
            "uniquetranslateprobe 中英翻译",
        );
        let context = context(nested);
        let params = json!({ "query": "翻译" });
        let turn_context = TurnContextOverride {
            metadata: HashMap::from([(
                "project_root".to_string(),
                json!(project.path().display().to_string()),
            )]),
            ..TurnContextOverride::default()
        };

        let result = runtime_skill_search_executor_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: SKILL_SEARCH_TOOL_NAME,
                params: &params,
                context: &context,
                turn_context: Some(&turn_context),
            })
            .await
            .expect("search should work");

        let output: Value = serde_json::from_str(&result.output).expect("json output");
        assert_eq!(output["results"][0]["name"], "translate");
        assert_eq!(output["results"][0]["displayName"], "Translate");
        assert_eq!(
            output["projectRoot"],
            json!(project.path().canonicalize().unwrap().display().to_string())
        );
    }

    #[tokio::test]
    async fn search_consumes_orchestrator_skill_from_frozen_turn_snapshot() {
        let root = TempDir::new().expect("root");
        let context = context(root.path().to_path_buf());
        let params = json!({ "query": "release notes" });
        let turn_context = TurnContextOverride {
            metadata: HashMap::from([(
                SKILL_SNAPSHOT_TURN_METADATA_KEY.to_string(),
                json!({
                    "roots": [],
                    "skills": [{
                        "skill_id": "orchestrator:delivery:release-notes",
                        "name": "delivery:release-notes",
                        "description": "Draft source-backed release notes.",
                        "scope": "orchestrator",
                        "source": "orchestrator",
                        "authority": "orchestrator",
                        "enabled": true,
                        "interface": {
                            "display_name": "delivery:release-notes",
                            "execution_mode": "mcp_resource",
                            "provider": "codex_apps",
                            "model": null,
                            "argument_hint": null
                        },
                        "dependencies": { "tools": [] },
                        "policy": {
                            "allow_implicit_invocation": true,
                            "when_to_use": null
                        },
                        "capabilities": [],
                        "directory": "skill://delivery/release-notes",
                        "skill_file_path": "skill://delivery/release-notes/SKILL.md"
                    }]
                }),
            )]),
            ..TurnContextOverride::default()
        };

        let result = runtime_skill_search_executor_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: SKILL_SEARCH_TOOL_NAME,
                params: &params,
                context: &context,
                turn_context: Some(&turn_context),
            })
            .await
            .expect("orchestrator skill search");

        let output: Value = serde_json::from_str(&result.output).expect("json output");
        assert_eq!(output["snapshotSkillCount"], 1);
        assert_eq!(output["results"][0]["scope"], "orchestrator");
        assert_eq!(output["results"][0]["source"], "orchestrator");
        assert_eq!(output["results"][0]["authority"], "orchestrator");
        assert_eq!(
            output["results"][0]["locator"]["skillFilePath"],
            "skill://delivery/release-notes/SKILL.md"
        );
        assert!(output["usage"]
            .as_str()
            .expect("usage")
            .contains("read_mcp_resource"));
    }

    #[test]
    fn permission_check_rejects_empty_query() {
        let result = check_runtime_skill_search_permissions(&json!({ "query": "" }));

        assert!(result.is_err());
    }

    fn write_skill(root: &Path, name: &str, display_name: &str, description: &str) {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {display_name}\ndescription: {description}\n---\n\n# {display_name}\nSECRET BODY"
            ),
        )
        .expect("skill");
    }
}
