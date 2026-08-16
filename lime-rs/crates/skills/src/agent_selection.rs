//! Agent Skill 显式选择解析。

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent_body::{
    agent_skill_body_locator_from_metadata, evaluate_agent_skill_body, AgentSkillBody,
    AgentSkillBodyBudgetDecision, AgentSkillBodyBudgetDecisionKind, AgentSkillBodyLocator,
};
use crate::agent_snapshot::{
    agent_skill_stable_id, AgentSkillAuthority, AgentSkillMetadata, AgentSkillScope,
    AgentSkillSnapshot, AgentSkillSource,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillSelection {
    pub locator: AgentSkillBodyLocator,
    pub trigger: AgentSkillSelectionTrigger,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillSelectionEvaluation {
    pub selection: AgentSkillSelection,
    pub skill_id: String,
    pub decision: AgentSkillBodyBudgetDecisionKind,
    pub reason: String,
    pub source: Option<AgentSkillSource>,
    pub authority: Option<AgentSkillAuthority>,
    pub required_capabilities: Vec<String>,
    pub missing_capabilities: Vec<String>,
    pub body_budget: AgentSkillBodyBudgetDecision,
    pub body: Option<AgentSkillBody>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSkillSelectionTrigger {
    DollarMention,
    SlashMention,
    AtMention,
    SkillUri,
    SkillFilePath,
    CatalogBinding,
    ExpertBinding,
    ImplicitHighConfidence,
}

pub fn evaluate_agent_skill_selection_bodies(
    selections: &[AgentSkillSelection],
    snapshot: &AgentSkillSnapshot,
    max_visible_tokens: u32,
) -> Vec<AgentSkillSelectionEvaluation> {
    let mut remaining_tokens = max_visible_tokens;
    selections
        .iter()
        .cloned()
        .map(|selection| {
            let skill_id = agent_skill_stable_id(selection.locator.scope, &selection.locator.name);
            let metadata = snapshot
                .skills
                .iter()
                .find(|skill| skill.skill_id == skill_id);
            let body_evaluation = evaluate_agent_skill_body(&selection.locator, remaining_tokens);
            if body_evaluation.budget.decision == AgentSkillBodyBudgetDecisionKind::Allow {
                remaining_tokens = remaining_tokens
                    .saturating_sub(body_evaluation.budget.estimated_tokens.unwrap_or_default());
            }
            let reason = match body_evaluation.budget.decision {
                AgentSkillBodyBudgetDecisionKind::Allow => "skill_selection_allowed",
                AgentSkillBodyBudgetDecisionKind::Deny => body_evaluation.budget.reason.as_str(),
                AgentSkillBodyBudgetDecisionKind::Omitted => {
                    "skill_selection_body_omitted_by_token_budget"
                }
            }
            .to_string();

            AgentSkillSelectionEvaluation {
                selection,
                skill_id,
                decision: body_evaluation.budget.decision,
                reason,
                source: metadata.map(|skill| skill.source),
                authority: metadata.map(|skill| skill.authority),
                required_capabilities: metadata
                    .map(|skill| skill.capabilities.clone())
                    .unwrap_or_default(),
                missing_capabilities: Vec::new(),
                body_budget: body_evaluation.budget,
                body: body_evaluation.body,
                error: body_evaluation.error,
            }
        })
        .collect()
}

pub fn select_explicit_agent_skills(
    user_input: &str,
    snapshot: &AgentSkillSnapshot,
) -> Vec<AgentSkillSelection> {
    let mut selections = Vec::new();
    let mut seen = HashSet::<String>::new();
    for candidate in explicit_candidates(user_input) {
        for selection in select_candidate(candidate, snapshot) {
            let key = agent_skill_stable_id(selection.locator.scope, &selection.locator.name);
            if seen.insert(key) {
                selections.push(selection);
            }
        }
    }
    selections
}

pub fn select_agent_skills_by_name_candidates(
    candidates: impl IntoIterator<Item = impl AsRef<str>>,
    snapshot: &AgentSkillSnapshot,
    trigger: AgentSkillSelectionTrigger,
) -> Vec<AgentSkillSelection> {
    let mut selections = Vec::new();
    let mut seen = HashSet::<String>::new();
    for candidate in candidates {
        let candidate = candidate.as_ref().trim();
        if candidate.is_empty() {
            continue;
        }
        let Some(skill) = find_unique_skill_by_name(snapshot, candidate) else {
            continue;
        };
        let selection = AgentSkillSelection {
            locator: agent_skill_body_locator_from_metadata(skill),
            trigger: trigger.clone(),
            reason: format!("结构化 metadata 绑定 skill `{candidate}`"),
        };
        let key = agent_skill_stable_id(selection.locator.scope, &selection.locator.name);
        if seen.insert(key) {
            selections.push(selection);
        }
    }
    selections
}

pub fn select_structured_agent_skill(
    name: &str,
    path: &Path,
    snapshot: &AgentSkillSnapshot,
) -> Option<AgentSkillSelection> {
    let skill = find_skill_by_path(snapshot, path)?;
    Some(AgentSkillSelection {
        locator: agent_skill_body_locator_from_metadata(skill),
        trigger: AgentSkillSelectionTrigger::SkillFilePath,
        reason: format!("结构化 Skill `{name}` 显式引用 {}", path.display()),
    })
}

pub fn select_mentioned_agent_skill(
    name: &str,
    path: &str,
    snapshot: &AgentSkillSnapshot,
) -> Option<AgentSkillSelection> {
    let (path, trigger) = if path.starts_with("skill://") {
        (parse_skill_uri(path)?, AgentSkillSelectionTrigger::SkillUri)
    } else {
        (
            parse_skill_file_path(path)?,
            AgentSkillSelectionTrigger::SkillFilePath,
        )
    };
    let mut selection = select_structured_agent_skill(name, &path, snapshot)?;
    selection.trigger = trigger;
    selection.reason = format!("结构化 Mention `{name}` 显式引用 {}", path.display());
    Some(selection)
}

pub fn select_implicit_agent_skills(
    user_input: &str,
    snapshot: &AgentSkillSnapshot,
) -> Vec<AgentSkillSelection> {
    let user_input = user_input.trim();
    if user_input.chars().count() < 4 {
        return Vec::new();
    }

    let mut scored = snapshot
        .skills
        .iter()
        .filter(|skill| skill.enabled && skill.policy.allow_implicit_invocation)
        .filter_map(|skill| implicit_skill_score(user_input, skill).map(|score| (skill, score)))
        .collect::<Vec<_>>();
    scored.sort_by(|(_, left), (_, right)| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or_else(|| std::cmp::Ordering::Equal)
    });
    scored.sort_by(|(left_skill, left_score), (right_skill, right_score)| {
        if (left_score.score - right_score.score).abs() < 0.16 {
            scope_priority(right_skill.scope).cmp(&scope_priority(left_skill.scope))
        } else {
            right_score
                .score
                .partial_cmp(&left_score.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    let Some((skill, top_score)) = scored.first() else {
        return Vec::new();
    };
    if top_score.score < 0.6 {
        return Vec::new();
    }
    if let Some((second_skill, second_score)) = scored.get(1) {
        let project_preferred = skill.scope == AgentSkillScope::Project
            && second_skill.scope != AgentSkillScope::Project
            && top_score.score - second_score.score >= 0.04;
        let project_scope_tiebreak = skill.scope == AgentSkillScope::Project
            && second_skill.scope != AgentSkillScope::Project
            && (top_score.score - second_score.score).abs() < 0.16;
        if !project_preferred
            && !project_scope_tiebreak
            && top_score.score - second_score.score < 0.16
        {
            return Vec::new();
        }
    }

    vec![AgentSkillSelection {
        locator: agent_skill_body_locator_from_metadata(skill),
        trigger: AgentSkillSelectionTrigger::ImplicitHighConfidence,
        reason: format!(
            "高置信匹配 Agent Skill `{}`: {}",
            skill.name, top_score.reason
        ),
    }]
}

fn scope_priority(scope: AgentSkillScope) -> u8 {
    match scope {
        AgentSkillScope::Project => 3,
        AgentSkillScope::User => 2,
        AgentSkillScope::App => 1,
        AgentSkillScope::Orchestrator => 1,
        AgentSkillScope::Other => 0,
    }
}

#[derive(Debug, Clone)]
struct ImplicitSkillScore {
    score: f32,
    reason: String,
}

fn implicit_skill_score(
    user_input: &str,
    skill: &AgentSkillMetadata,
) -> Option<ImplicitSkillScore> {
    let input_text = normalize_match_text(user_input);
    if input_text.is_empty() {
        return None;
    }

    let aliases = skill_name_aliases(skill);
    if aliases.iter().any(|alias| {
        let alias = normalize_match_text(alias);
        !alias.is_empty() && input_text.contains(&alias)
    }) {
        return Some(ImplicitSkillScore {
            score: 1.0,
            reason: "用户输入包含 skill 名称或稳定别名".to_string(),
        });
    }

    let mut skill_text = format!("{} {}", skill.description, skill.interface.display_name);
    if let Some(when_to_use) = skill.policy.when_to_use.as_ref() {
        skill_text.push(' ');
        skill_text.push_str(when_to_use);
    }
    let phrase_matches = cjk_phrases(&skill_text)
        .into_iter()
        .filter(|phrase| input_text.contains(phrase))
        .collect::<Vec<_>>();
    if !phrase_matches.is_empty() {
        return Some(ImplicitSkillScore {
            score: 0.92,
            reason: format!("匹配短语 {}", phrase_matches.join("、")),
        });
    }

    let skill_tokens = meaningful_tokens(&skill_text);
    if skill_tokens.is_empty() {
        return None;
    }
    let input_tokens = meaningful_tokens(user_input);
    if input_tokens.is_empty() {
        return None;
    }

    let input_set = input_tokens.into_iter().collect::<HashSet<_>>();
    let skill_set = skill_tokens.into_iter().collect::<HashSet<_>>();
    let matches = input_set
        .iter()
        .filter(|token| skill_set.contains(*token))
        .cloned()
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }

    let overlap = matches.len() as f32;
    let input_coverage = overlap / input_set.len().min(8) as f32;
    let skill_coverage = overlap / skill_set.len().min(16) as f32;
    let score = (0.22 + input_coverage * 0.55 + skill_coverage * 0.35).min(0.95);
    Some(ImplicitSkillScore {
        score,
        reason: format!("匹配关键词 {}", matches.join("、")),
    })
}

fn cjk_phrases(value: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    for raw in value.split(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                ',' | '，'
                    | '.'
                    | '。'
                    | ':'
                    | '：'
                    | ';'
                    | '；'
                    | '!'
                    | '！'
                    | '?'
                    | '？'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '/'
                    | '\\'
                    | '|'
            )
    }) {
        let phrase = raw.trim();
        let chars = phrase.chars().collect::<Vec<_>>();
        if chars.len() >= 4 && chars.iter().any(|ch| is_cjk(*ch)) {
            phrases.push(phrase.to_ascii_lowercase());
            for window in chars.windows(4) {
                if window.iter().all(|ch| is_cjk(*ch)) {
                    phrases.push(window.iter().collect::<String>().to_ascii_lowercase());
                }
            }
        }
    }
    phrases.sort();
    phrases.dedup();
    phrases
}

fn select_candidate(
    candidate: ExplicitSkillCandidate,
    snapshot: &AgentSkillSnapshot,
) -> Vec<AgentSkillSelection> {
    match candidate {
        ExplicitSkillCandidate::Name { name, trigger } => {
            find_unique_skill_by_name(snapshot, &name)
                .into_iter()
                .map(|skill| AgentSkillSelection {
                    locator: agent_skill_body_locator_from_metadata(skill),
                    trigger: trigger.clone(),
                    reason: format!("用户显式点名 skill `{name}`"),
                })
                .collect()
        }
        ExplicitSkillCandidate::Path { path, trigger } => find_skill_by_path(snapshot, &path)
            .into_iter()
            .map(|skill| AgentSkillSelection {
                locator: agent_skill_body_locator_from_metadata(skill),
                trigger: trigger.clone(),
                reason: format!("用户显式引用 SKILL.md: {}", path.display()),
            })
            .collect(),
    }
}

fn find_unique_skill_by_name<'a>(
    snapshot: &'a AgentSkillSnapshot,
    name: &str,
) -> Option<&'a AgentSkillMetadata> {
    let normalized = normalize_skill_name(name);
    let mut matches = snapshot.skills.iter().filter(|skill| {
        skill.enabled
            && skill_name_aliases(skill)
                .into_iter()
                .any(|alias| normalize_skill_name(&alias) == normalized)
    });
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(first)
}

fn skill_name_aliases(skill: &AgentSkillMetadata) -> Vec<String> {
    let mut aliases = Vec::new();
    push_stable_skill_alias(&mut aliases, &skill.name);
    push_display_skill_alias(&mut aliases, &skill.interface.display_name);
    if let Some(directory_name) = skill.directory.file_name().and_then(|name| name.to_str()) {
        push_stable_skill_alias(&mut aliases, directory_name);
    }
    aliases
}

fn push_stable_skill_alias(aliases: &mut Vec<String>, alias: &str) {
    let normalized = normalize_skill_name(alias);
    if normalized.is_empty() {
        return;
    }
    aliases.push(normalized.clone());
    aliases.push(format!("local:{normalized}"));
    aliases.push(format!("project:{normalized}"));
}

fn push_display_skill_alias(aliases: &mut Vec<String>, alias: &str) {
    let normalized = normalize_skill_name(alias);
    if !normalized.is_empty() {
        aliases.push(normalized);
    }
}

fn find_skill_by_path<'a>(
    snapshot: &'a AgentSkillSnapshot,
    path: &Path,
) -> Option<&'a AgentSkillMetadata> {
    if path.file_name().and_then(|name| name.to_str()) != Some("SKILL.md") {
        return None;
    }
    let normalized = normalize_path(path);
    snapshot
        .skills
        .iter()
        .find(|skill| skill.enabled && normalize_path(&skill.skill_file_path) == normalized)
}

fn explicit_candidates(user_input: &str) -> Vec<ExplicitSkillCandidate> {
    let mut candidates = Vec::new();
    for token in tokenize_user_input(user_input) {
        if let Some(name) = token.strip_prefix('$').and_then(normalize_mention_name) {
            candidates.push(ExplicitSkillCandidate::Name {
                name,
                trigger: AgentSkillSelectionTrigger::DollarMention,
            });
            continue;
        }
        if let Some(name) = token.strip_prefix('/').and_then(normalize_mention_name) {
            candidates.push(ExplicitSkillCandidate::Name {
                name,
                trigger: AgentSkillSelectionTrigger::SlashMention,
            });
            continue;
        }
        if let Some(name) = token.strip_prefix('@').and_then(normalize_mention_name) {
            candidates.push(ExplicitSkillCandidate::Name {
                name,
                trigger: AgentSkillSelectionTrigger::AtMention,
            });
            continue;
        }
        if let Some(path) = parse_skill_uri(&token) {
            candidates.push(ExplicitSkillCandidate::Path {
                path,
                trigger: AgentSkillSelectionTrigger::SkillUri,
            });
            continue;
        }
        if let Some(path) = parse_skill_file_path(&token) {
            candidates.push(ExplicitSkillCandidate::Path {
                path,
                trigger: AgentSkillSelectionTrigger::SkillFilePath,
            });
        }
    }
    candidates
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExplicitSkillCandidate {
    Name {
        name: String,
        trigger: AgentSkillSelectionTrigger,
    },
    Path {
        path: PathBuf,
        trigger: AgentSkillSelectionTrigger,
    },
}

fn tokenize_user_input(user_input: &str) -> Vec<String> {
    user_input
        .replace("](", " ")
        .replace(')', " ")
        .split_whitespace()
        .map(trim_token)
        .filter(|token| !token.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn trim_token(token: &str) -> &str {
    token.trim_matches(|ch: char| {
        matches!(
            ch,
            ',' | '，'
                | '.'
                | '。'
                | ':'
                | '：'
                | ';'
                | '；'
                | '!'
                | '！'
                | '?'
                | '？'
                | '"'
                | '\''
                | '`'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
        )
    })
}

fn normalize_mention_name(value: &str) -> Option<String> {
    let value = value.trim_matches(|ch: char| {
        matches!(
            ch,
            ',' | '，' | '.' | '。' | ':' | '：' | ';' | '；' | '!' | '！' | '?' | '？'
        )
    });
    if value.is_empty() || !is_skill_name_like(value) {
        return None;
    }
    Some(value.to_string())
}

fn is_skill_name_like(value: &str) -> bool {
    value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

fn parse_skill_uri(token: &str) -> Option<PathBuf> {
    let payload = token.strip_prefix("skill://")?;
    if payload.is_empty() {
        return None;
    }
    if token.ends_with("SKILL.md")
        && url::Url::parse(token)
            .ok()
            .and_then(|url| url.host_str().map(ToOwned::to_owned))
            .is_some_and(|host| !host.contains('%'))
    {
        return Some(PathBuf::from(token));
    }
    let decoded = urlencoding::decode(payload).ok()?.into_owned();
    parse_skill_file_path(&decoded)
}

fn parse_skill_file_path(token: &str) -> Option<PathBuf> {
    if !token.ends_with("SKILL.md") {
        return None;
    }
    let path = PathBuf::from(token);
    if path.file_name().and_then(|name| name.to_str()) != Some("SKILL.md") {
        return None;
    }
    Some(path)
}

fn normalize_skill_name(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn normalize_match_text(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn meaningful_tokens(value: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "is", "are", "to", "of", "in", "for", "on", "with", "and", "or", "user",
        "need", "needs", "use", "using", "please", "help", "me", "my", "you", "的", "了", "和",
        "是", "在", "我", "你", "请", "帮我", "需要", "使用", "进行", "用户", "时", "内容", "当前",
        "这个", "那个", "文本", "文件",
    ];

    let lower = value.to_ascii_lowercase();
    let mut tokens = Vec::new();
    for raw in lower.split(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                ',' | '，'
                    | '.'
                    | '。'
                    | ':'
                    | '：'
                    | ';'
                    | '；'
                    | '!'
                    | '！'
                    | '?'
                    | '？'
                    | '"'
                    | '\''
                    | '`'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
                    | '/'
                    | '\\'
                    | '|'
            )
    }) {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        if token.chars().any(is_cjk) && token.chars().count() > 3 {
            let chars = token.chars().collect::<Vec<_>>();
            for window in chars.windows(2) {
                let bigram = window.iter().collect::<String>();
                push_meaningful_token(&mut tokens, &bigram, STOP_WORDS);
            }
            continue;
        }
        push_meaningful_token(&mut tokens, token, STOP_WORDS);
    }
    tokens.sort();
    tokens.dedup();
    tokens
}

fn push_meaningful_token(tokens: &mut Vec<String>, token: &str, stop_words: &[&str]) {
    let token = token.trim();
    if token.chars().count() < 2 || stop_words.contains(&token) {
        return;
    }
    tokens.push(token.to_string());
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch,
        '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}'
    )
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_agent_skill_snapshot_from_roots, AgentSkillRoot, AgentSkillScope};
    use tempfile::TempDir;

    #[test]
    fn selects_unique_skill_from_dollar_mention() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_explicit_agent_skills("请用 $writer 改写", &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "writer");
        assert_eq!(
            selections[0].trigger,
            AgentSkillSelectionTrigger::DollarMention
        );
    }

    #[test]
    fn selects_unique_skill_from_slash_mention() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "summary", "Summary");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_explicit_agent_skills("/summary 这段内容", &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "summary");
        assert_eq!(
            selections[0].trigger,
            AgentSkillSelectionTrigger::SlashMention
        );
    }

    #[test]
    fn selects_unique_skill_from_at_mention() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_explicit_agent_skills("请用 @writer 改写", &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "writer");
        assert_eq!(selections[0].trigger, AgentSkillSelectionTrigger::AtMention);
    }

    #[test]
    fn selects_skill_from_local_skill_file_path() {
        let root = TempDir::new().expect("root");
        let skill_file = write_skill(root.path(), "analysis", "Analysis");
        let snapshot = snapshot_from_root(root.path());

        let selections =
            select_explicit_agent_skills(&format!("读取 {}", skill_file.display()), &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "analysis");
        assert_eq!(
            selections[0].trigger,
            AgentSkillSelectionTrigger::SkillFilePath
        );
    }

    #[test]
    fn structured_selection_requires_an_enabled_snapshot_path() {
        let root = TempDir::new().expect("root");
        let skill_file = write_skill(root.path(), "analysis", "Analysis");
        let mut snapshot = snapshot_from_root(root.path());

        let selection = select_structured_agent_skill("wire-name", &skill_file, &snapshot)
            .expect("structured selection");
        assert_eq!(selection.locator.name, "analysis");
        assert_eq!(
            selection.locator.skill_file_path,
            std::fs::canonicalize(&skill_file).expect("canonical skill path")
        );

        snapshot.skills[0].enabled = false;
        assert!(select_structured_agent_skill("analysis", &skill_file, &snapshot).is_none());
        assert!(select_structured_agent_skill(
            "analysis",
            &root.path().join("missing/SKILL.md"),
            &snapshot,
        )
        .is_none());
    }

    #[test]
    fn selects_skill_from_skill_uri() {
        let root = TempDir::new().expect("root");
        let skill_file = write_skill(root.path(), "research", "Research");
        let snapshot = snapshot_from_root(root.path());
        let skill_file_text = skill_file.to_string_lossy();
        let encoded = urlencoding::encode(&skill_file_text);

        let selections = select_explicit_agent_skills(&format!("用 skill://{encoded}"), &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "research");
        assert_eq!(selections[0].trigger, AgentSkillSelectionTrigger::SkillUri);
    }

    #[test]
    fn selects_skill_from_markdown_skill_uri_link() {
        let root = TempDir::new().expect("root");
        let skill_file = write_skill(root.path(), "research", "Research");
        let snapshot = snapshot_from_root(root.path());
        let skill_file_text = skill_file.to_string_lossy();
        let encoded = urlencoding::encode(&skill_file_text);

        let selections =
            select_explicit_agent_skills(&format!("用 [research](skill://{encoded})"), &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "research");
        assert_eq!(selections[0].trigger, AgentSkillSelectionTrigger::SkillUri);
    }

    #[test]
    fn explicit_selection_deduplicates_same_stable_skill_across_project_roots() {
        let first = TempDir::new().expect("first");
        let second = TempDir::new().expect("second");
        let first_skill = write_skill(first.path(), "research", "Research");
        let second_skill = write_skill(second.path(), "research", "Research");
        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: first.path().to_path_buf(),
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: second.path().to_path_buf(),
                scope: AgentSkillScope::Project,
            },
        ]);

        let selections = select_explicit_agent_skills(
            &format!("{} {}", first_skill.display(), second_skill.display()),
            &snapshot,
        );

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "research");
    }

    #[test]
    fn selects_unique_skill_from_structured_catalog_candidates() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_agent_skills_by_name_candidates(
            ["local:writer", "project:writer", "writer"],
            &snapshot,
            AgentSkillSelectionTrigger::CatalogBinding,
        );

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "writer");
        assert_eq!(
            selections[0].trigger,
            AgentSkillSelectionTrigger::CatalogBinding
        );
    }

    #[test]
    fn structured_candidates_do_not_select_unknown_skill() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_agent_skills_by_name_candidates(
            ["service-skill-writer"],
            &snapshot,
            AgentSkillSelectionTrigger::CatalogBinding,
        );

        assert!(selections.is_empty());
    }

    #[test]
    fn duplicate_skill_name_is_not_auto_selected() {
        let first = TempDir::new().expect("first");
        let second = TempDir::new().expect("second");
        write_skill(first.path(), "writer", "Writer");
        write_skill(second.path(), "writer", "Writer");
        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: first.path().to_path_buf(),
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: second.path().to_path_buf(),
                scope: AgentSkillScope::User,
            },
        ]);

        let selections = select_explicit_agent_skills("$writer", &snapshot);

        assert!(selections.is_empty());
    }

    #[test]
    fn implicit_selection_loads_unique_high_confidence_skill() {
        let root = TempDir::new().expect("root");
        write_skill_with_description(
            root.path(),
            "research",
            "Research",
            "联网信息检索与事实核验",
        );
        write_skill_with_description(root.path(), "writer", "Writer", "写作润色");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_implicit_agent_skills("帮我做最新事实核验和趋势调研", &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "research");
        assert_eq!(
            selections[0].trigger,
            AgentSkillSelectionTrigger::ImplicitHighConfidence
        );
    }

    #[test]
    fn implicit_selection_ignores_ambiguous_candidates() {
        let root = TempDir::new().expect("root");
        write_skill_with_description(root.path(), "summary", "Summary", "总结提炼文本重点");
        write_skill_with_description(root.path(), "analysis", "Analysis", "分析文本重点");
        let snapshot = snapshot_from_root(root.path());

        let selections = select_implicit_agent_skills("处理文本重点", &snapshot);

        assert!(selections.is_empty());
    }

    #[test]
    fn implicit_selection_prefers_project_candidate_when_scores_are_close() {
        let project = TempDir::new().expect("project");
        let app = TempDir::new().expect("app");
        write_skill_with_description(
            project.path(),
            "fact-check-lab",
            "Fact Check Lab",
            "联网信息检索与事实核验",
        );
        write_skill_with_description(app.path(), "research", "Research", "事实核验与趋势调研");
        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: project.path().to_path_buf(),
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: app.path().to_path_buf(),
                scope: AgentSkillScope::App,
            },
        ]);

        let selections = select_implicit_agent_skills("帮我做最新事实核验和趋势调研", &snapshot);

        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].locator.name, "fact-check-lab");
    }

    #[test]
    fn selection_skips_disabled_skill_metadata() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer");
        let mut snapshot = snapshot_from_root(root.path());
        snapshot.skills[0].enabled = false;
        snapshot.skills[0].policy.allow_implicit_invocation = false;

        assert!(select_explicit_agent_skills("$writer", &snapshot).is_empty());
        assert!(select_implicit_agent_skills("writer rewrite", &snapshot).is_empty());
    }

    #[test]
    fn selection_body_budget_uses_real_bodies_and_omits_after_budget_is_consumed() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer");
        write_skill(root.path(), "summary", "Summary");
        let snapshot = snapshot_from_root(root.path());
        let selections = select_agent_skills_by_name_candidates(
            ["writer", "summary"],
            &snapshot,
            AgentSkillSelectionTrigger::CatalogBinding,
        );
        let first_budget = evaluate_agent_skill_body(&selections[0].locator, u32::MAX)
            .budget
            .estimated_tokens
            .expect("first body tokens");

        let evaluations =
            evaluate_agent_skill_selection_bodies(&selections, &snapshot, first_budget);

        assert_eq!(evaluations.len(), 2);
        assert_eq!(
            evaluations[0].decision,
            AgentSkillBodyBudgetDecisionKind::Allow
        );
        assert_eq!(evaluations[0].reason, "skill_selection_allowed");
        assert_eq!(evaluations[0].skill_id, "project:writer");
        assert_eq!(evaluations[0].source, Some(AgentSkillSource::Project));
        assert_eq!(
            evaluations[0].authority,
            Some(AgentSkillAuthority::Workspace)
        );
        assert_eq!(
            evaluations[1].decision,
            AgentSkillBodyBudgetDecisionKind::Omitted
        );
        assert_eq!(
            evaluations[1].reason,
            "skill_selection_body_omitted_by_token_budget"
        );
    }

    fn snapshot_from_root(root: &Path) -> AgentSkillSnapshot {
        build_agent_skill_snapshot_from_roots([AgentSkillRoot {
            path: root.to_path_buf(),
            scope: AgentSkillScope::Project,
        }])
    }

    fn write_skill(root: &Path, name: &str, display_name: &str) -> PathBuf {
        write_skill_with_description(root, name, display_name, "Test skill")
    }

    fn write_skill_with_description(
        root: &Path,
        name: &str,
        display_name: &str,
        description: &str,
    ) -> PathBuf {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        let skill_file = skill_dir.join("SKILL.md");
        std::fs::write(
            &skill_file,
            format!("---\nname: {display_name}\ndescription: {description}\n---\n\n# Body"),
        )
        .expect("skill");
        skill_file
    }
}
