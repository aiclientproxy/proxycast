//! Agent Skills 轻量 metadata 检索。
//!
//! 检索只用于候选排序和 prompt 预算裁剪，不读取 `SKILL.md` 正文，也不改变执行授权。

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::agent_snapshot::{AgentSkillMetadata, AgentSkillScope, AgentSkillSnapshot};

pub const DEFAULT_AGENT_SKILL_SEARCH_LIMIT: usize = 8;
const DEFAULT_MIN_SCORE: f32 = 0.34;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentSkillSearchOptions {
    pub limit: usize,
    pub min_score: f32,
}

impl Default for AgentSkillSearchOptions {
    fn default() -> Self {
        Self {
            limit: DEFAULT_AGENT_SKILL_SEARCH_LIMIT,
            min_score: DEFAULT_MIN_SCORE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSkillSearchResult {
    pub skill: AgentSkillMetadata,
    pub score: f32,
    pub matched_terms: Vec<String>,
    pub reason: String,
}

pub fn search_agent_skills(
    snapshot: &AgentSkillSnapshot,
    query: &str,
    options: AgentSkillSearchOptions,
) -> Vec<AgentSkillSearchResult> {
    let query_tokens = meaningful_tokens(query);
    if query_tokens.is_empty() || snapshot.skills.is_empty() || options.limit == 0 {
        return Vec::new();
    }

    let document_frequency = document_frequency(&snapshot.skills);
    let mut results = snapshot
        .skills
        .iter()
        .filter_map(|skill| {
            score_skill(
                skill,
                &query_tokens,
                &document_frequency,
                snapshot.skills.len(),
            )
            .filter(|score| score.score >= options.min_score)
            .map(|score| AgentSkillSearchResult {
                skill: skill.clone(),
                score: score.score,
                matched_terms: score.matched_terms,
                reason: score.reason,
            })
        })
        .collect::<Vec<_>>();

    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| scope_priority(right.skill.scope).cmp(&scope_priority(left.skill.scope)))
            .then_with(|| left.skill.name.cmp(&right.skill.name))
    });
    results.truncate(options.limit);
    results
}

pub fn reorder_agent_skill_snapshot_for_query(
    snapshot: &AgentSkillSnapshot,
    query: &str,
    options: AgentSkillSearchOptions,
) -> AgentSkillSnapshot {
    let results = search_agent_skills(snapshot, query, options);
    if results.is_empty() {
        return snapshot.clone();
    }

    let ranked_paths = results
        .iter()
        .map(|result| result.skill.skill_file_path.clone())
        .collect::<HashSet<_>>();
    let mut skills = results
        .into_iter()
        .map(|result| result.skill)
        .collect::<Vec<_>>();
    skills.extend(
        snapshot
            .skills
            .iter()
            .filter(|skill| !ranked_paths.contains(&skill.skill_file_path))
            .cloned(),
    );

    AgentSkillSnapshot {
        roots: snapshot.roots.clone(),
        skills,
    }
}

#[derive(Debug, Clone)]
struct SkillSearchScore {
    score: f32,
    matched_terms: Vec<String>,
    reason: String,
}

fn score_skill(
    skill: &AgentSkillMetadata,
    query_tokens: &[String],
    document_frequency: &HashMap<String, usize>,
    document_count: usize,
) -> Option<SkillSearchScore> {
    let fields = skill_search_fields(skill);
    let mut field_tokens = HashMap::<String, f32>::new();
    for (text, weight) in fields {
        for token in meaningful_tokens(&text) {
            field_tokens
                .entry(token)
                .and_modify(|current| *current = current.max(weight))
                .or_insert(weight);
        }
    }

    let mut matched_terms = Vec::new();
    let mut weighted_overlap = 0.0f32;
    for token in query_tokens {
        let Some(field_weight) = field_tokens.get(token) else {
            continue;
        };
        let df = *document_frequency.get(token).unwrap_or(&1) as f32;
        let idf = ((document_count as f32 + 1.0) / (df + 0.5)).ln().max(0.1);
        weighted_overlap += field_weight * idf;
        matched_terms.push(token.clone());
    }

    if matched_terms.is_empty() {
        return None;
    }

    matched_terms.sort();
    matched_terms.dedup();

    let query_coverage = matched_terms.len() as f32 / query_tokens.len().min(8) as f32;
    let field_coverage = matched_terms.len() as f32 / field_tokens.len().min(16) as f32;
    let alias_boost = stable_alias_tokens(skill)
        .into_iter()
        .any(|token| query_tokens.iter().any(|query_token| query_token == &token))
        .then_some(0.28)
        .unwrap_or(0.0);
    let score = (0.18
        + weighted_overlap * 0.18
        + query_coverage * 0.42
        + field_coverage * 0.22
        + alias_boost)
        .min(1.0);

    Some(SkillSearchScore {
        score,
        reason: format!("metadata 匹配 {}", matched_terms.join("、")),
        matched_terms,
    })
}

fn document_frequency(skills: &[AgentSkillMetadata]) -> HashMap<String, usize> {
    let mut frequency = HashMap::<String, usize>::new();
    for skill in skills {
        let tokens = skill_search_fields(skill)
            .into_iter()
            .flat_map(|(text, _)| meaningful_tokens(&text))
            .collect::<HashSet<_>>();
        for token in tokens {
            *frequency.entry(token).or_insert(0) += 1;
        }
    }
    frequency
}

fn skill_search_fields(skill: &AgentSkillMetadata) -> Vec<(String, f32)> {
    let mut fields = vec![
        (skill.name.clone(), 2.4),
        (skill.interface.display_name.clone(), 2.0),
        (skill.description.clone(), 1.4),
    ];
    if let Some(when_to_use) = skill.policy.when_to_use.as_ref() {
        fields.push((when_to_use.clone(), 1.5));
    }
    if let Some(argument_hint) = skill.interface.argument_hint.as_ref() {
        fields.push((argument_hint.clone(), 0.8));
    }
    if !skill.capabilities.is_empty() {
        fields.push((skill.capabilities.join(" "), 0.5));
    }
    fields
}

fn stable_alias_tokens(skill: &AgentSkillMetadata) -> Vec<String> {
    let mut aliases = meaningful_tokens(&skill.name);
    if let Some(directory_name) = skill.directory.file_name().and_then(|name| name.to_str()) {
        aliases.extend(meaningful_tokens(directory_name));
    }
    aliases.sort();
    aliases.dedup();
    aliases
}

fn meaningful_tokens(value: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "is", "are", "to", "of", "in", "for", "on", "with", "and", "or", "user",
        "need", "needs", "use", "using", "please", "help", "me", "my", "you", "skill", "agent",
        "local", "project", "的", "了", "和", "是", "在", "我", "你", "请", "帮我", "需要", "使用",
        "进行", "用户", "时", "内容", "当前", "这个", "那个", "文本", "文件",
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
                    | '_'
                    | '-'
            )
    }) {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        if token.chars().any(is_cjk) && token.chars().count() > 3 {
            let chars = token.chars().collect::<Vec<_>>();
            for window in chars.windows(2) {
                push_meaningful_token(&mut tokens, &window.iter().collect::<String>(), STOP_WORDS);
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

fn scope_priority(scope: AgentSkillScope) -> u8 {
    match scope {
        AgentSkillScope::Project => 3,
        AgentSkillScope::User => 2,
        AgentSkillScope::App => 1,
        AgentSkillScope::Orchestrator => 1,
        AgentSkillScope::Other => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_agent_skill_snapshot_from_roots, AgentSkillRoot, AgentSkillScope};
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn search_ranks_relevant_skill_from_metadata() {
        let root = TempDir::new().expect("root");
        write_skill(
            root.path(),
            "research",
            "Research",
            "联网信息检索与事实核验",
        );
        write_skill(root.path(), "writer", "Writer", "写作润色与语气调整");
        let snapshot = snapshot_from_root(root.path());

        let results = search_agent_skills(
            &snapshot,
            "帮我做最新事实核验和趋势调研",
            AgentSkillSearchOptions::default(),
        );

        assert_eq!(results[0].skill.name, "research");
        assert!(results[0].score >= DEFAULT_MIN_SCORE);
        assert!(results[0].matched_terms.iter().any(|term| term == "核验"));
    }

    #[test]
    fn search_respects_limit() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "research-a", "Research A", "事实核验 调研");
        write_skill(root.path(), "research-b", "Research B", "事实核验 调研");
        write_skill(root.path(), "research-c", "Research C", "事实核验 调研");
        let snapshot = snapshot_from_root(root.path());

        let results = search_agent_skills(
            &snapshot,
            "事实核验",
            AgentSkillSearchOptions {
                limit: 2,
                min_score: 0.1,
            },
        );

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_returns_empty_for_generic_query() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer", "写作润色");
        let snapshot = snapshot_from_root(root.path());

        let results = search_agent_skills(
            &snapshot,
            "请帮我处理这个内容",
            AgentSkillSearchOptions::default(),
        );

        assert!(results.is_empty());
    }

    #[test]
    fn reorder_places_relevant_skill_before_budget_tail() {
        let root = TempDir::new().expect("root");
        for index in 0..12 {
            write_skill(
                root.path(),
                &format!("generic-{index}"),
                &format!("Generic {index}"),
                "通用处理",
            );
        }
        write_skill(
            root.path(),
            "research",
            "Research",
            "联网信息检索与事实核验",
        );
        let snapshot = snapshot_from_root(root.path());

        let reordered = reorder_agent_skill_snapshot_for_query(
            &snapshot,
            "最新事实核验",
            AgentSkillSearchOptions::default(),
        );

        assert_eq!(reordered.skills[0].name, "research");
        assert_eq!(reordered.skills.len(), snapshot.skills.len());
    }

    fn snapshot_from_root(root: &Path) -> AgentSkillSnapshot {
        build_agent_skill_snapshot_from_roots([AgentSkillRoot {
            path: root.to_path_buf(),
            scope: AgentSkillScope::Project,
        }])
    }

    fn write_skill(root: &Path, name: &str, display_name: &str, description: &str) {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {display_name}\ndescription: {description}\n---\n\n# {display_name}"
            ),
        )
        .expect("skill");
    }
}
