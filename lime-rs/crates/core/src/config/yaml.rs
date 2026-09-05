//! YAML 配置文件支持
//!
//! 提供 YAML 配置的加载、保存和管理功能
//! 支持保留注释的配置保存

#![allow(dead_code)]

use super::types::Config;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

const CONFIG_PATH_ENV: &str = "LIME_CONFIG_PATH";

/// 配置错误类型
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// 文件读取错误
    ReadError(String),
    /// 文件写入错误
    WriteError(String),
    /// YAML 解析错误
    ParseError(String),
    /// YAML 序列化错误
    SerializeError(String),
    /// 配置验证错误
    ValidationError(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::ReadError(msg) => write!(f, "配置读取错误: {msg}"),
            ConfigError::WriteError(msg) => write!(f, "配置写入错误: {msg}"),
            ConfigError::ParseError(msg) => write!(f, "YAML 解析错误: {msg}"),
            ConfigError::SerializeError(msg) => write!(f, "YAML 序列化错误: {msg}"),
            ConfigError::ValidationError(msg) => write!(f, "配置验证错误: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// 配置管理器
///
/// 管理 YAML 配置文件的加载、保存和热重载
#[derive(Debug)]
pub struct ConfigManager {
    /// 当前配置
    config: Config,
    /// 配置文件路径
    config_path: PathBuf,
}

impl ConfigManager {
    /// 创建新的配置管理器
    pub fn new(config_path: PathBuf) -> Self {
        Self {
            config: Config::default(),
            config_path,
        }
    }

    /// 使用指定配置创建配置管理器
    pub fn with_config(config: Config, config_path: PathBuf) -> Self {
        Self {
            config,
            config_path,
        }
    }

    /// 从文件加载配置
    ///
    /// 如果文件不存在，返回默认配置
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let mut config = if path.exists() {
            let content =
                std::fs::read_to_string(path).map_err(|e| ConfigError::ReadError(e.to_string()))?;
            Self::parse_yaml(&content)?
        } else {
            Config::default()
        };
        config.normalize_local_server_surface();
        config.normalize_workspace_preferences();

        Ok(Self {
            config,
            config_path: path.to_path_buf(),
        })
    }

    /// 从 YAML 字符串解析配置
    pub fn parse_yaml(yaml: &str) -> Result<Config, ConfigError> {
        parse_yaml_config_with_legacy_tracking(yaml).map(|(config, _)| config)
    }

    /// 将配置序列化为 YAML 字符串
    pub fn to_yaml(config: &Config) -> Result<String, ConfigError> {
        serde_yaml::to_string(config).map_err(|e| ConfigError::SerializeError(e.to_string()))
    }

    /// 保存配置到文件
    pub fn save(&self) -> Result<(), ConfigError> {
        self.save_to(&self.config_path)
    }

    /// 保存配置到指定路径
    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        // 确保父目录存在
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ConfigError::WriteError(e.to_string()))?;
        }

        if path.exists() {
            let backup_path = path.with_extension("yaml.backup");
            let _ = std::fs::copy(path, backup_path);
        }
        let yaml = Self::to_yaml(&normalized_config_for_persistence(&self.config))?;
        std::fs::write(path, yaml).map_err(|e| ConfigError::WriteError(e.to_string()))
    }

    /// 重新加载配置
    pub fn reload(&mut self) -> Result<(), ConfigError> {
        let content = std::fs::read_to_string(&self.config_path)
            .map_err(|e| ConfigError::ReadError(e.to_string()))?;
        self.config = Self::parse_yaml(&content)?;
        self.config.normalize_local_server_surface();
        self.config.normalize_workspace_preferences();
        Ok(())
    }

    /// 获取当前配置
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// 获取可变配置引用
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// 设置配置
    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }

    /// 获取配置文件路径
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    /// 导出配置为 YAML 字符串
    ///
    /// # Arguments
    /// * `redact_secrets` - 是否脱敏敏感信息（API 密钥等）
    pub fn export(&self, redact_secrets: bool) -> Result<String, ConfigError> {
        if redact_secrets {
            let mut config = self.config.clone();
            // 脱敏 API 密钥
            config.server.api_key = "***REDACTED***".to_string();
            if config.providers.openai.api_key.is_some() {
                config.providers.openai.api_key = Some("***REDACTED***".to_string());
            }
            if config.providers.claude.api_key.is_some() {
                config.providers.claude.api_key = Some("***REDACTED***".to_string());
            }
            Self::to_yaml(&config)
        } else {
            Self::to_yaml(&self.config)
        }
    }

    /// 从 YAML 字符串导入配置
    ///
    /// # Arguments
    /// * `yaml` - YAML 配置字符串
    /// * `merge` - 是否合并到现有配置（true）或替换（false）
    pub fn import(&mut self, yaml: &str, merge: bool) -> Result<(), ConfigError> {
        let imported = Self::parse_yaml(yaml)?;

        if merge {
            // 合并配置：只更新导入配置中非默认的字段
            self.merge_config(imported);
        } else {
            // 替换配置
            self.config = imported;
        }

        Ok(())
    }

    /// 合并配置
    fn merge_config(&mut self, other: Config) {
        // 合并服务器配置
        if other.server != ServerConfig::default() {
            self.config.server = other.server;
        }

        // 合并 Provider 配置
        if other.providers.kiro.enabled || other.providers.kiro.credentials_path.is_some() {
            self.config.providers.kiro = other.providers.kiro;
        }
        if other.providers.gemini.enabled || other.providers.gemini.credentials_path.is_some() {
            self.config.providers.gemini = other.providers.gemini;
        }
        if other.providers.qwen.enabled || other.providers.qwen.credentials_path.is_some() {
            self.config.providers.qwen = other.providers.qwen;
        }
        if other.providers.openai.enabled || other.providers.openai.api_key.is_some() {
            self.config.providers.openai = other.providers.openai;
        }
        if other.providers.claude.enabled || other.providers.claude.api_key.is_some() {
            self.config.providers.claude = other.providers.claude;
        }

        if other.default_provider != Config::default().default_provider {
            self.config.default_provider = other.default_provider;
        }
        if other.default_permissions.is_some() {
            self.config.default_permissions = other.default_permissions;
        }
        if !other.permissions.is_empty() {
            self.config.permissions.extend(other.permissions);
        }

        // 合并路由辅助配置
        if !other.routing.model_aliases.is_empty() {
            self.config
                .routing
                .model_aliases
                .extend(other.routing.model_aliases);
        }

        // 合并重试配置
        if other.retry != RetrySettings::default() {
            self.config.retry = other.retry;
        }

        // 合并日志配置
        if other.logging != LoggingConfig::default() {
            self.config.logging = other.logging;
        }
    }

    /// 获取默认配置文件路径
    pub fn default_config_path() -> PathBuf {
        if let Some(path) = config_path_from_override(std::env::var_os(CONFIG_PATH_ENV)) {
            return path;
        }

        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("lime")
            .join("config.yaml")
    }
}

use super::types::{LoggingConfig, RetrySettings, ServerConfig};

fn normalize_legacy_workspace_preferences_yaml_value(value: &mut serde_yaml::Value) -> bool {
    let Some(mapping) = value.as_mapping_mut() else {
        return false;
    };

    let workspace_key = serde_yaml::Value::String("workspace_preferences".to_string());
    let legacy_key = serde_yaml::Value::String("content_creator".to_string());

    if mapping.contains_key(&workspace_key) {
        return mapping.remove(&legacy_key).is_some();
    }

    if let Some(legacy_value) = mapping.remove(&legacy_key) {
        mapping.insert(workspace_key, legacy_value);
        return true;
    }

    false
}

fn normalize_retired_nested_default_provider_yaml_value(value: &mut serde_yaml::Value) -> bool {
    let Some(mapping) = value.as_mapping_mut() else {
        return false;
    };

    let routing_key = serde_yaml::Value::String("routing".to_string());
    let provider_key = serde_yaml::Value::String("default_provider".to_string());
    let nested_provider = mapping
        .get_mut(&routing_key)
        .and_then(serde_yaml::Value::as_mapping_mut)
        .and_then(|routing| routing.remove(&provider_key));
    let Some(nested_provider) = nested_provider else {
        return false;
    };

    if !mapping.contains_key(&provider_key) {
        mapping.insert(provider_key, nested_provider);
    }
    true
}

fn parse_yaml_config_with_legacy_tracking(yaml: &str) -> Result<(Config, bool), ConfigError> {
    let mut value: serde_yaml::Value =
        serde_yaml::from_str(yaml).map_err(|e| ConfigError::ParseError(e.to_string()))?;
    let migrated_workspace_preferences =
        normalize_legacy_workspace_preferences_yaml_value(&mut value);
    let migrated_default_provider =
        normalize_retired_nested_default_provider_yaml_value(&mut value);
    let config =
        serde_yaml::from_value(value).map_err(|e| ConfigError::ParseError(e.to_string()))?;
    Ok((
        config,
        migrated_workspace_preferences || migrated_default_provider,
    ))
}

fn normalized_config_for_persistence(config: &Config) -> Config {
    let mut normalized = config.clone();
    normalized.normalize_local_server_surface();
    normalized.normalize_workspace_preferences();
    normalized
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new(Self::default_config_path())
    }
}

// ============ YAML 注释保留功能 ============

/// YAML 服务 - 提供保留注释的 YAML 操作
pub struct YamlService;

/// 注释信息
#[derive(Debug, Clone)]
struct CommentInfo {
    /// 行号（0-indexed）
    line: usize,
    /// 注释内容（包含 # 符号）
    content: String,
    /// 是否是行尾注释
    is_inline: bool,
    /// 关联的键路径（如果有）
    key_path: Option<String>,
}

impl YamlService {
    /// 保存配置到 YAML，保留原文件中的注释和格式
    ///
    /// # Arguments
    /// * `path` - 配置文件路径
    /// * `config` - 要保存的配置
    ///
    /// # Returns
    /// * `Ok(())` - 保存成功
    /// * `Err(ConfigError)` - 保存失败
    pub fn save_preserve_comments(path: &Path, config: &Config) -> Result<(), ConfigError> {
        // 读取原文件内容（如果存在）
        let original_content = if path.exists() {
            std::fs::read_to_string(path).ok()
        } else {
            None
        };

        // 序列化新配置
        let new_yaml = ConfigManager::to_yaml(config)?;

        // 如果原文件存在，尝试保留注释
        let final_content = if let Some(original) = original_content {
            Self::merge_comments(&original, &new_yaml)
        } else {
            new_yaml
        };

        // 确保父目录存在
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ConfigError::WriteError(e.to_string()))?;
        }

        // 写入文件
        std::fs::write(path, final_content).map_err(|e| ConfigError::WriteError(e.to_string()))
    }

    /// 合并原文件的注释到新 YAML 内容中
    ///
    /// 策略：
    /// 1. 提取原文件中的所有注释（独立行注释和行尾注释）
    /// 2. 尝试将注释与键路径关联
    /// 3. 在新 YAML 中找到对应位置插入注释
    /// 4. 无法关联的注释放在文件头部
    fn merge_comments(original: &str, new_yaml: &str) -> String {
        let original_lines: Vec<&str> = original.lines().collect();
        let new_lines: Vec<&str> = new_yaml.lines().collect();

        // 提取原文件中的注释
        let comments = Self::extract_comments(&original_lines);

        // 如果没有注释，直接返回新内容
        if comments.is_empty() {
            return new_yaml.to_string();
        }

        // 构建新 YAML 的键位置映射
        let new_key_positions = Self::build_key_positions(&new_lines);

        // 合并注释到新内容
        Self::insert_comments(&new_lines, &comments, &new_key_positions)
    }

    /// 提取所有独立行注释（不尝试关联键路径）
    pub fn extract_all_comments(yaml: &str) -> Vec<String> {
        yaml.lines()
            .filter(|line| line.trim().starts_with('#'))
            .map(|s| s.to_string())
            .collect()
    }

    /// 从 YAML 行中提取注释
    fn extract_comments(lines: &[&str]) -> Vec<CommentInfo> {
        let mut comments = Vec::new();
        let mut current_key_path: Vec<String> = Vec::new();
        let mut indent_stack: Vec<usize> = vec![0];

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // 跳过空行
            if trimmed.is_empty() {
                continue;
            }

            // 计算当前行的缩进
            let indent = line.len() - line.trim_start().len();

            // 检查是否是纯注释行
            if trimmed.starts_with('#') {
                // 确定注释关联的键路径
                let key_path = if current_key_path.is_empty() {
                    None
                } else {
                    Some(current_key_path.join("."))
                };

                comments.push(CommentInfo {
                    line: line_num,
                    content: line.to_string(),
                    is_inline: false,
                    key_path,
                });
                continue;
            }

            // 检查是否有行尾注释
            if let Some(comment_pos) = Self::find_comment_position(line) {
                let comment_content = line[comment_pos..].to_string();

                // 更新键路径
                Self::update_key_path(line, indent, &mut current_key_path, &mut indent_stack);

                comments.push(CommentInfo {
                    line: line_num,
                    content: comment_content,
                    is_inline: true,
                    key_path: Some(current_key_path.join(".")),
                });
            } else {
                // 普通键值行，更新键路径
                Self::update_key_path(line, indent, &mut current_key_path, &mut indent_stack);
            }
        }

        comments
    }

    /// 查找行中注释的位置（考虑字符串内的 # 符号）
    fn find_comment_position(line: &str) -> Option<usize> {
        let mut in_single_quote = false;
        let mut in_double_quote = false;
        let mut prev_char = ' ';

        for (i, c) in line.char_indices() {
            match c {
                '\'' if !in_double_quote && prev_char != '\\' => {
                    in_single_quote = !in_single_quote;
                }
                '"' if !in_single_quote && prev_char != '\\' => {
                    in_double_quote = !in_double_quote;
                }
                '#' if !in_single_quote && !in_double_quote => {
                    // 确保 # 前面有空格（YAML 注释规则）
                    if i == 0
                        || line
                            .chars()
                            .nth(i - 1)
                            .map(|c| c.is_whitespace())
                            .unwrap_or(false)
                    {
                        return Some(i);
                    }
                }
                _ => {}
            }
            prev_char = c;
        }
        None
    }

    /// 更新当前键路径
    fn update_key_path(
        line: &str,
        indent: usize,
        current_key_path: &mut Vec<String>,
        indent_stack: &mut Vec<usize>,
    ) {
        let trimmed = line.trim();

        // 提取键名
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim();

            // 跳过列表项
            if key.starts_with('-') {
                return;
            }

            // 根据缩进调整键路径
            while indent_stack.len() > 1 && indent <= indent_stack[indent_stack.len() - 1] {
                indent_stack.pop();
                current_key_path.pop();
            }

            // 添加新键
            current_key_path.push(key.to_string());
            indent_stack.push(indent);
        }
    }

    /// 构建新 YAML 的键位置映射
    fn build_key_positions(lines: &[&str]) -> HashMap<String, usize> {
        let mut positions = HashMap::new();
        let mut current_key_path: Vec<String> = Vec::new();
        let mut indent_stack: Vec<usize> = vec![0];

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let indent = line.len() - line.trim_start().len();

            if let Some(colon_pos) = trimmed.find(':') {
                let key = trimmed[..colon_pos].trim();

                if key.starts_with('-') {
                    continue;
                }

                // 根据缩进调整键路径
                while indent_stack.len() > 1 && indent <= indent_stack[indent_stack.len() - 1] {
                    indent_stack.pop();
                    current_key_path.pop();
                }

                current_key_path.push(key.to_string());
                indent_stack.push(indent);

                // 记录位置
                positions.insert(current_key_path.join("."), line_num);
            }
        }

        positions
    }

    /// 将注释插入到新 YAML 内容中
    fn insert_comments(
        new_lines: &[&str],
        comments: &[CommentInfo],
        key_positions: &HashMap<String, usize>,
    ) -> String {
        let mut result_lines: Vec<String> = new_lines.iter().map(|s| s.to_string()).collect();
        let mut insertions: Vec<(usize, String)> = Vec::new();
        let mut unmatched_comments: Vec<String> = Vec::new();

        for comment in comments {
            if comment.is_inline {
                // 行尾注释：找到对应的键并追加
                if let Some(key_path) = &comment.key_path {
                    if let Some(&line_num) = key_positions.get(key_path) {
                        if line_num < result_lines.len() {
                            // 追加行尾注释
                            let existing = &result_lines[line_num];
                            if !existing.contains('#') {
                                result_lines[line_num] =
                                    format!("{}  {}", existing, comment.content);
                            }
                        }
                    } else {
                        // 无法匹配的行尾注释，作为独立注释保留
                        unmatched_comments.push(comment.content.clone());
                    }
                }
            } else {
                // 独立行注释：尝试在对应位置插入
                if let Some(key_path) = &comment.key_path {
                    // 找到下一个键的位置
                    if let Some(&next_line) = key_positions.get(key_path) {
                        insertions.push((next_line, comment.content.clone()));
                    } else {
                        // 无法匹配的注释，放到头部
                        unmatched_comments.push(comment.content.clone());
                    }
                } else {
                    // 文件头部注释
                    insertions.push((0, comment.content.clone()));
                }
            }
        }

        // 按位置倒序插入，避免索引偏移
        insertions.sort_by(|a, b| b.0.cmp(&a.0));
        for (pos, content) in insertions {
            if pos <= result_lines.len() {
                result_lines.insert(pos, content);
            }
        }

        // 将无法匹配的注释放在文件头部
        if !unmatched_comments.is_empty() {
            let mut final_lines = unmatched_comments;
            final_lines.extend(result_lines);
            return final_lines.join("\n");
        }

        result_lines.join("\n")
    }

    /// 更新 YAML 中的特定字段
    ///
    /// # Arguments
    /// * `path` - 配置文件路径
    /// * `field_path` - 字段路径，如 ["server", "port"]
    /// * `value` - 新值（YAML 格式的字符串）
    ///
    /// # Returns
    /// * `Ok(())` - 更新成功
    /// * `Err(ConfigError)` - 更新失败
    pub fn update_field(path: &Path, field_path: &[&str], value: &str) -> Result<(), ConfigError> {
        // 读取原文件
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::ReadError(e.to_string()))?;

        let lines: Vec<&str> = content.lines().collect();
        let mut result_lines: Vec<String> = Vec::new();

        let target_key = field_path.last().copied().unwrap_or("");
        let parent_path = &field_path[..field_path.len().saturating_sub(1)];

        let mut current_path: Vec<String> = Vec::new();
        let mut indent_stack: Vec<usize> = vec![0];
        let mut found = false;

        for line in lines {
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                result_lines.push(line.to_string());
                continue;
            }

            let indent = line.len() - line.trim_start().len();

            if let Some(colon_pos) = trimmed.find(':') {
                let key = trimmed[..colon_pos].trim();

                if !key.starts_with('-') {
                    // 根据缩进调整路径
                    while indent_stack.len() > 1 && indent <= indent_stack[indent_stack.len() - 1] {
                        indent_stack.pop();
                        current_path.pop();
                    }

                    // 检查是否匹配目标字段
                    let parent_matches = current_path.len() == parent_path.len()
                        && current_path
                            .iter()
                            .zip(parent_path.iter())
                            .all(|(a, b)| a == *b);

                    if parent_matches && key == target_key {
                        // 找到目标字段，替换值
                        let new_line = format!("{}{}: {}", " ".repeat(indent), key, value);
                        result_lines.push(new_line);
                        found = true;

                        current_path.push(key.to_string());
                        indent_stack.push(indent);
                        continue;
                    }

                    current_path.push(key.to_string());
                    indent_stack.push(indent);
                }
            }

            result_lines.push(line.to_string());
        }

        if !found {
            return Err(ConfigError::ValidationError(format!(
                "字段 {} 未找到",
                field_path.join(".")
            )));
        }

        // 写入文件
        std::fs::write(path, result_lines.join("\n"))
            .map_err(|e| ConfigError::WriteError(e.to_string()))
    }
}

// ============ 当前 YAML 配置函数 ============

fn config_path_from_override(value: Option<OsString>) -> Option<PathBuf> {
    let value = value?;
    let trimmed = value.to_string_lossy().trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

/// 加载当前 YAML 配置。
///
/// 首次启动时自动生成强随机 API Key 并保存配置
pub fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let yaml_path = ConfigManager::default_config_path();
    load_config_from_path(&yaml_path)
}

fn normalize_loaded_config(mut config: Config, migrated_legacy_key: bool) -> (Config, bool) {
    use super::types::{generate_secure_api_key, is_default_api_key};

    let mut should_save = migrated_legacy_key || config.normalize_workspace_preferences();
    if config.normalize_local_server_surface() {
        should_save = true;
    }
    if is_default_api_key(&config.server.api_key) {
        let new_key = generate_secure_api_key();
        tracing::warn!("[CONFIG] 检测到默认 API Key，已自动生成强随机 Key");
        config.server.api_key = new_key;
        should_save = true;
    }

    (config, should_save)
}

fn persist_loaded_config(config: &Config, yaml_path: &Path) {
    if let Err(e) = save_config_yaml_to_path(config, yaml_path) {
        tracing::error!("[CONFIG] 保存配置失败: {}", e);
    }
}

fn load_config_from_path(yaml_path: &Path) -> Result<Config, Box<dyn std::error::Error>> {
    if yaml_path.exists() {
        let content = std::fs::read_to_string(&yaml_path)?;
        let (config, migrated_legacy_key) = parse_yaml_config_with_legacy_tracking(&content)?;
        let (config, should_save) = normalize_loaded_config(config, migrated_legacy_key);
        if should_save {
            persist_loaded_config(&config, yaml_path);
        }
        return Ok(config);
    }

    // 都不存在，创建默认配置并生成强随机 API Key
    let mut config = Config::default();
    let new_key = super::types::generate_secure_api_key();
    tracing::info!("[CONFIG] 首次启动，已生成强随机 API Key");
    config.server.api_key = new_key;
    // 保存初始配置
    if let Err(e) = save_config_yaml_to_path(&config, yaml_path) {
        tracing::error!("[CONFIG] 保存初始配置失败: {}", e);
    }
    Ok(config)
}

/// 保存当前 YAML 配置。
pub fn save_config(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let yaml_path = ConfigManager::default_config_path();
    save_config_yaml_to_path(config, &yaml_path)
}

/// 保存配置为 YAML 格式
pub fn save_config_yaml(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let path = ConfigManager::default_config_path();
    save_config_yaml_to_path(config, &path)
}

fn save_config_yaml_to_path(
    config: &Config,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let normalized = normalized_config_for_persistence(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if path.exists() {
        let backup_path = path.with_extension("yaml.backup");
        let _ = std::fs::copy(&path, &backup_path);
    }
    let content = serde_yaml::to_string(&normalized)?;
    std::fs::write(&path, content)?;
    Ok(())
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_parse_yaml_minimal() {
        let yaml = r#"
server:
  host: "127.0.0.1"
  port: 9000
  api_key: "test-key"
providers:
  openai:
    enabled: true
    base_url: "https://api.openai.com/v1"
"#;
        let config = ConfigManager::parse_yaml(yaml).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.api_key, "test-key");
        assert!(config.providers.openai.enabled);
    }

    #[test]
    fn test_parse_yaml_full() {
        let yaml = r#"
server:
  host: "127.0.0.1"
  port: 8999
  api_key: "proxy_cast"
providers:
  gemini:
    enabled: false
  qwen:
    enabled: false
  openai:
    enabled: true
    base_url: "https://api.openai.com/v1"
  claude:
    enabled: false
default_provider: "openai"
routing:
  model_aliases:
    gpt-4: "claude-sonnet-4-5-20250514"
retry:
  max_retries: 3
  base_delay_ms: 1000
  max_delay_ms: 30000
  auto_switch_provider: true
logging:
  enabled: true
  level: "info"
  retention_days: 7
"#;
        let config = ConfigManager::parse_yaml(yaml).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.default_provider, "openai");
        assert_eq!(
            config.routing.model_aliases.get("gpt-4"),
            Some(&"claude-sonnet-4-5-20250514".to_string())
        );
    }

    #[test]
    fn yaml_roundtrip_preserves_codex_permission_profile_shape() {
        let yaml = r#"
default_permissions: locked-down
permissions:
  locked-down:
    extends: ":workspace"
    description: Read selected documentation
    filesystem:
      entries:
        /docs: read
        /docs/private: none
      glob_scan_max_depth: 4
    network:
      enabled: true
"#;

        let config = ConfigManager::parse_yaml(yaml).expect("permission profile YAML");
        assert_eq!(config.default_permissions.as_deref(), Some("locked-down"));
        let profile = config
            .permissions
            .get("locked-down")
            .expect("named permission profile");
        assert_eq!(profile.extends.as_deref(), Some(":workspace"));
        assert_eq!(
            profile.description.as_deref(),
            Some("Read selected documentation")
        );
        assert_eq!(
            profile
                .filesystem
                .as_ref()
                .and_then(|filesystem| filesystem.entries.get("/docs"))
                .map(String::as_str),
            Some("read")
        );
        assert_eq!(
            profile.network.as_ref().and_then(|network| network.enabled),
            Some(true)
        );

        let serialized =
            ConfigManager::to_yaml(&config).expect("serialize permission profile YAML");
        let reparsed =
            ConfigManager::parse_yaml(&serialized).expect("reparse permission profile YAML");
        assert_eq!(reparsed.default_permissions, config.default_permissions);
        assert_eq!(reparsed.permissions, config.permissions);
    }

    #[test]
    fn yaml_roundtrip_preserves_windows_sandbox_mode() {
        let yaml = r#"
agent:
  workspace_sandbox:
    enabled: true
    strict: true
    mode: unelevated
"#;
        let config = ConfigManager::parse_yaml(yaml).expect("Windows sandbox mode YAML");
        assert_eq!(
            config.agent.workspace_sandbox.mode,
            Some(crate::config::WindowsSandboxMode::Unelevated)
        );
        let serialized = ConfigManager::to_yaml(&config).expect("serialize Windows sandbox YAML");
        let reparsed =
            ConfigManager::parse_yaml(&serialized).expect("reparse Windows sandbox YAML");
        assert_eq!(
            reparsed.agent.workspace_sandbox.mode,
            config.agent.workspace_sandbox.mode
        );
    }

    #[test]
    fn test_parse_yaml_migrates_retired_nested_default_provider() {
        let config = ConfigManager::parse_yaml(
            r#"
routing:
  default_provider: "gemini"
  model_aliases:
    fast: "gemini/gemini-2.5-flash"
"#,
        )
        .expect("retired nested provider should migrate at the YAML boundary");

        assert_eq!(config.default_provider, "gemini");
        assert_eq!(
            config.routing.model_aliases.get("fast"),
            Some(&"gemini/gemini-2.5-flash".to_string())
        );
    }

    #[test]
    fn test_parse_yaml_prefers_current_top_level_default_provider() {
        let config = ConfigManager::parse_yaml(
            r#"
default_provider: "anthropic"
routing:
  default_provider: "gemini"
"#,
        )
        .expect("current top-level provider should win during migration");

        assert_eq!(config.default_provider, "anthropic");
    }

    #[test]
    fn test_parse_yaml_rejects_other_unknown_routing_fields() {
        let error = ConfigManager::parse_yaml(
            r#"
routing:
  retired_field: true
"#,
        )
        .expect_err("unknown routing fields must still fail closed");

        assert!(error.to_string().contains("retired_field"));
    }

    #[test]
    fn test_parse_yaml_migrates_legacy_content_creator_key() {
        let yaml = r#"
content_creator:
  schema_version: 0
  media_defaults:
    image:
      preferred_provider_id: "fal"
"#;

        let config = ConfigManager::parse_yaml(yaml).unwrap();
        assert_eq!(config.workspace_preferences.schema_version, 0);
        assert_eq!(
            config
                .workspace_preferences
                .media_defaults
                .image
                .preferred_provider_id
                .as_deref(),
            Some("fal")
        );
    }

    #[test]
    fn test_to_yaml_roundtrip() {
        let config = Config::default();
        let yaml = ConfigManager::to_yaml(&config).unwrap();
        let parsed = ConfigManager::parse_yaml(&yaml).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_to_yaml_serializes_wechat_runtime_fields_once() {
        let mut config = Config::default();
        config.channels.wechat.default_model = Some("openai/gpt-4.1".to_string());

        let yaml = ConfigManager::to_yaml(&config).unwrap();
        let mut in_wechat = false;
        let mut wechat_lines = Vec::new();
        for line in yaml.lines() {
            if line == "  wechat:" {
                in_wechat = true;
            } else if in_wechat && line.starts_with("  ") && !line.starts_with("    ") {
                break;
            }
            if in_wechat {
                wechat_lines.push(line);
            }
        }
        let wechat_block = wechat_lines.join("\n");

        assert_eq!(wechat_block.matches("    streaming: off").count(), 1);
        assert_eq!(wechat_block.matches("    reply_to_mode: off").count(), 1);
        ConfigManager::parse_yaml(&yaml).unwrap();
    }

    #[test]
    fn test_current_yaml_parse_error_does_not_fallback_to_json() {
        let temp_dir = tempfile::tempdir().unwrap();
        let yaml_path = temp_dir.path().join("config.yaml");
        let json_path = temp_dir.path().join("config.json");
        std::fs::write(
            &yaml_path,
            "channels:\n  wechat:\n    streaming: off\n    streaming: partial\n",
        )
        .unwrap();
        std::fs::write(&json_path, r#"{"server":{"api_key":"json-key"}}"#).unwrap();

        let error = load_config_from_path(&yaml_path).unwrap_err().to_string();

        assert!(error.contains("duplicate entry with key"));
    }

    #[test]
    fn test_load_config_rewrites_nested_default_provider_to_current_shape() {
        let temp_dir = tempfile::tempdir().unwrap();
        let yaml_path = temp_dir.path().join("config.yaml");
        std::fs::write(
            &yaml_path,
            r#"
server:
  api_key: "custom-key"
routing:
  default_provider: "gemini"
  model_aliases:
    fast: "gemini/gemini-2.5-flash"
"#,
        )
        .unwrap();

        let config = load_config_from_path(&yaml_path).unwrap();
        let persisted: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&yaml_path).unwrap()).unwrap();

        assert_eq!(config.default_provider, "gemini");
        assert_eq!(
            persisted
                .get("default_provider")
                .and_then(serde_yaml::Value::as_str),
            Some("gemini")
        );
        assert!(persisted
            .get("routing")
            .and_then(serde_yaml::Value::as_mapping)
            .is_some_and(|routing| !routing.contains_key("default_provider")));
        assert!(yaml_path.with_extension("yaml.backup").is_file());
    }

    #[test]
    fn test_save_config_yaml_to_path_only_writes_current_yaml() {
        let temp_dir = tempfile::tempdir().unwrap();
        let yaml_path = temp_dir.path().join("config.yaml");
        let json_path = temp_dir.path().join("config.json");

        save_config_yaml_to_path(&Config::default(), &yaml_path).unwrap();

        assert!(yaml_path.is_file());
        assert!(!json_path.exists());
    }

    #[test]
    fn test_parse_yaml_with_defaults() {
        // 只提供部分配置，其他使用默认值
        let yaml = r#"
server:
  port: 9999
"#;
        let config = ConfigManager::parse_yaml(yaml).unwrap();
        assert_eq!(config.server.port, 9999);
        // 其他字段应使用默认值
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.retry.max_retries, 3);
        assert_eq!(
            config.server.response_cache.cacheable_status_codes,
            vec![200]
        );
    }

    #[test]
    fn test_parse_yaml_with_response_cache_settings() {
        let yaml = r#"
server:
  response_cache:
    enabled: true
    ttl_secs: 120
    max_entries: 64
    max_body_bytes: 262144
    cacheable_status_codes: [200, 201]
"#;
        let config = ConfigManager::parse_yaml(yaml).unwrap();
        assert!(config.server.response_cache.enabled);
        assert_eq!(config.server.response_cache.ttl_secs, 120);
        assert_eq!(config.server.response_cache.max_entries, 64);
        assert_eq!(config.server.response_cache.max_body_bytes, 262144);
        assert_eq!(
            config.server.response_cache.cacheable_status_codes,
            vec![200, 201]
        );
    }

    #[test]
    fn test_export_redacted() {
        let mut config = Config::default();
        config.server.api_key = "secret-key".to_string();
        config.providers.openai.api_key = Some("openai-secret".to_string());

        let manager = ConfigManager {
            config,
            config_path: PathBuf::from("test.yaml"),
        };

        let exported = manager.export(true).unwrap();
        assert!(exported.contains("***REDACTED***"));
        assert!(!exported.contains("secret-key"));
        assert!(!exported.contains("openai-secret"));
    }

    #[test]
    fn test_export_not_redacted() {
        let mut config = Config::default();
        config.server.api_key = "secret-key".to_string();

        let manager = ConfigManager {
            config,
            config_path: PathBuf::from("test.yaml"),
        };

        let exported = manager.export(false).unwrap();
        assert!(exported.contains("secret-key"));
        assert!(!exported.contains("***REDACTED***"));
    }

    #[test]
    fn test_import_replace() {
        let mut manager = ConfigManager::default();
        manager.config.server.port = 1234;

        let yaml = r#"
server:
  port: 5678
"#;
        manager.import(yaml, false).unwrap();
        assert_eq!(manager.config.server.port, 5678);
    }

    #[test]
    fn yaml_import_merge_preserves_existing_and_imported_permission_profiles() {
        let mut manager = ConfigManager::with_config(Config::default(), "config.yaml".into());
        manager
            .import(
                "permissions:\n  existing:\n    description: Existing\n",
                false,
            )
            .expect("seed permission profile");
        manager
            .import(
                "default_permissions: imported\npermissions:\n  imported:\n    description: Imported\n",
                true,
            )
            .expect("merge permission profile");

        assert_eq!(
            manager.config().default_permissions.as_deref(),
            Some("imported")
        );
        assert!(manager.config().permissions.contains_key("existing"));
        assert!(manager.config().permissions.contains_key("imported"));
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::ParseError("invalid yaml".to_string());
        assert!(err.to_string().contains("YAML 解析错误"));
        assert!(err.to_string().contains("invalid yaml"));
    }
}
