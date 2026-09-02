use semver::{BuildMetadata, Version};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// 更新检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    /// 当前版本
    pub current_version: String,
    /// 最新版本
    pub latest_version: Option<String>,
    /// 是否有更新
    pub has_update: bool,
    /// 下载链接
    pub download_url: Option<String>,
    /// 发布说明链接
    pub release_notes_url: Option<String>,
    /// 发布说明摘要
    pub release_notes: Option<String>,
    /// 发布时间
    pub pub_date: Option<String>,
    /// 检查时间（Unix 时间戳）
    pub checked_at: u64,
    /// 错误信息
    pub error: Option<String>,
}

/// 更新检查服务状态
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateCheckState {
    /// 是否正在检查
    pub is_checking: bool,
    /// 最后一次检查结果
    pub last_result: Option<UpdateInfo>,
    /// 下次检查时间（Unix 时间戳）
    pub next_check_at: Option<u64>,
}

/// 更新检查服务
pub struct UpdateCheckService {
    state: Arc<RwLock<UpdateCheckState>>,
}

impl UpdateCheckService {
    const CURRENT_VERSION: &'static str = env!("CARGO_PKG_VERSION");

    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(UpdateCheckState::default())),
        }
    }

    pub fn current_version() -> &'static str {
        Self::CURRENT_VERSION
    }

    /// 获取当前状态
    pub async fn get_state(&self) -> UpdateCheckState {
        self.state.read().await.clone()
    }

    /// 标记开始检查
    pub async fn begin_check(&self) {
        let mut state = self.state.write().await;
        state.is_checking = true;
    }

    /// 写回检查结果
    pub async fn finish_check(&self, result: UpdateInfo) -> UpdateInfo {
        let mut state = self.state.write().await;
        state.is_checking = false;
        state.last_result = Some(result.clone());
        result
    }

    /// 检查是否需要执行更新检查
    pub fn should_check(
        last_check_timestamp: u64,
        check_interval_hours: u32,
        skipped_version: Option<&str>,
        latest_version: Option<&str>,
    ) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let interval_secs = (check_interval_hours as u64) * 3600;

        // 检查时间间隔
        if now < last_check_timestamp + interval_secs {
            return false;
        }

        // 检查是否跳过了当前最新版本
        if let (Some(skipped), Some(latest)) = (skipped_version, latest_version) {
            if skipped == latest {
                return false;
            }
        }

        true
    }
    /// 版本比较：返回 true 如果 latest > current
    pub fn version_compare(current: &str, latest: &str) -> bool {
        let Some(mut current) = parse_semver(current) else {
            return false;
        };
        let Some(mut latest) = parse_semver(latest) else {
            return false;
        };

        if !latest.pre.is_empty() {
            return false;
        }

        current.build = BuildMetadata::EMPTY;
        latest.build = BuildMetadata::EMPTY;

        latest > current
    }
}

fn parse_semver(value: &str) -> Option<Version> {
    let trimmed = value.trim().trim_start_matches('v');
    Version::parse(trimmed).ok()
}

impl Default for UpdateCheckService {
    fn default() -> Self {
        Self::new()
    }
}
/// 更新检查服务共享状态
#[derive(Clone)]
pub struct UpdateCheckServiceState(pub Arc<RwLock<UpdateCheckService>>);

impl UpdateCheckServiceState {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(UpdateCheckService::new())))
    }
}

impl Default for UpdateCheckServiceState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_compare() {
        assert!(UpdateCheckService::version_compare("0.14.0", "0.14.1"));
        assert!(UpdateCheckService::version_compare("0.14.0", "0.15.0"));
        assert!(UpdateCheckService::version_compare("0.14.0", "1.0.0"));
        assert!(UpdateCheckService::version_compare("0.38.0", "0.39.0"));
        assert!(!UpdateCheckService::version_compare("0.14.1", "0.14.0"));
        assert!(!UpdateCheckService::version_compare("0.14.0", "0.14.0"));
        assert!(!UpdateCheckService::version_compare("1.0.0", "0.14.0"));
        assert!(!UpdateCheckService::version_compare(
            "1.0.0",
            "1.1.0-beta.1"
        ));
        assert!(UpdateCheckService::version_compare("1.1.0-beta.1", "1.1.0"));
        assert!(!UpdateCheckService::version_compare(
            "1.0.0+build.1",
            "1.0.0+build.2"
        ));
    }

    #[test]
    fn test_should_check() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // 从未检查过，应该检查
        assert!(UpdateCheckService::should_check(0, 24, None, None));

        // 刚检查过，不应该检查
        assert!(!UpdateCheckService::should_check(now, 24, None, None));

        // 超过间隔，应该检查
        let old_timestamp = now - 25 * 3600;
        assert!(UpdateCheckService::should_check(
            old_timestamp,
            24,
            None,
            None
        ));

        // 跳过了当前最新版本，不应该检查
        assert!(!UpdateCheckService::should_check(
            0,
            24,
            Some("1.0.0"),
            Some("1.0.0")
        ));

        // 跳过了旧版本，应该检查
        assert!(UpdateCheckService::should_check(
            0,
            24,
            Some("0.9.0"),
            Some("1.0.0")
        ));
    }
}
