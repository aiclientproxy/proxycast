//! 托盘菜单元数据模块
//!
//! 提供与桌面宿主无关的菜单 ID 和菜单元信息工具函数。

/// 菜单项 ID 常量
pub mod menu_ids {
    /// 当前模型信息
    pub const CURRENT_MODEL_INFO: &str = "current_model_info";
    /// 快速切换模型
    pub const QUICK_MODEL_ROOT: &str = "quick_model_root";
    /// 请求信息
    pub const REQUEST_INFO: &str = "request_info";
    /// 分隔符 1
    pub const SEPARATOR_1: &str = "sep_1";
    /// 分隔符 2
    pub const SEPARATOR_2: &str = "sep_2";
    /// 打开主窗口
    pub const OPEN_WINDOW: &str = "open_window";
    /// 打开日志目录
    pub const OPEN_LOG_DIR: &str = "open_log_dir";
    /// 分隔符 3
    pub const SEPARATOR_3: &str = "sep_3";
    /// 开机自启
    pub const AUTO_START: &str = "auto_start";
    /// 分隔符 4
    pub const SEPARATOR_4: &str = "sep_4";
    /// 退出
    pub const QUIT: &str = "quit";

    /// 获取所有必需的菜单项 ID 列表
    pub fn all_required_ids() -> Vec<&'static str> {
        vec![REQUEST_INFO, OPEN_WINDOW, OPEN_LOG_DIR, AUTO_START, QUIT]
    }
}

/// 获取菜单中包含的所有菜单项 ID
pub fn get_menu_item_ids() -> Vec<&'static str> {
    menu_ids::all_required_ids()
}

const QUICK_MODEL_ID_PREFIX: &str = "quick_model";

/// 生成快速模型切换菜单项 ID
pub fn build_quick_model_item_id(provider_type: &str, model: &str) -> String {
    format!("{QUICK_MODEL_ID_PREFIX}::{provider_type}::{model}")
}

/// 解析快速模型切换菜单项 ID
pub fn parse_quick_model_item_id(id: &str) -> Option<(String, String)> {
    let mut parts = id.splitn(3, "::");
    let prefix = parts.next()?;
    let provider_type = parts.next()?;
    let model = parts.next()?;

    if prefix != QUICK_MODEL_ID_PREFIX || provider_type.is_empty() || model.is_empty() {
        return None;
    }

    Some((provider_type.to_string(), model.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_all_required_ids_not_empty() {
        let ids = menu_ids::all_required_ids();
        assert!(!ids.is_empty(), "必需的菜单项 ID 列表不应为空");
    }

    #[test]
    fn test_all_required_ids_unique() {
        let ids = menu_ids::all_required_ids();
        let mut unique_ids = ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        assert_eq!(ids.len(), unique_ids.len(), "菜单项 ID 应该唯一");
    }

    #[test]
    fn test_menu_ids_completeness() {
        let ids = menu_ids::all_required_ids();

        assert!(ids.contains(&menu_ids::REQUEST_INFO), "应包含 REQUEST_INFO");
        assert!(ids.contains(&menu_ids::OPEN_WINDOW), "应包含 OPEN_WINDOW");
        assert!(ids.contains(&menu_ids::OPEN_LOG_DIR), "应包含 OPEN_LOG_DIR");
        assert!(ids.contains(&menu_ids::AUTO_START), "应包含 AUTO_START");
        assert!(ids.contains(&menu_ids::QUIT), "应包含 QUIT");
    }

    #[test]
    fn test_get_menu_item_ids() {
        let ids = get_menu_item_ids();
        assert_eq!(ids.len(), 5, "应有 5 个必需的菜单项");
    }

    #[test]
    fn test_build_and_parse_quick_model_item_id() {
        let id = build_quick_model_item_id("claude", "claude-sonnet-4-5");
        assert_eq!(id, "quick_model::claude::claude-sonnet-4-5");

        let parsed = parse_quick_model_item_id(&id);
        assert_eq!(
            parsed,
            Some(("claude".to_string(), "claude-sonnet-4-5".to_string()))
        );
    }

    #[test]
    fn test_parse_quick_model_item_id_invalid() {
        assert_eq!(parse_quick_model_item_id("quick_model::claude"), None);
        assert_eq!(parse_quick_model_item_id("other::claude::model"), None);
    }

    proptest! {
        #[test]
        fn prop_menu_ids_completeness(
            _requests in 0u64..1000000,
            _auto_start in any::<bool>()
        ) {
            let ids = menu_ids::all_required_ids();

            let required = vec![
                menu_ids::REQUEST_INFO,
                menu_ids::OPEN_WINDOW,
                menu_ids::OPEN_LOG_DIR,
                menu_ids::AUTO_START,
                menu_ids::QUIT,
            ];

            for id in required {
                prop_assert!(ids.contains(&id), "菜单项列表应包含 {}", id);
            }

            let mut sorted_ids = ids.clone();
            sorted_ids.sort();
            sorted_ids.dedup();
            prop_assert_eq!(ids.len(), sorted_ids.len(), "菜单项 ID 应该唯一");
        }
    }
}
