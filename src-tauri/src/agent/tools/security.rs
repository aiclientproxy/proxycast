//! 安全管理器模块
//!
//! 提供路径验证、目录遍历防护、符号链接检查等安全功能
//! 符合 Requirements 8.1, 8.2, 8.3, 8.5

use std::path::{Component, Path, PathBuf};
use thiserror::Error;
use tracing::{debug, warn};

/// 安全错误类型
///
/// Requirements: 8.6 - IF a security violation is detected, THEN THE Security_Manager SHALL reject the operation with a clear error
#[derive(Debug, Error)]
pub enum SecurityError {
    /// 路径遍历攻击
    /// Requirements: 8.1, 8.2 - THE Security_Manager SHALL validate all file paths to prevent directory traversal attacks
    #[error("路径遍历攻击: 路径 '{0}' 包含 '..' 组件")]
    PathTraversal(PathBuf),

    /// 路径超出基础目录
    /// Requirements: 8.5 - THE Security_Manager SHALL enforce a configurable base directory for all file operations
    #[error("路径超出基础目录: '{0}' 不在允许的目录范围内")]
    OutsideBaseDir(PathBuf),

    /// 不允许操作符号链接
    /// Requirements: 8.3 - THE Security_Manager SHALL reject operations on symlinks to prevent escape attacks
    #[error("不允许操作符号链接: '{0}'")]
    SymlinkNotAllowed(PathBuf),

    /// 无效路径
    #[error("无效路径: {0}")]
    InvalidPath(String),

    /// IO 错误
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
}

/// 安全管理器
///
/// 负责验证所有文件操作的安全性
/// Requirements: 8.1, 8.2, 8.3, 8.5
#[derive(Debug, Clone)]
pub struct SecurityManager {
    /// 基础目录（所有文件操作必须在此目录内）
    base_dir: PathBuf,
}

impl SecurityManager {
    /// 创建新的安全管理器
    ///
    /// # Arguments
    /// * `base_dir` - 基础目录，所有文件操作必须在此目录内
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// 获取基础目录
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// 设置基础目录
    pub fn set_base_dir(&mut self, base_dir: impl Into<PathBuf>) {
        self.base_dir = base_dir.into();
        debug!("[SecurityManager] 设置基础目录: {:?}", self.base_dir);
    }

    /// 验证路径安全性
    ///
    /// 执行以下检查：
    /// 1. 检查路径是否包含 ".." 组件（Requirements 8.2）
    /// 2. 检查路径是否为符号链接（Requirements 8.3）- 在规范化之前检查
    /// 3. 检查路径是否在基础目录内（Requirements 8.5）
    ///
    /// # Arguments
    /// * `path` - 要验证的路径（可以是相对路径或绝对路径）
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - 规范化后的安全路径
    /// * `Err(SecurityError)` - 安全错误
    pub fn validate_path(&self, path: &Path) -> Result<PathBuf, SecurityError> {
        // 1. 检查 ".." 组件
        // Requirements: 8.2 - THE Security_Manager SHALL reject paths containing ".." components
        if self.contains_parent_dir(path) {
            warn!("[SecurityManager] 检测到路径遍历攻击: {:?}", path);
            return Err(SecurityError::PathTraversal(path.to_path_buf()));
        }

        // 2. 构建完整路径
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };

        // 3. 检查符号链接（在规范化之前检查，因为规范化会解析符号链接）
        // Requirements: 8.3 - THE Security_Manager SHALL reject operations on symlinks
        self.check_symlink(&full_path)?;

        // 4. 检查是否在基础目录内
        // Requirements: 8.5 - THE Security_Manager SHALL enforce a configurable base directory
        let validated_path = self.check_within_base_dir(&full_path)?;

        debug!(
            "[SecurityManager] 路径验证通过: {:?} -> {:?}",
            path, validated_path
        );
        Ok(validated_path)
    }

    /// 检查路径是否包含 ".." 组件
    ///
    /// Requirements: 8.2 - THE Security_Manager SHALL reject paths containing ".." components
    fn contains_parent_dir(&self, path: &Path) -> bool {
        path.components().any(|c| matches!(c, Component::ParentDir))
    }

    /// 检查路径是否在基础目录内
    ///
    /// Requirements: 8.5 - THE Security_Manager SHALL enforce a configurable base directory
    fn check_within_base_dir(&self, path: &Path) -> Result<PathBuf, SecurityError> {
        // 尝试规范化基础目录
        let canonical_base = self.base_dir.canonicalize().map_err(|e| {
            SecurityError::InvalidPath(format!("无法规范化基础目录 {:?}: {}", self.base_dir, e))
        })?;

        // 尝试规范化目标路径
        if path.exists() {
            // 文件存在，直接规范化
            let canonical_path = path.canonicalize()?;
            if !canonical_path.starts_with(&canonical_base) {
                return Err(SecurityError::OutsideBaseDir(path.to_path_buf()));
            }
            Ok(canonical_path)
        } else {
            // 文件不存在，检查父目录
            if let Some(parent) = path.parent() {
                if parent.as_os_str().is_empty() {
                    // 父目录为空，说明是相对路径的单个文件名
                    // 此时完整路径应该在基础目录内
                    return Ok(path.to_path_buf());
                }

                if parent.exists() {
                    let canonical_parent = parent.canonicalize()?;
                    if !canonical_parent.starts_with(&canonical_base) {
                        return Err(SecurityError::OutsideBaseDir(path.to_path_buf()));
                    }
                    // 返回规范化的父目录 + 文件名
                    if let Some(file_name) = path.file_name() {
                        return Ok(canonical_parent.join(file_name));
                    }
                }
            }
            // 父目录也不存在，返回原路径（后续创建时会再次验证）
            Ok(path.to_path_buf())
        }
    }

    /// 检查路径是否为符号链接
    ///
    /// Requirements: 8.3 - THE Security_Manager SHALL reject operations on symlinks
    fn check_symlink(&self, path: &Path) -> Result<(), SecurityError> {
        if path.exists() {
            let metadata = path.symlink_metadata()?;
            if metadata.is_symlink() {
                warn!("[SecurityManager] 检测到符号链接: {:?}", path);
                return Err(SecurityError::SymlinkNotAllowed(path.to_path_buf()));
            }
        }
        Ok(())
    }

    /// 验证路径安全性（不检查符号链接）
    ///
    /// 用于某些只需要检查路径遍历和基础目录的场景
    pub fn validate_path_no_symlink_check(&self, path: &Path) -> Result<PathBuf, SecurityError> {
        // 1. 检查 ".." 组件
        if self.contains_parent_dir(path) {
            return Err(SecurityError::PathTraversal(path.to_path_buf()));
        }

        // 2. 构建完整路径
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };

        // 3. 检查是否在基础目录内
        self.check_within_base_dir(&full_path)
    }

    /// 检查路径是否安全（快速检查，不规范化）
    ///
    /// 仅检查是否包含 ".." 组件，用于快速过滤明显的攻击
    pub fn quick_check(&self, path: &Path) -> bool {
        !self.contains_parent_dir(path)
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        // 默认使用当前目录作为基础目录
        Self::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// 创建测试用的临时目录结构
    fn setup_test_dir() -> TempDir {
        let temp_dir = TempDir::new().unwrap();

        // 创建一些测试文件和目录
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").unwrap();

        let sub_dir = temp_dir.path().join("subdir");
        fs::create_dir(&sub_dir).unwrap();

        let sub_file = sub_dir.join("nested.txt");
        fs::write(&sub_file, "nested content").unwrap();

        temp_dir
    }

    #[test]
    fn test_security_manager_creation() {
        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        assert_eq!(security.base_dir(), temp_dir.path());
    }

    #[test]
    fn test_validate_path_within_base_dir() {
        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        // 相对路径应该通过
        let result = security.validate_path(Path::new("test.txt"));
        assert!(result.is_ok());

        // 子目录中的文件也应该通过
        let result = security.validate_path(Path::new("subdir/nested.txt"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_path_traversal() {
        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        // 包含 ".." 的路径应该被拒绝
        let result = security.validate_path(Path::new("../etc/passwd"));
        assert!(matches!(result, Err(SecurityError::PathTraversal(_))));

        // 嵌套的 ".." 也应该被拒绝
        let result = security.validate_path(Path::new("subdir/../../etc/passwd"));
        assert!(matches!(result, Err(SecurityError::PathTraversal(_))));

        // 中间包含 ".." 的路径也应该被拒绝
        let result = security.validate_path(Path::new("subdir/../../../etc/passwd"));
        assert!(matches!(result, Err(SecurityError::PathTraversal(_))));
    }

    #[test]
    fn test_reject_outside_base_dir() {
        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        // 绝对路径指向基础目录外应该被拒绝
        let result = security.validate_path(Path::new("/etc/passwd"));
        assert!(matches!(result, Err(SecurityError::OutsideBaseDir(_))));

        // 另一个临时目录也应该被拒绝
        let other_temp = TempDir::new().unwrap();
        let other_file = other_temp.path().join("other.txt");
        fs::write(&other_file, "other content").unwrap();

        let result = security.validate_path(&other_file);
        assert!(matches!(result, Err(SecurityError::OutsideBaseDir(_))));
    }

    #[test]
    #[cfg(unix)]
    fn test_reject_symlink() {
        use std::os::unix::fs::symlink;

        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        // 创建符号链接
        let link_path = temp_dir.path().join("link.txt");
        let target_path = temp_dir.path().join("test.txt");
        symlink(&target_path, &link_path).unwrap();

        // 符号链接应该被拒绝
        let result = security.validate_path(Path::new("link.txt"));
        assert!(matches!(result, Err(SecurityError::SymlinkNotAllowed(_))));
    }

    #[test]
    fn test_quick_check() {
        let security = SecurityManager::default();

        // 正常路径应该通过
        assert!(security.quick_check(Path::new("test.txt")));
        assert!(security.quick_check(Path::new("subdir/file.txt")));

        // 包含 ".." 的路径应该失败
        assert!(!security.quick_check(Path::new("../test.txt")));
        assert!(!security.quick_check(Path::new("subdir/../test.txt")));
    }

    #[test]
    fn test_set_base_dir() {
        let temp_dir = setup_test_dir();
        let mut security = SecurityManager::default();

        security.set_base_dir(temp_dir.path());
        assert_eq!(security.base_dir(), temp_dir.path());
    }

    #[test]
    fn test_validate_new_file_path() {
        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        // 新文件（不存在）在基础目录内应该通过
        let result = security.validate_path(Path::new("new_file.txt"));
        assert!(result.is_ok());

        // 新文件在子目录内也应该通过
        let result = security.validate_path(Path::new("subdir/new_file.txt"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_no_symlink_check() {
        let temp_dir = setup_test_dir();
        let security = SecurityManager::new(temp_dir.path());

        // 正常路径应该通过
        let result = security.validate_path_no_symlink_check(Path::new("test.txt"));
        assert!(result.is_ok());

        // 包含 ".." 的路径应该被拒绝
        let result = security.validate_path_no_symlink_check(Path::new("../test.txt"));
        assert!(matches!(result, Err(SecurityError::PathTraversal(_))));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    /// 生成有效的文件名（不包含特殊字符）
    fn arb_valid_filename() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,20}\\.[a-z]{1,4}"
    }

    /// 生成有效的目录名
    fn arb_valid_dirname() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,15}"
    }

    /// 生成包含 ".." 的路径
    fn arb_path_with_parent_dir() -> impl Strategy<Value = PathBuf> {
        prop_oneof![
            // 开头的 ..
            arb_valid_filename().prop_map(|f| PathBuf::from(format!("../{}", f))),
            // 中间的 ..
            (arb_valid_dirname(), arb_valid_filename())
                .prop_map(|(d, f)| PathBuf::from(format!("{}/../{}", d, f))),
            // 多个 ..
            arb_valid_filename().prop_map(|f| PathBuf::from(format!("../../{}", f))),
            // 嵌套的 ..
            (arb_valid_dirname(), arb_valid_filename())
                .prop_map(|(d, f)| PathBuf::from(format!("{}/../../{}", d, f))),
        ]
    }

    /// 生成不包含 ".." 的相对路径
    fn arb_safe_relative_path() -> impl Strategy<Value = PathBuf> {
        prop_oneof![
            // 单个文件名
            arb_valid_filename().prop_map(PathBuf::from),
            // 一级子目录
            (arb_valid_dirname(), arb_valid_filename())
                .prop_map(|(d, f)| PathBuf::from(format!("{}/{}", d, f))),
            // 两级子目录
            (
                arb_valid_dirname(),
                arb_valid_dirname(),
                arb_valid_filename()
            )
                .prop_map(|(d1, d2, f)| PathBuf::from(format!("{}/{}/{}", d1, d2, f))),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 14: 路径安全验证**
        /// **Validates: Requirements 8.1, 8.2, 8.5**
        ///
        /// *For any* 包含 ".." 组件或指向基础目录外的路径，
        /// Security Manager 应该拒绝操作并返回安全错误。
        #[test]
        fn prop_path_traversal_rejected(path in arb_path_with_parent_dir()) {
            let temp_dir = TempDir::new().unwrap();
            let security = SecurityManager::new(temp_dir.path());

            let result = security.validate_path(&path);

            prop_assert!(
                matches!(result, Err(SecurityError::PathTraversal(_))),
                "包含 '..' 的路径 {:?} 应该被拒绝，但结果是 {:?}",
                path,
                result
            );
        }

        /// **Feature: agent-tool-calling, Property 14: 路径安全验证 - 安全路径通过**
        /// **Validates: Requirements 8.1, 8.2, 8.5**
        ///
        /// *For any* 不包含 ".." 且在基础目录内的路径，
        /// Security Manager 应该允许操作。
        #[test]
        fn prop_safe_path_accepted(path in arb_safe_relative_path()) {
            let temp_dir = TempDir::new().unwrap();

            // 创建必要的目录结构
            let full_path = temp_dir.path().join(&path);
            if let Some(parent) = full_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            // 创建文件
            let _ = fs::write(&full_path, "test content");

            let security = SecurityManager::new(temp_dir.path());
            let result = security.validate_path(&path);

            prop_assert!(
                result.is_ok(),
                "安全路径 {:?} 应该通过验证，但结果是 {:?}",
                path,
                result
            );
        }

        /// **Feature: agent-tool-calling, Property 14: 路径安全验证 - 快速检查一致性**
        /// **Validates: Requirements 8.1, 8.2**
        ///
        /// *For any* 路径，quick_check 返回 false 当且仅当路径包含 ".." 组件。
        #[test]
        fn prop_quick_check_consistency(path in arb_path_with_parent_dir()) {
            let security = SecurityManager::default();

            prop_assert!(
                !security.quick_check(&path),
                "包含 '..' 的路径 {:?} 的 quick_check 应该返回 false",
                path
            );
        }

        /// **Feature: agent-tool-calling, Property 14: 路径安全验证 - 安全路径快速检查**
        /// **Validates: Requirements 8.1, 8.2**
        #[test]
        fn prop_safe_path_quick_check(path in arb_safe_relative_path()) {
            let security = SecurityManager::default();

            prop_assert!(
                security.quick_check(&path),
                "不包含 '..' 的路径 {:?} 的 quick_check 应该返回 true",
                path
            );
        }
    }

    /// Property 15 符号链接拒绝测试（仅 Unix 平台）
    #[cfg(unix)]
    mod symlink_proptests {
        use super::*;
        use std::os::unix::fs::symlink;

        /// 生成符号链接测试场景
        fn arb_symlink_scenario() -> impl Strategy<Value = (String, String)> {
            // 生成目标文件名和链接文件名
            (
                "[a-zA-Z][a-zA-Z0-9_-]{0,10}\\.[a-z]{1,3}",
                "[a-zA-Z][a-zA-Z0-9_-]{0,10}_link\\.[a-z]{1,3}",
            )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            /// **Feature: agent-tool-calling, Property 15: 符号链接拒绝**
            /// **Validates: Requirements 8.3**
            ///
            /// *For any* 指向符号链接的路径，Security Manager 应该拒绝操作并返回安全错误。
            #[test]
            fn prop_symlink_rejected((target_name, link_name) in arb_symlink_scenario()) {
                let temp_dir = TempDir::new().unwrap();

                // 创建目标文件
                let target_path = temp_dir.path().join(&target_name);
                fs::write(&target_path, "target content").unwrap();

                // 创建符号链接
                let link_path = temp_dir.path().join(&link_name);
                symlink(&target_path, &link_path).unwrap();

                let security = SecurityManager::new(temp_dir.path());
                let result = security.validate_path(Path::new(&link_name));

                prop_assert!(
                    matches!(result, Err(SecurityError::SymlinkNotAllowed(_))),
                    "符号链接 {:?} 应该被拒绝，但结果是 {:?}",
                    link_name,
                    result
                );
            }

            /// **Feature: agent-tool-calling, Property 15: 符号链接拒绝 - 普通文件通过**
            /// **Validates: Requirements 8.3**
            ///
            /// *For any* 普通文件（非符号链接），Security Manager 应该允许操作。
            #[test]
            fn prop_regular_file_accepted(filename in "[a-zA-Z][a-zA-Z0-9_-]{0,15}\\.[a-z]{1,4}") {
                let temp_dir = TempDir::new().unwrap();

                // 创建普通文件
                let file_path = temp_dir.path().join(&filename);
                fs::write(&file_path, "regular file content").unwrap();

                let security = SecurityManager::new(temp_dir.path());
                let result = security.validate_path(Path::new(&filename));

                prop_assert!(
                    result.is_ok(),
                    "普通文件 {:?} 应该通过验证，但结果是 {:?}",
                    filename,
                    result
                );
            }

            /// **Feature: agent-tool-calling, Property 15: 符号链接拒绝 - 目录符号链接**
            /// **Validates: Requirements 8.3**
            ///
            /// *For any* 指向目录的符号链接，Security Manager 应该拒绝操作。
            #[test]
            fn prop_dir_symlink_rejected(
                (dir_name, link_name) in (
                    "[a-zA-Z][a-zA-Z0-9_-]{0,10}",
                    "[a-zA-Z][a-zA-Z0-9_-]{0,10}_dirlink"
                )
            ) {
                let temp_dir = TempDir::new().unwrap();

                // 创建目标目录
                let target_dir = temp_dir.path().join(&dir_name);
                fs::create_dir(&target_dir).unwrap();

                // 创建指向目录的符号链接
                let link_path = temp_dir.path().join(&link_name);
                symlink(&target_dir, &link_path).unwrap();

                let security = SecurityManager::new(temp_dir.path());
                let result = security.validate_path(Path::new(&link_name));

                prop_assert!(
                    matches!(result, Err(SecurityError::SymlinkNotAllowed(_))),
                    "目录符号链接 {:?} 应该被拒绝，但结果是 {:?}",
                    link_name,
                    result
                );
            }
        }
    }
}
