//! 项目相关错误类型
//!
//! 定义统一内容创作系统中的错误类型，包括：
//! - ProjectError（项目错误）
//! - PersonaError（人设错误）
//! - MaterialError（素材错误）
//! - MigrationError（迁移错误）
//!
//! ## 设计原则
//! - 使用 thiserror 派生 Error trait
//! - 支持 From 转换以便错误传播
//! - 实现 Serialize 以支持 App Server 协议返回
//!
//! ## 相关需求
//! - Requirements 2.4: 迁移错误处理
//! - Requirements 11.6: 默认项目保护

use thiserror::Error;

// ============================================================================
// 项目错误
// ============================================================================

/// 项目操作错误
///
/// 涵盖项目 CRUD 操作中可能出现的所有错误情况。
#[derive(Error, Debug)]
pub enum ProjectError {
    /// 项目不存在
    #[error("项目不存在: {0}")]
    NotFound(String),

    /// 无法删除默认项目
    #[error("无法删除默认项目")]
    CannotDeleteDefault,

    /// 无法归档默认项目
    #[error("无法归档默认项目")]
    CannotArchiveDefault,

    /// 项目名称已存在
    #[error("项目名称已存在: {0}")]
    NameAlreadyExists(String),

    /// 数据库错误
    #[error("数据库错误: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<ProjectError> for String {
    fn from(err: ProjectError) -> Self {
        err.to_string()
    }
}

impl serde::Serialize for ProjectError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// 人设错误
// ============================================================================

/// 人设操作错误
///
/// 涵盖人设 CRUD 操作中可能出现的所有错误情况。
#[derive(Error, Debug)]
pub enum PersonaError {
    /// 人设不存在
    #[error("人设不存在: {0}")]
    NotFound(String),

    /// 项目不存在
    #[error("项目不存在: {0}")]
    ProjectNotFound(String),

    /// 人设名称已存在
    #[error("人设名称已存在: {0}")]
    NameAlreadyExists(String),

    /// 数据库错误
    #[error("数据库错误: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl From<PersonaError> for String {
    fn from(err: PersonaError) -> Self {
        err.to_string()
    }
}

impl serde::Serialize for PersonaError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// 素材错误
// ============================================================================

/// 素材操作错误
///
/// 涵盖素材上传、存储、删除等操作中可能出现的所有错误情况。
#[derive(Error, Debug)]
pub enum MaterialError {
    /// 素材不存在
    #[error("素材不存在: {0}")]
    NotFound(String),

    /// 项目不存在
    #[error("项目不存在: {0}")]
    ProjectNotFound(String),

    /// 不支持的文件类型
    #[error("不支持的文件类型: {0}")]
    UnsupportedFileType(String),

    /// 文件过大
    #[error("文件过大: {0} bytes (最大 {1} bytes)")]
    FileTooLarge(u64, u64),

    /// 文件读取失败
    #[error("文件读取失败: {0}")]
    FileReadError(String),

    /// 数据库错误
    #[error("数据库错误: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<MaterialError> for String {
    fn from(err: MaterialError) -> Self {
        err.to_string()
    }
}

impl serde::Serialize for MaterialError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// 发布配置错误
// ============================================================================

/// 发布配置操作错误
///
/// 涵盖发布配置 CRUD 操作中可能出现的所有错误情况。
#[derive(Error, Debug)]
pub enum PublishConfigError {
    /// 发布配置不存在
    #[error("发布配置不存在: {0}")]
    NotFound(String),

    /// 项目不存在
    #[error("项目不存在: {0}")]
    ProjectNotFound(String),

    /// 平台配置已存在
    #[error("平台配置已存在: 项目 {0} 平台 {1}")]
    PlatformAlreadyExists(String, String),

    /// 不支持的平台
    #[error("不支持的平台: {0}")]
    UnsupportedPlatform(String),

    /// 数据库错误
    #[error("数据库错误: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl From<PublishConfigError> for String {
    fn from(err: PublishConfigError) -> Self {
        err.to_string()
    }
}

impl serde::Serialize for PublishConfigError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// 迁移错误
// ============================================================================

/// 数据迁移错误
///
/// 涵盖数据库迁移过程中可能出现的所有错误情况。
/// 主要用于现有话题迁移到默认项目的场景。
#[derive(Error, Debug)]
pub enum MigrationError {
    /// 迁移失败
    #[error("迁移失败: {0}")]
    MigrationFailed(String),

    /// 数据库错误
    #[error("数据库错误: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl From<MigrationError> for String {
    fn from(err: MigrationError) -> Self {
        err.to_string()
    }
}

impl serde::Serialize for MigrationError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_error_display() {
        let err = ProjectError::NotFound("test-project".to_string());
        assert_eq!(err.to_string(), "项目不存在: test-project");

        let err = ProjectError::CannotDeleteDefault;
        assert_eq!(err.to_string(), "无法删除默认项目");

        let err = ProjectError::CannotArchiveDefault;
        assert_eq!(err.to_string(), "无法归档默认项目");

        let err = ProjectError::NameAlreadyExists("我的项目".to_string());
        assert_eq!(err.to_string(), "项目名称已存在: 我的项目");
    }

    #[test]
    fn test_persona_error_display() {
        let err = PersonaError::NotFound("persona-1".to_string());
        assert_eq!(err.to_string(), "人设不存在: persona-1");

        let err = PersonaError::ProjectNotFound("project-1".to_string());
        assert_eq!(err.to_string(), "项目不存在: project-1");

        let err = PersonaError::NameAlreadyExists("专业写手".to_string());
        assert_eq!(err.to_string(), "人设名称已存在: 专业写手");
    }

    #[test]
    fn test_material_error_display() {
        let err = MaterialError::NotFound("mat-1".to_string());
        assert_eq!(err.to_string(), "素材不存在: mat-1");

        let err = MaterialError::ProjectNotFound("project-1".to_string());
        assert_eq!(err.to_string(), "项目不存在: project-1");

        let err = MaterialError::UnsupportedFileType(".exe".to_string());
        assert_eq!(err.to_string(), "不支持的文件类型: .exe");

        let err = MaterialError::FileTooLarge(10_000_000, 5_000_000);
        assert_eq!(
            err.to_string(),
            "文件过大: 10000000 bytes (最大 5000000 bytes)"
        );

        let err = MaterialError::FileReadError("权限不足".to_string());
        assert_eq!(err.to_string(), "文件读取失败: 权限不足");
    }

    #[test]
    fn test_migration_error_display() {
        let err = MigrationError::MigrationFailed("表不存在".to_string());
        assert_eq!(err.to_string(), "迁移失败: 表不存在");
    }

    #[test]
    fn test_project_error_to_string() {
        let err = ProjectError::NotFound("test".to_string());
        let s: String = err.into();
        assert_eq!(s, "项目不存在: test");
    }

    #[test]
    fn test_persona_error_to_string() {
        let err = PersonaError::NotFound("test".to_string());
        let s: String = err.into();
        assert_eq!(s, "人设不存在: test");
    }

    #[test]
    fn test_material_error_to_string() {
        let err = MaterialError::NotFound("test".to_string());
        let s: String = err.into();
        assert_eq!(s, "素材不存在: test");
    }

    #[test]
    fn test_migration_error_to_string() {
        let err = MigrationError::MigrationFailed("test".to_string());
        let s: String = err.into();
        assert_eq!(s, "迁移失败: test");
    }

    #[test]
    fn test_project_error_serialize() {
        let err = ProjectError::CannotDeleteDefault;
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(json, "\"无法删除默认项目\"");
    }

    #[test]
    fn test_persona_error_serialize() {
        let err = PersonaError::NotFound("test".to_string());
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(json, "\"人设不存在: test\"");
    }

    #[test]
    fn test_material_error_serialize() {
        let err = MaterialError::FileTooLarge(100, 50);
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(json, "\"文件过大: 100 bytes (最大 50 bytes)\"");
    }

    #[test]
    fn test_migration_error_serialize() {
        let err = MigrationError::MigrationFailed("test".to_string());
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(json, "\"迁移失败: test\"");
    }

    #[test]
    fn test_publish_config_error_display() {
        let err = PublishConfigError::NotFound("config-1".to_string());
        assert_eq!(err.to_string(), "发布配置不存在: config-1");

        let err = PublishConfigError::ProjectNotFound("project-1".to_string());
        assert_eq!(err.to_string(), "项目不存在: project-1");

        let err = PublishConfigError::PlatformAlreadyExists(
            "project-1".to_string(),
            "xiaohongshu".to_string(),
        );
        assert_eq!(
            err.to_string(),
            "平台配置已存在: 项目 project-1 平台 xiaohongshu"
        );

        let err = PublishConfigError::UnsupportedPlatform("unknown".to_string());
        assert_eq!(err.to_string(), "不支持的平台: unknown");
    }

    #[test]
    fn test_publish_config_error_to_string() {
        let err = PublishConfigError::NotFound("test".to_string());
        let s: String = err.into();
        assert_eq!(s, "发布配置不存在: test");
    }

    #[test]
    fn test_publish_config_error_serialize() {
        let err = PublishConfigError::PlatformAlreadyExists("p1".to_string(), "wechat".to_string());
        let json = serde_json::to_string(&err).unwrap();
        assert_eq!(json, "\"平台配置已存在: 项目 p1 平台 wechat\"");
    }
}
