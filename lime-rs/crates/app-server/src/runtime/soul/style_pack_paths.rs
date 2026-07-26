use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub(super) const STYLE_PACK_DATA_DIR: &str = "soul/style-packs";
pub(super) const REGISTRY_FILE_NAME: &str = "registry.json";
pub(super) const REQUIRED_LOCALES: [&str; 5] = ["zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"];

/// Soul style pack 是 AppDataRoot 下的产品资产。根由组合根注入一次，
/// 消费方（RuntimeCore soul 方法与 prompt style profile 链路）都只读取注入值。
/// 未注入时 fail closed，不解析平台默认根——旧的平台根旁路已删除。
static STYLE_PACK_DATA_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// 由 AppDataRoot 派生 style pack 资产根。只做拼接，不解析平台根。
pub fn style_pack_data_root_for_app_data_root(app_data_root: impl AsRef<Path>) -> PathBuf {
    app_data_root.as_ref().join(STYLE_PACK_DATA_DIR)
}

/// 组合根注入入口。同根重复安装幂等；不同根必须失败，避免进程继续写入首次注入的错误目录。
pub fn install_style_pack_data_root(app_data_root: impl AsRef<Path>) -> Result<(), String> {
    install_style_pack_root(
        &STYLE_PACK_DATA_ROOT,
        style_pack_data_root_for_app_data_root(app_data_root),
    )
}

fn install_style_pack_root(root: &OnceLock<PathBuf>, requested: PathBuf) -> Result<(), String> {
    if let Some(installed) = root.get() {
        return if installed == &requested {
            Ok(())
        } else {
            Err(format!(
                "Soul style pack 资产根已固定为 {}，拒绝切换到 {}",
                installed.display(),
                requested.display()
            ))
        };
    }

    root.set(requested)
        .map_err(|_| "Soul style pack 资产根并发安装失败".to_string())
}

pub(crate) fn style_pack_data_root() -> Result<PathBuf, String> {
    STYLE_PACK_DATA_ROOT
        .get()
        .cloned()
        .ok_or_else(|| "Soul style pack 资产根未注入；拒绝解析平台默认目录".to_string())
}

pub(super) fn validate_storage_id(id: &str) -> Result<(), String> {
    if !id.is_empty()
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
    {
        return Ok(());
    }
    Err(format!("Soul Style Pack id 不安全: {id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_pack_root_is_derived_from_injected_app_data_root() {
        assert_eq!(
            style_pack_data_root_for_app_data_root("/tmp/machine-data"),
            PathBuf::from("/tmp/machine-data/soul/style-packs")
        );
    }

    #[test]
    fn style_pack_root_install_is_idempotent_and_rejects_mismatch() {
        let root = OnceLock::new();
        let first = PathBuf::from("/tmp/machine-data/soul/style-packs");
        let second = PathBuf::from("/tmp/other-data/soul/style-packs");

        install_style_pack_root(&root, first.clone()).expect("first install");
        install_style_pack_root(&root, first).expect("same root should be idempotent");

        let error =
            install_style_pack_root(&root, second).expect_err("different root must fail closed");
        assert!(error.contains("拒绝切换"));
    }
}
