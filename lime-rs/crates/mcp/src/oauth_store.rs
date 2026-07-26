use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const STORE_PARENT_DIR_NAME: &str = "mcp";
const STORE_DIR_NAME: &str = "oauth";
const STORE_VERSION: u32 = 1;

/// MCP OAuth 凭据的唯一 owner 目录，必须由 AppDataRoot 显式派生。
/// 这里只做拼接，不解析平台根——解析只属于组合根。
pub fn oauth_store_root_for_app_data_root(app_data_root: impl AsRef<Path>) -> PathBuf {
    app_data_root
        .as_ref()
        .join(STORE_PARENT_DIR_NAME)
        .join(STORE_DIR_NAME)
}

/// MCP OAuth 凭据持久化存储。
///
/// key 同时包含 server name 与 URL，避免同名服务器换 URL 后复用旧 token。
#[derive(Debug, Clone)]
pub struct PersistentCredentialStore {
    server_name: String,
    server_url: String,
    root_dir: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct CredentialEnvelope {
    version: u32,
    server_name: String,
    server_url: String,
    credentials: StoredCredentials,
}

impl PersistentCredentialStore {
    pub fn new_in(
        root_dir: impl Into<PathBuf>,
        server_name: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Self {
        Self {
            server_name: server_name.into(),
            server_url: server_url.into(),
            root_dir: root_dir.into(),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.root_dir.join(format!(
            "{}-{}.json",
            sanitize_file_stem(&self.server_name),
            fnv1a64_hex(self.server_url.as_bytes())
        ))
    }

    pub async fn has_credentials(&self) -> Result<bool, AuthError> {
        self.load().await.map(|credentials| {
            credentials.is_some_and(|credentials| credentials.token_response.is_some())
        })
    }

    fn envelope(&self, credentials: StoredCredentials) -> CredentialEnvelope {
        CredentialEnvelope {
            version: STORE_VERSION,
            server_name: self.server_name.clone(),
            server_url: self.server_url.clone(),
            credentials,
        }
    }

    fn validate_envelope(
        &self,
        envelope: CredentialEnvelope,
    ) -> Result<StoredCredentials, AuthError> {
        if envelope.version != STORE_VERSION {
            return Err(AuthError::InternalError(format!(
                "unsupported MCP OAuth credential version: {}",
                envelope.version
            )));
        }
        if envelope.server_name != self.server_name || envelope.server_url != self.server_url {
            return Err(AuthError::InternalError(
                "MCP OAuth credential key mismatch".to_string(),
            ));
        }
        Ok(envelope.credentials)
    }
}

#[async_trait::async_trait]
impl CredentialStore for PersistentCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        let path = self.path();
        if !path.exists() {
            return Ok(None);
        }

        let contents = fs::read_to_string(&path).map_err(|error| {
            AuthError::InternalError(format!(
                "failed to read MCP OAuth credentials {}: {error}",
                path.display()
            ))
        })?;
        let envelope: CredentialEnvelope = serde_json::from_str(&contents).map_err(|error| {
            AuthError::InternalError(format!(
                "failed to parse MCP OAuth credentials {}: {error}",
                path.display()
            ))
        })?;
        self.validate_envelope(envelope).map(Some)
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        fs::create_dir_all(&self.root_dir).map_err(|error| {
            AuthError::InternalError(format!(
                "failed to create MCP OAuth credential dir {}: {error}",
                self.root_dir.display()
            ))
        })?;
        let path = self.path();
        let contents = serde_json::to_vec_pretty(&self.envelope(credentials)).map_err(|error| {
            AuthError::InternalError(format!(
                "failed to serialize MCP OAuth credentials: {error}"
            ))
        })?;
        write_private_file(&path, &contents)
    }

    async fn clear(&self) -> Result<(), AuthError> {
        let path = self.path();
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(AuthError::InternalError(format!(
                "failed to clear MCP OAuth credentials {}: {error}",
                path.display()
            ))),
        }
    }
}

fn sanitize_file_stem(value: &str) -> String {
    let mut stem = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while stem.contains("--") {
        stem = stem.replace("--", "-");
    }
    let stem = stem.trim_matches('-');
    if stem.is_empty() {
        "server".to_string()
    } else {
        stem.chars().take(80).collect()
    }
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(unix)]
fn write_private_file(path: &Path, contents: &[u8]) -> Result<(), AuthError> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| {
            AuthError::InternalError(format!(
                "failed to open MCP OAuth credentials {}: {error}",
                path.display()
            ))
        })?;
    file.write_all(contents).map_err(|error| {
        AuthError::InternalError(format!(
            "failed to write MCP OAuth credentials {}: {error}",
            path.display()
        ))
    })?;
    file.sync_all().map_err(|error| {
        AuthError::InternalError(format!(
            "failed to sync MCP OAuth credentials {}: {error}",
            path.display()
        ))
    })
}

#[cfg(not(unix))]
fn write_private_file(path: &Path, contents: &[u8]) -> Result<(), AuthError> {
    fs::write(path, contents).map_err(|error| {
        AuthError::InternalError(format!(
            "failed to write MCP OAuth credentials {}: {error}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn credentials(client_id: &str) -> StoredCredentials {
        StoredCredentials {
            client_id: client_id.to_string(),
            token_response: None,
        }
    }

    #[test]
    fn store_root_is_derived_from_injected_app_data_root() {
        assert_eq!(
            oauth_store_root_for_app_data_root("/tmp/machine-data"),
            PathBuf::from("/tmp/machine-data/mcp/oauth")
        );
    }

    #[tokio::test]
    async fn saves_and_loads_credentials_by_server_and_url() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = PersistentCredentialStore::new_in(
            temp.path(),
            "Docs Server",
            "https://example.com/mcp",
        );

        store.save(credentials("client-1")).await.expect("save");

        let restored = PersistentCredentialStore::new_in(
            temp.path(),
            "Docs Server",
            "https://example.com/mcp",
        )
        .load()
        .await
        .expect("load")
        .expect("credentials");

        assert_eq!(restored.client_id, "client-1");
    }

    #[tokio::test]
    async fn separates_same_server_name_by_url() {
        let temp = tempfile::tempdir().expect("tempdir");
        let first = PersistentCredentialStore::new_in(temp.path(), "docs", "https://one.test/mcp");
        let second = PersistentCredentialStore::new_in(temp.path(), "docs", "https://two.test/mcp");

        first
            .save(credentials("client-1"))
            .await
            .expect("save first");
        second
            .save(credentials("client-2"))
            .await
            .expect("save second");

        assert_eq!(
            first.load().await.expect("load first").unwrap().client_id,
            "client-1"
        );
        assert_eq!(
            second.load().await.expect("load second").unwrap().client_id,
            "client-2"
        );
    }

    #[tokio::test]
    async fn damaged_json_fails_closed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = PersistentCredentialStore::new_in(temp.path(), "docs", "https://example.com");
        fs::create_dir_all(temp.path()).expect("create dir");
        fs::write(store.path(), "{not-json").expect("write damaged json");

        let error = store
            .load()
            .await
            .expect_err("damaged JSON should fail closed");

        assert!(error
            .to_string()
            .contains("failed to parse MCP OAuth credentials"));
    }

    #[tokio::test]
    async fn clear_removes_credentials() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = PersistentCredentialStore::new_in(temp.path(), "docs", "https://example.com");

        store.save(credentials("client-1")).await.expect("save");
        assert!(store.load().await.expect("load").is_some());

        store.clear().await.expect("clear");

        assert!(store.load().await.expect("load after clear").is_none());
    }

    #[tokio::test]
    async fn has_credentials_requires_token_response() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = PersistentCredentialStore::new_in(temp.path(), "docs", "https://example.com");

        store.save(credentials("client-1")).await.expect("save");
        assert!(!store
            .has_credentials()
            .await
            .expect("empty token response should not authorize"));

        fs::write(
            store.path(),
            serde_json::json!({
                "version": 1,
                "server_name": "docs",
                "server_url": "https://example.com",
                "credentials": {
                    "client_id": "client-1",
                    "token_response": {
                        "access_token": "access",
                        "token_type": "Bearer",
                        "refresh_token": "refresh"
                    }
                }
            })
            .to_string(),
        )
        .expect("write token credentials");

        assert!(store
            .has_credentials()
            .await
            .expect("token response should authorize"));
    }
}
