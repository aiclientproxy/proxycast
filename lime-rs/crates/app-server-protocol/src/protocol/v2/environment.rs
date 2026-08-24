use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::path::Path;
use url::Url;

/// Canonical file URI used for environment-owned paths.
///
/// The URI form preserves the target environment's path convention across
/// desktop platforms; it is intentionally narrower than a general URL.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(description = "Canonical file URI owned by an execution environment")]
pub struct PathUri(String);

impl PathUri {
    pub fn parse(value: &str) -> Result<Self, String> {
        let url = Url::parse(value).map_err(|error| format!("invalid path URI: {error}"))?;
        if url.scheme() != "file" {
            return Err(format!("unsupported path URI scheme `{}`", url.scheme()));
        }
        if !url.username().is_empty()
            || url.password().is_some()
            || url.port().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err("path URI cannot contain credentials, port, query, or fragment".into());
        }
        if urlencoding::decode_binary(url.path().as_bytes()).contains(&0) {
            return Err("path URI cannot contain NUL".into());
        }
        let mut url = url;
        if url.host_str() == Some("localhost") {
            url.set_host(None)
                .map_err(|_| "invalid localhost path URI".to_string())?;
        }
        Ok(Self(url.to_string()))
    }

    pub fn from_host_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(format!("path '{}' must be absolute", path.display()));
        }
        let url = Url::from_file_path(path).map_err(|()| {
            format!(
                "path '{}' cannot be represented as a file URI",
                path.display()
            )
        })?;
        Self::parse(url.as_str())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PathUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for PathUri {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for PathUri {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentAddParams {
    pub environment_id: String,
    pub exec_server_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentAddResponse {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentConnectionNotification {
    pub thread_id: String,
    pub environment_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentInfoParams {
    pub environment_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentInfoResponse {
    pub shell: EnvironmentShellInfo,
    pub cwd: Option<PathUri>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentStatusParams {
    pub environment_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentStatusResponse {
    pub status: EnvironmentStatusKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum EnvironmentStatusKind {
    Ready,
    Pending,
    Disconnected,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentShellInfo {
    pub name: String,
    pub path: String,
}
