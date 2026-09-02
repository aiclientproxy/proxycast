//! 语音输入配置服务
//!
//! 管理语音输入配置、ASR 凭证与润色指令。
//! 不依赖桌面宿主，可被 App Server 复用。

use lime_core::config::{
    load_config, save_config, AsrCredentialEntry, AsrProviderType, BaiduConfig, OpenAIAsrConfig,
    SenseVoiceLocalConfig, VoiceInputConfig, VoiceInstruction, VoiceOutputMode, WhisperLocalConfig,
    XunfeiConfig,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const SENSEVOICE_MODEL_FILE: &str = "model.int8.onnx";
const SENSEVOICE_TOKENS_FILE: &str = "tokens.txt";
const SENSEVOICE_VAD_FILE: &str = "silero_vad.onnx";

/// 添加 ASR 凭证的服务层输入。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddAsrCredentialRequest {
    pub provider: AsrProviderType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default = "default_asr_language_input")]
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whisper_config: Option<WhisperLocalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensevoice_config: Option<SenseVoiceLocalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xunfei_config: Option<XunfeiConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baidu_config: Option<BaiduConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_config: Option<OpenAIAsrConfig>,
}

/// ASR 凭证测试结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrCredentialTestResult {
    pub success: bool,
    pub message: String,
}

fn default_asr_language_input() -> String {
    "zh".to_string()
}

/// 加载语音输入配置
pub fn load_voice_config() -> Result<VoiceInputConfig, String> {
    let config = load_config().map_err(|e| e.to_string())?;
    Ok(config.experimental.voice_input)
}

/// 保存语音输入配置
pub fn save_voice_config(voice_config: VoiceInputConfig) -> Result<(), String> {
    let mut config = load_config().map_err(|e| e.to_string())?;
    config.experimental.voice_input = voice_config;
    save_config(&config).map_err(|e| e.to_string())?;
    Ok(())
}

/// 获取默认 ASR 凭证
pub fn get_default_asr_credential() -> Result<Option<AsrCredentialEntry>, String> {
    let config = load_config().map_err(|e| e.to_string())?;
    Ok(config
        .experimental
        .voice_input
        .asr_credentials
        .into_iter()
        .find(|credential| credential.is_default && !credential.disabled))
}

/// 获取指定 ID 的 ASR 凭证
pub fn get_asr_credential(id: &str) -> Result<Option<AsrCredentialEntry>, String> {
    let config = load_config().map_err(|e| e.to_string())?;
    Ok(config
        .experimental
        .voice_input
        .asr_credentials
        .into_iter()
        .find(|credential| credential.id == id))
}

/// 列出所有 ASR 凭证
pub fn list_asr_credentials() -> Result<Vec<AsrCredentialEntry>, String> {
    let config = load_config().map_err(|e| e.to_string())?;
    Ok(config.experimental.voice_input.asr_credentials)
}

/// 添加 ASR 凭证。
pub fn add_asr_credential(entry: AddAsrCredentialRequest) -> Result<AsrCredentialEntry, String> {
    let mut config = load_config().map_err(|e| e.to_string())?;
    let credentials = &mut config.experimental.voice_input.asr_credentials;
    let mut new_entry = AsrCredentialEntry {
        id: Uuid::new_v4().to_string(),
        provider: entry.provider,
        name: entry.name,
        is_default: entry.is_default || credentials.is_empty(),
        disabled: entry.disabled,
        language: entry.language,
        whisper_config: entry.whisper_config,
        sensevoice_config: entry.sensevoice_config,
        xunfei_config: entry.xunfei_config,
        baidu_config: entry.baidu_config,
        openai_config: entry.openai_config,
    };

    if new_entry.is_default {
        for credential in credentials.iter_mut() {
            credential.is_default = false;
        }
    }

    credentials.push(new_entry.clone());
    normalize_asr_defaults(credentials);
    new_entry = credentials
        .iter()
        .find(|credential| credential.id == new_entry.id)
        .cloned()
        .ok_or_else(|| "ASR 凭证创建后未找到".to_string())?;
    save_config(&config).map_err(|e| e.to_string())?;
    Ok(new_entry)
}

/// 更新 ASR 凭证。
pub fn update_asr_credential(entry: AsrCredentialEntry) -> Result<(), String> {
    let mut config = load_config().map_err(|e| e.to_string())?;
    let credentials = &mut config.experimental.voice_input.asr_credentials;
    let index = credentials
        .iter()
        .position(|credential| credential.id == entry.id)
        .ok_or_else(|| format!("凭证不存在: {}", entry.id))?;

    let is_default = entry.is_default;
    credentials[index] = entry;
    if is_default {
        let id = credentials[index].id.clone();
        for credential in credentials.iter_mut() {
            credential.is_default = credential.id == id;
        }
    }
    normalize_asr_defaults(credentials);
    save_config(&config).map_err(|e| e.to_string())
}

/// 删除 ASR 凭证。
pub fn delete_asr_credential(id: &str) -> Result<(), String> {
    let mut config = load_config().map_err(|e| e.to_string())?;
    let credentials = &mut config.experimental.voice_input.asr_credentials;
    let index = credentials
        .iter()
        .position(|credential| credential.id == id)
        .ok_or_else(|| format!("凭证不存在: {id}"))?;

    credentials.remove(index);
    normalize_asr_defaults(credentials);
    save_config(&config).map_err(|e| e.to_string())
}

/// 设置默认 ASR 凭证。
pub fn set_default_asr_credential(id: &str) -> Result<(), String> {
    let mut config = load_config().map_err(|e| e.to_string())?;
    let credentials = &mut config.experimental.voice_input.asr_credentials;
    if !credentials.iter().any(|credential| credential.id == id) {
        return Err(format!("凭证不存在: {id}"));
    }

    for credential in credentials.iter_mut() {
        credential.is_default = credential.id == id;
    }
    save_config(&config).map_err(|e| e.to_string())
}

/// 将已安装的 SenseVoice 本地模型设置为默认 ASR 凭证。
pub fn upsert_sensevoice_local_default(
    model_id: &str,
    model_dir: &str,
    vad_model_id: &str,
) -> Result<AsrCredentialEntry, String> {
    let mut config = load_config().map_err(|e| e.to_string())?;
    let credentials = &mut config.experimental.voice_input.asr_credentials;
    for credential in credentials.iter_mut() {
        credential.is_default = false;
    }

    let credential = match credentials.iter_mut().find(|credential| {
        credential.provider == AsrProviderType::SenseVoiceLocal
            && credential
                .sensevoice_config
                .as_ref()
                .map(|config| config.model_id.as_str() == model_id)
                .unwrap_or(false)
    }) {
        Some(existing) => {
            existing.name = Some("SenseVoice Small 本地".to_string());
            existing.disabled = false;
            existing.is_default = true;
            existing.language = "auto".to_string();
            existing.sensevoice_config = Some(SenseVoiceLocalConfig {
                model_id: model_id.to_string(),
                model_dir: Some(model_dir.to_string()),
                use_itn: true,
                num_threads: 4,
                vad_model_id: Some(vad_model_id.to_string()),
            });
            existing.clone()
        }
        None => {
            let entry = AsrCredentialEntry {
                id: format!("sensevoice-local-{model_id}"),
                provider: AsrProviderType::SenseVoiceLocal,
                name: Some("SenseVoice Small 本地".to_string()),
                is_default: true,
                disabled: false,
                language: "auto".to_string(),
                whisper_config: None,
                sensevoice_config: Some(SenseVoiceLocalConfig {
                    model_id: model_id.to_string(),
                    model_dir: Some(model_dir.to_string()),
                    use_itn: true,
                    num_threads: 4,
                    vad_model_id: Some(vad_model_id.to_string()),
                }),
                xunfei_config: None,
                baidu_config: None,
                openai_config: None,
            };
            credentials.push(entry.clone());
            entry
        }
    };

    save_config(&config).map_err(|e| e.to_string())?;
    Ok(credential)
}

/// 测试 ASR 凭证。
///
/// 该入口只返回真实可判断的本地 readiness；云端 Provider 在未接入真实 probe 前
/// 必须 fail closed，避免把“配置字段存在”伪装成连通成功。
pub fn test_asr_credential(id: &str) -> Result<AsrCredentialTestResult, String> {
    let credential = get_asr_credential(id)?.ok_or_else(|| format!("凭证不存在: {id}"))?;

    let result = match credential.provider {
        AsrProviderType::WhisperLocal => test_whisper_local_credential(&credential),
        AsrProviderType::SenseVoiceLocal => test_sensevoice_local_credential(&credential),
        AsrProviderType::Xunfei => match credential.xunfei_config {
            Some(_) => AsrCredentialTestResult {
                success: false,
                message: "讯飞 ASR 真实连通性探测尚未接入 App Server current，已拒绝返回假成功"
                    .to_string(),
            },
            None => AsrCredentialTestResult {
                success: false,
                message: "讯飞配置缺失".to_string(),
            },
        },
        AsrProviderType::Baidu => match credential.baidu_config {
            Some(_) => AsrCredentialTestResult {
                success: false,
                message: "百度 ASR 真实连通性探测尚未接入 App Server current，已拒绝返回假成功"
                    .to_string(),
            },
            None => AsrCredentialTestResult {
                success: false,
                message: "百度配置缺失".to_string(),
            },
        },
        AsrProviderType::OpenAI => match credential.openai_config {
            Some(_) => AsrCredentialTestResult {
                success: false,
                message: "OpenAI ASR 真实连通性探测尚未接入 App Server current，已拒绝返回假成功"
                    .to_string(),
            },
            None => AsrCredentialTestResult {
                success: false,
                message: "OpenAI 配置缺失".to_string(),
            },
        },
    };

    Ok(result)
}

/// 获取首个启用的指定 Provider 凭证
pub fn get_enabled_asr_credential_by_provider(
    provider: AsrProviderType,
) -> Result<Option<AsrCredentialEntry>, String> {
    let config = load_config().map_err(|e| e.to_string())?;
    Ok(config
        .experimental
        .voice_input
        .asr_credentials
        .into_iter()
        .find(|credential| credential.provider == provider && !credential.disabled))
}

/// 获取指令列表
pub fn get_instructions() -> Result<Vec<VoiceInstruction>, String> {
    let config = load_config().map_err(|e| e.to_string())?;
    Ok(config.experimental.voice_input.instructions)
}

/// 获取指定 ID 的指令
pub fn get_instruction(id: &str) -> Result<Option<VoiceInstruction>, String> {
    let instructions = get_instructions()?;
    Ok(instructions
        .into_iter()
        .find(|instruction| instruction.id == id))
}

/// 保存或更新语音指令
pub fn save_voice_instruction(instruction: VoiceInstruction) -> Result<(), String> {
    let mut voice_config = load_voice_config()?;

    if let Some(index) = voice_config
        .instructions
        .iter()
        .position(|item| item.id == instruction.id)
    {
        voice_config.instructions[index] = instruction;
    } else {
        voice_config.instructions.push(instruction);
    }

    save_voice_config(voice_config)
}

/// 删除语音指令（预设指令不可删除）
pub fn delete_voice_instruction(id: &str) -> Result<(), String> {
    let mut voice_config = load_voice_config()?;

    if let Some(instruction) = voice_config.instructions.iter().find(|item| item.id == id) {
        if instruction.is_preset {
            return Err("无法删除预设指令".to_string());
        }
    }

    voice_config.instructions.retain(|item| item.id != id);

    if !voice_config
        .instructions
        .iter()
        .any(|item| item.id == voice_config.processor.default_instruction_id)
    {
        voice_config.processor.default_instruction_id = voice_config
            .instructions
            .iter()
            .find(|item| item.id == "default")
            .or_else(|| voice_config.instructions.first())
            .map(|item| item.id.clone())
            .unwrap_or_else(|| "default".to_string());
    }

    if !voice_config
        .instructions
        .iter()
        .any(|item| item.id == voice_config.translate_instruction_id)
    {
        voice_config.translate_instruction_id = voice_config
            .instructions
            .iter()
            .find(|item| item.id == "translate_en")
            .map(|item| item.id.clone())
            .unwrap_or_else(|| voice_config.processor.default_instruction_id.clone());
    }

    save_voice_config(voice_config)
}

/// 解析输出模式
///
/// 当 `mode` 为 `None` 时，返回配置中的默认输出模式。
pub fn resolve_output_mode(mode: Option<&str>) -> Result<VoiceOutputMode, String> {
    match mode {
        Some("type") => Ok(VoiceOutputMode::Type),
        Some("clipboard") => Ok(VoiceOutputMode::Clipboard),
        Some("both") => Ok(VoiceOutputMode::Both),
        None => {
            let voice_config = load_voice_config()?;
            Ok(voice_config.output.mode)
        }
        Some(other) => Err(format!("未知的输出模式: {other}")),
    }
}

/// 获取 ASR Provider 展示名
pub fn asr_provider_name(provider: AsrProviderType) -> &'static str {
    match provider {
        AsrProviderType::WhisperLocal => "本地 Whisper",
        AsrProviderType::SenseVoiceLocal => "SenseVoice Small 本地",
        AsrProviderType::OpenAI => "OpenAI Whisper",
        AsrProviderType::Baidu => "百度语音",
        AsrProviderType::Xunfei => "讯飞语音",
    }
}

fn normalize_asr_defaults(credentials: &mut [AsrCredentialEntry]) {
    let mut seen_default = false;
    for credential in credentials.iter_mut() {
        if credential.is_default {
            if seen_default {
                credential.is_default = false;
            } else {
                seen_default = true;
            }
        }
    }

    if !seen_default {
        if let Some(first) = credentials.first_mut() {
            first.is_default = true;
        }
    }
}

fn test_whisper_local_credential(credential: &AsrCredentialEntry) -> AsrCredentialTestResult {
    let Some(config) = credential.whisper_config.as_ref() else {
        return AsrCredentialTestResult {
            success: false,
            message: "Whisper 本地配置缺失".to_string(),
        };
    };

    match resolve_whisper_model_path(config).and_then(|model_path| {
        if model_path.is_file() {
            Ok(())
        } else {
            Err(format!("Whisper 模型文件不存在: {}", model_path.display()))
        }
    }) {
        Ok(()) => AsrCredentialTestResult {
            success: true,
            message: "本地 Whisper 模型已就绪".to_string(),
        },
        Err(error) => AsrCredentialTestResult {
            success: false,
            message: error,
        },
    }
}

fn test_sensevoice_local_credential(credential: &AsrCredentialEntry) -> AsrCredentialTestResult {
    let Some(config) = credential.sensevoice_config.as_ref() else {
        return AsrCredentialTestResult {
            success: false,
            message: "SenseVoice 本地配置缺失".to_string(),
        };
    };

    match resolve_sensevoice_model_dir(config).and_then(|model_dir| {
        ensure_required_files(
            &model_dir,
            &[
                SENSEVOICE_MODEL_FILE,
                SENSEVOICE_TOKENS_FILE,
                SENSEVOICE_VAD_FILE,
            ],
        )
    }) {
        Ok(()) => AsrCredentialTestResult {
            success: true,
            message: "SenseVoice Small 本地模型已就绪".to_string(),
        },
        Err(error) => AsrCredentialTestResult {
            success: false,
            message: error,
        },
    }
}

pub(crate) fn resolve_whisper_model_path(config: &WhisperLocalConfig) -> Result<PathBuf, String> {
    resolve_absolute_model_path(config.model_path.as_deref(), "Whisper model_path")
}

pub(crate) fn resolve_sensevoice_model_dir(
    config: &SenseVoiceLocalConfig,
) -> Result<PathBuf, String> {
    resolve_absolute_model_path(config.model_dir.as_deref(), "SenseVoice model_dir")
}

fn resolve_absolute_model_path(value: Option<&str>, field: &str) -> Result<PathBuf, String> {
    let value = value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{field} 未配置，模型路径必须由 Desktop Host 或显式配置提供"))?;
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{field} 必须是绝对路径: {}", path.display()));
    }
    Ok(path)
}

fn ensure_required_files(model_dir: &Path, required_files: &[&str]) -> Result<(), String> {
    let missing_files = required_files
        .iter()
        .filter(|file_name| !model_dir.join(file_name).is_file())
        .copied()
        .collect::<Vec<_>>();

    if missing_files.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "本地模型文件不完整，请先在设置 -> 语音模型中下载；缺失文件: {}",
            missing_files.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_model_paths_require_explicit_absolute_config() {
        let whisper = WhisperLocalConfig::default();
        let sensevoice = SenseVoiceLocalConfig::default();

        assert!(resolve_whisper_model_path(&whisper)
            .expect_err("missing Whisper path must fail")
            .contains("Whisper model_path 未配置"));
        assert!(resolve_sensevoice_model_dir(&sensevoice)
            .expect_err("missing SenseVoice path must fail")
            .contains("SenseVoice model_dir 未配置"));

        let whisper = WhisperLocalConfig {
            model_path: Some("models/whisper/model.bin".to_string()),
            ..WhisperLocalConfig::default()
        };
        let sensevoice = SenseVoiceLocalConfig {
            model_dir: Some("models/voice/sensevoice".to_string()),
            ..SenseVoiceLocalConfig::default()
        };
        assert!(resolve_whisper_model_path(&whisper)
            .expect_err("relative Whisper path must fail")
            .contains("必须是绝对路径"));
        assert!(resolve_sensevoice_model_dir(&sensevoice)
            .expect_err("relative SenseVoice path must fail")
            .contains("必须是绝对路径"));
    }

    #[test]
    fn local_model_paths_accept_explicit_absolute_config() {
        let temp = tempfile::tempdir().expect("tempdir");
        let whisper_path = temp.path().join("ggml-base.bin");
        let sensevoice_dir = temp.path().join("sensevoice");
        let whisper = WhisperLocalConfig {
            model_path: Some(whisper_path.to_string_lossy().to_string()),
            ..WhisperLocalConfig::default()
        };
        let sensevoice = SenseVoiceLocalConfig {
            model_dir: Some(sensevoice_dir.to_string_lossy().to_string()),
            ..SenseVoiceLocalConfig::default()
        };

        assert_eq!(
            resolve_whisper_model_path(&whisper).expect("Whisper path"),
            whisper_path
        );
        assert_eq!(
            resolve_sensevoice_model_dir(&sensevoice).expect("SenseVoice path"),
            sensevoice_dir
        );
    }
}
