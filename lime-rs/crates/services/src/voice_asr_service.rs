//! ASR 服务
//!
//! 统一管理语音识别服务，支持本地 Whisper 和云端 ASR。
//!
//! ## 功能
//! - 本地 Whisper 识别（离线、隐私）
//! - OpenAI Whisper API
//! - 百度语音识别
//! - 讯飞语音识别（WebSocket 流式）
//!
//! ## 模型文件路径
//! 本地模型路径由 Desktop Host 或显式 credential 配置提供；服务层不解析平台数据根。
//!
//! 支持的模型：
//! - `ggml-tiny.bin` (~75MB)
//! - `ggml-base.bin` (~142MB)
//! - `ggml-small.bin` (~466MB)
//! - `ggml-medium.bin` (~1.5GB)
//!
//! ## 使用示例
//! ```rust,ignore
//! let credential = AsrService::get_default_credential()?.unwrap();
//! let text = AsrService::transcribe(&credential, &audio_data, 16000).await?;
//! ```

#[cfg(feature = "local-sensevoice")]
use std::path::PathBuf;

#[cfg(feature = "local-whisper")]
use lime_core::config::WhisperModelSize;
use lime_core::config::{AsrCredentialEntry, AsrProviderType};
#[cfg(feature = "local-sensevoice")]
use once_cell::sync::Lazy;
#[cfg(feature = "local-sensevoice")]
use parking_lot::Mutex;

use super::voice_config_service;
use model_provider::audio::{
    execute_transcription, AudioProviderConfig, TranscriptionProviderRequest,
};
use voice_core::asr_client::{AsrClient, BaiduClient, XunfeiClient};
use voice_core::types::AudioData;

#[cfg(feature = "local-sensevoice")]
const SENSEVOICE_MODEL_FILE: &str = "model.int8.onnx";
#[cfg(feature = "local-sensevoice")]
const SENSEVOICE_TOKENS_FILE: &str = "tokens.txt";
#[cfg(feature = "local-sensevoice")]
const SENSEVOICE_VAD_FILE: &str = "silero_vad.onnx";

#[cfg(feature = "local-sensevoice")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct SenseVoiceCacheKey {
    model_path: PathBuf,
    tokens_path: PathBuf,
    language: String,
    use_itn: bool,
    num_threads: u16,
}

#[cfg(feature = "local-sensevoice")]
struct CachedSenseVoiceTranscriber {
    key: SenseVoiceCacheKey,
    transcriber: voice_core::SenseVoiceTranscriber,
}

#[cfg(feature = "local-sensevoice")]
static SENSEVOICE_TRANSCRIBER_CACHE: Lazy<Mutex<Option<CachedSenseVoiceTranscriber>>> =
    Lazy::new(|| Mutex::new(None));

/// ASR 服务
pub struct AsrService;

impl AsrService {
    /// 获取默认 ASR 凭证
    pub fn get_default_credential() -> Result<Option<AsrCredentialEntry>, String> {
        voice_config_service::get_default_asr_credential()
    }

    /// 获取指定 ID 的 ASR 凭证
    pub fn get_credential(id: &str) -> Result<Option<AsrCredentialEntry>, String> {
        voice_config_service::get_asr_credential(id)
    }

    /// 使用指定凭证进行语音识别
    ///
    /// 当云端服务失败时，自动回退到本地 Whisper（需求 3.4）
    pub async fn transcribe(
        credential: &AsrCredentialEntry,
        audio_data: &[u8],
        sample_rate: u32,
    ) -> Result<String, String> {
        // 本地服务直接调用，不走云端回退链。
        match credential.provider {
            AsrProviderType::WhisperLocal => {
                return Self::transcribe_whisper_local(credential, audio_data, sample_rate).await;
            }
            AsrProviderType::SenseVoiceLocal => {
                return Self::transcribe_sensevoice_local(credential, audio_data, sample_rate)
                    .await;
            }
            _ => {}
        }

        // 云端服务：先尝试云端，失败则回退到本地 Whisper
        let cloud_result = match credential.provider {
            AsrProviderType::OpenAI => {
                Self::transcribe_openai(credential, audio_data, sample_rate).await
            }
            AsrProviderType::Baidu => {
                Self::transcribe_baidu(credential, audio_data, sample_rate).await
            }
            AsrProviderType::Xunfei => {
                Self::transcribe_xunfei(credential, audio_data, sample_rate).await
            }
            AsrProviderType::WhisperLocal => unreachable!(), // 已在上面处理
            AsrProviderType::SenseVoiceLocal => unreachable!(), // 已在上面处理
        };

        // 云端成功，直接返回
        if cloud_result.is_ok() {
            return cloud_result;
        }

        // 云端失败，尝试回退到本地 Whisper
        let cloud_error = cloud_result.unwrap_err();
        tracing::warn!(
            "云端 ASR 服务 ({:?}) 失败: {}，尝试回退到本地 Whisper",
            credential.provider,
            cloud_error
        );

        // 尝试获取本地 Whisper 凭证
        match Self::get_whisper_local_credential() {
            Ok(Some(whisper_credential)) => {
                tracing::info!("正在使用本地 Whisper 进行回退识别...");
                match Self::transcribe_whisper_local(&whisper_credential, audio_data, sample_rate)
                    .await
                {
                    Ok(text) => {
                        tracing::info!("本地 Whisper 回退识别成功");
                        Ok(text)
                    }
                    Err(whisper_error) => {
                        tracing::error!("本地 Whisper 回退也失败: {}", whisper_error);
                        // 返回原始云端错误，因为那是用户选择的服务
                        Err(format!(
                            "云端服务失败: {cloud_error}；本地 Whisper 回退也失败: {whisper_error}"
                        ))
                    }
                }
            }
            Ok(None) => {
                tracing::warn!("未找到本地 Whisper 凭证，无法回退");
                Err(format!(
                    "云端服务失败: {cloud_error}；未配置本地 Whisper，无法回退"
                ))
            }
            Err(e) => {
                tracing::error!("获取本地 Whisper 凭证失败: {}", e);
                Err(format!(
                    "云端服务失败: {cloud_error}；获取本地 Whisper 凭证失败: {e}"
                ))
            }
        }
    }

    /// 获取本地 Whisper 凭证（用于回退）
    fn get_whisper_local_credential() -> Result<Option<AsrCredentialEntry>, String> {
        voice_config_service::get_enabled_asr_credential_by_provider(AsrProviderType::WhisperLocal)
    }

    /// 本地 Whisper 识别
    #[cfg(feature = "local-whisper")]
    async fn transcribe_whisper_local(
        credential: &AsrCredentialEntry,
        audio_data: &[u8],
        sample_rate: u32,
    ) -> Result<String, String> {
        // 获取 Whisper 配置
        let whisper_config = credential
            .whisper_config
            .as_ref()
            .ok_or("Whisper 本地配置缺失")?;

        // 获取模型文件路径
        let model_path = voice_config_service::resolve_whisper_model_path(whisper_config)?;

        // 将 PCM 字节转换为 i16 采样
        let audio = Self::build_audio_data(audio_data, sample_rate)?;

        // 检查录音时长
        if !audio.is_valid() {
            return Err("录音时间过短（需要至少 0.5 秒）".to_string());
        }

        // 转换模型大小枚举
        let model = Self::convert_model_size(&whisper_config.model);

        // 创建 Whisper 识别器
        let transcriber =
            voice_core::WhisperTranscriber::new(model_path, model, &credential.language)
                .map_err(|e| format!("Whisper 模型加载失败: {e}"))?;

        // 执行识别
        let result = transcriber
            .transcribe(&audio)
            .map_err(|e| format!("Whisper 识别失败: {e}"))?;

        Ok(result.text)
    }

    /// 本地 Whisper 识别（未启用 local-whisper feature 时的 stub）
    #[cfg(not(feature = "local-whisper"))]
    async fn transcribe_whisper_local(
        _credential: &AsrCredentialEntry,
        _audio_data: &[u8],
        _sample_rate: u32,
    ) -> Result<String, String> {
        Err("本地 Whisper 功能未启用。请使用云端 ASR 服务（OpenAI、百度、讯飞）".to_string())
    }

    #[cfg(feature = "local-sensevoice")]
    /// 本地 SenseVoice 识别。
    async fn transcribe_sensevoice_local(
        credential: &AsrCredentialEntry,
        audio_data: &[u8],
        sample_rate: u32,
    ) -> Result<String, String> {
        let config = credential
            .sensevoice_config
            .as_ref()
            .ok_or("SenseVoice 本地配置缺失")?;

        let model_dir = voice_config_service::resolve_sensevoice_model_dir(config)?;
        let model_path = model_dir.join(SENSEVOICE_MODEL_FILE);
        let tokens_path = model_dir.join(SENSEVOICE_TOKENS_FILE);
        let vad_path = model_dir.join(SENSEVOICE_VAD_FILE);
        Self::ensure_sensevoice_required_files(&[
            (&model_path, SENSEVOICE_MODEL_FILE),
            (&tokens_path, SENSEVOICE_TOKENS_FILE),
            (&vad_path, SENSEVOICE_VAD_FILE),
        ])?;

        let audio = Self::build_audio_data(audio_data, sample_rate)?;
        if !audio.is_valid() {
            return Err("录音时间过短（需要至少 0.5 秒）".to_string());
        }

        let language = if credential.language.trim().is_empty() {
            "auto".to_string()
        } else {
            credential.language.clone()
        };
        let use_itn = config.use_itn;
        let num_threads = config.num_threads;
        let key = SenseVoiceCacheKey {
            model_path,
            tokens_path,
            language,
            use_itn,
            num_threads,
        };

        tokio::task::spawn_blocking(move || {
            let mut cache = SENSEVOICE_TRANSCRIBER_CACHE.lock();
            let should_reload = cache
                .as_ref()
                .map(|cached| cached.key != key)
                .unwrap_or(true);

            if should_reload {
                tracing::info!(
                    "[SenseVoice] 加载本地识别器: model={}, language={}, threads={}",
                    key.model_path.display(),
                    key.language,
                    key.num_threads
                );
                let transcriber = voice_core::SenseVoiceTranscriber::new(
                    key.model_path.clone(),
                    key.tokens_path.clone(),
                    &key.language,
                    key.use_itn,
                    key.num_threads,
                )
                .map_err(|error| format!("SenseVoice 模型加载失败: {error}"))?;
                *cache = Some(CachedSenseVoiceTranscriber {
                    key: key.clone(),
                    transcriber,
                });
            }

            let transcriber = cache
                .as_ref()
                .map(|cached| &cached.transcriber)
                .ok_or("SenseVoice 识别器缓存初始化失败")?;

            let result = transcriber
                .transcribe(&audio)
                .map_err(|error| format!("SenseVoice 识别失败: {error}"))?;

            Ok::<_, String>(result.text)
        })
        .await
        .map_err(|error| format!("SenseVoice 识别任务执行失败: {error}"))?
    }

    #[cfg(not(feature = "local-sensevoice"))]
    /// 本地 SenseVoice 识别（未启用 local-sensevoice feature 时的 stub）
    async fn transcribe_sensevoice_local(
        _credential: &AsrCredentialEntry,
        _audio_data: &[u8],
        _sample_rate: u32,
    ) -> Result<String, String> {
        Err(
            "本地 SenseVoice 功能未启用。请启用 local-sensevoice 构建特性，或使用云端 ASR 服务"
                .to_string(),
        )
    }

    #[cfg(feature = "local-sensevoice")]
    fn ensure_sensevoice_required_files(files: &[(&PathBuf, &str)]) -> Result<(), String> {
        let missing_files = files
            .iter()
            .filter_map(
                |(path, name)| {
                    if path.is_file() {
                        None
                    } else {
                        Some(*name)
                    }
                },
            )
            .collect::<Vec<_>>();

        if missing_files.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "本地 SenseVoice Small 尚未安装或文件不完整，请先在设置 -> 语音模型中下载；缺失文件: {}",
                missing_files.join(", ")
            ))
        }
    }

    /// 转换模型大小枚举
    #[cfg(feature = "local-whisper")]
    fn convert_model_size(size: &WhisperModelSize) -> voice_core::types::WhisperModel {
        match size {
            WhisperModelSize::Tiny => voice_core::types::WhisperModel::Tiny,
            WhisperModelSize::Base => voice_core::types::WhisperModel::Base,
            WhisperModelSize::Small => voice_core::types::WhisperModel::Small,
            WhisperModelSize::Medium => voice_core::types::WhisperModel::Medium,
        }
    }

    /// OpenAI Whisper API 识别
    async fn transcribe_openai(
        credential: &AsrCredentialEntry,
        audio_data: &[u8],
        sample_rate: u32,
    ) -> Result<String, String> {
        let config = credential.openai_config.as_ref().ok_or("OpenAI 配置缺失")?;
        let audio = Self::build_audio_data(audio_data, sample_rate)?;

        let endpoint = Self::openai_transcription_endpoint(config.base_url.as_deref());
        let output = execute_transcription(
            &AudioProviderConfig::with_endpoint(endpoint, config.api_key.clone()),
            &TranscriptionProviderRequest {
                model_id: "whisper-1".to_string(),
                audio: audio.to_wav_bytes(),
                filename: "audio.wav".to_string(),
                mime_type: "audio/wav".to_string(),
                language: (!credential.language.is_empty()).then(|| credential.language.clone()),
                prompt: None,
                response_format: Some("json".to_string()),
            },
        )
        .await
        .map_err(|error| format!("OpenAI Whisper 识别失败: {}: {}", error.code, error.message))?;

        Ok(output.text)
    }

    fn openai_transcription_endpoint(base_url: Option<&str>) -> String {
        let base = base_url
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/');
        if base.ends_with("/audio/transcriptions") {
            base.to_string()
        } else if base.ends_with("/v1") {
            format!("{base}/audio/transcriptions")
        } else {
            format!("{base}/v1/audio/transcriptions")
        }
    }

    /// 百度语音识别
    async fn transcribe_baidu(
        credential: &AsrCredentialEntry,
        audio_data: &[u8],
        sample_rate: u32,
    ) -> Result<String, String> {
        let config = credential.baidu_config.as_ref().ok_or("百度配置缺失")?;
        let audio = Self::build_audio_data(audio_data, sample_rate)?;

        let client = BaiduClient::new(config.api_key.clone(), config.secret_key.clone());
        let result = client
            .transcribe(&audio)
            .await
            .map_err(|e| format!("百度识别失败: {e}"))?;

        Ok(result.text)
    }

    /// 讯飞语音识别
    ///
    /// 使用 WebSocket 流式识别，支持实时语音转文字
    async fn transcribe_xunfei(
        credential: &AsrCredentialEntry,
        audio_data: &[u8],
        sample_rate: u32,
    ) -> Result<String, String> {
        let config = credential.xunfei_config.as_ref().ok_or("讯飞配置缺失")?;
        let audio = Self::build_audio_data(audio_data, sample_rate)?;

        // 创建讯飞客户端
        // 讯飞语言代码转换：zh -> zh_cn, en -> en_us
        let xunfei_language = match credential.language.as_str() {
            "zh" => "zh_cn".to_string(),
            "en" => "en_us".to_string(),
            other => other.to_string(),
        };

        let client = XunfeiClient::new(
            config.app_id.clone(),
            config.api_key.clone(),
            config.api_secret.clone(),
        )
        .with_language(xunfei_language);

        let result = client
            .transcribe(&audio)
            .await
            .map_err(|e| format!("讯飞识别失败: {e}"))?;

        Ok(result.text)
    }

    /// 将 PCM 字节构造成 voice-core 的 AudioData
    fn build_audio_data(audio_data: &[u8], sample_rate: u32) -> Result<AudioData, String> {
        let audio = AudioData::from_pcm16le_bytes(audio_data, sample_rate, 1);
        if audio.samples.is_empty() {
            return Err("音频数据为空".to_string());
        }

        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "local-sensevoice")]
    use super::{AsrService, SENSEVOICE_MODEL_FILE, SENSEVOICE_TOKENS_FILE, SENSEVOICE_VAD_FILE};
    #[cfg(feature = "local-sensevoice")]
    #[test]
    fn sensevoice_required_files_reports_missing_names() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model_path = temp.path().join(SENSEVOICE_MODEL_FILE);
        std::fs::write(&model_path, b"mock").expect("write model");
        let tokens_path = temp.path().join(SENSEVOICE_TOKENS_FILE);
        let vad_path = temp.path().join(SENSEVOICE_VAD_FILE);

        let error = AsrService::ensure_sensevoice_required_files(&[
            (&model_path, SENSEVOICE_MODEL_FILE),
            (&tokens_path, SENSEVOICE_TOKENS_FILE),
            (&vad_path, SENSEVOICE_VAD_FILE),
        ])
        .expect_err("missing files should fail");

        assert!(error.contains(SENSEVOICE_TOKENS_FILE));
        assert!(error.contains(SENSEVOICE_VAD_FILE));
    }
}
