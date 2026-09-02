//! voice-core - 语音输入核心库
//!
//! 提供音频录制、语音识别、文字输出等功能。
//! 不依赖桌面宿主，可被任何 Rust 项目使用。

pub mod asr_client;
pub mod device;
pub mod error;
pub mod output;
pub mod recorder;
#[cfg(feature = "local-sensevoice")]
pub mod sensevoice_transcriber;
pub mod text_polish;
pub mod threaded_recorder;
#[cfg(feature = "local-whisper")]
pub mod transcriber;
pub mod types;

pub use device::{list_audio_devices, AudioDeviceInfo};
pub use error::{Result, VoiceError};
pub use output::OutputHandler;
pub use recorder::AudioRecorder;
#[cfg(feature = "local-sensevoice")]
pub use sensevoice_transcriber::SenseVoiceTranscriber;
pub use threaded_recorder::{RecordingCommand, RecordingResponse, RecordingService};
#[cfg(feature = "local-whisper")]
pub use transcriber::WhisperTranscriber;
pub use types::*;
