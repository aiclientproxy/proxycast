//! 录音服务
//!
//! 管理录音状态，提供录音控制接口。
//!
//! ## 线程安全设计
//!
//! 由于 `cpal::Stream` 不实现 `Send` trait，无法跨异步任务直接共享。
//! 本模块采用**独立线程 + channel 通信**的方案：
//!
//! ```text
//! ┌─────────────────┐     Command      ┌─────────────────┐
//! │  App Server API │ ───────────────> │  Recording      │
//! │  (async)        │                  │  Thread         │
//! │                 │ <─────────────── │  (owns Stream)  │
//! └─────────────────┘     Response     └─────────────────┘
//! ```
//!
//! - 录音线程拥有 `cpal::Stream`，在独立线程中运行
//! - App Server 通过 channel 发送控制指令
//! - 录音线程通过 channel 返回结果

use crate::types::AudioData;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

const MAX_RECORDING_DURATION_SECS: usize = 300;
const INITIAL_RECORDING_CAPACITY_SECS: usize = 30;
const DEFAULT_SEGMENT_DURATION_SECS: f32 = 1.25;
const MAX_SEGMENT_DURATION_SECS: f32 = 2.0;

/// 录音控制命令
#[derive(Debug)]
pub enum RecordingCommand {
    /// 开始录音（可选指定设备 ID）
    Start(Option<String>),
    /// 停止录音
    Stop,
    /// 获取当前录音快照，不停止录音
    Snapshot,
    /// 获取当前录音片段，不停止录音
    Segment {
        start_sample: usize,
        max_duration_secs: Option<f32>,
    },
    /// 取消录音
    Cancel,
    /// 关闭录音线程
    Shutdown,
}

/// 录音响应
#[derive(Debug)]
pub enum RecordingResponse {
    /// 操作成功
    Ok,
    /// 停止录音成功，返回音频数据
    AudioData(AudioData),
    /// 录音片段数据
    AudioSegment {
        audio: AudioData,
        start_sample: usize,
        end_sample: usize,
        total_samples: usize,
    },
    /// 操作失败
    Error(String),
}

/// 录音服务
///
/// 使用独立线程管理 cpal::Stream，通过 channel 与 App Server 通信
pub struct RecordingService {
    /// 命令发送端
    command_tx: Option<Sender<RecordingCommand>>,
    /// 响应接收端
    response_rx: Option<Receiver<RecordingResponse>>,
    /// 录音线程句柄
    thread_handle: Option<JoinHandle<()>>,
    /// 是否正在录音（共享状态，用于快速查询）
    is_recording: Arc<AtomicBool>,
    /// 当前音量级别（共享状态，用于快速查询）
    volume_level: Arc<AtomicU32>,
    /// 录音开始时间（共享状态）
    start_time: Arc<Mutex<Option<Instant>>>,
}

impl RecordingService {
    /// 创建新的录音服务
    pub fn new() -> Self {
        Self {
            command_tx: None,
            response_rx: None,
            thread_handle: None,
            is_recording: Arc::new(AtomicBool::new(false)),
            volume_level: Arc::new(AtomicU32::new(0)),
            start_time: Arc::new(Mutex::new(None)),
        }
    }

    /// 确保录音线程已启动
    fn ensure_thread_started(&mut self) {
        if self.command_tx.is_some() {
            return;
        }

        let (cmd_tx, cmd_rx) = mpsc::channel::<RecordingCommand>();
        let (resp_tx, resp_rx) = mpsc::channel::<RecordingResponse>();

        let is_recording = Arc::clone(&self.is_recording);
        let volume_level = Arc::clone(&self.volume_level);
        let start_time = Arc::clone(&self.start_time);

        let handle = thread::spawn(move || {
            recording_thread_main(cmd_rx, resp_tx, is_recording, volume_level, start_time);
        });

        self.command_tx = Some(cmd_tx);
        self.response_rx = Some(resp_rx);
        self.thread_handle = Some(handle);

        tracing::info!("[录音服务] 录音线程已启动");
    }

    /// 开始录音（可选指定设备 ID）
    pub fn start(&mut self, device_id: Option<String>) -> Result<(), String> {
        self.ensure_thread_started();

        let tx = self.command_tx.as_ref().ok_or("录音线程未启动")?;
        let rx = self.response_rx.as_ref().ok_or("录音线程未启动")?;

        tx.send(RecordingCommand::Start(device_id))
            .map_err(|e| format!("发送命令失败: {e}"))?;

        match rx.recv() {
            Ok(RecordingResponse::Ok) => {
                tracing::info!("[录音服务] 开始录音");
                Ok(())
            }
            Ok(RecordingResponse::Error(e)) => Err(e),
            Ok(_) => Err("意外的响应".to_string()),
            Err(e) => Err(format!("接收响应失败: {e}")),
        }
    }

    /// 停止录音并返回音频数据
    pub fn stop(&mut self) -> Result<AudioData, String> {
        let tx = self.command_tx.as_ref().ok_or("录音线程未启动")?;
        let rx = self.response_rx.as_ref().ok_or("录音线程未启动")?;

        tx.send(RecordingCommand::Stop)
            .map_err(|e| format!("发送命令失败: {e}"))?;

        match rx.recv() {
            Ok(RecordingResponse::AudioData(audio)) => {
                tracing::info!("[录音服务] 停止录音，时长: {:.2}s", audio.duration_secs);
                Ok(audio)
            }
            Ok(RecordingResponse::Error(e)) => Err(e),
            Ok(_) => Err("意外的响应".to_string()),
            Err(e) => Err(format!("接收响应失败: {e}")),
        }
    }

    /// 获取当前录音快照，不停止录音
    pub fn snapshot(&mut self) -> Result<AudioData, String> {
        let tx = self.command_tx.as_ref().ok_or("录音线程未启动")?;
        let rx = self.response_rx.as_ref().ok_or("录音线程未启动")?;

        tx.send(RecordingCommand::Snapshot)
            .map_err(|e| format!("发送命令失败: {e}"))?;

        match rx.recv() {
            Ok(RecordingResponse::AudioData(audio)) => Ok(audio),
            Ok(RecordingResponse::Error(e)) => Err(e),
            Ok(_) => Err("意外的响应".to_string()),
            Err(e) => Err(format!("接收响应失败: {e}")),
        }
    }

    /// 获取当前录音片段，不停止录音
    pub fn segment(
        &mut self,
        start_sample: usize,
        max_duration_secs: Option<f32>,
    ) -> Result<(AudioData, usize, usize, usize), String> {
        let tx = self.command_tx.as_ref().ok_or("录音线程未启动")?;
        let rx = self.response_rx.as_ref().ok_or("录音线程未启动")?;

        tx.send(RecordingCommand::Segment {
            start_sample,
            max_duration_secs,
        })
        .map_err(|e| format!("发送命令失败: {e}"))?;

        match rx.recv() {
            Ok(RecordingResponse::AudioSegment {
                audio,
                start_sample,
                end_sample,
                total_samples,
            }) => Ok((audio, start_sample, end_sample, total_samples)),
            Ok(RecordingResponse::Error(e)) => Err(e),
            Ok(_) => Err("意外的响应".to_string()),
            Err(e) => Err(format!("接收响应失败: {e}")),
        }
    }

    /// 取消录音
    pub fn cancel(&mut self) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(RecordingCommand::Cancel);
            // 使用 try_recv 避免阻塞，或者设置超时
            if let Some(rx) = &self.response_rx {
                // 尝试接收响应，但不阻塞太久
                use std::time::Duration;
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(_) => tracing::info!("[录音服务] 取消录音成功"),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        tracing::warn!("[录音服务] 取消录音超时，强制继续");
                    }
                    Err(e) => {
                        tracing::warn!("[录音服务] 取消录音响应错误: {}", e);
                    }
                }
            }
        }
        // 无论如何都重置状态
        self.is_recording.store(false, Ordering::SeqCst);
        self.volume_level.store(0, Ordering::SeqCst);
        *self.start_time.lock() = None;
    }

    /// 获取当前音量级别（0-100）
    pub fn get_volume(&self) -> u32 {
        self.volume_level.load(Ordering::SeqCst)
    }

    /// 获取录音时长（秒）
    pub fn get_duration(&self) -> f32 {
        self.start_time
            .lock()
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0)
    }

    /// 是否正在录音
    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    /// 关闭录音服务
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.command_tx.take() {
            let _ = tx.send(RecordingCommand::Shutdown);
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        self.response_rx = None;
        tracing::info!("[录音服务] 已关闭");
    }
}

impl Default for RecordingService {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RecordingService {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn max_recording_samples(sample_rate: u32) -> usize {
    (sample_rate as usize).saturating_mul(MAX_RECORDING_DURATION_SECS)
}

fn initial_recording_capacity(sample_rate: u32) -> usize {
    (sample_rate as usize).saturating_mul(INITIAL_RECORDING_CAPACITY_SECS)
}

fn reset_sample_buffer(samples: &mut Vec<i16>, sample_rate: u32) {
    let initial_capacity = initial_recording_capacity(sample_rate);
    if samples.capacity() > initial_capacity.saturating_mul(2) {
        *samples = Vec::with_capacity(initial_capacity);
        return;
    }

    samples.clear();
    if samples.capacity() < initial_capacity {
        samples.reserve(initial_capacity);
    }
}

fn clamp_segment_duration(max_duration_secs: Option<f32>) -> f32 {
    max_duration_secs
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .unwrap_or(DEFAULT_SEGMENT_DURATION_SECS)
        .min(MAX_SEGMENT_DURATION_SECS)
}

fn sample_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn append_callback_samples(
    target: &mut Vec<i16>,
    data: &[f32],
    channels: u16,
    sample_cap: usize,
) -> usize {
    if data.is_empty() || target.len() >= sample_cap {
        return 0;
    }

    let remaining = sample_cap - target.len();
    let channels = usize::from(channels.max(1));

    if channels == 1 {
        let sample_count = data.len().min(remaining);
        for sample in data.iter().take(sample_count) {
            target.push(sample_to_i16(*sample));
        }
        return sample_count;
    }

    let frame_count = data.chunks_exact(channels).len().min(remaining);
    for frame in data.chunks_exact(channels).take(frame_count) {
        let mono = frame.iter().copied().sum::<f32>() / channels as f32;
        target.push(sample_to_i16(mono));
    }
    frame_count
}

/// 录音线程主函数
///
/// 在独立线程中运行，拥有 cpal::Stream
fn recording_thread_main(
    cmd_rx: Receiver<RecordingCommand>,
    resp_tx: Sender<RecordingResponse>,
    is_recording: Arc<AtomicBool>,
    volume_level: Arc<AtomicU32>,
    start_time: Arc<Mutex<Option<Instant>>>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    // 录音数据缓冲区
    let samples: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
    // 当前活跃的音频流
    let mut active_stream: Option<cpal::Stream> = None;
    // 实际使用的采样率和声道数
    let mut actual_sample_rate: u32 = 16000;
    #[allow(unused_assignments)]
    let mut actual_channels: u16 = 1;

    tracing::debug!("[录音线程] 开始运行");

    loop {
        match cmd_rx.recv() {
            Ok(RecordingCommand::Start(device_id)) => {
                // 如果已在录音，返回错误
                if is_recording.load(Ordering::SeqCst) {
                    let _ = resp_tx.send(RecordingResponse::Error("已在录音中".to_string()));
                    continue;
                }

                // 获取输入设备
                let host = cpal::default_host();
                let device = if let Some(ref id) = device_id {
                    // 查找指定设备
                    host.input_devices()
                        .ok()
                        .and_then(|mut devices| {
                            devices.find(|d| d.name().ok().as_ref() == Some(id))
                        })
                        .or_else(|| {
                            tracing::warn!("[录音线程] 未找到指定设备 {}，使用默认设备", id);
                            host.default_input_device()
                        })
                } else {
                    host.default_input_device()
                };

                let device = match device {
                    Some(d) => d,
                    None => {
                        let _ =
                            resp_tx.send(RecordingResponse::Error("未找到麦克风设备".to_string()));
                        continue;
                    }
                };

                tracing::info!("[录音线程] 使用麦克风: {:?}", device.name());

                // 获取设备支持的配置
                let supported_config = match device.default_input_config() {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = resp_tx
                            .send(RecordingResponse::Error(format!("获取音频配置失败: {e}")));
                        continue;
                    }
                };

                tracing::info!(
                    "[录音线程] 设备支持配置: 采样率={}, 声道={}",
                    supported_config.sample_rate().0,
                    supported_config.channels()
                );

                // 使用设备默认配置
                actual_sample_rate = supported_config.sample_rate().0;
                actual_channels = supported_config.channels();

                let config = cpal::StreamConfig {
                    channels: actual_channels,
                    sample_rate: supported_config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                };

                {
                    let mut sample_buffer = samples.lock();
                    reset_sample_buffer(&mut sample_buffer, actual_sample_rate);
                }

                // 创建共享状态的克隆
                let samples_clone = Arc::clone(&samples);
                let volume_clone = Arc::clone(&volume_level);
                let is_rec_clone = Arc::clone(&is_recording);
                let channels = actual_channels;
                let sample_cap = max_recording_samples(actual_sample_rate);

                // 避免每个 callback 创建临时 Vec，降低实时音频线程分配抖动。
                let mut callback_count = 0_u32;
                let mut sample_cap_logged = false;

                // 创建输入流
                let stream = match device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !is_rec_clone.load(Ordering::SeqCst) {
                            return;
                        }
                        if data.is_empty() {
                            return;
                        }

                        if callback_count == 0 {
                            tracing::info!("[录音线程] 首次收到音频数据，数据长度: {}", data.len());
                        } else if callback_count.is_multiple_of(500) {
                            tracing::trace!("[录音线程] 已收到 {} 次音频回调", callback_count);
                        }
                        callback_count = callback_count.wrapping_add(1);

                        // 计算音量级别（使用 RMS 均方根，更准确反映音量）
                        let sum_sq: f32 = data.iter().map(|s| s * s).sum();
                        let rms = (sum_sq / data.len() as f32).sqrt();
                        // 将 RMS 值映射到 0-100 范围
                        // 静音时 RMS 约 0.001-0.01，说话时约 0.02-0.1
                        // 使用更高的系数来提高灵敏度
                        let level = ((rms * 1500.0).min(100.0)) as u32;

                        // 降低实时线程日志频率，避免录音时被日志 I/O 干扰。
                        if callback_count.is_multiple_of(500) {
                            tracing::debug!("[录音线程] RMS: {:.6}, 音量: {}%", rms, level);
                        }

                        volume_clone.store(level, Ordering::SeqCst);

                        let reached_sample_cap = {
                            let mut sample_buffer = samples_clone.lock();
                            let was_below_cap = sample_buffer.len() < sample_cap;
                            append_callback_samples(&mut sample_buffer, data, channels, sample_cap);
                            was_below_cap && sample_buffer.len() >= sample_cap
                        };
                        if reached_sample_cap && !sample_cap_logged {
                            tracing::warn!(
                                "[录音线程] 已达到单次录音上限 {} 秒，后续音频不再写入内存",
                                MAX_RECORDING_DURATION_SECS
                            );
                            sample_cap_logged = true;
                        }
                    },
                    |err| {
                        tracing::error!("[录音线程] 录音流错误: {}", err);
                    },
                    None,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ =
                            resp_tx.send(RecordingResponse::Error(format!("创建音频流失败: {e}")));
                        continue;
                    }
                };

                // 开始播放（录音）
                if let Err(e) = stream.play() {
                    let _ = resp_tx.send(RecordingResponse::Error(format!("启动录音失败: {e}")));
                    continue;
                }

                tracing::info!("[录音线程] stream.play() 成功，等待音频数据...");

                // 保存流和状态
                active_stream = Some(stream);
                is_recording.store(true, Ordering::SeqCst);
                *start_time.lock() = Some(Instant::now());

                let _ = resp_tx.send(RecordingResponse::Ok);
                tracing::info!(
                    "[录音线程] 开始录音，采样率: {}, 声道: {}",
                    actual_sample_rate,
                    actual_channels
                );
            }

            Ok(RecordingCommand::Stop) => {
                if !is_recording.load(Ordering::SeqCst) {
                    let _ = resp_tx.send(RecordingResponse::Error("未在录音中".to_string()));
                    continue;
                }

                // 停止录音
                is_recording.store(false, Ordering::SeqCst);

                // 停止并释放流
                if let Some(stream) = active_stream.take() {
                    drop(stream);
                }

                // 获取录音数据（已转换为单声道）
                let audio_samples = {
                    let mut sample_buffer = samples.lock();
                    std::mem::take(&mut *sample_buffer)
                };
                let audio = AudioData::new(audio_samples, actual_sample_rate, 1);

                // 重置开始时间
                *start_time.lock() = None;
                volume_level.store(0, Ordering::SeqCst);

                // 检查录音时长
                if !audio.is_valid() {
                    let _ = resp_tx.send(RecordingResponse::Error(
                        "录音时间过短（需要至少 0.5 秒）".to_string(),
                    ));
                    continue;
                }

                let _ = resp_tx.send(RecordingResponse::AudioData(audio));
                tracing::info!("[录音线程] 停止录音");
            }

            Ok(RecordingCommand::Snapshot) => {
                if !is_recording.load(Ordering::SeqCst) {
                    let _ = resp_tx.send(RecordingResponse::Error("未在录音中".to_string()));
                    continue;
                }

                let audio_samples = samples.lock().clone();
                let audio = AudioData::new(audio_samples, actual_sample_rate, 1);
                let _ = resp_tx.send(RecordingResponse::AudioData(audio));
            }

            Ok(RecordingCommand::Segment {
                start_sample,
                max_duration_secs,
            }) => {
                if !is_recording.load(Ordering::SeqCst) {
                    let _ = resp_tx.send(RecordingResponse::Error("未在录音中".to_string()));
                    continue;
                }

                let locked_samples = samples.lock();
                let total_samples = locked_samples.len();
                let safe_start = start_sample.min(total_samples);
                let max_samples = (clamp_segment_duration(max_duration_secs)
                    * actual_sample_rate as f32)
                    .ceil() as usize;
                let end_sample = safe_start.saturating_add(max_samples).min(total_samples);
                let audio_samples = locked_samples[safe_start..end_sample].to_vec();
                drop(locked_samples);

                let audio = AudioData::new(audio_samples, actual_sample_rate, 1);
                let _ = resp_tx.send(RecordingResponse::AudioSegment {
                    audio,
                    start_sample: safe_start,
                    end_sample,
                    total_samples,
                });
            }

            Ok(RecordingCommand::Cancel) => {
                // 停止录音
                is_recording.store(false, Ordering::SeqCst);

                // 停止并释放流
                if let Some(stream) = active_stream.take() {
                    drop(stream);
                }

                // 清空缓冲区
                *samples.lock() = Vec::new();

                // 重置状态
                *start_time.lock() = None;
                volume_level.store(0, Ordering::SeqCst);

                let _ = resp_tx.send(RecordingResponse::Ok);
                tracing::info!("[录音线程] 取消录音");
            }

            Ok(RecordingCommand::Shutdown) => {
                // 清理资源
                is_recording.store(false, Ordering::SeqCst);
                if let Some(stream) = active_stream.take() {
                    drop(stream);
                }
                tracing::info!("[录音线程] 收到关闭命令，退出");
                break;
            }

            Err(_) => {
                // channel 已关闭，退出线程
                tracing::info!("[录音线程] channel 已关闭，退出");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_callback_samples_converts_mono_without_exceeding_cap() {
        let mut samples = Vec::new();
        let written = append_callback_samples(&mut samples, &[0.0, 0.5, -1.0], 1, 2);

        assert_eq!(written, 2);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0);
        assert!(samples[1] > 16_000);
    }

    #[test]
    fn append_callback_samples_downmixes_stereo_without_temp_vectors() {
        let mut samples = Vec::new();
        let written = append_callback_samples(&mut samples, &[1.0, -1.0, 0.5, 0.5], 2, 8);

        assert_eq!(written, 2);
        assert_eq!(samples[0], 0);
        assert!(samples[1] > 16_000);
    }

    #[test]
    fn clamp_segment_duration_defaults_and_caps_live_segments() {
        assert_eq!(clamp_segment_duration(None), DEFAULT_SEGMENT_DURATION_SECS);
        assert_eq!(
            clamp_segment_duration(Some(10.0)),
            MAX_SEGMENT_DURATION_SECS
        );
        assert_eq!(
            clamp_segment_duration(Some(f32::NAN)),
            DEFAULT_SEGMENT_DURATION_SECS
        );
    }
}
