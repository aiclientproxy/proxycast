//! 系统信息服务
//!
//! 提供 CPU、内存等系统信息的获取和订阅功能。
//! 不依赖桌面宿主，事件发送通过回调注入。

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info};

/// 系统信息事件回调
pub type SysinfoEmitter = Arc<dyn Fn(&SysinfoData) -> Result<(), String> + Send + Sync>;

/// 系统信息数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SysinfoData {
    /// 时间戳（毫秒）
    pub ts: u64,
    /// 总 CPU 使用率（0-100）
    pub cpu: f32,
    /// 已用内存（GB）
    #[serde(rename = "mem:used")]
    pub mem_used: f64,
    /// 总内存（GB）
    #[serde(rename = "mem:total")]
    pub mem_total: f64,
    /// 各核心 CPU 使用率
    #[serde(flatten)]
    pub cpu_cores: std::collections::HashMap<String, f32>,
}

/// 系统信息服务状态
pub struct SysinfoService {
    /// 是否正在运行订阅
    running: AtomicBool,
    /// 系统信息实例
    system: Mutex<System>,
}

impl SysinfoService {
    /// 创建新的系统信息服务
    pub fn new() -> Self {
        let system = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        Self {
            running: AtomicBool::new(false),
            system: Mutex::new(system),
        }
    }

    /// 获取当前系统信息快照
    pub async fn get_sysinfo(&self) -> SysinfoData {
        let mut system = self.system.lock().await;

        system.refresh_cpu_usage();
        system.refresh_memory();

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let cpu = system.global_cpu_usage();
        let mem_total = system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let mem_used = system.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        let mut cpu_cores = std::collections::HashMap::new();
        for (index, core) in system.cpus().iter().enumerate() {
            cpu_cores.insert(format!("cpu:{index}"), core.cpu_usage());
        }

        SysinfoData {
            ts,
            cpu,
            mem_used,
            mem_total,
            cpu_cores,
        }
    }

    /// 检查是否正在运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// 停止订阅
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("系统信息订阅已停止");
    }
}

impl Default for SysinfoService {
    fn default() -> Self {
        Self::new()
    }
}

/// 全局系统信息服务实例
static SYSINFO_SERVICE: once_cell::sync::Lazy<Arc<SysinfoService>> =
    once_cell::sync::Lazy::new(|| Arc::new(SysinfoService::new()));

/// 获取全局系统信息服务
pub fn get_sysinfo_service() -> Arc<SysinfoService> {
    SYSINFO_SERVICE.clone()
}

/// 获取当前系统信息
pub async fn get_sysinfo() -> Result<SysinfoData, String> {
    let service = get_sysinfo_service();
    Ok(service.get_sysinfo().await)
}

/// 开始订阅系统信息（每秒回调一次）
pub async fn subscribe_sysinfo(emitter: SysinfoEmitter) -> Result<(), String> {
    let service = get_sysinfo_service();

    if service.is_running() {
        debug!("系统信息订阅已在运行");
        return Ok(());
    }

    service.running.store(true, Ordering::SeqCst);
    info!("开始系统信息订阅");

    let service_clone = service.clone();
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(1));

        ticker.tick().await;

        while service_clone.is_running() {
            ticker.tick().await;

            if !service_clone.is_running() {
                break;
            }

            let data = service_clone.get_sysinfo().await;
            if let Err(err) = emitter(&data) {
                error!("发送系统信息事件失败: {}", err);
            }
        }

        info!("系统信息订阅任务结束");
    });

    Ok(())
}

/// 停止订阅系统信息
pub async fn unsubscribe_sysinfo() -> Result<(), String> {
    let service = get_sysinfo_service();
    service.stop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_sysinfo() {
        let service = SysinfoService::new();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let data = service.get_sysinfo().await;
        assert!(data.ts > 0);
        assert!(data.cpu >= 0.0 && data.cpu <= 100.0);
        assert!(data.mem_total > 0.0);
        assert!(data.mem_used >= 0.0);
        assert!(data.mem_used <= data.mem_total);
    }
}
