//! 日志管理模块
use chrono::{Local, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
}

pub struct LogStore {
    logs: Vec<LogEntry>,
    max_logs: usize,
    log_file_path: Option<PathBuf>,
}

impl Default for LogStore {
    fn default() -> Self {
        // 默认日志文件路径: ~/.proxycast/logs/proxycast.log
        let log_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".proxycast")
            .join("logs");

        // 创建日志目录
        let _ = fs::create_dir_all(&log_dir);

        let log_file = log_dir.join("proxycast.log");

        Self {
            logs: Vec::new(),
            max_logs: 1000,
            log_file_path: Some(log_file),
        }
    }
}

impl LogStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, level: &str, message: &str) {
        let now = Utc::now();
        let entry = LogEntry {
            timestamp: now.to_rfc3339(),
            level: level.to_string(),
            message: message.to_string(),
        };

        self.logs.push(entry.clone());

        // 写入日志文件
        if let Some(ref path) = self.log_file_path {
            let local_time = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let log_line = format!("{} [{}] {}\n", local_time, level.to_uppercase(), message);

            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = file.write_all(log_line.as_bytes());
            }
        }

        // 保持日志数量在限制内
        if self.logs.len() > self.max_logs {
            self.logs.remove(0);
        }
    }

    /// 记录原始响应到单独的文件（用于调试）
    pub fn log_raw_response(&self, request_id: &str, body: &str) {
        if let Some(ref log_path) = self.log_file_path {
            let log_dir = log_path.parent().unwrap_or(std::path::Path::new("."));
            let raw_file = log_dir.join(format!("raw_response_{request_id}.txt"));

            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&raw_file)
            {
                let _ = file.write_all(body.as_bytes());
            }
        }
    }

    pub fn get_logs(&self) -> Vec<LogEntry> {
        self.logs.clone()
    }

    pub fn clear(&mut self) {
        self.logs.clear();
    }

    pub fn get_log_file_path(&self) -> Option<String> {
        self.log_file_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
    }
}

#[allow(dead_code)]
pub type SharedLogStore = Arc<RwLock<LogStore>>;
