//! Code Mode host transport。
//!
//! Host 只负责握手、请求路由和 delegate 回调；V8 执行由
//! `code-mode-runtime` 提供。当前 stdio transport 与既有 sidecar 协议保持兼容。

pub use code_mode_protocol::host;
pub use code_mode_runtime::V8CodeModeSessionProvider;
pub use grpc::GrpcCodeModeHost;
pub use transport::DEFAULT_LISTEN_URL;

mod delegate;
mod grpc;
mod grpc_transport;
mod peer;
mod stdio;
mod transport;

/// 在 stdio transport 上运行 host。
pub async fn run_stdio() -> Result<(), String> {
    stdio::run_stdio().await
}

/// 运行 host transport。`stdio`（或空字符串）是当前受支持的本地 transport。
pub async fn run_main(listen_url: &str) -> Result<(), String> {
    transport::run_transport(listen_url).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn host_rejects_unknown_transport_without_starting_a_process() {
        let result = tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(super::run_main("unknown"));
        assert!(result
            .expect_err("unknown transport must fail")
            .contains("unsupported"));
    }
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod host_tests;

#[cfg(test)]
#[path = "peer_tests.rs"]
mod peer_tests;
