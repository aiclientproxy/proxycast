use super::super::*;
use app_server_protocol::SupportBundleExportParams;
use async_trait::async_trait;

#[async_trait]
impl DiagnosticsAppDataSource for LocalAppDataSource {
    async fn list_logs(&self) -> Result<LogListResponse, RuntimeCoreError> {
        let (current_log_path, _) = log_storage_snapshot(self).await?;
        Ok(LogListResponse {
            entries: diagnostics::read_persisted_logs_tail_from_path(&current_log_path, 1_000),
        })
    }

    async fn read_persisted_log_tail(
        &self,
        params: LogPersistedTailParams,
    ) -> Result<LogPersistedTailResponse, RuntimeCoreError> {
        let limit = params.lines.unwrap_or(200).clamp(20, 1_000);
        let (current_log_path, _) = log_storage_snapshot(self).await?;
        Ok(LogPersistedTailResponse {
            entries: diagnostics::read_persisted_logs_tail_from_path(&current_log_path, limit),
        })
    }

    async fn clear_logs(&self) -> Result<LogClearResponse, RuntimeCoreError> {
        let (current_log_path, _) = log_storage_snapshot(self).await?;
        diagnostics::clear_persisted_log_artifacts_from_path(&current_log_path)
            .map_err(data_error)?;
        Ok(LogClearResponse { cleared: true })
    }

    async fn clear_diagnostic_log_history(&self) -> Result<LogClearResponse, RuntimeCoreError> {
        let (current_log_path, _) = log_storage_snapshot(self).await?;
        diagnostics::clear_diagnostic_log_artifacts_from_path(&current_log_path)
            .map_err(data_error)?;
        Ok(LogClearResponse { cleared: true })
    }

    async fn read_log_storage_diagnostics(
        &self,
    ) -> Result<LogStorageDiagnosticsResponse, RuntimeCoreError> {
        let (current_log_path, in_memory_log_count) = log_storage_snapshot(self).await?;
        Ok(diagnostics::read_log_storage_diagnostics_from_path(
            &current_log_path,
            in_memory_log_count,
        ))
    }

    async fn export_support_bundle(
        &self,
        params: SupportBundleExportParams,
        trace_store_root: Option<std::path::PathBuf>,
    ) -> Result<SupportBundleExportResponse, RuntimeCoreError> {
        let (current_log_path, _) = log_storage_snapshot(self).await?;
        diagnostics::export_support_bundle(
            self.app_data_root(),
            self.product_db_path(),
            &current_log_path,
            params,
            trace_store_root.as_deref(),
        )
        .map_err(data_error)
    }

    async fn read_windows_startup_diagnostics(
        &self,
    ) -> Result<WindowsStartupDiagnosticsResponse, RuntimeCoreError> {
        diagnostics::read_windows_startup_diagnostics(self.app_data_root(), self.product_db_path())
            .map_err(data_error)
    }
}

async fn log_storage_snapshot(
    data_source: &LocalAppDataSource,
) -> Result<(std::path::PathBuf, usize), RuntimeCoreError> {
    let logs = data_source.logs.read().await;
    let current_log_path = logs
        .log_file_path()
        .ok_or_else(|| RuntimeCoreError::Backend("日志文件存储未配置".to_string()))?
        .to_path_buf();
    Ok((current_log_path, logs.entry_count()))
}
