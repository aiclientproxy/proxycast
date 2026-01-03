//! 原生 Agent 命令模块
//!
//! 提供原生 Rust Agent 的 Tauri 命令，替代 aster sidecar 方案

use crate::agent::{
    AgentSession, ImageData, NativeAgentState, NativeChatRequest, NativeChatResponse, ProviderType,
    StreamEvent, ToolLoopEngine,
};
use crate::AppState;
use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};
use tokio::sync::mpsc;

#[derive(Debug, Serialize)]
pub struct NativeAgentStatus {
    pub initialized: bool,
    pub base_url: Option<String>,
}

#[tauri::command]
pub async fn native_agent_init(
    agent_state: State<'_, NativeAgentState>,
    app_state: State<'_, AppState>,
) -> Result<NativeAgentStatus, String> {
    tracing::info!("[NativeAgent] 初始化 Agent");

    let (port, api_key, running, default_provider) = {
        let state = app_state.read().await;
        (
            state.config.server.port,
            state.running_api_key.clone(),
            state.running,
            state.config.routing.default_provider.clone(),
        )
    };

    if !running {
        return Err("ProxyCast API Server 未运行，请先启动服务器".to_string());
    }

    let api_key = api_key.ok_or_else(|| "ProxyCast API Server 未配置 API Key".to_string())?;

    let base_url = format!("http://127.0.0.1:{}", port);
    let provider_type = ProviderType::from_str(&default_provider);

    tracing::info!(
        "[NativeAgent] 初始化 Agent: base_url={}, provider={:?}",
        base_url,
        provider_type
    );

    agent_state.init(base_url.clone(), api_key, provider_type)?;

    tracing::info!("[NativeAgent] Agent 初始化成功: {}", base_url);

    Ok(NativeAgentStatus {
        initialized: true,
        base_url: Some(base_url),
    })
}

#[tauri::command]
pub async fn native_agent_status(
    agent_state: State<'_, NativeAgentState>,
) -> Result<NativeAgentStatus, String> {
    Ok(NativeAgentStatus {
        initialized: agent_state.is_initialized(),
        base_url: None,
    })
}

#[tauri::command]
pub async fn native_agent_reset(agent_state: State<'_, NativeAgentState>) -> Result<(), String> {
    agent_state.reset();
    tracing::info!("[NativeAgent] Agent 已重置");
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct ImageInputParam {
    pub data: String,
    pub media_type: String,
}

#[tauri::command]
pub async fn native_agent_chat(
    agent_state: State<'_, NativeAgentState>,
    app_state: State<'_, AppState>,
    message: String,
    model: Option<String>,
    images: Option<Vec<ImageInputParam>>,
) -> Result<NativeChatResponse, String> {
    tracing::info!(
        "[NativeAgent] 发送消息: message_len={}, model={:?}",
        message.len(),
        model
    );

    // 如果 Agent 未初始化，自动初始化
    if !agent_state.is_initialized() {
        let (port, api_key, running, default_provider) = {
            let state = app_state.read().await;
            (
                state.config.server.port,
                state.running_api_key.clone(),
                state.running,
                state.config.routing.default_provider.clone(),
            )
        };

        if !running {
            return Err("ProxyCast API Server 未运行".to_string());
        }

        let api_key = api_key.ok_or_else(|| "未配置 API Key".to_string())?;
        let base_url = format!("http://127.0.0.1:{}", port);
        let provider_type = ProviderType::from_str(&default_provider);
        agent_state.init(base_url, api_key, provider_type)?;
    }

    let request = NativeChatRequest {
        session_id: None,
        message,
        model,
        images: images.map(|imgs| {
            imgs.into_iter()
                .map(|img| ImageData {
                    data: img.data,
                    media_type: img.media_type,
                })
                .collect()
        }),
        stream: false,
    };

    // 使用 chat_sync 方法避免跨 await 持有锁
    agent_state.chat(request).await
}

#[tauri::command]
pub async fn native_agent_chat_stream(
    app_handle: tauri::AppHandle,
    agent_state: State<'_, NativeAgentState>,
    app_state: State<'_, AppState>,
    message: String,
    event_name: String,
    session_id: Option<String>,
    model: Option<String>,
    images: Option<Vec<ImageInputParam>>,
    provider: Option<String>,
) -> Result<(), String> {
    tracing::info!(
        "[NativeAgent] 发送流式消息: message_len={}, model={:?}, provider={:?}, event={}, session={:?}",
        message.len(),
        model,
        provider,
        event_name,
        session_id
    );

    // 获取配置信息
    let (port, api_key, running, default_provider) = {
        let state = app_state.read().await;
        (
            state.config.server.port,
            state.running_api_key.clone(),
            state.running,
            state.config.routing.default_provider.clone(),
        )
    };

    if !running {
        return Err("ProxyCast API Server 未运行".to_string());
    }

    let api_key = api_key.ok_or_else(|| "未配置 API Key".to_string())?;

    // 使用前端传递的 provider，如果没有则使用默认值
    let provider_str = provider.unwrap_or(default_provider);
    let provider_type = ProviderType::from_str(&provider_str);

    tracing::info!(
        "[NativeAgent] 使用 provider: {:?} (原始值: {})",
        provider_type,
        provider_str
    );

    // 如果 Agent 未初始化，或者 provider 发生变化，重新初始化
    let need_reinit = if !agent_state.is_initialized() {
        tracing::info!("[NativeAgent] Agent 未初始化，需要初始化");
        true
    } else if let Some(current_provider) = agent_state.get_provider_type() {
        if current_provider != provider_type {
            tracing::info!(
                "[NativeAgent] Provider 发生变化: {:?} -> {:?}，需要重新初始化",
                current_provider,
                provider_type
            );
            true
        } else {
            false
        }
    } else {
        true
    };

    if need_reinit {
        let base_url = format!("http://127.0.0.1:{}", port);
        agent_state.init(base_url, api_key, provider_type)?;
    }

    // 获取工具注册表（用于创建 ToolLoopEngine）
    let tool_registry = agent_state.get_tool_registry()?;

    let request = NativeChatRequest {
        session_id, // 使用前端传递的 session_id 以保持上下文
        message,
        model,
        images: images.map(|imgs| {
            imgs.into_iter()
                .map(|img| ImageData {
                    data: img.data,
                    media_type: img.media_type,
                })
                .collect()
        }),
        stream: true,
    };

    // 克隆 agent_state 用于后台任务（共享 sessions）
    let agent_state_clone = agent_state.inner().clone();

    // 在后台任务中处理流式响应
    let event_name_clone = event_name.clone();
    eprintln!(
        "[native_agent_chat_stream] 启动后台任务, event_name={}",
        event_name_clone
    );
    tauri::async_runtime::spawn(async move {
        eprintln!("[native_agent_chat_stream] 后台任务开始执行");

        // 创建工具循环引擎（使用共享的 tool_registry）
        let tool_loop_engine = ToolLoopEngine::new(tool_registry);
        eprintln!("[native_agent_chat_stream] 工具循环引擎创建成功");

        let (tx, mut rx) = mpsc::channel::<StreamEvent>(100);

        // 使用 agent_state 的方法（共享 sessions）
        eprintln!(
            "[native_agent_chat_stream] 开始 chat_stream_with_tools, request.session_id={:?}",
            request.session_id
        );
        let stream_task = tokio::spawn(async move {
            agent_state_clone
                .chat_stream_with_tools(request, tx, &tool_loop_engine)
                .await
        });

        eprintln!("[native_agent_chat_stream] 开始接收流式事件...");
        // 注意：不要在收到 Done 事件后立即 break，因为工具循环可能还在执行
        // 继续接收直到 channel 关闭（stream_task 完成）
        while let Some(event) = rx.recv().await {
            eprintln!("[native_agent_chat_stream] 收到事件: {:?}", event);
            tracing::debug!(
                "[NativeAgent] 收到流式事件: {:?}, 发送到: {}",
                event,
                event_name_clone
            );
            if let Err(e) = app_handle.emit(&event_name_clone, &event) {
                tracing::error!("[NativeAgent] 发送事件失败: {}", e);
                eprintln!("[native_agent_chat_stream] 发送事件失败: {}", e);
                break;
            }
            tracing::debug!("[NativeAgent] 事件发送成功");

            // 只在 Error 时 break，Done 不 break 因为工具循环可能还会发送更多事件
            if matches!(event, StreamEvent::Error { .. }) {
                tracing::info!("[NativeAgent] 流式响应错误，停止接收");
                eprintln!("[native_agent_chat_stream] 流式响应错误");
                break;
            }
        }
        eprintln!("[native_agent_chat_stream] channel 关闭，事件接收完成");

        eprintln!("[native_agent_chat_stream] 等待 stream_task 完成...");
        match stream_task.await {
            Ok(result) => {
                eprintln!("[native_agent_chat_stream] stream_task 完成: {:?}", result);
            }
            Err(e) => {
                eprintln!("[native_agent_chat_stream] stream_task 错误: {}", e);
            }
        }
        eprintln!("[native_agent_chat_stream] 后台任务结束");
    });

    Ok(())
}

#[tauri::command]
pub async fn native_agent_create_session(
    agent_state: State<'_, NativeAgentState>,
    model: Option<String>,
    system_prompt: Option<String>,
) -> Result<String, String> {
    agent_state.create_session(model, system_prompt)
}

#[tauri::command]
pub async fn native_agent_get_session(
    agent_state: State<'_, NativeAgentState>,
    session_id: String,
) -> Result<Option<AgentSession>, String> {
    agent_state.get_session(&session_id)
}

#[tauri::command]
pub async fn native_agent_delete_session(
    agent_state: State<'_, NativeAgentState>,
    session_id: String,
) -> Result<bool, String> {
    Ok(agent_state.delete_session(&session_id))
}

#[tauri::command]
pub async fn native_agent_list_sessions(
    agent_state: State<'_, NativeAgentState>,
) -> Result<Vec<AgentSession>, String> {
    Ok(agent_state.list_sessions())
}
