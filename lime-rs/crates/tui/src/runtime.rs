use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use app_server_client::{SessionEvent, StdioTransportConfig};
use app_server_protocol::protocol::v2::{ServerNotification, ServerRequest};
use crossterm::event::EventStream;
use futures::StreamExt;
use serde::Serialize;

use crate::app::{App, AppAction};
use crate::app_server_session::AppServerSession;
use crate::bottom_pane::AppServerResponse;
use crate::external_editor::edit_draft;
use crate::projection::ConversationProjection;
use crate::reconnect::reconnect_session;
use crate::session_picker::pick_session;
use crate::settings::{
    cycle_setting, parse_settings_command, SettingsCommand, EFFORTS, PERMISSION_PROFILES,
};
use crate::terminal::TerminalGuard;
use crate::view;

#[derive(Debug, Clone)]
pub struct TuiOptions {
    pub app_server_bin: PathBuf,
    pub app_server_args: Vec<OsString>,
    pub cwd: PathBuf,
    pub model: Option<String>,
    pub model_provider: Option<String>,
    pub reasoning_effort: Option<String>,
    pub permissions: Option<String>,
    pub resume_thread: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExecOptions {
    pub tui: TuiOptions,
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecResult {
    pub thread_id: String,
    pub turn_id: String,
    pub status: String,
    pub output: String,
}

fn validate_model_route(options: &TuiOptions) -> Result<()> {
    match (&options.model, &options.model_provider) {
        (Some(model), Some(provider)) if model.trim().is_empty() || provider.trim().is_empty() => {
            bail!("model and provider must not be empty")
        }
        (Some(_), None) | (None, Some(_)) => {
            bail!("--model and --provider must be specified together")
        }
        _ => Ok(()),
    }
}

pub async fn run_tui(options: TuiOptions) -> Result<()> {
    let config = stdio_config(&options)?;
    let mut session = Some(AppServerSession::connect(config.clone()).await?);
    let mut app = App::default();
    let mut model = options.model.clone();
    let mut model_provider = options.model_provider.clone();
    let mut effort = options.reasoning_effort.clone();
    let mut permissions = options.permissions.clone();
    let setup_result: Result<()> = async {
        if let Some(thread_id) = options.resume_thread.clone() {
            let response = session
                .as_mut()
                .expect("session available during setup")
                .resume_thread(thread_id)
                .await?;
            app.projection.hydrate_thread(response.thread);
            if model.is_none() {
                model = Some(response.model);
            }
            if model_provider.is_none() {
                model_provider = Some(response.model_provider);
            }
            if effort.is_none() {
                effort = response.reasoning_effort;
            }
        } else {
            session
                .as_mut()
                .expect("session available during setup")
                .start_thread(options.cwd.clone(), model.clone(), model_provider.clone())
                .await?;
        }
        session
            .as_ref()
            .expect("session available during setup")
            .update_settings(
                model.clone(),
                model_provider.clone(),
                effort.clone(),
                permissions.clone(),
            )
            .await?;
        app.set_settings(
            model.clone(),
            model_provider.clone(),
            effort.clone(),
            permissions.clone(),
        );
        match session
            .as_ref()
            .expect("session available during setup")
            .read_prompt_history(200)
            .await
        {
            Ok(history) => app
                .composer
                .load_history(history.data.into_iter().map(|entry| entry.text)),
            Err(error) => app
                .projection
                .set_status(format!("prompt history unavailable: {error}")),
        }
        Ok(())
    }
    .await;
    if let Err(error) = setup_result {
        let _ = session
            .take()
            .expect("session available after setup failure")
            .shutdown()
            .await;
        return Err(error);
    }
    let mut terminal = match TerminalGuard::enter().context("failed to initialize terminal") {
        Ok(terminal) => terminal,
        Err(error) => {
            let _ = session
                .take()
                .expect("session available before terminal setup")
                .shutdown()
                .await;
            return Err(error);
        }
    };
    let mut input = EventStream::new();

    let run_result: Result<()> = async {
        loop {
            terminal
                .terminal_mut()
                .draw(|frame| view::render(frame, &app))
                .context("failed to render terminal")?;

            tokio::select! {
                event = input.next() => {
                    let Some(event) = event else {
                        break;
                    };
                    match app.handle_terminal_event(event.context("failed to read terminal event")?) {
                        AppAction::Submit(prompt) => {
                            if let Some(command) = parse_settings_command(&prompt) {
                                match command {
                                    Ok(SettingsCommand::ModelPicker) => {
                                        match session
                                            .as_ref()
                                            .expect("session available during TUI")
                                            .list_models(100)
                                            .await
                                        {
                                            Ok(response) => {
                                                app.open_model_picker(response.data);
                                                app.projection.set_status("choose model");
                                            }
                                            Err(error) => {
                                                app.projection.set_status(error.to_string());
                                            }
                                        }
                                    }
                                    Ok(SettingsCommand::Model {
                                        model: model_value,
                                        provider,
                                    }) => {
                                        match session
                                            .as_ref()
                                            .expect("session available during TUI")
                                            .update_settings(
                                                Some(model_value.clone()),
                                                provider.clone(),
                                                None,
                                                None,
                                            )
                                            .await
                                        {
                                            Ok(()) => {
                                                model = Some(model_value);
                                                model_provider = provider;
                                                app.set_settings(
                                                    model.clone(),
                                                    model_provider.clone(),
                                                    effort.clone(),
                                                    permissions.clone(),
                                                );
                                                app.projection.set_status("settings updated");
                                            }
                                            Err(error) => app.projection.set_status(error.to_string()),
                                        }
                                    }
                                    Ok(SettingsCommand::Effort(value)) => {
                                        match session
                                            .as_ref()
                                            .expect("session available during TUI")
                                            .update_settings(None, None, Some(value.clone()), None)
                                            .await
                                        {
                                            Ok(()) => {
                                                effort = Some(value);
                                                app.set_settings(
                                                    model.clone(),
                                                    model_provider.clone(),
                                                    effort.clone(),
                                                    permissions.clone(),
                                                );
                                                app.projection.set_status("settings updated");
                                            }
                                            Err(error) => app.projection.set_status(error.to_string()),
                                        }
                                    }
                                    Ok(SettingsCommand::Permissions(value)) => {
                                        match session
                                            .as_ref()
                                            .expect("session available during TUI")
                                            .update_settings(None, None, None, Some(value.clone()))
                                            .await
                                        {
                                            Ok(()) => {
                                                permissions = Some(value);
                                                app.set_settings(
                                                    model.clone(),
                                                    model_provider.clone(),
                                                    effort.clone(),
                                                    permissions.clone(),
                                                );
                                                app.projection.set_status("settings updated");
                                            }
                                            Err(error) => app.projection.set_status(error.to_string()),
                                        }
                                    }
                                    Err(error) => app.projection.set_status(error.to_string()),
                                }
                                continue;
                            }
                            if let Some(turn_id) = app.projection.active_turn_id().map(str::to_owned) {
                                match session
                                    .as_ref()
                                    .expect("session available during TUI")
                                    .steer_turn(&turn_id, prompt.clone())
                                    .await
                                {
                                    Ok(_) => {
                                        app.projection.set_status("steering");
                                        persist_prompt(
                                            session.as_ref().expect("session available during TUI"),
                                            &mut app,
                                            prompt,
                                        )
                                        .await;
                                    }
                                    Err(steer_error) => match session
                                        .as_ref()
                                        .expect("session available during TUI")
                                        .queue_prompt(prompt.clone())
                                        .await
                                    {
                                        Ok(_) => {
                                            app.projection.set_status("queued");
                                            persist_prompt(
                                                session.as_ref().expect("session available during TUI"),
                                                &mut app,
                                                prompt,
                                            )
                                            .await;
                                        }
                                        Err(queue_error) => app.projection.set_status(format!(
                                            "{steer_error}; queue failed: {queue_error}"
                                        )),
                                    },
                                }
                            } else {
                                match session
                                    .as_ref()
                                    .expect("session available during TUI")
                                    .start_turn(prompt.clone())
                                    .await
                                {
                                    Ok(turn_id) => {
                                        app.projection.start_turn(turn_id);
                                        persist_prompt(
                                            session.as_ref().expect("session available during TUI"),
                                            &mut app,
                                            prompt,
                                        )
                                        .await;
                                    }
                                    Err(error) => app.projection.set_status(error.to_string()),
                                }
                            }
                        }
                        AppAction::Queue(prompt) => {
                            match session
                                .as_ref()
                                .expect("session available during TUI")
                                .queue_prompt(prompt.clone())
                                .await
                            {
                                Ok(_) => {
                                    app.projection.set_status("queued");
                                    persist_prompt(
                                        session.as_ref().expect("session available during TUI"),
                                        &mut app,
                                        prompt,
                                    )
                                    .await;
                                }
                                Err(error) => app.projection.set_status(error.to_string()),
                            }
                        }
                        action @ (AppAction::DecreaseEffort | AppAction::IncreaseEffort) => {
                            let direction = if matches!(action, AppAction::DecreaseEffort) {
                                -1
                            } else {
                                1
                            };
                            let next = cycle_setting(&EFFORTS, effort.as_deref(), direction);
                            match session
                                .as_ref()
                                .expect("session available during TUI")
                                .update_settings(None, None, Some(next.clone()), None)
                                .await
                            {
                                Ok(()) => {
                                    effort = Some(next.clone());
                                    app.set_settings(
                                        model.clone(),
                                        model_provider.clone(),
                                        effort.clone(),
                                        permissions.clone(),
                                    );
                                    app.projection.set_status(format!("effort: {next}"));
                                }
                                Err(error) => app.projection.set_status(error.to_string()),
                            }
                        }
                        action @ (AppAction::PreviousPermissions | AppAction::NextPermissions) => {
                            let direction = if matches!(action, AppAction::PreviousPermissions) {
                                -1
                            } else {
                                1
                            };
                            let next = cycle_setting(
                                &PERMISSION_PROFILES,
                                permissions.as_deref(),
                                direction,
                            );
                            match session
                                .as_ref()
                                .expect("session available during TUI")
                                .update_settings(None, None, None, Some(next.clone()))
                                .await
                            {
                                Ok(()) => {
                                    permissions = Some(next.clone());
                                    app.set_settings(
                                        model.clone(),
                                        model_provider.clone(),
                                        effort.clone(),
                                        permissions.clone(),
                                    );
                                    app.projection.set_status(format!("permissions: {next}"));
                                }
                                Err(error) => app.projection.set_status(error.to_string()),
                            }
                        }
                        AppAction::Interrupt => {
                            if let Some(turn_id) = app.projection.active_turn_id() {
                                let turn_id = turn_id.to_string();
                                match session
                                    .as_ref()
                                    .expect("session available during TUI")
                                    .interrupt(&turn_id)
                                    .await
                                {
                                    Ok(()) => app.projection.set_status("interrupting"),
                                    Err(error) if is_no_active_turn_error(&error) => break,
                                    Err(error) => return Err(error),
                                }
                            } else {
                                break;
                            }
                        }
                        AppAction::OpenExternalEditor => {
                            let draft = app.composer.text().to_string();
                            if let Err(error) = terminal.suspend() {
                                app.projection.set_status(format!("editor unavailable: {error}"));
                                continue;
                            }
                            let edited = edit_draft(&draft, &options.cwd);
                            let resume_result = terminal.resume();
                            match (edited, resume_result) {
                                (Ok(Some(text)), Ok(())) => app.composer.replace(text),
                                (Ok(None), Ok(())) => app.projection.set_status("editor draft empty"),
                                (Err(error), Ok(())) => app.projection.set_status(error.to_string()),
                                (_, Err(error)) => return Err(anyhow!("failed to resume terminal after editor: {error}")),
                            }
                        }
                        AppAction::ScrollUp => {
                            let page_size = current_transcript_page_size(&mut terminal, &app)?;
                            app.scroll_up(page_size);
                        }
                        AppAction::ScrollDown => {
                            let page_size = current_transcript_page_size(&mut terminal, &app)?;
                            app.scroll_down(page_size);
                        }
                        AppAction::ScrollTop => app.scroll_top(),
                        AppAction::ScrollBottom => app.scroll_bottom(),
                        AppAction::Respond(response) => {
                            if let Err(error) = session
                                .as_ref()
                                .expect("session available during TUI")
                                .respond(response)
                                .await
                            {
                                app.projection.set_status(error.to_string());
                            }
                        }
                        AppAction::SelectModel(selection) => {
                            match session
                                .as_ref()
                                .expect("session available during TUI")
                                .update_settings(
                                    Some(selection.model.clone()),
                                    Some(selection.provider.clone()),
                                    None,
                                    None,
                                )
                                .await
                            {
                                Ok(()) => {
                                    model = Some(selection.model.clone());
                                    model_provider = Some(selection.provider.clone());
                                    app.set_settings(
                                        model.clone(),
                                        model_provider.clone(),
                                        effort.clone(),
                                        permissions.clone(),
                                    );
                                    app.projection.set_status("settings updated");
                                }
                                Err(error) => app.projection.set_status(error.to_string()),
                            }
                        }
                        AppAction::Quit => break,
                        AppAction::None => {}
                    }
                }
                event = session
                    .as_mut()
                    .expect("session available during TUI")
                    .next_event() => {
                    let event = event.unwrap_or_else(|| SessionEvent::Disconnected {
                        message: "app-server event stream closed".to_string(),
                    });
                    match event {
                        SessionEvent::Notification(notification) => app.projection.apply(*notification),
                        SessionEvent::ServerRequest(request) => {
                            match *request {
                                ServerRequest::CurrentTimeRead { id, .. } => {
                                    if let Err(error) = session
                                        .as_ref()
                                        .expect("session available during TUI")
                                        .respond_current_time(id)
                                        .await
                                    {
                                        app.projection.set_status(error.to_string());
                                    }
                                }
                                request => {
                                    if let Err(request) = app.bottom_pane.enqueue(request) {
                                        if let Err(error) = session
                                            .as_ref()
                                            .expect("session available during TUI")
                                            .reject_server_request(request)
                                            .await
                                        {
                                            app.projection.set_status(error.to_string());
                                        }
                                    }
                                }
                            }
                        }
                        SessionEvent::RawServerRequest(request) => {
                            if let Err(error) = session
                                .as_ref()
                                .expect("session available during TUI")
                                .reject_raw_server_request(request)
                                .await
                            {
                                app.projection.set_status(error.to_string());
                            }
                        }
                        SessionEvent::Disconnected { message } => {
                            let thread_id = session
                                .as_ref()
                                .expect("session available during TUI")
                                .thread_id()?
                            .to_string();
                            app.bottom_pane.clear();
                            app.model_picker = None;
                            app.projection.set_status(format!("reconnecting: {message}"));
                            terminal
                                .terminal_mut()
                                .draw(|frame| view::render(frame, &app))
                                .context("failed to render reconnecting state")?;
                            let old_session = session
                                .take()
                                .expect("session available during reconnect");
                            let _ = old_session.shutdown().await;
                            session = Some(reconnect_session(
                                &config,
                                &mut app,
                                &thread_id,
                                model.clone(),
                                model_provider.clone(),
                                effort.clone(),
                                permissions.clone(),
                            )
                            .await?);
                        }
                        SessionEvent::RawNotification(_) => {}
                    }
                }
            }
        }
        Ok(())
    }
    .await;

    let restore_result = terminal.restore().context("failed to restore terminal");
    let shutdown_result = match session {
        Some(session) => session.shutdown().await,
        None => Ok(()),
    };
    run_result?;
    restore_result?;
    shutdown_result
}

fn is_no_active_turn_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string().to_ascii_lowercase();
        message.contains("no active turn") || message.contains("turn_not_active")
    })
}

fn current_transcript_page_size(terminal: &mut TerminalGuard, app: &App) -> Result<usize> {
    let size = terminal
        .terminal_mut()
        .size()
        .context("failed to read terminal size")?;
    Ok(view::transcript_page_size(size.width, size.height, app))
}

/// Resume without an explicit id using the same canonical thread picker as Codex.
/// The picker session is short-lived; the selected thread is then resumed by the
/// normal TUI runtime so there is only one conversation owner.
pub async fn run_resume(mut options: TuiOptions) -> Result<()> {
    if options.resume_thread.is_none() {
        let selected = pick_session(&options).await?;
        let Some(thread_id) = selected else {
            return Ok(());
        };
        options.resume_thread = Some(thread_id);
    }
    run_tui(options).await
}

async fn persist_prompt(session: &AppServerSession, app: &mut App, prompt: String) {
    if let Err(error) = session.append_prompt_history(prompt).await {
        app.projection
            .set_status(format!("prompt history unavailable: {error}"));
    }
}

pub async fn run_exec(options: ExecOptions) -> Result<ExecResult> {
    let config = stdio_config(&options.tui)?;
    run_exec_with_config(options, config).await
}

pub(crate) fn stdio_config(options: &TuiOptions) -> Result<StdioTransportConfig> {
    validate_model_route(options)?;
    let mut config = StdioTransportConfig::runtime(&options.app_server_bin);
    config.args.extend(options.app_server_args.iter().cloned());
    Ok(config)
}

async fn run_exec_with_config(
    options: ExecOptions,
    config: StdioTransportConfig,
) -> Result<ExecResult> {
    if options.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    let mut session = AppServerSession::connect(config).await?;
    let execution = async {
        let thread_id = if let Some(thread_id) = options.tui.resume_thread.clone() {
            let response = session.resume_thread(thread_id).await?;
            let thread_id = response.thread.id.clone();
            let mut projection = ConversationProjection::default();
            projection.hydrate_thread(response.thread);
            (thread_id, projection)
        } else {
            let thread_id = session
                .start_thread(
                    options.tui.cwd.clone(),
                    options.tui.model.clone(),
                    options.tui.model_provider.clone(),
                )
                .await?;
            (thread_id, ConversationProjection::default())
        };
        session
            .update_settings(
                options.tui.model.clone(),
                options.tui.model_provider.clone(),
                options.tui.reasoning_effort.clone(),
                options.tui.permissions.clone(),
            )
            .await?;
        let (thread_id, mut projection) = thread_id;
        let turn_id = session.start_turn(options.prompt).await?;

        loop {
            let event = session
                .next_event()
                .await
                .ok_or_else(|| anyhow!("App Server disconnected before turn completion"))?;
            match event {
                SessionEvent::Notification(notification) => {
                    let completed = matches!(
                        notification.as_ref(),
                        ServerNotification::TurnCompleted(params) if params.turn.id == turn_id
                    );
                    projection.apply(*notification);
                    if completed {
                        break;
                    }
                }
                SessionEvent::ServerRequest(request) => match *request {
                    ServerRequest::CurrentTimeRead { id, .. } => {
                        session.respond_current_time(id).await?;
                    }
                    request => match AppServerResponse::fail_closed(request) {
                        Ok(response) => session.respond(response).await?,
                        Err(request) => session.reject_server_request(request).await?,
                    },
                },
                SessionEvent::RawServerRequest(request) => {
                    session.reject_raw_server_request(request).await?;
                }
                SessionEvent::Disconnected { message } => bail!(message),
                SessionEvent::RawNotification(_) => {}
            }
        }

        Ok(ExecResult {
            thread_id,
            turn_id,
            status: projection.status().to_string(),
            output: projection.final_answer(),
        })
    };
    let execution = execution.await;
    let shutdown = session.shutdown().await;
    match execution {
        Ok(result) => {
            shutdown?;
            Ok(result)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "runtime_pty_tests.rs"]
mod pty_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[tokio::test]
    async fn real_stdio_unavailable_backend_fails_closed_without_provider() {
        let Some(app_server_bin) = std::env::var_os("LIME_TEST_APP_SERVER_BIN") else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp data directory");
        let tui = TuiOptions {
            app_server_bin: PathBuf::from(&app_server_bin),
            app_server_args: Vec::new(),
            cwd: temp_dir.path().to_path_buf(),
            model: None,
            model_provider: None,
            reasoning_effort: None,
            permissions: None,
            resume_thread: None,
        };
        let config = StdioTransportConfig {
            app_server_bin: PathBuf::from(app_server_bin),
            args: vec![
                OsString::from("--stdio"),
                OsString::from("--backend"),
                OsString::from("unavailable"),
                OsString::from("--data-dir"),
                temp_dir.path().as_os_str().to_os_string(),
            ],
        };

        let error = run_exec_with_config(
            ExecOptions {
                tui,
                prompt: "stdio contract probe".to_string(),
            },
            config,
        )
        .await
        .expect_err("unavailable backend must reject turn/start");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to start App Server thread")
                && rendered.contains("runtime model route is not executable"),
            "unexpected fail-closed error: {rendered}"
        );
    }

    #[test]
    fn model_route_requires_model_and_provider_together() {
        let options = TuiOptions {
            app_server_bin: PathBuf::from("app-server"),
            app_server_args: Vec::new(),
            cwd: PathBuf::from("."),
            model: Some("gpt-test".to_string()),
            model_provider: None,
            reasoning_effort: None,
            permissions: None,
            resume_thread: None,
        };

        assert_eq!(
            validate_model_route(&options)
                .expect_err("partial route must fail")
                .to_string(),
            "--model and --provider must be specified together"
        );
    }

    #[test]
    fn stdio_config_appends_host_arguments_after_the_current_runtime_default() {
        let options = TuiOptions {
            app_server_bin: PathBuf::from("custom-app-server"),
            app_server_args: vec![OsString::from("--backend"), OsString::from("external")],
            cwd: PathBuf::from("."),
            model: Some("fixture-model".to_string()),
            model_provider: Some("fixture-provider".to_string()),
            reasoning_effort: None,
            permissions: None,
            resume_thread: None,
        };

        let config = stdio_config(&options).expect("stdio config");
        assert_eq!(
            config.args,
            vec![
                OsString::from("--stdio"),
                OsString::from("--backend"),
                OsString::from("runtime"),
                OsString::from("--backend"),
                OsString::from("external"),
            ]
        );
    }
}
