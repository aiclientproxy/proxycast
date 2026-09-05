use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use app_server_client::{RemoteTransportConfig, SessionEvent, StdioTransportConfig};
use app_server_protocol::protocol::v2::{ServerNotification, ServerRequest, UserInput};
use futures::StreamExt;
use serde::Serialize;

use crate::app::{App, AppAction};
use crate::app_server_session::AppServerSession;
use crate::bottom_pane::AppServerResponse;
use crate::clipboard_copy::copy_to_clipboard;
use crate::clipboard_paste::paste_image_to_temp_png;
use crate::external_editor::edit_draft;
use crate::locale::Locale;
use crate::projection::ConversationProjection;
use crate::reconnect::reconnect_session;
use crate::resume_picker::run_resume_picker_with_app_server;
use crate::settings::{EFFORTS, SettingsCommand, cycle_setting, parse_settings_command};
use crate::tui::{TerminalGuard, TuiEvent};
use crate::view;

#[derive(Debug, Clone)]
pub struct TuiOptions {
    pub app_server_bin: PathBuf,
    pub app_server_args: Vec<OsString>,
    pub remote: Option<RemoteTransportConfig>,
    pub cwd: PathBuf,
    pub model: Option<String>,
    pub model_provider: Option<String>,
    pub reasoning_effort: Option<String>,
    pub permissions: Option<String>,
    pub locale: Option<String>,
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
    validate_model_route(&options)?;
    let mut session = Some(connect_session(&options).await?);
    let mut app = App::default();
    app.set_cwd(options.cwd.clone());
    app.set_locale(Locale::resolve(options.locale.as_deref()));
    let mut model = options.model.clone();
    let mut model_provider = options.model_provider.clone();
    let mut effort = options.reasoning_effort.clone();
    let mut permissions = options.permissions.clone();
    let setup_result: Result<()> = async {
        let mut permission_cwd = options.cwd.to_string_lossy().into_owned();
        if let Some(thread_id) = options.resume_thread.clone() {
            let response = session
                .as_mut()
                .expect("session available during setup")
                .resume_thread(thread_id)
                .await?;
            permission_cwd = response.cwd.clone();
            app.hydrate_thread(response.thread);
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
        let permission_profiles = session
            .as_ref()
            .expect("session available during setup")
            .list_permission_profiles(Some(permission_cwd))
            .await?;
        app.set_permission_profiles(
            permission_profiles
                .data
                .into_iter()
                .filter(|profile| profile.allowed)
                .map(|profile| profile.id),
        );
        if permissions.is_none() {
            permissions = session
                .as_ref()
                .expect("session available during setup")
                .active_permission_profile()
                .map(str::to_string);
        }
        app.set_thread_id(
            session
                .as_ref()
                .expect("session available during setup")
                .thread_id()?
                .to_string(),
        );
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
        refresh_queued_submissions(
            session.as_ref().expect("session available during setup"),
            &mut app,
        )
        .await;
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
    let mut input = terminal.event_stream();
    let frame_requester = terminal.frame_requester();
    frame_requester.schedule_frame();
    let mut status_tick = tokio::time::interval(Duration::from_secs(1));
    status_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let run_result: Result<()> = async {
        loop {
            terminal
                .terminal_mut()
                .draw(|frame| view::render(frame, &app))
                .context("failed to render terminal")?;

            tokio::select! {
                _ = status_tick.tick(), if app.projection.active_turn_id().is_some() => {
                    frame_requester.schedule_frame();
                }
                event = input.next() => {
                    let Some(event) = event else {
                        break;
                    };
                    let event = match event {
                        TuiEvent::Draw => {
                            if app.projection.active_turn_id().is_some() {
                                frame_requester.schedule_frame_in(Duration::from_secs(1));
                            }
                            continue;
                        }
                        TuiEvent::Key(key) => crossterm::event::Event::Key(key),
                        TuiEvent::Paste(text) => crossterm::event::Event::Paste(text),
                        TuiEvent::Resize(size) => {
                            crossterm::event::Event::Resize(size.width, size.height)
                        }
                        TuiEvent::Resume => continue,
                        TuiEvent::FocusGained => crossterm::event::Event::FocusGained,
                        TuiEvent::FocusLost => crossterm::event::Event::FocusLost,
                    };
                    match app.handle_terminal_event(event) {
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
                            let images = app.take_pending_images();
                            let turn_input = submission_input(prompt.clone(), &images);
                            if let Some(turn_id) = app.projection.active_turn_id().map(str::to_owned) {
                                match session
                                    .as_ref()
                                    .expect("session available during TUI")
                                    .steer_turn_input(&turn_id, turn_input.clone())
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
                                        .queue_input(turn_input)
                                        .await
                                    {
                                    Ok(submission) => {
                                        app.upsert_queued_submission(submission);
                                        app.projection.set_status("queued");
                                            persist_prompt(
                                                session.as_ref().expect("session available during TUI"),
                                                &mut app,
                                                prompt,
                                            )
                                            .await;
                                        }
                                        Err(queue_error) => {
                                            app.restore_pending_images(images);
                                            app.projection.set_status(format!(
                                                "{steer_error}; queue failed: {queue_error}"
                                            ));
                                        }
                                    },
                                }
                            } else {
                                match session
                                    .as_ref()
                                    .expect("session available during TUI")
                                    .start_turn_input(turn_input)
                                    .await
                                {
                                    Ok(turn_id) => {
                                        app.start_turn(turn_id);
                                        persist_prompt(
                                            session.as_ref().expect("session available during TUI"),
                                            &mut app,
                                            prompt,
                                        )
                                        .await;
                                    }
                                    Err(error) => {
                                        app.restore_pending_images(images);
                                        app.projection.set_status(error.to_string());
                                    }
                                }
                            }
                        }
                        AppAction::Queue(prompt) => {
                            let images = app.take_pending_images();
                            let input = submission_input(prompt.clone(), &images);
                            match session
                                .as_ref()
                                .expect("session available during TUI")
                                .queue_input(input)
                                .await
                            {
                                Ok(submission) => {
                                    app.upsert_queued_submission(submission);
                                    app.projection.set_status("queued");
                                    persist_prompt(
                                        session.as_ref().expect("session available during TUI"),
                                        &mut app,
                                        prompt,
                                    )
                                    .await;
                                }
                                Err(error) => {
                                    app.restore_pending_images(images);
                                    app.projection.set_status(error.to_string());
                                }
                            }
                        }
                        AppAction::EditQueuedSubmission(submission) => {
                            match session
                                .as_ref()
                                .expect("session available during TUI")
                                .delete_queued_submission(submission.id.clone())
                                .await
                            {
                                Ok(true) if app.restore_queued_submission_for_edit(submission) => {
                                    app.projection.set_status("queued input editing");
                                }
                                Ok(true) => {
                                    refresh_queued_submissions(
                                        session.as_ref().expect("session available during TUI"),
                                        &mut app,
                                    )
                                    .await;
                                    app.projection.set_status("queued input unavailable");
                                }
                                Ok(false) => {
                                    refresh_queued_submissions(
                                        session.as_ref().expect("session available during TUI"),
                                        &mut app,
                                    )
                                    .await;
                                    app.projection.set_status("queued input unavailable");
                                }
                                Err(error) => app
                                    .projection
                                    .set_status(format!("queue edit failed: {error}")),
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
                            let next = app
                                .cycle_permission_profile(permissions.as_deref(), direction);
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
                        AppAction::CopyLastResponse => {
                            copy_last_response_with(&mut app, copy_to_clipboard);
                        }
                        AppAction::PasteImage => match paste_image_to_temp_png() {
                            Ok((path, info)) => {
                                app.attach_image(path);
                                app.projection.set_status(format!(
                                    "image attached: {}x{}",
                                    info.width, info.height
                                ));
                            }
                            Err(error) => app
                                .projection
                                .set_status(format!("image paste failed: {error}")),
                        },
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
                            let edited = edit_draft(&draft, &options.cwd).await;
                            let resume_result = terminal.resume();
                            match (edited, resume_result) {
                                (Ok(Some(text)), Ok(())) => app.replace_composer(text),
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
                        SessionEvent::Notification(notification) => {
                            let queue_changed = matches!(
                                notification.as_ref(),
                                ServerNotification::ThreadQueueChanged(params)
                                    if app.thread_id.as_deref() == Some(params.thread_id.as_str())
                            );
                            app.apply_notification(*notification);
                            if queue_changed {
                                refresh_queued_submissions(
                                    session.as_ref().expect("session available during TUI"),
                                    &mut app,
                                )
                                .await;
                            }
                        }
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
                                    match app.bottom_pane.enqueue(request) {
                                        Ok(()) => app.pager_overlay = None,
                                        Err(request) => {
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
                            app.pager_overlay = None;
                            app.projection.set_status(format!("reconnecting: {message}"));
                            terminal
                                .terminal_mut()
                                .draw(|frame| view::render(frame, &app))
                                .context("failed to render reconnecting state")?;
                            let old_session = session
                                .take()
                                .expect("session available during reconnect");
                            let _ = old_session.shutdown().await;
                            let reconnected = reconnect_session(
                                &options,
                                &mut app,
                                &thread_id,
                                model.clone(),
                                model_provider.clone(),
                                effort.clone(),
                                permissions.clone(),
                            )
                            .await?;
                            if options.permissions.is_none() {
                                if let Some(active_profile) =
                                    reconnected.active_permission_profile()
                                {
                                    permissions = Some(active_profile.to_string());
                                    app.set_settings(
                                        model.clone(),
                                        model_provider.clone(),
                                        effort.clone(),
                                        permissions.clone(),
                                    );
                                }
                            }
                            refresh_queued_submissions(&reconnected, &mut app).await;
                            session = Some(reconnected);
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
        let selected = run_resume_picker_with_app_server(&options).await?;
        let Some(thread_id) = selected else {
            return Ok(());
        };
        options.resume_thread = Some(thread_id);
    }
    run_tui(options).await
}

async fn persist_prompt(session: &AppServerSession, app: &mut App, prompt: String) {
    if prompt.trim().is_empty() {
        return;
    }
    if let Err(error) = session.append_prompt_history(prompt).await {
        app.projection
            .set_status(format!("prompt history unavailable: {error}"));
    }
}

async fn refresh_queued_submissions(session: &AppServerSession, app: &mut App) {
    match session.list_queued_submissions(100).await {
        Ok(submissions) => app.set_queued_submissions(submissions),
        Err(error) => {
            app.set_queued_submissions(Vec::new());
            app.projection
                .set_status(format!("queue unavailable: {error}"));
        }
    }
}

fn submission_input(prompt: String, images: &[PathBuf]) -> Vec<UserInput> {
    let mut input = images
        .iter()
        .map(|path| UserInput::LocalImage {
            detail: None,
            path: path.to_string_lossy().into_owned(),
        })
        .collect::<Vec<_>>();
    if !prompt.is_empty() {
        input.push(UserInput::Text {
            text: prompt,
            text_elements: Vec::new(),
        });
    }
    input
}

fn copy_last_response_with(
    app: &mut App,
    copy: impl FnOnce(&str) -> Result<Option<crate::clipboard_copy::ClipboardLease>, String>,
) {
    let response = app.projection.final_answer();
    if response.is_empty() {
        app.projection.set_status("no agent response to copy");
        return;
    }
    match copy(&response) {
        Ok(lease) => {
            app.clipboard_lease = lease;
            app.projection.set_status("copied last response");
        }
        Err(error) => app.projection.set_status(format!("copy failed: {error}")),
    }
}

pub async fn run_exec(options: ExecOptions) -> Result<ExecResult> {
    validate_model_route(&options.tui)?;
    if let Some(remote) = options.tui.remote.clone() {
        let session = AppServerSession::connect_remote(remote).await?;
        run_exec_with_session(options, session).await
    } else {
        let config = stdio_config(&options.tui)?;
        run_exec_with_config(options, config).await
    }
}

pub(crate) async fn connect_session(options: &TuiOptions) -> Result<AppServerSession> {
    if let Some(remote) = options.remote.clone() {
        AppServerSession::connect_remote(remote).await
    } else {
        AppServerSession::connect(stdio_config(options)?).await
    }
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
    let session = AppServerSession::connect(config).await?;
    run_exec_with_session(options, session).await
}

async fn run_exec_with_session(
    options: ExecOptions,
    mut session: AppServerSession,
) -> Result<ExecResult> {
    if options.prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
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
    use app_server_protocol::protocol::v2::AgentMessageDeltaNotification;
    use std::cell::RefCell;
    use std::ffi::OsString;

    #[test]
    fn submission_input_keeps_codex_image_then_text_order() {
        let input = submission_input(
            "describe these".to_string(),
            &[PathBuf::from("one.png"), PathBuf::from("two.png")],
        );

        assert_eq!(
            input,
            vec![
                UserInput::LocalImage {
                    detail: None,
                    path: "one.png".to_string(),
                },
                UserInput::LocalImage {
                    detail: None,
                    path: "two.png".to_string(),
                },
                UserInput::Text {
                    text: "describe these".to_string(),
                    text_elements: Vec::new(),
                },
            ]
        );
        assert_eq!(
            submission_input(String::new(), &[PathBuf::from("only.png")]),
            vec![UserInput::LocalImage {
                detail: None,
                path: "only.png".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn real_stdio_unavailable_backend_fails_closed_without_provider() {
        let Some(app_server_bin) = std::env::var_os("LIME_TEST_APP_SERVER_BIN") else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp data directory");
        let tui = TuiOptions {
            app_server_bin: PathBuf::from(&app_server_bin),
            app_server_args: Vec::new(),
            remote: None,
            cwd: temp_dir.path().to_path_buf(),
            model: None,
            model_provider: None,
            reasoning_effort: None,
            permissions: None,
            locale: None,
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
            remote: None,
            cwd: PathBuf::from("."),
            model: Some("gpt-test".to_string()),
            model_provider: None,
            reasoning_effort: None,
            permissions: None,
            locale: None,
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
            remote: None,
            cwd: PathBuf::from("."),
            model: Some("fixture-model".to_string()),
            model_provider: Some("fixture-provider".to_string()),
            reasoning_effort: None,
            permissions: None,
            locale: None,
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

    #[test]
    fn copy_uses_last_canonical_agent_markdown_and_reports_outcome() {
        let mut app = App::default();
        app.projection.apply(ServerNotification::AgentMessageDelta(
            AgentMessageDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "message-1".to_string(),
                delta: "**answer** with `code`".to_string(),
            },
        ));
        let copied = RefCell::new(String::new());

        copy_last_response_with(&mut app, |text| {
            copied.replace(text.to_string());
            Ok(Some(crate::clipboard_copy::ClipboardLease::test()))
        });

        assert_eq!(copied.into_inner(), "**answer** with `code`");
        assert_eq!(app.projection.status(), "copied last response");
        assert!(app.clipboard_lease.is_some());

        let mut empty = App::default();
        copy_last_response_with(&mut empty, |_| panic!("clipboard must not be called"));
        assert_eq!(empty.projection.status(), "no agent response to copy");
    }
}
