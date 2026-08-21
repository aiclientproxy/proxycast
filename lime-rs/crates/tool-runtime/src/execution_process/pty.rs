use super::*;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::io::Read as _;
use std::thread;

pub(super) fn start_local_pty_execution_process(
    request: LocalExecutionRequest,
) -> io::Result<LocalExecutionProcessHandle> {
    let Some(program) = request.command.first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local execution command must not be empty",
        ));
    };

    let mut command = CommandBuilder::new(program);
    for argument in request.command.iter().skip(1) {
        command.arg(argument);
    }
    if let Some(cwd) = &request.cwd {
        command.cwd(cwd);
    }
    command.env_clear();
    for (key, value) in &request.env {
        command.env(key, value);
    }
    if !request.env.contains_key("TERM") {
        command.env("TERM", "xterm-256color");
    }

    let (rows, cols) = request.pty_size.unwrap_or((24, 120));
    let pair = native_pty_system()
        .openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(io_error)?;
    let reader = pair.master.try_clone_reader().map_err(io_error)?;
    let writer = pair.master.take_writer().map_err(io_error)?;
    let master = pair.master;
    let child = pair.slave.spawn_command(command).map_err(io_error)?;
    drop(pair.slave);

    let start = ExecutionProcessStart {
        process_id: request.process_id.clone(),
        tool_id: request.tool_id.clone(),
        tool_name: request.tool_name.clone(),
        command: Some(request.command.join(" ")),
        cwd: request
            .cwd
            .as_ref()
            .map(|cwd| cwd.to_string_lossy().to_string()),
    };
    let process_state = ExecutionProcess::start(start);
    let initial_snapshot = process_state.snapshot();
    let process = Arc::new(Mutex::new(process_state));
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let (control_tx, control_rx) = std::sync::mpsc::channel();
    let (state_tx, state_rx) = watch::channel(initial_snapshot);
    let (final_tx, final_rx) = oneshot::channel();
    let reader_process = Arc::clone(&process);
    let reader_state_tx = state_tx.clone();
    let reader_handle = thread::spawn(move || {
        read_pty_process_stream(reader, reader_process, output_tx, reader_state_tx)
    });
    thread::spawn(move || {
        supervise_local_pty_process(
            child,
            Some(writer),
            master,
            reader_handle,
            process,
            state_tx,
            final_tx,
            control_rx,
        )
    });

    Ok(LocalExecutionProcessHandle {
        process_id: request.process_id,
        control_tx: LocalExecutionControlSender::Blocking(control_tx),
        output_rx,
        state_rx,
        final_rx: Some(final_rx),
        final_snapshot: None,
    })
}

fn read_pty_process_stream(
    mut reader: Box<dyn io::Read + Send>,
    process: Arc<Mutex<ExecutionProcess>>,
    output_tx: mpsc::UnboundedSender<ExecutionOutputDelta>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
) {
    let mut buffer = vec![0; PROCESS_OUTPUT_CHUNK_BYTES];
    loop {
        let bytes_read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(bytes_read) => bytes_read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let (delta, snapshot) = {
            let mut guard = process.blocking_lock();
            let delta = guard.append_output(ExecutionOutputKind::Combined, &buffer[..bytes_read]);
            let snapshot = guard.snapshot();
            (delta, snapshot)
        };
        let _ = output_tx.send(delta);
        let _ = state_tx.send(snapshot);
    }
}

#[allow(clippy::too_many_arguments)]
fn supervise_local_pty_process(
    mut child: Box<dyn portable_pty::Child + Send + Sync>,
    mut writer: Option<Box<dyn io::Write + Send>>,
    master: Box<dyn portable_pty::MasterPty + Send>,
    reader_handle: thread::JoinHandle<()>,
    process: Arc<Mutex<ExecutionProcess>>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
    final_tx: oneshot::Sender<ExecutionProcessSnapshot>,
    control_rx: std::sync::mpsc::Receiver<LocalExecutionControl>,
) {
    let wait_result = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => break Err(error),
        }
        drain_controls(
            &mut *child,
            &mut writer,
            &*master,
            &process,
            &state_tx,
            &control_rx,
        );
        thread::sleep(Duration::from_millis(25));
    };

    drop(writer);
    let _ = reader_handle.join();
    let final_snapshot = {
        let mut guard = process.blocking_lock();
        if !guard.status().is_terminal() {
            match wait_result {
                Ok(status) => guard.exit(i32::try_from(status.exit_code()).unwrap_or(-1)),
                Err(error) => guard.fail(error.to_string()),
            }
        }
        guard.snapshot()
    };
    let _ = state_tx.send(final_snapshot.clone());
    let _ = final_tx.send(final_snapshot);
}

fn drain_controls(
    child: &mut dyn portable_pty::Child,
    writer: &mut Option<Box<dyn io::Write + Send>>,
    master: &dyn portable_pty::MasterPty,
    process: &Arc<Mutex<ExecutionProcess>>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
    control_rx: &std::sync::mpsc::Receiver<LocalExecutionControl>,
) {
    loop {
        match control_rx.try_recv() {
            Ok(LocalExecutionControl::WriteStdin(bytes)) => {
                if let Some(writer) = writer.as_mut() {
                    let _ = writer.write_all(&bytes);
                    let _ = writer.flush();
                }
            }
            Ok(LocalExecutionControl::CloseStdin) => {
                writer.take();
            }
            Ok(LocalExecutionControl::Resize { rows, cols }) => {
                let _ = master.resize(PtySize {
                    rows,
                    cols,
                    pixel_width: 0,
                    pixel_height: 0,
                });
            }
            Ok(LocalExecutionControl::Interrupt) => {
                update_process_status_blocking(
                    process,
                    state_tx,
                    ExecutionProcessStatus::Interrupted,
                );
                let _ = child.kill();
            }
            Ok(LocalExecutionControl::Terminate) => {
                update_process_status_blocking(
                    process,
                    state_tx,
                    ExecutionProcessStatus::Terminated,
                );
                let _ = child.kill();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                update_process_status_blocking(
                    process,
                    state_tx,
                    ExecutionProcessStatus::Terminated,
                );
                let _ = child.kill();
                break;
            }
        }
    }
}

fn update_process_status_blocking(
    process: &Arc<Mutex<ExecutionProcess>>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
    status: ExecutionProcessStatus,
) {
    let snapshot = {
        let mut guard = process.blocking_lock();
        match status {
            ExecutionProcessStatus::Interrupted => guard.interrupt(),
            ExecutionProcessStatus::Terminated => guard.terminate(),
            ExecutionProcessStatus::Failed => guard.fail("process failed"),
            ExecutionProcessStatus::Exited => guard.exit(-1),
            ExecutionProcessStatus::Starting | ExecutionProcessStatus::Running => {}
        }
        guard.snapshot()
    };
    let _ = state_tx.send(snapshot);
}

fn io_error(error: impl ToString) -> io::Error {
    io::Error::other(error.to_string())
}
