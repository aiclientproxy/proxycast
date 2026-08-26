use super::windows_acl::AclLease;
use super::windows_null::NullDeviceLease;
use super::windows_runner_host::RunnerTransport;
use super::windows_runner_protocol::{
    decode_bytes, encode_bytes, read_frame, write_frame, RunnerMessage,
};
use super::{
    ExecutionOutputDelta, ExecutionProcess, ExecutionProcessSnapshot, ExecutionProcessStatus,
    LocalExecutionControl, OwnedHandle, CONTROL_POLL_MILLIS,
};
use std::fs::File;
use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio::sync::{mpsc as tokio_mpsc, oneshot, watch, Mutex};
use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
use windows_sys::Win32::System::Threading::{TerminateProcess, WaitForSingleObject};

#[allow(clippy::too_many_arguments)]
pub(super) fn supervise(
    transport: RunnerTransport,
    acl_lease: AclLease,
    null_device_lease: NullDeviceLease,
    process: Arc<Mutex<ExecutionProcess>>,
    output_tx: tokio_mpsc::UnboundedSender<ExecutionOutputDelta>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
    final_tx: oneshot::Sender<ExecutionProcessSnapshot>,
    control_rx: Receiver<LocalExecutionControl>,
) {
    let RunnerTransport {
        mut pipe_write,
        mut pipe_read,
        process: runner_process,
    } = transport;
    let (event_tx, event_rx) = mpsc::channel();
    thread::spawn(move || loop {
        let event = read_frame(&mut pipe_read);
        let terminal = matches!(
            event,
            Ok(Some(
                RunnerMessage::Exit { .. } | RunnerMessage::Error { .. }
            )) | Ok(None)
                | Err(_)
        );
        if event_tx.send(event).is_err() || terminal {
            break;
        }
    });

    let mut terminal_exit = None;
    let mut failure = None;
    let mut control_disconnected = false;
    while terminal_exit.is_none() && failure.is_none() {
        if !control_disconnected {
            match control_rx.recv_timeout(Duration::from_millis(CONTROL_POLL_MILLIS as u64)) {
                Ok(control) => {
                    if let Err(error) =
                        forward_control(control, &mut pipe_write, &process, &state_tx)
                    {
                        failure = Some(error.to_string());
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    control_disconnected = true;
                    update_status(&process, &state_tx, ExecutionProcessStatus::Terminated);
                    if let Err(error) = write_frame(&mut pipe_write, RunnerMessage::Terminate) {
                        failure = Some(error.to_string());
                    }
                }
            }
        } else {
            thread::sleep(Duration::from_millis(CONTROL_POLL_MILLIS as u64));
        }

        loop {
            match event_rx.try_recv() {
                Ok(Ok(Some(RunnerMessage::Output { kind, data_base64 }))) => {
                    match decode_bytes(&data_base64) {
                        Ok(bytes) => append_output(&process, &output_tx, &state_tx, kind, &bytes),
                        Err(error) => {
                            failure = Some(error.to_string());
                            break;
                        }
                    }
                }
                Ok(Ok(Some(RunnerMessage::Exit { exit_code }))) => {
                    terminal_exit = Some(exit_code);
                    break;
                }
                Ok(Ok(Some(RunnerMessage::Error {
                    message,
                    windows_error_code,
                }))) => {
                    failure = Some(super::windows_runner_host::format_runner_error(
                        &message,
                        windows_error_code,
                    ));
                    break;
                }
                Ok(Ok(Some(_))) => {
                    failure = Some("Windows sandbox runner sent an unexpected frame".to_string());
                    break;
                }
                Ok(Ok(None)) => {
                    failure = Some("Windows sandbox runner closed before exit".to_string());
                    break;
                }
                Ok(Err(error)) => {
                    failure = Some(format!("Windows sandbox runner IPC failed: {error}"));
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    failure = Some("Windows sandbox runner event reader closed".to_string());
                    break;
                }
            }
        }
    }

    let final_snapshot = {
        let mut guard = process.blocking_lock();
        if !guard.status().is_terminal() {
            if let Some(error) = failure.as_ref() {
                guard.fail(error.clone());
            } else if let Some(exit_code) = terminal_exit {
                guard.exit(exit_code);
            }
        }
        guard.snapshot()
    };
    let _ = state_tx.send(final_snapshot.clone());
    let _ = final_tx.send(final_snapshot);

    if failure.is_some() {
        unsafe {
            TerminateProcess(runner_process.raw(), 1);
        }
    }
    thread::spawn(move || reap_runner(runner_process, pipe_write, acl_lease, null_device_lease));
}

fn forward_control(
    control: LocalExecutionControl,
    pipe_write: &mut File,
    process: &Arc<Mutex<ExecutionProcess>>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
) -> io::Result<()> {
    let message = match control {
        LocalExecutionControl::WriteStdin(bytes) => RunnerMessage::Stdin {
            data_base64: encode_bytes(&bytes),
        },
        LocalExecutionControl::CloseStdin => RunnerMessage::CloseStdin,
        LocalExecutionControl::Resize { rows, cols } => RunnerMessage::Resize { rows, cols },
        LocalExecutionControl::Interrupt => {
            update_status(process, state_tx, ExecutionProcessStatus::Interrupted);
            RunnerMessage::Terminate
        }
        LocalExecutionControl::Terminate => {
            update_status(process, state_tx, ExecutionProcessStatus::Terminated);
            RunnerMessage::Terminate
        }
    };
    write_frame(pipe_write, message)
}

fn append_output(
    process: &Arc<Mutex<ExecutionProcess>>,
    output_tx: &tokio_mpsc::UnboundedSender<ExecutionOutputDelta>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
    kind: super::ExecutionOutputKind,
    bytes: &[u8],
) {
    let (delta, snapshot) = {
        let mut guard = process.blocking_lock();
        let delta = guard.append_output(kind, bytes);
        (delta, guard.snapshot())
    };
    let _ = output_tx.send(delta);
    let _ = state_tx.send(snapshot);
}

fn update_status(
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

fn reap_runner(
    process: OwnedHandle,
    _pipe_write: File,
    _acl_lease: AclLease,
    _null_device_lease: NullDeviceLease,
) {
    let _ = unsafe { WaitForSingleObject(process.raw(), u32::MAX) } == WAIT_OBJECT_0;
}
