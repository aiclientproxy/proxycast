use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, watch, Mutex};
use tokio::task::JoinHandle;

pub mod live;
mod pty;

const DEFAULT_OUTPUT_RETAIN_BYTES: usize = 128 * 1024;
const PROCESS_OUTPUT_CHUNK_BYTES: usize = 8 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProcessStatus {
    Starting,
    Running,
    Exited,
    Interrupted,
    Terminated,
    Failed,
}

impl ExecutionProcessStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Exited => "exited",
            Self::Interrupted => "interrupted",
            Self::Terminated => "terminated",
            Self::Failed => "failed",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Exited | Self::Interrupted | Self::Terminated | Self::Failed
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOutputKind {
    Stdout,
    Stderr,
    Combined,
}

impl ExecutionOutputKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
            Self::Combined => "combined",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionOutputDelta {
    pub process_id: String,
    pub tool_id: String,
    pub sequence: u64,
    pub kind: ExecutionOutputKind,
    pub delta: String,
    pub bytes: u64,
    pub omitted_bytes: u64,
    pub truncated: bool,
    #[serde(skip)]
    pub raw_bytes: Vec<u8>,
}

impl ExecutionOutputDelta {
    pub fn metadata(&self) -> HashMap<String, Value> {
        HashMap::from([
            ("processId".to_string(), json!(self.process_id)),
            ("outputSequence".to_string(), json!(self.sequence)),
            ("outputKind".to_string(), json!(self.kind.label())),
            ("outputBytes".to_string(), json!(self.bytes)),
            ("outputOmittedBytes".to_string(), json!(self.omitted_bytes)),
            ("outputTruncated".to_string(), json!(self.truncated)),
            (
                "executionProcessStatus".to_string(),
                json!(ExecutionProcessStatus::Running.label()),
            ),
            ("stdinWritable".to_string(), json!(true)),
            ("stdin_writable".to_string(), json!(true)),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionProcessSnapshot {
    pub process_id: String,
    pub tool_id: String,
    pub tool_name: String,
    pub status: ExecutionProcessStatus,
    pub exit_code: Option<i32>,
    pub elapsed_ms: u64,
    pub output_bytes: u64,
    pub output_omitted_bytes: u64,
    pub output_truncated: bool,
    pub retained_output: String,
    pub failure: Option<String>,
}

impl ExecutionProcessSnapshot {
    pub fn metadata(&self) -> HashMap<String, Value> {
        HashMap::from([
            ("processId".to_string(), json!(self.process_id)),
            (
                "executionProcessStatus".to_string(),
                json!(self.status.label()),
            ),
            (
                "executionProcessElapsedMs".to_string(),
                json!(self.elapsed_ms),
            ),
            ("outputBytes".to_string(), json!(self.output_bytes)),
            (
                "outputOmittedBytes".to_string(),
                json!(self.output_omitted_bytes),
            ),
            ("outputTruncated".to_string(), json!(self.output_truncated)),
            ("exitCode".to_string(), json!(self.exit_code)),
            ("exit_code".to_string(), json!(self.exit_code)),
            (
                "stdinWritable".to_string(),
                json!(!self.status.is_terminal()),
            ),
            (
                "stdin_writable".to_string(),
                json!(!self.status.is_terminal()),
            ),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProcessStart {
    pub process_id: String,
    pub tool_id: String,
    pub tool_name: String,
    pub command: Option<String>,
    pub cwd: Option<String>,
}

impl ExecutionProcessStart {
    pub fn metadata(&self) -> HashMap<String, Value> {
        let mut metadata = HashMap::from([
            ("processId".to_string(), json!(self.process_id)),
            (
                "executionProcessStatus".to_string(),
                json!(ExecutionProcessStatus::Running.label()),
            ),
            ("stdinWritable".to_string(), json!(true)),
            ("stdin_writable".to_string(), json!(true)),
        ]);
        if let Some(command) = &self.command {
            metadata.insert("command".to_string(), json!(command));
        }
        if let Some(cwd) = &self.cwd {
            metadata.insert("cwd".to_string(), json!(cwd));
        }
        metadata
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProcess {
    process_id: String,
    tool_id: String,
    tool_name: String,
    command: Option<String>,
    cwd: Option<String>,
    status: ExecutionProcessStatus,
    exit_code: Option<i32>,
    failure: Option<String>,
    started_at: Instant,
    output: BoundedProcessOutput,
    sequence: u64,
}

impl ExecutionProcess {
    pub fn start(start: ExecutionProcessStart) -> Self {
        Self {
            process_id: start.process_id,
            tool_id: start.tool_id,
            tool_name: start.tool_name,
            command: start.command,
            cwd: start.cwd,
            status: ExecutionProcessStatus::Running,
            exit_code: None,
            failure: None,
            started_at: Instant::now(),
            output: BoundedProcessOutput::new(DEFAULT_OUTPUT_RETAIN_BYTES),
            sequence: 0,
        }
    }

    pub fn process_id(&self) -> &str {
        &self.process_id
    }

    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    pub fn status(&self) -> ExecutionProcessStatus {
        self.status
    }

    pub fn append_output(
        &mut self,
        kind: ExecutionOutputKind,
        bytes: &[u8],
    ) -> ExecutionOutputDelta {
        self.sequence = self.sequence.saturating_add(1);
        self.output.push(bytes);
        let snapshot = self.output.snapshot();
        ExecutionOutputDelta {
            process_id: self.process_id.clone(),
            tool_id: self.tool_id.clone(),
            sequence: self.sequence,
            kind,
            delta: String::from_utf8_lossy(bytes).to_string(),
            bytes: snapshot.bytes,
            omitted_bytes: snapshot.omitted_bytes,
            truncated: snapshot.truncated,
            raw_bytes: bytes.to_vec(),
        }
    }

    pub fn interrupt(&mut self) {
        if !self.status.is_terminal() {
            self.status = ExecutionProcessStatus::Interrupted;
        }
    }

    pub fn terminate(&mut self) {
        if !self.status.is_terminal() {
            self.status = ExecutionProcessStatus::Terminated;
        }
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        if !self.status.is_terminal() {
            self.status = ExecutionProcessStatus::Failed;
            self.failure = Some(message.into());
        }
    }

    pub fn exit(&mut self, exit_code: i32) {
        if !self.status.is_terminal() {
            self.status = ExecutionProcessStatus::Exited;
            self.exit_code = Some(exit_code);
        }
    }

    pub fn snapshot(&self) -> ExecutionProcessSnapshot {
        let output = self.output.snapshot();
        ExecutionProcessSnapshot {
            process_id: self.process_id.clone(),
            tool_id: self.tool_id.clone(),
            tool_name: self.tool_name.clone(),
            status: self.status,
            exit_code: self.exit_code,
            elapsed_ms: duration_millis(self.started_at.elapsed()),
            output_bytes: output.bytes,
            output_omitted_bytes: output.omitted_bytes,
            output_truncated: output.truncated,
            retained_output: output.text,
            failure: self.failure.clone(),
        }
    }
}

#[derive(Debug, Default)]
pub struct ExecutionProcessManager {
    processes: HashMap<String, ExecutionProcess>,
}

impl ExecutionProcessManager {
    pub fn start(&mut self, start: ExecutionProcessStart) -> ExecutionProcessSnapshot {
        let process_id = start.process_id.clone();
        let process = ExecutionProcess::start(start);
        let snapshot = process.snapshot();
        self.processes.insert(process_id, process);
        snapshot
    }

    pub fn append_output(
        &mut self,
        process_id: &str,
        kind: ExecutionOutputKind,
        bytes: &[u8],
    ) -> Option<ExecutionOutputDelta> {
        self.processes
            .get_mut(process_id)
            .map(|process| process.append_output(kind, bytes))
    }

    pub fn interrupt(&mut self, process_id: &str) -> Option<ExecutionProcessSnapshot> {
        let process = self.processes.get_mut(process_id)?;
        process.interrupt();
        Some(process.snapshot())
    }

    pub fn terminate(&mut self, process_id: &str) -> Option<ExecutionProcessSnapshot> {
        let process = self.processes.get_mut(process_id)?;
        process.terminate();
        Some(process.snapshot())
    }

    pub fn exit(&mut self, process_id: &str, exit_code: i32) -> Option<ExecutionProcessSnapshot> {
        let process = self.processes.get_mut(process_id)?;
        process.exit(exit_code);
        Some(process.snapshot())
    }

    pub fn fail(
        &mut self,
        process_id: &str,
        message: impl Into<String>,
    ) -> Option<ExecutionProcessSnapshot> {
        let process = self.processes.get_mut(process_id)?;
        process.fail(message);
        Some(process.snapshot())
    }

    pub fn snapshot(&self, process_id: &str) -> Option<ExecutionProcessSnapshot> {
        self.processes
            .get(process_id)
            .map(ExecutionProcess::snapshot)
    }

    pub fn remove(&mut self, process_id: &str) -> Option<ExecutionProcessSnapshot> {
        self.processes
            .remove(process_id)
            .map(|process| process.snapshot())
    }

    pub fn len(&self) -> usize {
        self.processes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.processes.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalExecutionRequest {
    pub process_id: String,
    pub tool_id: String,
    pub tool_name: String,
    pub command: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: HashMap<String, String>,
    pub tty: bool,
    pub stdin: bool,
    pub env_clear: bool,
    pub pty_size: Option<(u16, u16)>,
}

impl LocalExecutionRequest {
    pub fn new(
        process_id: impl Into<String>,
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        command: Vec<String>,
    ) -> Self {
        Self {
            process_id: process_id.into(),
            tool_id: tool_id.into(),
            tool_name: tool_name.into(),
            command,
            cwd: None,
            env: HashMap::new(),
            tty: false,
            stdin: true,
            env_clear: false,
            pty_size: None,
        }
    }
}

pub trait LiveExecutionProcessRegistry: Send + Sync {
    fn register_live_process(
        &self,
        handle: LocalExecutionProcessControlHandle,
        snapshot: ExecutionProcessSnapshot,
    ) -> Result<(), String>;

    fn record_live_process_output(&self, delta: ExecutionOutputDelta) -> Result<(), String>;

    fn finish_live_process(&self, snapshot: ExecutionProcessSnapshot) -> Result<(), String>;
}

#[derive(Debug)]
pub struct LocalExecutionProcessHandle {
    process_id: String,
    control_tx: LocalExecutionControlSender,
    output_rx: mpsc::UnboundedReceiver<ExecutionOutputDelta>,
    state_rx: watch::Receiver<ExecutionProcessSnapshot>,
    final_rx: Option<oneshot::Receiver<ExecutionProcessSnapshot>>,
    final_snapshot: Option<ExecutionProcessSnapshot>,
}

#[derive(Debug, Clone)]
pub struct LocalExecutionProcessControlHandle {
    process_id: String,
    control_tx: LocalExecutionControlSender,
    state_rx: watch::Receiver<ExecutionProcessSnapshot>,
}

#[derive(Debug)]
pub enum LocalExecutionProcessEvent {
    Output(ExecutionOutputDelta),
    Exited(ExecutionProcessSnapshot),
}

impl LocalExecutionProcessHandle {
    pub fn process_id(&self) -> &str {
        &self.process_id
    }

    pub fn control_handle(&self) -> LocalExecutionProcessControlHandle {
        LocalExecutionProcessControlHandle {
            process_id: self.process_id.clone(),
            control_tx: self.control_tx.clone(),
            state_rx: self.state_rx.clone(),
        }
    }

    pub fn status(&self) -> ExecutionProcessSnapshot {
        self.state_rx.borrow().clone()
    }

    pub async fn recv_output(&mut self) -> Option<ExecutionOutputDelta> {
        self.output_rx.recv().await
    }

    pub async fn next_event(&mut self) -> Option<LocalExecutionProcessEvent> {
        loop {
            if let Ok(delta) = self.output_rx.try_recv() {
                return Some(LocalExecutionProcessEvent::Output(delta));
            }
            if self.final_snapshot.is_some() {
                return None;
            }
            let mut final_rx = self.final_rx.take()?;
            tokio::select! {
                biased;
                delta = self.output_rx.recv() => {
                    match delta {
                        Some(delta) => {
                            self.final_rx = Some(final_rx);
                            return Some(LocalExecutionProcessEvent::Output(delta));
                        }
                        None => {
                            let snapshot = final_rx.await.ok()?;
                            self.final_snapshot = Some(snapshot.clone());
                            return Some(LocalExecutionProcessEvent::Exited(snapshot));
                        }
                    }
                }
                result = &mut final_rx => {
                    let snapshot = result.ok()?;
                    self.final_snapshot = Some(snapshot.clone());
                    return Some(LocalExecutionProcessEvent::Exited(snapshot));
                }
            }
        }
    }

    pub fn write_stdin(&self, bytes: impl Into<Vec<u8>>) -> Result<(), LocalExecutionError> {
        self.control_tx
            .send(LocalExecutionControl::WriteStdin(bytes.into()))
    }

    pub fn interrupt(&self) -> Result<(), LocalExecutionError> {
        self.control_tx.send(LocalExecutionControl::Interrupt)
    }

    pub fn terminate(&self) -> Result<(), LocalExecutionError> {
        self.control_tx.send(LocalExecutionControl::Terminate)
    }

    pub fn close_stdin(&self) -> Result<(), LocalExecutionError> {
        self.control_tx.send(LocalExecutionControl::CloseStdin)
    }

    pub fn resize(&self, rows: u16, cols: u16) -> Result<(), LocalExecutionError> {
        self.control_tx
            .send(LocalExecutionControl::Resize { rows, cols })
    }

    pub async fn wait(&mut self) -> Result<ExecutionProcessSnapshot, LocalExecutionError> {
        if let Some(snapshot) = &self.final_snapshot {
            return Ok(snapshot.clone());
        }
        let final_rx = self
            .final_rx
            .take()
            .ok_or(LocalExecutionError::SupervisorClosed)?;
        let snapshot = final_rx
            .await
            .map_err(|_| LocalExecutionError::SupervisorClosed)?;
        self.final_snapshot = Some(snapshot.clone());
        Ok(snapshot)
    }
}

impl LocalExecutionProcessControlHandle {
    pub fn process_id(&self) -> &str {
        &self.process_id
    }

    pub fn status(&self) -> ExecutionProcessSnapshot {
        self.state_rx.borrow().clone()
    }

    pub fn write_stdin(&self, bytes: impl Into<Vec<u8>>) -> Result<(), LocalExecutionError> {
        self.control_tx
            .send(LocalExecutionControl::WriteStdin(bytes.into()))
    }

    pub fn interrupt(&self) -> Result<(), LocalExecutionError> {
        self.control_tx.send(LocalExecutionControl::Interrupt)
    }

    pub fn terminate(&self) -> Result<(), LocalExecutionError> {
        self.control_tx.send(LocalExecutionControl::Terminate)
    }

    pub fn close_stdin(&self) -> Result<(), LocalExecutionError> {
        self.control_tx.send(LocalExecutionControl::CloseStdin)
    }

    pub fn resize(&self, rows: u16, cols: u16) -> Result<(), LocalExecutionError> {
        self.control_tx
            .send(LocalExecutionControl::Resize { rows, cols })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalExecutionError {
    ControlClosed,
    SupervisorClosed,
}

impl std::fmt::Display for LocalExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ControlClosed => "process control channel closed",
            Self::SupervisorClosed => "process supervisor closed",
        })
    }
}

impl std::error::Error for LocalExecutionError {}

enum LocalExecutionControl {
    WriteStdin(Vec<u8>),
    CloseStdin,
    Resize { rows: u16, cols: u16 },
    Interrupt,
    Terminate,
}

#[derive(Debug, Clone)]
enum LocalExecutionControlSender {
    Async(mpsc::UnboundedSender<LocalExecutionControl>),
    Blocking(std::sync::mpsc::Sender<LocalExecutionControl>),
}

impl LocalExecutionControlSender {
    fn send(&self, control: LocalExecutionControl) -> Result<(), LocalExecutionError> {
        match self {
            Self::Async(sender) => sender
                .send(control)
                .map_err(|_| LocalExecutionError::ControlClosed),
            Self::Blocking(sender) => sender
                .send(control)
                .map_err(|_| LocalExecutionError::ControlClosed),
        }
    }
}

pub fn start_local_execution_process(
    request: LocalExecutionRequest,
) -> io::Result<LocalExecutionProcessHandle> {
    if request.tty {
        return pty::start_local_pty_execution_process(request);
    }
    let Some(program) = request.command.first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local execution command must not be empty",
        ));
    };

    let mut command = Command::new(program);
    command
        .args(request.command.iter().skip(1))
        .stdin(if request.stdin {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = &request.cwd {
        command.current_dir(cwd);
    }
    if request.env_clear {
        command.env_clear();
    }
    if !request.env.is_empty() {
        command.envs(&request.env);
    }

    let mut child = command.spawn()?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdin = child.stdin.take();

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
    let (control_tx, control_rx) = mpsc::unbounded_channel();
    let (state_tx, state_rx) = watch::channel(initial_snapshot);
    let (final_tx, final_rx) = oneshot::channel();

    tokio::spawn(supervise_local_process(
        child, stdin, stdout, stderr, process, output_tx, state_tx, final_tx, control_rx,
    ));

    Ok(LocalExecutionProcessHandle {
        process_id: request.process_id,
        control_tx: LocalExecutionControlSender::Async(control_tx),
        output_rx,
        state_rx,
        final_rx: Some(final_rx),
        final_snapshot: None,
    })
}

#[allow(clippy::too_many_arguments)]
async fn supervise_local_process(
    mut child: Child,
    mut stdin: Option<ChildStdin>,
    stdout: Option<impl AsyncRead + Unpin + Send + 'static>,
    stderr: Option<impl AsyncRead + Unpin + Send + 'static>,
    process: Arc<Mutex<ExecutionProcess>>,
    output_tx: mpsc::UnboundedSender<ExecutionOutputDelta>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
    final_tx: oneshot::Sender<ExecutionProcessSnapshot>,
    mut control_rx: mpsc::UnboundedReceiver<LocalExecutionControl>,
) {
    let mut control_open = true;
    let stdout_task = stdout.map(|reader| {
        tokio::spawn(read_process_stream(
            reader,
            ExecutionOutputKind::Stdout,
            Arc::clone(&process),
            output_tx.clone(),
            state_tx.clone(),
        ))
    });
    let stderr_task = stderr.map(|reader| {
        tokio::spawn(read_process_stream(
            reader,
            ExecutionOutputKind::Stderr,
            Arc::clone(&process),
            output_tx,
            state_tx.clone(),
        ))
    });

    let wait_result = loop {
        tokio::select! {
            result = child.wait() => break result,
            control = control_rx.recv(), if control_open => {
                let Some(control) = control else {
                    control_open = false;
                    update_process_status(&process, &state_tx, ExecutionProcessStatus::Terminated).await;
                    let _ = child.start_kill();
                    continue;
                };
                match control {
                    LocalExecutionControl::WriteStdin(bytes) => {
                        if let Some(stdin) = stdin.as_mut() {
                            let _ = stdin.write_all(&bytes).await;
                            let _ = stdin.flush().await;
                        }
                    }
                    LocalExecutionControl::CloseStdin => {
                        stdin.take();
                    }
                    LocalExecutionControl::Resize { .. } => {}
                    LocalExecutionControl::Interrupt => {
                        update_process_status(&process, &state_tx, ExecutionProcessStatus::Interrupted).await;
                        let _ = child.start_kill();
                    }
                    LocalExecutionControl::Terminate => {
                        update_process_status(&process, &state_tx, ExecutionProcessStatus::Terminated).await;
                        let _ = child.start_kill();
                    }
                }
            }
        }
    };

    join_output_task(stdout_task).await;
    join_output_task(stderr_task).await;

    let final_snapshot = {
        let mut guard = process.lock().await;
        if !guard.status().is_terminal() {
            match wait_result {
                Ok(status) => guard.exit(status.code().unwrap_or(-1)),
                Err(error) => guard.fail(error.to_string()),
            }
        }
        guard.snapshot()
    };
    let _ = state_tx.send(final_snapshot.clone());
    let _ = final_tx.send(final_snapshot);
}

async fn join_output_task(task: Option<JoinHandle<()>>) {
    if let Some(task) = task {
        let _ = task.await;
    }
}

async fn update_process_status(
    process: &Arc<Mutex<ExecutionProcess>>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
    status: ExecutionProcessStatus,
) {
    let snapshot = {
        let mut guard = process.lock().await;
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

async fn read_process_stream<R>(
    mut reader: R,
    kind: ExecutionOutputKind,
    process: Arc<Mutex<ExecutionProcess>>,
    output_tx: mpsc::UnboundedSender<ExecutionOutputDelta>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut buffer = vec![0; PROCESS_OUTPUT_CHUNK_BYTES];
    loop {
        let bytes_read = match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(bytes_read) => bytes_read,
            Err(_) => break,
        };
        let (delta, snapshot) = {
            let mut guard = process.lock().await;
            let delta = guard.append_output(kind, &buffer[..bytes_read]);
            let snapshot = guard.snapshot();
            (delta, snapshot)
        };
        let _ = output_tx.send(delta);
        let _ = state_tx.send(snapshot);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundedProcessOutput {
    retain_bytes: usize,
    bytes: u64,
    retained_bytes: usize,
    omitted_bytes: u64,
    chunks: VecDeque<Vec<u8>>,
}

impl BoundedProcessOutput {
    fn new(retain_bytes: usize) -> Self {
        Self {
            retain_bytes,
            bytes: 0,
            retained_bytes: 0,
            omitted_bytes: 0,
            chunks: VecDeque::new(),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.bytes = self.bytes.saturating_add(bytes.len() as u64);
        if self.retain_bytes == 0 {
            self.omitted_bytes = self.omitted_bytes.saturating_add(bytes.len() as u64);
            return;
        }

        let chunk = if bytes.len() > self.retain_bytes {
            let omitted = bytes.len() - self.retain_bytes;
            self.omitted_bytes = self.omitted_bytes.saturating_add(omitted as u64);
            bytes[omitted..].to_vec()
        } else {
            bytes.to_vec()
        };
        self.retained_bytes += chunk.len();
        self.chunks.push_back(chunk);

        while self.retained_bytes > self.retain_bytes {
            let overflow = self.retained_bytes - self.retain_bytes;
            let Some(front) = self.chunks.front_mut() else {
                break;
            };
            if front.len() <= overflow {
                let removed = self.chunks.pop_front().unwrap_or_default();
                self.retained_bytes -= removed.len();
                self.omitted_bytes = self.omitted_bytes.saturating_add(removed.len() as u64);
            } else {
                front.drain(..overflow);
                self.retained_bytes -= overflow;
                self.omitted_bytes = self.omitted_bytes.saturating_add(overflow as u64);
            }
        }
    }

    fn snapshot(&self) -> BoundedProcessOutputSnapshot {
        let retained = self
            .chunks
            .iter()
            .flat_map(|chunk| chunk.iter().copied())
            .collect::<Vec<_>>();
        BoundedProcessOutputSnapshot {
            bytes: self.bytes,
            omitted_bytes: self.omitted_bytes,
            truncated: self.omitted_bytes > 0,
            text: String::from_utf8_lossy(&retained).to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundedProcessOutputSnapshot {
    bytes: u64,
    omitted_bytes: u64,
    truncated: bool,
    text: String,
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
