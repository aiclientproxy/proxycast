use super::*;
use app_server_protocol::{
    ExecutionProcessDrainOutputResponse, ExecutionProcessEmptyResponse,
    ExecutionProcessOutputDelta, ExecutionProcessOutputKind, ExecutionProcessSnapshot,
    ExecutionProcessStartResponse, ExecutionProcessStatusResponse,
};
use async_trait::async_trait;

#[derive(Clone, Default)]
struct FixtureGateway {
    state: Arc<Mutex<HashMap<String, FixtureProcess>>>,
    starts: Arc<Mutex<Vec<ExecutionProcessStartParams>>>,
}

#[derive(Clone)]
struct FixtureProcess {
    snapshot: ExecutionProcessSnapshot,
    deltas: Vec<ExecutionProcessOutputDelta>,
}

#[async_trait]
impl RuntimeLiveExecutionGateway for FixtureGateway {
    async fn start_process(
        &self,
        _thread_id: &str,
        _display_command: &str,
        params: ExecutionProcessStartParams,
    ) -> Result<ExecutionProcessStartResponse, String> {
        self.starts.lock().unwrap().push(params.clone());
        let command = params.command.last().cloned().unwrap_or_default();
        let running = command == "long-running";
        let initial = snapshot(&params, ExecutionProcessStatus::Running, None, "");
        let stored = snapshot(
            &params,
            if running {
                ExecutionProcessStatus::Running
            } else {
                ExecutionProcessStatus::Exited
            },
            (!running).then_some(0),
            if running { "started\n" } else { "completed\n" },
        );
        self.state.lock().unwrap().insert(
            params.process_id.clone(),
            FixtureProcess {
                snapshot: stored,
                deltas: vec![delta(
                    &params,
                    1,
                    if running { "started\n" } else { "completed\n" },
                )],
            },
        );
        Ok(ExecutionProcessStartResponse { snapshot: initial })
    }

    fn write_stdin(
        &self,
        params: ExecutionProcessWriteStdinParams,
    ) -> Result<ExecutionProcessEmptyResponse, String> {
        let mut state = self.state.lock().unwrap();
        let process = state
            .get_mut(&params.process_id)
            .ok_or_else(|| "missing fixture process".to_string())?;
        if params.data.is_empty() {
            return Ok(ExecutionProcessEmptyResponse {});
        }
        process.snapshot.status = ExecutionProcessStatus::Exited;
        process.snapshot.exit_code = Some(0);
        process.snapshot.retained_output.push_str("finished\n");
        let sequence = process
            .deltas
            .last()
            .map(|delta| delta.sequence.saturating_add(1))
            .unwrap_or(1);
        process.deltas.push(ExecutionProcessOutputDelta {
            process_id: params.process_id,
            tool_id: process.snapshot.tool_id.clone(),
            sequence,
            kind: ExecutionProcessOutputKind::Stdout,
            delta: "finished\n".to_string(),
            bytes: 17,
            omitted_bytes: 0,
            truncated: false,
        });
        Ok(ExecutionProcessEmptyResponse {})
    }

    fn terminate(
        &self,
        params: ExecutionProcessIdParams,
    ) -> Result<ExecutionProcessStatusResponse, String> {
        let mut state = self.state.lock().unwrap();
        let process = state
            .get_mut(&params.process_id)
            .ok_or_else(|| "missing fixture process".to_string())?;
        process.snapshot.status = ExecutionProcessStatus::Terminated;
        Ok(ExecutionProcessStatusResponse {
            snapshot: process.snapshot.clone(),
        })
    }

    fn status(
        &self,
        params: ExecutionProcessIdParams,
    ) -> Result<ExecutionProcessStatusResponse, String> {
        let state = self.state.lock().unwrap();
        let process = state
            .get(&params.process_id)
            .ok_or_else(|| "missing fixture process".to_string())?;
        Ok(ExecutionProcessStatusResponse {
            snapshot: process.snapshot.clone(),
        })
    }

    fn drain_output(
        &self,
        params: ExecutionProcessDrainOutputParams,
    ) -> Result<ExecutionProcessDrainOutputResponse, String> {
        let process_id = params
            .process_id
            .ok_or_else(|| "fixture process id is required".to_string())?;
        let state = self.state.lock().unwrap();
        let process = state
            .get(&process_id)
            .ok_or_else(|| "missing fixture process".to_string())?;
        let deltas = process
            .deltas
            .iter()
            .filter(|delta| {
                params
                    .after_sequence
                    .is_none_or(|after| delta.sequence > after)
            })
            .cloned()
            .collect::<Vec<_>>();
        let next_sequence = deltas
            .last()
            .map(|delta| delta.sequence)
            .or(params.after_sequence);
        Ok(ExecutionProcessDrainOutputResponse {
            deltas,
            next_sequence,
        })
    }
}

impl FixtureGateway {
    fn append_output_to_only_process(&self, output: &str) {
        let mut state = self.state.lock().unwrap();
        let process = state
            .values_mut()
            .next()
            .expect("fixture should contain one process");
        let sequence = process
            .deltas
            .last()
            .map(|delta| delta.sequence.saturating_add(1))
            .unwrap_or(1);
        process.snapshot.retained_output.push_str(output);
        process.snapshot.output_bytes = process.snapshot.retained_output.len() as u64;
        process.deltas.push(ExecutionProcessOutputDelta {
            process_id: process.snapshot.process_id.clone(),
            tool_id: process.snapshot.tool_id.clone(),
            sequence,
            kind: ExecutionProcessOutputKind::Stdout,
            delta: output.to_string(),
            bytes: process.snapshot.output_bytes,
            omitted_bytes: 0,
            truncated: false,
        });
    }
}

fn snapshot(
    params: &ExecutionProcessStartParams,
    status: ExecutionProcessStatus,
    exit_code: Option<i32>,
    output: &str,
) -> ExecutionProcessSnapshot {
    ExecutionProcessSnapshot {
        process_id: params.process_id.clone(),
        tool_id: params.tool_id.clone(),
        tool_name: params.tool_name.clone(),
        status,
        exit_code,
        elapsed_ms: 1,
        output_bytes: output.len() as u64,
        output_omitted_bytes: 0,
        output_truncated: false,
        retained_output: output.to_string(),
        failure: None,
    }
}

fn delta(
    params: &ExecutionProcessStartParams,
    sequence: u64,
    output: &str,
) -> ExecutionProcessOutputDelta {
    ExecutionProcessOutputDelta {
        process_id: params.process_id.clone(),
        tool_id: params.tool_id.clone(),
        sequence,
        kind: ExecutionProcessOutputKind::Stdout,
        delta: output.to_string(),
        bytes: output.len() as u64,
        omitted_bytes: 0,
        truncated: false,
    }
}

fn request<'a>(
    tool_name: &'a str,
    params: &'a Value,
    call_id: &str,
) -> RuntimeUnifiedExecToolRequest<'a> {
    RuntimeUnifiedExecToolRequest {
        tool_name,
        params,
        thread_id: "thread-1",
        working_directory: std::env::current_dir().unwrap(),
        environment: HashMap::new(),
        tool_call_id: call_id.to_string(),
        cancel_token: None,
        turn_context: None,
    }
}

#[test]
fn definitions_expose_only_codex_unified_exec_tools() {
    let definitions = unified_exec_tool_definitions();
    let names = definitions
        .iter()
        .map(|definition| definition.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, [EXEC_COMMAND_TOOL_NAME, WRITE_STDIN_TOOL_NAME]);
    assert_eq!(definitions[0].input_schema["required"], json!(["cmd"]));
    assert_eq!(
        definitions[1].input_schema["required"],
        json!(["session_id"])
    );
}

#[tokio::test]
async fn exec_command_returns_terminal_output_for_short_process() {
    let gateway = Arc::new(FixtureGateway::default());
    let params = json!({
        "cmd": "short",
        "login": false,
        "yield_time_ms": 250
    });

    let result = execute_runtime_unified_exec_tool(
        gateway,
        request(EXEC_COMMAND_TOOL_NAME, &params, "call-short"),
    )
    .await
    .expect("short command result");

    assert!(result.success);
    let structured = result.structured_content.expect("structured output");
    assert_eq!(structured["exit_code"], json!(0));
    assert_eq!(structured["output"], json!("completed\n"));
    assert_eq!(structured["observation"]["kind"], json!("terminal"));
    assert_eq!(structured["observation"]["process_active"], json!(false));
    assert!(structured.get("session_id").is_none());
    assert_eq!(
        result.metadata.get("exec_command_call_id"),
        Some(&json!("call-short"))
    );
}

#[tokio::test]
async fn exec_command_forwards_tty_to_execution_process_gateway() {
    let gateway = Arc::new(FixtureGateway::default());
    let params = json!({
        "cmd": "short",
        "login": false,
        "tty": true,
        "yield_time_ms": 250
    });

    execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(EXEC_COMMAND_TOOL_NAME, &params, "call-tty"),
    )
    .await
    .expect("TTY command result");

    let starts = gateway.starts.lock().unwrap();
    assert_eq!(starts.len(), 1);
    assert!(starts[0].tty);
}

#[tokio::test]
async fn write_stdin_resumes_and_completes_original_exec_command() {
    let gateway = Arc::new(FixtureGateway::default());
    let exec_params = json!({
        "cmd": "long-running",
        "login": false,
        "yield_time_ms": 250
    });
    let running = execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(EXEC_COMMAND_TOOL_NAME, &exec_params, "call-long"),
    )
    .await
    .expect("running command result");
    let session_id = running
        .structured_content
        .as_ref()
        .and_then(|value| value.get("session_id"))
        .and_then(Value::as_i64)
        .expect("session id") as i32;

    let write_params = json!({
        "session_id": session_id,
        "chars": "continue\n",
        "yield_time_ms": 250
    });
    let completed = execute_runtime_unified_exec_tool(
        gateway,
        request(WRITE_STDIN_TOOL_NAME, &write_params, "write-call"),
    )
    .await
    .expect("write stdin result");

    assert!(completed.success);
    let structured = completed.structured_content.expect("structured output");
    assert_eq!(structured["exit_code"], json!(0));
    assert_eq!(structured["output"], json!("finished\n"));
    assert_eq!(structured["observation"]["kind"], json!("terminal"));
    assert_eq!(
        completed.metadata.get("exec_command_call_id"),
        Some(&json!("call-long"))
    );
}

#[tokio::test]
async fn write_stdin_rejects_a_session_owned_by_another_thread() {
    let gateway = Arc::new(FixtureGateway::default());
    let exec_params = json!({
        "cmd": "long-running",
        "login": false,
        "yield_time_ms": 250
    });
    let running = execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(EXEC_COMMAND_TOOL_NAME, &exec_params, "call-thread-scoped"),
    )
    .await
    .expect("running command result");
    let session_id = running
        .structured_content
        .as_ref()
        .and_then(|value| value.get("session_id"))
        .and_then(Value::as_i64)
        .expect("session id") as i32;
    let write_params = json!({
        "session_id": session_id,
        "chars": "continue\n",
        "yield_time_ms": 250
    });
    let mut wrong_thread = request(WRITE_STDIN_TOOL_NAME, &write_params, "cross-thread-write");
    wrong_thread.thread_id = "thread-2";
    let error = execute_runtime_unified_exec_tool(gateway.clone(), wrong_thread)
        .await
        .expect_err("cross-thread write must fail closed");
    assert!(error
        .message()
        .contains("does not belong to thread thread-2"));

    execute_runtime_unified_exec_tool(
        gateway,
        request(WRITE_STDIN_TOOL_NAME, &write_params, "owning-thread-write"),
    )
    .await
    .expect("owning thread completes session");
}

#[tokio::test]
async fn empty_polls_are_structured_and_aggregate_for_active_process() {
    let gateway = Arc::new(FixtureGateway::default());
    let exec_params = json!({
        "cmd": "long-running",
        "login": false,
        "yield_time_ms": 250
    });
    let running = execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(EXEC_COMMAND_TOOL_NAME, &exec_params, "call-wait"),
    )
    .await
    .expect("running command result");
    let session_id = running.structured_content.as_ref().unwrap()["session_id"]
        .as_i64()
        .expect("session id") as i32;

    for expected_count in [1, 2] {
        let poll_params = json!({
            "session_id": session_id,
            "chars": "",
            "yield_time_ms": 250
        });
        let poll = execute_runtime_unified_exec_tool(
            gateway.clone(),
            request(WRITE_STDIN_TOOL_NAME, &poll_params, "poll-call"),
        )
        .await
        .expect("empty poll result");
        let structured = poll.structured_content.expect("structured poll");

        assert_eq!(structured["output"], json!(""));
        assert_eq!(structured["session_id"], json!(session_id));
        assert_eq!(structured["observation"]["kind"], json!("waiting"));
        assert_eq!(
            structured["observation"]["empty_poll_count"],
            json!(expected_count)
        );
        assert_eq!(
            poll.metadata.get("unified_exec_observation"),
            Some(&json!("waiting"))
        );
    }
}

#[tokio::test]
async fn output_breaks_wait_streak_before_terminal_output() {
    let gateway = Arc::new(FixtureGateway::default());
    let exec_params = json!({
        "cmd": "long-running",
        "login": false,
        "yield_time_ms": 250
    });
    let running = execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(EXEC_COMMAND_TOOL_NAME, &exec_params, "call-later-output"),
    )
    .await
    .expect("running command result");
    let session_id = running.structured_content.as_ref().unwrap()["session_id"]
        .as_i64()
        .expect("session id") as i32;
    let empty_poll = json!({
        "session_id": session_id,
        "chars": "",
        "yield_time_ms": 250
    });

    execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(WRITE_STDIN_TOOL_NAME, &empty_poll, "poll-before-output"),
    )
    .await
    .expect("empty poll result");
    gateway.append_output_to_only_process("later\n");

    let with_output = execute_runtime_unified_exec_tool(
        gateway.clone(),
        request(WRITE_STDIN_TOOL_NAME, &empty_poll, "poll-with-output"),
    )
    .await
    .expect("output poll result");
    let structured = with_output
        .structured_content
        .expect("structured output poll");
    assert_eq!(structured["output"], json!("later\n"));
    assert_eq!(structured["observation"]["kind"], json!("output"));
    assert_eq!(structured["observation"]["empty_poll_count"], json!(0));

    let finish_params = json!({
        "session_id": session_id,
        "chars": "continue\n",
        "yield_time_ms": 250
    });
    let terminal = execute_runtime_unified_exec_tool(
        gateway,
        request(WRITE_STDIN_TOOL_NAME, &finish_params, "finish-call"),
    )
    .await
    .expect("terminal result");
    let structured = terminal.structured_content.expect("structured terminal");
    assert_eq!(structured["output"], json!("finished\n"));
    assert_eq!(structured["observation"]["kind"], json!("terminal"));
    assert_eq!(structured["observation"]["process_active"], json!(false));
}

#[tokio::test]
async fn unknown_session_fails_without_waiting_observation() {
    let gateway = Arc::new(FixtureGateway::default());
    let params = json!({
        "session_id": i32::MAX,
        "chars": "",
        "yield_time_ms": 250
    });

    let error = execute_runtime_unified_exec_tool(
        gateway,
        request(WRITE_STDIN_TOOL_NAME, &params, "unknown-session"),
    )
    .await
    .expect_err("unknown session should fail");

    assert!(error.message().contains("unified exec session not found"));
}
