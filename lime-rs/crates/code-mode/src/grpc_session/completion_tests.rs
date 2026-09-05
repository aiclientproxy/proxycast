use super::completion::decode_outcome_event;
use code_mode_protocol::grpc as proto;

#[test]
fn completion_requires_an_outcome_event() {
    let error = decode_outcome_event(proto::ExecuteEvent { event: None })
        .expect_err("missing outcome must fail closed");
    assert!(error.contains("invalid outcome"));
}

#[test]
fn completion_decodes_outcome_event() {
    let event = proto::ExecuteEvent {
        event: Some(proto::execute_event::Event::Outcome(
            proto::ExecutionOutcome {
                cell_id: "cell".to_string(),
                content_items: Vec::new(),
                code_mode_host_duration_ns: 0,
                outcome: Some(proto::execution_outcome::Outcome::Yielded(
                    proto::ExecutionYielded {},
                )),
            },
        )),
    };
    assert_eq!(
        decode_outcome_event(event).expect("outcome").cell_id,
        "cell"
    );
}
