use super::generation::{next_execution_id, next_wait_id, remote_cell_id};
use code_mode_protocol::RuntimeCodeModeCellId;

#[test]
fn generated_request_ids_are_unique_and_uuid_shaped() {
    let execution = next_execution_id();
    let wait = next_wait_id();
    assert_ne!(execution, wait);
    assert!(uuid::Uuid::parse_str(&execution).is_ok());
    assert!(uuid::Uuid::parse_str(&wait).is_ok());
}

#[test]
fn generation_parser_rejects_missing_or_stale_prefix() {
    assert!(remote_cell_id(2, &RuntimeCodeModeCellId::new("cell")).is_err());
    assert!(remote_cell_id(2, &RuntimeCodeModeCellId::new("g1:cell")).is_err());
    assert_eq!(
        remote_cell_id(2, &RuntimeCodeModeCellId::new("g2:cell"))
            .expect("matching generation")
            .as_str(),
        "cell"
    );
}
