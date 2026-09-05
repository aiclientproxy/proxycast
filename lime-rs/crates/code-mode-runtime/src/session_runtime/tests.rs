use super::{CellId, Error, ObserveMode};
use std::time::Duration;

#[test]
fn cell_ids_are_displayable_and_observe_mode_is_stable() {
    let cell_id = CellId::new("cell-1");
    assert_eq!(cell_id.to_string(), "cell-1");
    assert_eq!(
        ObserveMode::YieldAfter(Duration::from_millis(1)),
        ObserveMode::YieldAfter(Duration::from_millis(1))
    );
    assert_eq!(
        Error::MissingCell(cell_id).to_string(),
        "exec cell cell-1 not found"
    );
}
