use super::input_queue::RuntimeSessionMailboxDeliveryPhase;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeSessionStepContext {
    pub session_id: String,
    pub turn_id: String,
    pub step_index: u64,
    pub context_epoch: u64,
    pub mailbox_delivery_phase: RuntimeSessionMailboxDeliveryPhase,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeSessionTokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cache_write_input_tokens: u64,
}
