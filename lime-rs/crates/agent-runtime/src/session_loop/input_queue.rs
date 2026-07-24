use super::{
    RuntimeSessionInterAgentInput, RuntimeSessionLoopError, RuntimeSessionStepContext,
    RuntimeSessionTaskFailure, RuntimeSessionTokenUsage, RuntimeSessionTraceContext,
};
use crate::reply_input::RuntimeReplyInput;
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{oneshot, watch, Mutex};
use tokio_util::sync::CancellationToken;
#[derive(Clone, Debug)]
pub enum RuntimeSessionInput {
    User(RuntimeReplyInput),
    InterAgent(RuntimeSessionInterAgentInput),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSessionInputActivity {
    Mailbox,
    Steer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSessionTaskKind {
    Regular,
    Review,
    Compact,
    RunShell,
}

impl RuntimeSessionTaskKind {
    pub fn accepts_steer(self) -> bool {
        matches!(self, Self::Regular)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSessionTaskOutcome {
    Completed,
    Interrupted,
    Replaced,
    Shutdown,
}
/// 控制当前 task 是否还允许把 session mailbox 合并到本回合。
///
/// 可见最终回答已经发出后，迟到的 mailbox 必须留给下一回合；显式 steer
/// 或工具续行会重新打开当前回合。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSessionMailboxDeliveryPhase {
    CurrentTurn,
    NextTurn,
}

impl Default for RuntimeSessionMailboxDeliveryPhase {
    fn default() -> Self {
        Self::CurrentTurn
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeSessionResponseKind {
    Approval,
    AskUser,
    Permission,
    DynamicTool,
    McpElicitation,
}

pub struct RuntimeSessionPendingResponse {
    receiver: Option<oneshot::Receiver<Value>>,
    state: Arc<RuntimeSessionTaskState>,
    key: (RuntimeSessionResponseKind, String),
}

impl RuntimeSessionPendingResponse {
    pub async fn wait(mut self) -> Result<Value, RuntimeSessionTaskFailure> {
        let result = self
            .receiver
            .as_mut()
            .expect("runtime session response receiver is consumed once")
            .await
            .map_err(|_| RuntimeSessionTaskFailure {
                message: "runtime session response waiter was canceled".to_string(),
                ..Default::default()
            });
        self.receiver.take();
        self.state.responses.lock().await.remove(&self.key);
        result
    }
}

impl Drop for RuntimeSessionPendingResponse {
    fn drop(&mut self) {
        if self.receiver.is_none() {
            return;
        }
        if let Ok(mut responses) = self.state.responses.try_lock() {
            responses.remove(&self.key);
            return;
        }
        let state = Arc::clone(&self.state);
        let key = self.key.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                state.responses.lock().await.remove(&key);
            });
        }
    }
}

/// A task executed by the session loop.
pub trait RuntimeSessionTask: Send + Sync + 'static {
    fn turn_id(&self) -> &str;

    fn kind(&self) -> RuntimeSessionTaskKind {
        RuntimeSessionTaskKind::Regular
    }

    fn initial_input(&self) -> Vec<RuntimeSessionInput> {
        Vec::new()
    }

    fn mailbox_loader(&self) -> Option<RuntimeSessionMailboxLoader> {
        None
    }

    fn run(
        self: Arc<Self>,
        context: RuntimeSessionTaskContext,
        input: Vec<RuntimeSessionInput>,
        cancellation_token: CancellationToken,
    ) -> BoxFuture<'static, Result<(), RuntimeSessionTaskFailure>>;

    fn abort(&self, _context: RuntimeSessionTaskContext) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

#[derive(Clone)]
pub struct RuntimeSessionClosureTask {
    turn_id: String,
    kind: RuntimeSessionTaskKind,
    initial_input: Vec<RuntimeSessionInput>,
    mailbox_loader: Option<RuntimeSessionMailboxLoader>,
    run: Arc<
        dyn Fn(
                RuntimeSessionTaskContext,
                Vec<RuntimeSessionInput>,
                CancellationToken,
            ) -> BoxFuture<'static, Result<(), RuntimeSessionTaskFailure>>
            + Send
            + Sync,
    >,
    abort: Arc<dyn Fn(RuntimeSessionTaskContext) -> BoxFuture<'static, ()> + Send + Sync>,
}

impl RuntimeSessionClosureTask {
    pub fn new(
        turn_id: impl Into<String>,
        initial_input: Vec<RuntimeSessionInput>,
        run: impl Fn(
                RuntimeSessionTaskContext,
                Vec<RuntimeSessionInput>,
                CancellationToken,
            ) -> BoxFuture<'static, Result<(), RuntimeSessionTaskFailure>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            turn_id: turn_id.into(),
            kind: RuntimeSessionTaskKind::Regular,
            initial_input,
            mailbox_loader: None,
            run: Arc::new(run),
            abort: Arc::new(|_context| Box::pin(async {})),
        }
    }

    pub fn with_kind(mut self, kind: RuntimeSessionTaskKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_mailbox_loader(
        mut self,
        loader: impl Fn() -> BoxFuture<'static, Result<Vec<RuntimeSessionInput>, String>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.mailbox_loader = Some(Arc::new(loader));
        self
    }

    pub fn with_abort(
        mut self,
        abort: impl Fn(RuntimeSessionTaskContext) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    ) -> Self {
        self.abort = Arc::new(abort);
        self
    }
}

impl RuntimeSessionTask for RuntimeSessionClosureTask {
    fn turn_id(&self) -> &str {
        &self.turn_id
    }

    fn kind(&self) -> RuntimeSessionTaskKind {
        self.kind
    }

    fn initial_input(&self) -> Vec<RuntimeSessionInput> {
        self.initial_input.clone()
    }

    fn mailbox_loader(&self) -> Option<RuntimeSessionMailboxLoader> {
        self.mailbox_loader.clone()
    }

    fn run(
        self: Arc<Self>,
        context: RuntimeSessionTaskContext,
        input: Vec<RuntimeSessionInput>,
        cancellation_token: CancellationToken,
    ) -> BoxFuture<'static, Result<(), RuntimeSessionTaskFailure>> {
        (self.run)(context, input, cancellation_token)
    }

    fn abort(&self, context: RuntimeSessionTaskContext) -> BoxFuture<'static, ()> {
        (self.abort)(context)
    }
}

#[derive(Clone)]
pub struct RuntimeSessionTaskContext {
    session_id: Arc<str>,
    turn_id: Arc<str>,
    kind: RuntimeSessionTaskKind,
    metadata: RuntimeSessionTaskMetadata,
    pending_input: Arc<PendingInputQueue>,
    mailbox_loader: Option<RuntimeSessionMailboxLoader>,
    state: Arc<RuntimeSessionTaskState>,
}

#[derive(Clone)]
pub(super) struct RuntimeSessionTaskMetadata {
    submission_id: Arc<str>,
    client_user_message_id: Option<Arc<str>>,
    trace: Option<RuntimeSessionTraceContext>,
}

impl RuntimeSessionTaskMetadata {
    pub(super) fn new(
        submission_id: String,
        client_user_message_id: Option<String>,
        trace: Option<RuntimeSessionTraceContext>,
    ) -> Self {
        Self {
            submission_id: Arc::from(submission_id),
            client_user_message_id: client_user_message_id.map(Arc::from),
            trace,
        }
    }

    fn submission_id(&self) -> &str {
        &self.submission_id
    }

    fn client_user_message_id(&self) -> Option<&str> {
        self.client_user_message_id.as_deref()
    }

    fn trace(&self) -> Option<&RuntimeSessionTraceContext> {
        self.trace.as_ref()
    }
}

pub type RuntimeSessionMailboxLoader =
    Arc<dyn Fn() -> BoxFuture<'static, Result<Vec<RuntimeSessionInput>, String>> + Send + Sync>;

/// Session task 与 provider step 共享的输入视图。
///
/// 句柄只暴露当前 task 的 pending 输入，不允许调用方绕过 session actor
/// 直接替换活动 task 或清理队列。
#[derive(Clone)]
pub struct RuntimeSessionInputHandle {
    pub(super) session_id: Arc<str>,
    pub(super) pending_input: Arc<PendingInputQueue>,
    pub(super) turn_id: Arc<str>,
    pub(super) kind: RuntimeSessionTaskKind,
    pub(super) mailbox_loader: Option<RuntimeSessionMailboxLoader>,
    pub(super) state: Arc<RuntimeSessionTaskState>,
}

#[derive(Default)]
pub(super) struct RuntimeSessionTaskState {
    input: Mutex<RuntimeSessionTurnInputState>,
    step: Mutex<RuntimeSessionStepState>,
    responses: Mutex<HashMap<(RuntimeSessionResponseKind, String), oneshot::Sender<Value>>>,
}

#[derive(Default)]
struct RuntimeSessionTurnInputState {
    steer: VecDeque<RuntimeSessionInput>,
    mailbox_delivery_phase: RuntimeSessionMailboxDeliveryPhase,
    finishing: bool,
}

#[derive(Default)]
struct RuntimeSessionStepState {
    next_step_index: u64,
    context_epoch: u64,
    token_usage: RuntimeSessionTokenUsage,
    rollover_requested: bool,
}

impl RuntimeSessionTaskContext {
    pub(super) fn new(
        session_id: Arc<str>,
        turn_id: Arc<str>,
        kind: RuntimeSessionTaskKind,
        metadata: RuntimeSessionTaskMetadata,
        pending_input: Arc<PendingInputQueue>,
        mailbox_loader: Option<RuntimeSessionMailboxLoader>,
        state: Arc<RuntimeSessionTaskState>,
    ) -> Self {
        Self {
            session_id,
            turn_id,
            kind,
            metadata,
            pending_input,
            mailbox_loader,
            state,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }

    pub fn kind(&self) -> RuntimeSessionTaskKind {
        self.kind
    }

    pub fn submission_id(&self) -> &str {
        self.metadata.submission_id()
    }

    pub fn client_user_message_id(&self) -> Option<&str> {
        self.metadata.client_user_message_id()
    }

    pub fn trace(&self) -> Option<&RuntimeSessionTraceContext> {
        self.metadata.trace()
    }

    pub fn input_handle(&self) -> RuntimeSessionInputHandle {
        RuntimeSessionInputHandle {
            session_id: Arc::clone(&self.session_id),
            pending_input: Arc::clone(&self.pending_input),
            turn_id: Arc::clone(&self.turn_id),
            kind: self.kind,
            mailbox_loader: self.mailbox_loader.clone(),
            state: Arc::clone(&self.state),
        }
    }

    /// 取出当前 turn 的 pending 输入；steer 永远优先于 mailbox。
    pub async fn take_pending_input(&self, accept_mailbox: bool) -> Vec<RuntimeSessionInput> {
        self.input_handle().take_pending_input(accept_mailbox).await
    }

    pub async fn try_take_pending_input(
        &self,
        accept_mailbox: bool,
    ) -> Result<Vec<RuntimeSessionInput>, String> {
        self.input_handle()
            .try_take_pending_input(accept_mailbox)
            .await
    }

    pub async fn has_pending_input(&self, accept_mailbox: bool) -> bool {
        self.input_handle().has_pending_input(accept_mailbox).await
    }

    /// 等待 steer/mailbox 新输入，避免 task 在无输入时 busy-loop。
    pub async fn wait_for_pending_input(&self) {
        self.input_handle().wait_for_pending_input().await;
    }

    pub async fn subscribe_activity(
        &self,
    ) -> (
        watch::Receiver<RuntimeSessionInputActivity>,
        Option<RuntimeSessionInputActivity>,
    ) {
        self.input_handle().subscribe_activity().await
    }

    pub async fn capture_step_context(&self) -> RuntimeSessionStepContext {
        self.input_handle().capture_step_context().await
    }

    pub async fn mark_mailbox_delivery_for_next_turn(&self) {
        self.input_handle()
            .mark_mailbox_delivery_for_next_turn()
            .await;
    }

    pub async fn accept_mailbox_delivery_for_current_turn(&self) {
        self.input_handle()
            .accept_mailbox_delivery_for_current_turn()
            .await;
    }

    pub async fn advance_context_epoch(&self) -> u64 {
        self.input_handle().advance_context_epoch().await
    }

    pub async fn record_token_usage(
        &self,
        input: u64,
        output: u64,
        reasoning: u64,
        cache_write_input: u64,
    ) {
        self.input_handle()
            .record_token_usage(input, output, reasoning, cache_write_input)
            .await;
    }

    pub async fn token_usage(&self) -> RuntimeSessionTokenUsage {
        self.input_handle().token_usage().await
    }

    pub async fn request_context_rollover(&self) {
        self.input_handle().request_context_rollover().await;
    }

    pub async fn context_rollover_requested(&self) -> bool {
        self.input_handle().context_rollover_requested().await
    }

    pub async fn mark_finishing(&self) -> bool {
        self.input_handle().mark_finishing().await
    }

    pub async fn wait_for_response(
        &self,
        kind: RuntimeSessionResponseKind,
        request_id: impl Into<String>,
    ) -> Result<Value, RuntimeSessionTaskFailure> {
        self.input_handle()
            .wait_for_response(kind, request_id)
            .await
    }

    pub async fn register_response(
        &self,
        kind: RuntimeSessionResponseKind,
        request_id: impl Into<String>,
    ) -> Result<RuntimeSessionPendingResponse, RuntimeSessionTaskFailure> {
        self.input_handle()
            .register_response(kind, request_id)
            .await
    }
}

impl RuntimeSessionInputHandle {
    pub fn kind(&self) -> RuntimeSessionTaskKind {
        self.kind
    }

    pub async fn take_pending_input(&self, accept_mailbox: bool) -> Vec<RuntimeSessionInput> {
        self.try_take_pending_input(accept_mailbox)
            .await
            .unwrap_or_default()
    }

    pub async fn try_take_pending_input(
        &self,
        accept_mailbox: bool,
    ) -> Result<Vec<RuntimeSessionInput>, String> {
        let mut turn_input = self.state.input.lock().await;
        let accepts_mailbox_delivery =
            turn_input.mailbox_delivery_phase == RuntimeSessionMailboxDeliveryPhase::CurrentTurn;
        if !accept_mailbox || !accepts_mailbox_delivery {
            return Ok(turn_input.steer.drain(..).collect());
        }
        let mailbox_generation = self.pending_input.pending_mailbox_generation().await;
        // Keep the turn-input guard across the durable load so final-answer deferral and steer
        // reopening cannot cross the loader's delivery/ack side effect.
        let loaded_mailbox = match self.mailbox_loader.as_ref() {
            Some(loader) => loader().await?,
            None => Vec::new(),
        };
        let mut input = turn_input.steer.drain(..).collect::<Vec<_>>();
        drop(turn_input);
        input.extend(
            self.pending_input
                .take_mailbox_through(mailbox_generation)
                .await,
        );
        input.extend(loaded_mailbox);
        self.pending_input
            .acknowledge_mailbox_activity(mailbox_generation)
            .await;
        Ok(input)
    }

    pub async fn has_pending_input(&self, accept_mailbox: bool) -> bool {
        let (has_steer, accepts_mailbox_delivery) = {
            let state = self.state.input.lock().await;
            (
                !state.steer.is_empty(),
                state.mailbox_delivery_phase == RuntimeSessionMailboxDeliveryPhase::CurrentTurn,
            )
        };
        has_steer
            || (accept_mailbox
                && accepts_mailbox_delivery
                && self.pending_input.has_mailbox_activity().await)
    }

    pub async fn wait_for_pending_input(&self) {
        let (mut activity, _) = self.subscribe_activity().await;
        loop {
            if self.has_pending_input(true).await {
                return;
            }
            if activity.changed().await.is_err() {
                return;
            }
        }
    }

    pub async fn subscribe_activity(
        &self,
    ) -> (
        watch::Receiver<RuntimeSessionInputActivity>,
        Option<RuntimeSessionInputActivity>,
    ) {
        // Subscribe first so activity arriving while the pending snapshot is read
        // remains visible through the receiver.
        let activity = self.pending_input.subscribe_activity_receiver();
        let has_steer = !self.state.input.lock().await.steer.is_empty();
        let pending_activity = if has_steer {
            Some(RuntimeSessionInputActivity::Steer)
        } else if self.pending_input.has_mailbox_activity().await {
            Some(RuntimeSessionInputActivity::Mailbox)
        } else {
            None
        };
        (activity, pending_activity)
    }

    pub async fn mailbox_delivery_phase(&self) -> RuntimeSessionMailboxDeliveryPhase {
        self.state.input.lock().await.mailbox_delivery_phase
    }

    pub async fn mark_mailbox_delivery_for_next_turn(&self) {
        let mut state = self.state.input.lock().await;
        if state.steer.is_empty() {
            state.mailbox_delivery_phase = RuntimeSessionMailboxDeliveryPhase::NextTurn;
        }
    }

    pub async fn accept_mailbox_delivery_for_current_turn(&self) {
        self.state.input.lock().await.mailbox_delivery_phase =
            RuntimeSessionMailboxDeliveryPhase::CurrentTurn;
    }

    pub(super) async fn push_steer(&self, input: Vec<RuntimeSessionInput>) -> bool {
        let mut state = self.state.input.lock().await;
        if state.finishing {
            return false;
        }
        state.steer.extend(input);
        state.mailbox_delivery_phase = RuntimeSessionMailboxDeliveryPhase::CurrentTurn;
        drop(state);
        self.pending_input
            .publish_activity(RuntimeSessionInputActivity::Steer);
        true
    }

    pub(super) async fn clear_turn_state(&self) {
        let mut input = self.state.input.lock().await;
        input.steer.clear();
        input.finishing = true;
        drop(input);
        self.state.responses.lock().await.clear();
    }

    pub async fn mark_finishing(&self) -> bool {
        let mut input = self.state.input.lock().await;
        if !input.steer.is_empty() {
            input.mailbox_delivery_phase = RuntimeSessionMailboxDeliveryPhase::CurrentTurn;
            return false;
        }
        input.finishing = true;
        true
    }

    pub async fn capture_step_context(&self) -> RuntimeSessionStepContext {
        let phase = self.mailbox_delivery_phase().await;
        let mut step = self.state.step.lock().await;
        if step.rollover_requested {
            step.context_epoch = step.context_epoch.saturating_add(1);
            step.rollover_requested = false;
        }
        step.next_step_index = step.next_step_index.saturating_add(1);
        RuntimeSessionStepContext {
            session_id: self.session_id.to_string(),
            turn_id: self.turn_id.to_string(),
            step_index: step.next_step_index,
            context_epoch: step.context_epoch,
            mailbox_delivery_phase: phase,
        }
    }

    pub async fn advance_context_epoch(&self) -> u64 {
        let mut step = self.state.step.lock().await;
        step.context_epoch = step.context_epoch.saturating_add(1);
        step.rollover_requested = false;
        step.context_epoch
    }

    pub async fn record_token_usage(
        &self,
        input: u64,
        output: u64,
        reasoning: u64,
        cache_write_input: u64,
    ) {
        let mut step = self.state.step.lock().await;
        step.token_usage.input_tokens = step.token_usage.input_tokens.saturating_add(input);
        step.token_usage.output_tokens = step.token_usage.output_tokens.saturating_add(output);
        step.token_usage.reasoning_tokens =
            step.token_usage.reasoning_tokens.saturating_add(reasoning);
        step.token_usage.cache_write_input_tokens = step
            .token_usage
            .cache_write_input_tokens
            .saturating_add(cache_write_input);
    }

    pub async fn token_usage(&self) -> RuntimeSessionTokenUsage {
        self.state.step.lock().await.token_usage.clone()
    }

    pub async fn request_context_rollover(&self) {
        self.state.step.lock().await.rollover_requested = true;
    }

    pub async fn context_rollover_requested(&self) -> bool {
        self.state.step.lock().await.rollover_requested
    }

    pub async fn wait_for_response(
        &self,
        kind: RuntimeSessionResponseKind,
        request_id: impl Into<String>,
    ) -> Result<Value, RuntimeSessionTaskFailure> {
        self.register_response(kind, request_id).await?.wait().await
    }

    pub async fn register_response(
        &self,
        kind: RuntimeSessionResponseKind,
        request_id: impl Into<String>,
    ) -> Result<RuntimeSessionPendingResponse, RuntimeSessionTaskFailure> {
        let (sender, receiver) = oneshot::channel();
        let key = (kind, request_id.into());
        let mut responses = self.state.responses.lock().await;
        if responses.contains_key(&key) {
            return Err(RuntimeSessionTaskFailure {
                message: "runtime session response request is already pending".to_string(),
                ..Default::default()
            });
        }
        responses.insert(key.clone(), sender);
        drop(responses);
        Ok(RuntimeSessionPendingResponse {
            receiver: Some(receiver),
            state: Arc::clone(&self.state),
            key,
        })
    }

    pub(super) async fn resolve_response(
        &self,
        kind: RuntimeSessionResponseKind,
        request_id: &str,
        response: Value,
    ) -> Result<(), RuntimeSessionLoopError> {
        let sender = self
            .state
            .responses
            .lock()
            .await
            .remove(&(kind, request_id.to_string()))
            .ok_or_else(|| {
                RuntimeSessionLoopError::InvalidTask(
                    "runtime session response request is not pending".to_string(),
                )
            })?;
        sender.send(response).map_err(|_| {
            RuntimeSessionLoopError::InvalidTask(
                "runtime session response waiter is closed".to_string(),
            )
        })
    }
}

pub(super) struct PendingInputQueue {
    state: Mutex<PendingInputState>,
    activity: watch::Sender<RuntimeSessionInputActivity>,
}

#[derive(Default)]
struct PendingInputState {
    mailbox: VecDeque<(u64, RuntimeSessionInput)>,
    mailbox_generation: u64,
    acknowledged_mailbox_generation: u64,
}

impl Default for PendingInputQueue {
    fn default() -> Self {
        let (activity, _) = watch::channel(RuntimeSessionInputActivity::Mailbox);
        Self {
            state: Mutex::new(PendingInputState::default()),
            activity,
        }
    }
}

impl PendingInputQueue {
    pub(super) async fn notify_mailbox_activity(&self) {
        let mut state = self.state.lock().await;
        state.mailbox_generation = state.mailbox_generation.saturating_add(1);
        drop(state);
        self.publish_activity(RuntimeSessionInputActivity::Mailbox);
    }

    fn subscribe_activity_receiver(&self) -> watch::Receiver<RuntimeSessionInputActivity> {
        self.activity.subscribe()
    }

    pub(super) async fn subscribe_activity_snapshot(
        &self,
    ) -> (
        watch::Receiver<RuntimeSessionInputActivity>,
        Option<RuntimeSessionInputActivity>,
    ) {
        let activity = self.subscribe_activity_receiver();
        let pending_activity = self
            .has_mailbox_activity()
            .await
            .then_some(RuntimeSessionInputActivity::Mailbox);
        (activity, pending_activity)
    }

    fn publish_activity(&self, activity: RuntimeSessionInputActivity) {
        self.activity.send_replace(activity);
    }

    pub(super) fn publish_steer_activity(&self) {
        self.publish_activity(RuntimeSessionInputActivity::Steer);
    }

    async fn pending_mailbox_generation(&self) -> Option<u64> {
        let state = self.state.lock().await;
        (state.mailbox_generation > state.acknowledged_mailbox_generation)
            .then_some(state.mailbox_generation)
    }

    async fn take_mailbox_through(&self, generation: Option<u64>) -> Vec<RuntimeSessionInput> {
        let Some(generation) = generation else {
            return Vec::new();
        };
        let mut state = self.state.lock().await;
        let count = state
            .mailbox
            .iter()
            .take_while(|(message_generation, _)| *message_generation <= generation)
            .count();
        state
            .mailbox
            .drain(..count)
            .map(|(_, input)| input)
            .collect()
    }

    async fn acknowledge_mailbox_activity(&self, generation: Option<u64>) {
        let Some(generation) = generation else {
            return;
        };
        let mut state = self.state.lock().await;
        state.acknowledged_mailbox_generation = state
            .acknowledged_mailbox_generation
            .max(generation.min(state.mailbox_generation));
    }

    async fn has_mailbox_activity(&self) -> bool {
        self.pending_mailbox_generation().await.is_some()
    }

    pub(super) async fn clear(&self) {
        let mut state = self.state.lock().await;
        state.mailbox.clear();
        state.acknowledged_mailbox_generation = state.mailbox_generation;
    }
}

pub(super) struct QueuedTask {
    pub(super) task: Arc<dyn RuntimeSessionTask>,
    pub(super) input: Vec<RuntimeSessionInput>,
    pub(super) completion:
        oneshot::Sender<Result<RuntimeSessionTaskOutcome, RuntimeSessionTaskFailure>>,
    pub(super) metadata: RuntimeSessionTaskMetadata,
}
