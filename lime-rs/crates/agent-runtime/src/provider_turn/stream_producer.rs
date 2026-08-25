use futures::{Stream, StreamExt};
use model_provider::current_client::{
    CanonicalLlmEvent, CurrentProvider, CurrentProviderError, CurrentProviderRequest,
    CurrentProviderStream,
};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

const PROVIDER_EVENT_CHANNEL_CAPACITY: usize = 1600;

pub(super) enum ProviderStreamStart {
    Started(CurrentProviderStream),
    Cancelled,
    FirstVisibleOutputDeadlineElapsed,
    ProviderStepDeadlineElapsed,
}

pub(super) async fn start_provider_stream(
    provider: &Arc<dyn CurrentProvider>,
    request: CurrentProviderRequest,
    cancel_token: Option<&CancellationToken>,
    first_visible_output_deadline: Instant,
    provider_step_deadline: Instant,
) -> Result<ProviderStreamStart, CurrentProviderError> {
    let (started_sender, started_receiver) = oneshot::channel();
    let (event_sender, event_receiver) = mpsc::channel(PROVIDER_EVENT_CHANNEL_CAPACITY);
    let provider = Arc::clone(provider);
    let producer = tokio::spawn(async move {
        let mut stream = match provider.stream(request).await {
            Ok(stream) => {
                if started_sender.send(Ok(())).is_err() {
                    return;
                }
                stream
            }
            Err(error) => {
                let _ = started_sender.send(Err(error));
                return;
            }
        };
        while let Some(event) = stream.next().await {
            match event_sender.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(event)) => {
                    if event_sender.send(event).await.is_err() {
                        return;
                    }
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return,
            }
        }
    });
    let producer = AbortOnDropHandle::new(producer);
    let started = match cancel_token {
        Some(cancel_token) => {
            tokio::select! {
                biased;
                result = started_receiver => result,
                _ = cancel_token.cancelled() => return Ok(ProviderStreamStart::Cancelled),
                _ = tokio::time::sleep_until(first_visible_output_deadline) => {
                    return Ok(ProviderStreamStart::FirstVisibleOutputDeadlineElapsed);
                }
                _ = tokio::time::sleep_until(provider_step_deadline) => {
                    return Ok(ProviderStreamStart::ProviderStepDeadlineElapsed);
                }
            }
        }
        None => {
            tokio::select! {
                biased;
                result = started_receiver => result,
                _ = tokio::time::sleep_until(first_visible_output_deadline) => {
                    return Ok(ProviderStreamStart::FirstVisibleOutputDeadlineElapsed);
                }
                _ = tokio::time::sleep_until(provider_step_deadline) => {
                    return Ok(ProviderStreamStart::ProviderStepDeadlineElapsed);
                }
            }
        }
    };
    match started {
        Ok(Ok(())) => Ok(ProviderStreamStart::Started(Box::pin(
            ProviderEventReceiverStream {
                receiver: event_receiver,
                _producer: producer,
            },
        ))),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(CurrentProviderError::new(
            "Provider stream producer stopped before initialization",
        )),
    }
}

struct ProviderEventReceiverStream {
    receiver: mpsc::Receiver<Result<CanonicalLlmEvent, CurrentProviderError>>,
    _producer: AbortOnDropHandle<()>,
}

impl Stream for ProviderEventReceiverStream {
    type Item = Result<CanonicalLlmEvent, CurrentProviderError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(context)
    }
}
