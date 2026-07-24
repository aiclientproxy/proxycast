use super::ConnectionRequestId;
use app_server_protocol::METHOD_THREAD_RESUME;

pub(super) fn request_id_for_thread_resume(
    method: &str,
    connection_request_id: Option<&ConnectionRequestId>,
) -> Option<ConnectionRequestId> {
    (method == METHOD_THREAD_RESUME)
        .then(|| connection_request_id.cloned())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::request_id_for_thread_resume;
    use crate::processor::ConnectionRequestId;
    use app_server_protocol::{RequestId, METHOD_THREAD_READ, METHOD_THREAD_RESUME};
    use app_server_transport::ConnectionId;

    #[test]
    fn direct_request_has_no_thread_resume_connection_context() {
        assert_eq!(
            request_id_for_thread_resume(METHOD_THREAD_RESUME, None),
            None
        );
    }

    #[test]
    fn transport_thread_resume_keeps_exact_connection_and_request_ids() {
        let context = ConnectionRequestId {
            connection_id: ConnectionId(42),
            request_id: RequestId::String("resume-7".to_string()),
        };

        assert_eq!(
            request_id_for_thread_resume(METHOD_THREAD_RESUME, Some(&context)),
            Some(context)
        );
        assert_eq!(
            request_id_for_thread_resume(
                METHOD_THREAD_READ,
                Some(&ConnectionRequestId {
                    connection_id: ConnectionId(42),
                    request_id: RequestId::String("read-8".to_string()),
                }),
            ),
            None
        );
    }
}
