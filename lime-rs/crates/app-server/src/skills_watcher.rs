use crate::{AppServerEventBridge, JsonRpcMessage};
use app_server_protocol::protocol::v2::{ServerNotification, SkillsChangedNotification};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task::AbortHandle;

const WATCHER_THROTTLE_INTERVAL: Duration = Duration::from_secs(10);

pub(crate) struct SkillsWatcher {
    _watcher: Arc<Mutex<RecommendedWatcher>>,
    _watched_roots: Arc<Mutex<HashSet<PathBuf>>>,
    notifier: CatalogChangeNotifier,
    reconcile_task: tokio::task::JoinHandle<()>,
}

impl Drop for SkillsWatcher {
    fn drop(&mut self) {
        self.reconcile_task.abort();
        self.notifier.cancel_pending();
    }
}

#[derive(Default)]
struct NotificationState {
    last_notification: Option<Instant>,
    trailing_task: Option<AbortHandle>,
}

#[derive(Clone)]
struct CatalogChangeNotifier {
    bridge: AppServerEventBridge,
    handle: tokio::runtime::Handle,
    throttle_interval: Duration,
    state: Arc<Mutex<NotificationState>>,
}

impl CatalogChangeNotifier {
    fn new(
        bridge: AppServerEventBridge,
        handle: tokio::runtime::Handle,
        throttle_interval: Duration,
    ) -> Self {
        Self {
            bridge,
            handle,
            throttle_interval,
            state: Arc::new(Mutex::new(NotificationState::default())),
        }
    }

    fn notify(&self) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());

        if state.trailing_task.is_some() {
            return;
        }

        let Some(last_notification) = state.last_notification else {
            state.last_notification = Some(now);
            drop(state);
            self.broadcast();
            return;
        };
        let elapsed = now.duration_since(last_notification);
        if elapsed >= self.throttle_interval {
            state.last_notification = Some(now);
            drop(state);
            self.broadcast();
            return;
        }

        let delay = self.throttle_interval - elapsed;
        let notifier = self.clone();
        let task = self.handle.spawn(async move {
            tokio::time::sleep(delay).await;
            notifier.flush_trailing();
        });
        state.trailing_task = Some(task.abort_handle());
    }

    fn flush_trailing(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.trailing_task.take().is_none() {
            return;
        }
        state.last_notification = Some(Instant::now());
        drop(state);
        self.broadcast();
    }

    fn cancel_pending(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(task) = state.trailing_task.take() {
            task.abort();
        }
    }

    fn broadcast(&self) {
        lime_skills::invalidate_agent_skill_snapshot_cache();
        let bridge = self.bridge.clone();
        self.handle.spawn(async move {
            bridge
                .broadcast_message(JsonRpcMessage::Notification(
                    ServerNotification::SkillsChanged(SkillsChangedNotification {}).into(),
                ))
                .await;
        });
    }
}

impl SkillsWatcher {
    pub(crate) fn start_default(bridge: AppServerEventBridge) -> Option<Arc<Self>> {
        Self::start(
            bridge,
            lime_skills::default_agent_skill_roots()
                .into_iter()
                .map(|root| root.path),
            WATCHER_THROTTLE_INTERVAL,
        )
    }

    fn start(
        bridge: AppServerEventBridge,
        roots: impl IntoIterator<Item = PathBuf>,
        throttle_interval: Duration,
    ) -> Option<Arc<Self>> {
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle,
            Err(error) => {
                tracing::warn!(%error, "skills watcher skipped without Tokio runtime");
                return None;
            }
        };
        let roots = roots.into_iter().collect::<HashSet<_>>();
        if roots.is_empty() {
            tracing::warn!("skills watcher has no default Skill roots");
            return None;
        }
        let notifier = CatalogChangeNotifier::new(bridge, handle, throttle_interval);
        let callback_notifier = notifier.clone();
        let watcher = match notify::recommended_watcher(move |result: notify::Result<Event>| {
            let event = match result {
                Ok(event) if is_catalog_change(&event.kind) => event,
                Ok(_) => return,
                Err(error) => {
                    tracing::warn!(%error, "skills watcher received an error");
                    return;
                }
            };
            if event.paths.is_empty() {
                return;
            }
            callback_notifier.notify();
        }) {
            Ok(watcher) => watcher,
            Err(error) => {
                tracing::warn!(%error, "failed to initialize skills watcher");
                return None;
            }
        };
        let watcher = Arc::new(Mutex::new(watcher));
        let watched_roots = Arc::new(Mutex::new(HashSet::new()));
        reconcile_watches(&watcher, &watched_roots, &roots);

        let reconcile_watcher = watcher.clone();
        let reconcile_watched_roots = watched_roots.clone();
        let reconcile_notifier = notifier.clone();
        let reconcile_task = notifier.handle.spawn(async move {
            let mut interval = tokio::time::interval(throttle_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            interval.tick().await;
            loop {
                interval.tick().await;
                if reconcile_watches(&reconcile_watcher, &reconcile_watched_roots, &roots) {
                    reconcile_notifier.notify();
                }
            }
        });

        Some(Arc::new(Self {
            _watcher: watcher,
            _watched_roots: watched_roots,
            notifier,
            reconcile_task,
        }))
    }
}

fn reconcile_watches(
    watcher: &Mutex<RecommendedWatcher>,
    watched_roots: &Mutex<HashSet<PathBuf>>,
    roots: &HashSet<PathBuf>,
) -> bool {
    let existing_roots = roots
        .iter()
        .filter(|root| root.is_dir())
        .map(|root| std::fs::canonicalize(root).unwrap_or_else(|_| root.clone()))
        .collect::<HashSet<_>>();
    let mut watcher = watcher.lock().unwrap_or_else(|error| error.into_inner());
    let mut watched_roots = watched_roots
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut changed = false;

    for removed in watched_roots
        .difference(&existing_roots)
        .cloned()
        .collect::<Vec<_>>()
    {
        if let Err(error) = watcher.unwatch(&removed) {
            tracing::debug!(path = %removed.display(), %error, "Skill root was already unwatched");
        }
        watched_roots.remove(&removed);
        changed = true;
    }
    for added in existing_roots
        .difference(&watched_roots)
        .cloned()
        .collect::<Vec<_>>()
    {
        match watcher.watch(&added, RecursiveMode::Recursive) {
            Ok(()) => {
                watched_roots.insert(added);
                changed = true;
            }
            Err(error) => {
                tracing::warn!(path = %added.display(), %error, "failed to watch Skill root");
            }
        }
    }
    changed
}

fn is_catalog_change(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppServer;
    use std::fs;

    #[test]
    fn only_catalog_mutations_trigger_invalidation() {
        assert!(is_catalog_change(&EventKind::Create(
            notify::event::CreateKind::File
        )));
        assert!(is_catalog_change(&EventKind::Modify(
            notify::event::ModifyKind::Data(notify::event::DataChange::Content)
        )));
        assert!(is_catalog_change(&EventKind::Remove(
            notify::event::RemoveKind::Folder
        )));
        assert!(!is_catalog_change(&EventKind::Access(
            notify::event::AccessKind::Read
        )));
    }

    #[tokio::test]
    async fn watched_skill_file_change_broadcasts_typed_invalidation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        let skill_dir = root.join("writer");
        fs::create_dir_all(&skill_dir).expect("create Skill directory");
        let skill_file = skill_dir.join("SKILL.md");
        fs::write(&skill_file, "---\nname: writer\ndescription: before\n---\n")
            .expect("write Skill fixture");

        let server = AppServer::new();
        let mut outbound = server.subscribe_outbound_messages();
        let _watcher =
            SkillsWatcher::start(server.event_bridge(), [root], Duration::from_millis(20))
                .expect("start Skill watcher");
        fs::write(&skill_file, "---\nname: writer\ndescription: after\n---\n")
            .expect("update Skill fixture");

        let notification = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let message = outbound.recv().await.expect("watcher notification");
                if matches!(
                    &message,
                    JsonRpcMessage::Notification(notification)
                        if notification.method == "skills/changed"
                ) {
                    break message;
                }
            }
        })
        .await
        .expect("wait for skills/changed");
        assert_eq!(
            notification,
            JsonRpcMessage::Notification(
                ServerNotification::SkillsChanged(SkillsChangedNotification {}).into()
            )
        );
    }

    #[tokio::test]
    async fn throttled_change_is_coalesced_into_trailing_notification() {
        let server = AppServer::new();
        let mut outbound = server.subscribe_outbound_messages();
        let notifier = CatalogChangeNotifier::new(
            server.event_bridge(),
            tokio::runtime::Handle::current(),
            Duration::from_millis(500),
        );

        notifier.notify();
        receive_skills_changed(&mut outbound, Duration::from_secs(1)).await;

        notifier.notify();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), outbound.recv())
                .await
                .is_err(),
            "throttled change must not broadcast immediately"
        );
        receive_skills_changed(&mut outbound, Duration::from_secs(1)).await;
        notifier.cancel_pending();
    }

    #[tokio::test]
    async fn skill_root_created_after_start_broadcasts_typed_invalidation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        let server = AppServer::new();
        let mut outbound = server.subscribe_outbound_messages();
        let _watcher = SkillsWatcher::start(
            server.event_bridge(),
            [root.clone()],
            Duration::from_millis(20),
        )
        .expect("start Skill watcher before root exists");

        let skill_dir = root.join("writer");
        fs::create_dir_all(&skill_dir).expect("create Skill root and directory");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: writer\ndescription: created later\n---\n",
        )
        .expect("write Skill fixture");

        let notification = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let message = outbound.recv().await.expect("watcher notification");
                if matches!(
                    &message,
                    JsonRpcMessage::Notification(notification)
                        if notification.method == "skills/changed"
                ) {
                    break message;
                }
            }
        })
        .await
        .expect("wait for root creation skills/changed");
        assert_eq!(
            notification,
            JsonRpcMessage::Notification(
                ServerNotification::SkillsChanged(SkillsChangedNotification {}).into()
            )
        );
    }

    async fn receive_skills_changed(
        outbound: &mut tokio::sync::broadcast::Receiver<JsonRpcMessage>,
        timeout: Duration,
    ) -> JsonRpcMessage {
        tokio::time::timeout(timeout, async {
            loop {
                let message = outbound.recv().await.expect("watcher notification");
                if matches!(
                    &message,
                    JsonRpcMessage::Notification(notification)
                        if notification.method == "skills/changed"
                ) {
                    break message;
                }
            }
        })
        .await
        .expect("wait for skills/changed")
    }
}
