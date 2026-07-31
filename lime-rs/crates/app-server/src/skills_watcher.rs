use crate::{AppServerEventBridge, JsonRpcMessage};
use app_server_protocol::protocol::v2::{ServerNotification, SkillsChangedNotification};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WATCHER_THROTTLE_INTERVAL: Duration = Duration::from_secs(10);

pub(crate) struct SkillsWatcher {
    _watcher: Arc<Mutex<RecommendedWatcher>>,
    _watched_roots: Arc<Mutex<HashSet<PathBuf>>>,
    reconcile_task: tokio::task::JoinHandle<()>,
}

impl Drop for SkillsWatcher {
    fn drop(&mut self) {
        self.reconcile_task.abort();
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
        let last_notification = Arc::new(Mutex::new(None::<Instant>));
        let callback_last_notification = last_notification.clone();
        let callback_bridge = bridge.clone();
        let callback_handle = handle.clone();
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
            notify_catalog_changed(
                &callback_bridge,
                &callback_handle,
                &callback_last_notification,
                throttle_interval,
            );
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
        let reconcile_last_notification = last_notification.clone();
        let reconcile_handle = handle.clone();
        let reconcile_task = handle.spawn(async move {
            let mut interval = tokio::time::interval(throttle_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            interval.tick().await;
            loop {
                interval.tick().await;
                if reconcile_watches(&reconcile_watcher, &reconcile_watched_roots, &roots) {
                    notify_catalog_changed(
                        &bridge,
                        &reconcile_handle,
                        &reconcile_last_notification,
                        throttle_interval,
                    );
                }
            }
        });

        Some(Arc::new(Self {
            _watcher: watcher,
            _watched_roots: watched_roots,
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

fn notify_catalog_changed(
    bridge: &AppServerEventBridge,
    handle: &tokio::runtime::Handle,
    last_notification: &Mutex<Option<Instant>>,
    throttle_interval: Duration,
) {
    let now = Instant::now();
    let mut last = last_notification
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if last.is_some_and(|last| now.duration_since(last) < throttle_interval) {
        return;
    }
    *last = Some(now);
    drop(last);

    lime_skills::invalidate_agent_skill_snapshot_cache();
    let bridge = bridge.clone();
    handle.spawn(async move {
        bridge
            .broadcast_message(JsonRpcMessage::Notification(
                ServerNotification::SkillsChanged(SkillsChangedNotification {}).into(),
            ))
            .await;
    });
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
}
