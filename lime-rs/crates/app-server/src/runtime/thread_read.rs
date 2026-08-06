use super::session_lifecycle::stored_session_hidden_from_user_recents;
use super::status::{agent_turn_blocks_queue_resume, agent_turn_is_active};
use super::{RuntimeCore, RuntimeCoreError};
use agent_protocol::PageCursor;
use app_server_protocol::protocol::v2::{
    ThreadLoadedListParams, ThreadLoadedListResponse, ThreadMetadataGitInfoUpdateParams,
    ThreadMetadataUpdateParams, ThreadSearchOccurrence, ThreadSearchOccurrencesParams,
    ThreadSearchOccurrencesResponse, ThreadSearchTextRange, ThreadSection,
    ThreadSectionCreateParams, ThreadSectionCreateResponse, ThreadSectionDeleteParams,
    ThreadSectionDeleteResponse, ThreadSectionListParams, ThreadSectionListResponse,
    ThreadSectionMoveParams, ThreadSectionMoveResponse, ThreadSectionUpdateParams,
    ThreadSectionUpdateResponse, ThreadSetNameParams, ThreadSetNameResponse,
};
use app_server_protocol::{
    ThreadItemsListParams, ThreadItemsListResponse, ThreadListParams, ThreadListResponse,
    ThreadReadParams, ThreadReadResponse, ThreadTurnsListParams, ThreadTurnsListResponse,
};
use thread_store::{
    ArchiveThreadParams, CreateThreadSectionParams as StoreCreateThreadSectionParams,
    DeleteThreadSectionParams as StoreDeleteThreadSectionParams, ListItemsParams,
    ListThreadSectionsParams as StoreListThreadSectionsParams, ListThreadsParams, ListTurnsParams,
    MoveThreadToSectionParams as StoreMoveThreadToSectionParams, PageRequest, ReadThreadParams,
    RenameThreadSectionParams as StoreRenameThreadSectionParams,
    SearchThreadOccurrencesParams as StoreSearchThreadOccurrencesParams, StoreCursor,
    StoredThreadSection, ThreadMetadataPatch, ThreadStore, ThreadStoreErrorKind,
    UpdateThreadMetadataParams,
};

const DEFAULT_PAGE_LIMIT: u32 = 100;
const THREAD_SEARCH_OCCURRENCES_DEFAULT_LIMIT: usize = 50;
const THREAD_SEARCH_OCCURRENCES_MAX_LIMIT: usize = 250;

pub(crate) struct RuntimeThreadResumeSnapshot {
    pub thread: agent_protocol::Thread,
    pub active_turn_id: Option<agent_protocol::TurnId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThreadUnloadResult {
    Active,
    NotLoaded,
    Unloaded,
}

impl RuntimeCore {
    pub(crate) fn loaded_session_id_for_thread(&self, thread_id: &str) -> Option<String> {
        self.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .values()
            .find(|stored| stored.session.thread_id == thread_id)
            .map(|stored| stored.session.session_id.clone())
    }

    pub(crate) fn loaded_thread_is_active(&self, thread_id: &str) -> Option<bool> {
        self.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .values()
            .find(|stored| stored.session.thread_id == thread_id)
            .map(|stored| {
                stored
                    .turns
                    .iter()
                    .any(|turn| agent_turn_is_active(turn.status))
            })
    }

    pub(crate) async fn unload_thread_if_idle(
        &self,
        thread_id: &str,
    ) -> Result<ThreadUnloadResult, RuntimeCoreError> {
        let (session_id, stored) = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let Some(session_id) = state
                .sessions
                .values()
                .find(|stored| stored.session.thread_id == thread_id)
                .map(|stored| stored.session.session_id.clone())
            else {
                return Ok(ThreadUnloadResult::NotLoaded);
            };
            let is_active = state.sessions.get(&session_id).is_some_and(|stored| {
                stored
                    .turns
                    .iter()
                    .any(|turn| agent_turn_is_active(turn.status))
            });
            if is_active {
                return Ok(ThreadUnloadResult::Active);
            }
            let stored = state
                .sessions
                .remove(&session_id)
                .expect("loaded thread session disappeared while locked");
            (session_id, stored)
        };

        if let Err(error) = self.session_loops.shutdown(&session_id).await {
            self.state
                .lock()
                .expect("runtime core state mutex poisoned")
                .sessions
                .insert(session_id, stored);
            return Err(RuntimeCoreError::Backend(error.to_string()));
        }
        if let Err(error) = self.backend.close_session(&session_id, thread_id).await {
            self.state
                .lock()
                .expect("runtime core state mutex poisoned")
                .sessions
                .insert(session_id, stored);
            return Err(error);
        }

        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        super::approval_cache::remove_session(&mut state.session_approval_cache, &session_id);
        state.thread_elicitation_counts.remove(thread_id);
        state.thread_goal_continuations.remove(&session_id);
        Ok(ThreadUnloadResult::Unloaded)
    }

    pub(crate) async fn can_accept_direct_input(
        &self,
        thread_id: &str,
    ) -> Result<bool, RuntimeCoreError> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "threadId is required to check direct-input policy".to_string(),
            ));
        }
        let thread = self
            .canonical_thread_store()?
            .read_thread(ReadThreadParams {
                thread_id: agent_protocol::ThreadId::new(thread_id),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| RuntimeCoreError::Backend(format!("thread not found: {thread_id}")))?;
        Ok(thread.parent_thread_id.is_none())
    }

    pub fn list_loaded_threads(
        &self,
        params: ThreadLoadedListParams,
    ) -> Result<ThreadLoadedListResponse, RuntimeCoreError> {
        let mut data = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .values()
            .filter(|stored| !stored_session_hidden_from_user_recents(stored))
            .map(|stored| stored.session.thread_id.clone())
            .collect::<Vec<_>>();
        if data.is_empty() {
            return Ok(ThreadLoadedListResponse {
                data,
                next_cursor: None,
            });
        }

        data.sort_unstable();
        data.dedup();
        let total = data.len();
        let start = match params.cursor {
            Some(cursor) => {
                let cursor = uuid::Uuid::parse_str(&cursor)
                    .map_err(|_| {
                        RuntimeCoreError::InvalidRequest(format!("invalid cursor: {cursor}"))
                    })?
                    .to_string();
                match data.binary_search(&cursor) {
                    Ok(index) => index + 1,
                    Err(index) => index,
                }
            }
            None => 0,
        };
        let limit = params.limit.unwrap_or(total as u32).max(1) as usize;
        let end = start.saturating_add(limit).min(total);
        let page = data[start..end].to_vec();
        let next_cursor = page.last().filter(|_| end < total).cloned();
        Ok(ThreadLoadedListResponse {
            data: page,
            next_cursor,
        })
    }

    pub async fn set_thread_name(
        &self,
        params: ThreadSetNameParams,
    ) -> Result<ThreadSetNameResponse, RuntimeCoreError> {
        let thread_id = params.thread_id.trim();
        if thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "threadId is required for thread/name/set".to_string(),
            ));
        }
        let name = params.name.trim();
        if name.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread name must not be empty".to_string(),
            ));
        }
        let store = self.canonical_thread_store()?;
        store
            .update_thread_metadata(UpdateThreadMetadataParams {
                thread_id: agent_protocol::ThreadId::new(thread_id.to_string()),
                patch: ThreadMetadataPatch {
                    name: Some(Some(name.to_string())),
                    ..Default::default()
                },
                include_archived: false,
            })
            .await
            .map_err(store_error)?;
        Ok(ThreadSetNameResponse {})
    }

    pub async fn update_thread_metadata(
        &self,
        params: ThreadMetadataUpdateParams,
    ) -> Result<agent_protocol::Thread, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let thread_id = agent_protocol::ThreadId::new(params.thread_id);
        let current = store
            .read_thread(ReadThreadParams {
                thread_id: thread_id.clone(),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| RuntimeCoreError::Backend(format!("thread not found: {thread_id}")))?;
        let mut metadata = current.metadata;
        let metadata = metadata_object(&mut metadata);
        if let Some(git_info) = params.git_info {
            apply_git_info_patch(metadata, git_info);
        }
        store
            .update_thread_metadata(UpdateThreadMetadataParams {
                thread_id,
                patch: ThreadMetadataPatch {
                    metadata: Some(Some(serde_json::Value::Object(metadata.clone()))),
                    ..Default::default()
                },
                include_archived: true,
            })
            .await
            .map_err(store_error)
    }

    pub async fn archive_thread(
        &self,
        thread_id: agent_protocol::ThreadId,
    ) -> Result<bool, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let current = store
            .read_thread(ReadThreadParams {
                thread_id: thread_id.clone(),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| RuntimeCoreError::Backend(format!("thread not found: {thread_id}")))?;
        if current.archived {
            return Ok(false);
        }
        let session_id = current.session_id.to_string();
        let loaded = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .contains_key(&session_id);
        if loaded {
            self.session_loops
                .shutdown(&session_id)
                .await
                .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
            self.backend
                .close_session(&session_id, thread_id.as_str())
                .await?;
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            super::approval_cache::remove_session(&mut state.session_approval_cache, &session_id);
            state.thread_elicitation_counts.remove(thread_id.as_str());
            state.thread_goal_continuations.remove(&session_id);
            state.sessions.remove(&session_id);
        }
        store
            .archive_thread(ArchiveThreadParams { thread_id })
            .await
            .map_err(store_error)?;
        Ok(true)
    }

    pub async fn unarchive_thread(
        &self,
        thread_id: agent_protocol::ThreadId,
    ) -> Result<(agent_protocol::Thread, bool), RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let current = store
            .read_thread(ReadThreadParams {
                thread_id: thread_id.clone(),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| RuntimeCoreError::Backend(format!("thread not found: {thread_id}")))?;
        let thread = store
            .unarchive_thread(ArchiveThreadParams { thread_id })
            .await
            .map_err(store_error)?;
        Ok((thread, current.archived))
    }

    pub(crate) async fn resume_thread(
        &self,
        thread_id: agent_protocol::ThreadId,
    ) -> Result<RuntimeThreadResumeSnapshot, RuntimeCoreError> {
        let response = self
            .read_thread(ThreadReadParams {
                thread_id,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        let session_id = response.thread.session_id.clone();
        if response.thread.forked_from_id.is_some()
            && response
                .thread
                .metadata
                .get("forkSequence")
                .and_then(serde_json::Value::as_u64)
                .is_some()
        {
            let canonical = self
                .read_thread(ThreadReadParams {
                    thread_id: response.thread.thread_id.clone(),
                    turns_view: agent_protocol::ThreadTurnsView::Full,
                })
                .await?;
            self.hydrate_fork_session_from_canonical(&canonical.thread)?;
        } else {
            self.ensure_current_session_hydrated(session_id.as_str())
                .await?;
        }
        let active_turn_id = self
            .session_loops
            .snapshot(session_id.as_str())
            .await
            .map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "read runtime session snapshot for thread resume: {error}"
                ))
            })?
            .and_then(|snapshot| snapshot.active_turn_id)
            .map(agent_protocol::TurnId::new);
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let thread_is_idle = active_turn_id.is_none()
            && state
                .sessions
                .get(session_id.as_str())
                .is_some_and(|stored| {
                    !stored
                        .turns
                        .iter()
                        .any(|turn| agent_turn_blocks_queue_resume(turn.status))
                });
        self.projection_store
            .as_deref()
            .ok_or_else(|| {
                RuntimeCoreError::Backend("thread goal store is unavailable".to_string())
            })?
            .restore_thread_goal_accounting_sync(response.thread.thread_id.as_str(), thread_is_idle)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        drop(state);
        Ok(RuntimeThreadResumeSnapshot {
            thread: response.thread,
            active_turn_id,
        })
    }

    pub async fn read_thread(
        &self,
        params: ThreadReadParams,
    ) -> Result<ThreadReadResponse, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let thread_id = params.thread_id.clone();
        let turns_view = params.turns_view;
        let mut thread = store
            .read_thread(ReadThreadParams {
                thread_id: params.thread_id,
                include_archived: true,
                turns_view,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| RuntimeCoreError::Backend(format!("thread not found: {thread_id}")))?;
        if let Some(product) = self
            .projection_store
            .as_deref()
            .map(|store| store.read_thread_product_projection(thread.session_id.as_str()))
            .transpose()
            .map_err(RuntimeCoreError::Backend)?
            .flatten()
        {
            merge_thread_product_projection(&mut thread.metadata, product);
        }
        Ok(ThreadReadResponse { thread })
    }

    pub async fn list_threads(
        &self,
        params: ThreadListParams,
    ) -> Result<ThreadListResponse, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let turns_view = params.turns_view;
        let page = store
            .list_threads(ListThreadsParams {
                include_archived: params.include_archived,
                page: store_page(params.page)?,
                section: params.section,
                sort_by_section_position: params.sort_by_section_position,
            })
            .await
            .map_err(store_error)?;
        let mut data = page.data;
        if !matches!(turns_view, agent_protocol::ThreadTurnsView::NotLoaded) {
            for thread in &mut data {
                let Some(hydrated) = store
                    .read_thread(ReadThreadParams {
                        thread_id: thread.thread_id.clone(),
                        include_archived: true,
                        turns_view,
                    })
                    .await
                    .map_err(store_error)?
                else {
                    return Err(RuntimeCoreError::Backend(format!(
                        "listed thread disappeared during hydration: {}",
                        thread.thread_id
                    )));
                };
                *thread = hydrated;
            }
        }
        Ok(ThreadListResponse {
            data,
            next_cursor: page.next_cursor.map(StoreCursor::into_string),
            backwards_cursor: page.backwards_cursor.map(StoreCursor::into_string),
        })
    }

    pub async fn list_thread_sections(
        &self,
        params: ThreadSectionListParams,
    ) -> Result<ThreadSectionListResponse, RuntimeCoreError> {
        let cursor = params
            .cursor
            .map(StoreCursor::new)
            .transpose()
            .map_err(|error| RuntimeCoreError::InvalidRequest(error.to_string()))?;
        let page = self
            .canonical_thread_store()?
            .list_thread_sections(StoreListThreadSectionsParams {
                cursor,
                limit: params.limit.unwrap_or(100).clamp(1, 500),
            })
            .await
            .map_err(store_error)?;
        Ok(ThreadSectionListResponse {
            data: page.data.into_iter().map(project_section).collect(),
            next_cursor: page.next_cursor.map(StoreCursor::into_string),
        })
    }

    pub async fn create_thread_section(
        &self,
        params: ThreadSectionCreateParams,
    ) -> Result<ThreadSectionCreateResponse, RuntimeCoreError> {
        let section = self
            .canonical_thread_store()?
            .create_thread_section(StoreCreateThreadSectionParams { name: params.name })
            .await
            .map_err(store_error)?;
        Ok(ThreadSectionCreateResponse {
            section: project_section(section),
        })
    }

    pub async fn update_thread_section(
        &self,
        params: ThreadSectionUpdateParams,
    ) -> Result<ThreadSectionUpdateResponse, RuntimeCoreError> {
        let section_id = params.section_id.clone();
        let section = self
            .canonical_thread_store()?
            .rename_thread_section(StoreRenameThreadSectionParams {
                section_id,
                name: params.name,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::InvalidRequest(format!(
                    "thread section not found: {}",
                    params.section_id
                ))
            })?;
        Ok(ThreadSectionUpdateResponse {
            section: project_section(section),
        })
    }

    pub async fn delete_thread_section(
        &self,
        params: ThreadSectionDeleteParams,
    ) -> Result<ThreadSectionDeleteResponse, RuntimeCoreError> {
        let deleted = self
            .canonical_thread_store()?
            .delete_thread_section(StoreDeleteThreadSectionParams {
                section_id: params.section_id.clone(),
            })
            .await
            .map_err(store_error)?;
        if !deleted {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "thread section not found: {}",
                params.section_id
            )));
        }
        Ok(ThreadSectionDeleteResponse {})
    }

    pub async fn move_thread_to_section(
        &self,
        params: ThreadSectionMoveParams,
    ) -> Result<ThreadSectionMoveResponse, RuntimeCoreError> {
        self.canonical_thread_store()?
            .move_thread_to_section(StoreMoveThreadToSectionParams {
                thread_id: agent_protocol::ThreadId::new(params.thread_id),
                section: params.section_id,
                before_thread_id: params.before_thread_id.map(agent_protocol::ThreadId::new),
            })
            .await
            .map_err(store_error)?;
        Ok(ThreadSectionMoveResponse {})
    }

    pub async fn list_thread_turns(
        &self,
        params: ThreadTurnsListParams,
    ) -> Result<ThreadTurnsListResponse, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let page = store
            .list_turns(ListTurnsParams {
                thread_id: params.thread_id,
                include_archived: true,
                page: store_page(params.page)?,
                items_view: params.items_view,
            })
            .await
            .map_err(store_error)?;
        Ok(ThreadTurnsListResponse {
            data: page.data,
            next_cursor: page.next_cursor.map(StoreCursor::into_string),
            backwards_cursor: page.backwards_cursor.map(StoreCursor::into_string),
        })
    }

    pub async fn list_thread_items(
        &self,
        params: ThreadItemsListParams,
    ) -> Result<ThreadItemsListResponse, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let page = store
            .list_items(ListItemsParams {
                thread_id: params.thread_id,
                turn_id: params.turn_id,
                include_archived: true,
                page: store_page(params.page)?,
            })
            .await
            .map_err(store_error)?;
        Ok(ThreadItemsListResponse {
            data: page.data,
            next_cursor: page.next_cursor.map(StoreCursor::into_string),
            backwards_cursor: page.backwards_cursor.map(StoreCursor::into_string),
        })
    }

    pub async fn search_thread_occurrences(
        &self,
        params: ThreadSearchOccurrencesParams,
    ) -> Result<ThreadSearchOccurrencesResponse, RuntimeCoreError> {
        if params.search_term.trim().is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread/searchOccurrences requires a non-empty searchTerm".to_string(),
            ));
        }
        let cursor = params
            .cursor
            .map(StoreCursor::new)
            .transpose()
            .map_err(|error| RuntimeCoreError::InvalidRequest(error.to_string()))?;
        let page_size = params
            .limit
            .map(|value| value as usize)
            .unwrap_or(THREAD_SEARCH_OCCURRENCES_DEFAULT_LIMIT)
            .clamp(1, THREAD_SEARCH_OCCURRENCES_MAX_LIMIT);
        let page = self
            .canonical_thread_store()?
            .search_thread_occurrences(StoreSearchThreadOccurrencesParams {
                thread_id: agent_protocol::ThreadId::new(params.thread_id),
                search_term: params.search_term,
                cursor,
                page_size,
            })
            .await
            .map_err(search_store_error)?;
        Ok(ThreadSearchOccurrencesResponse {
            data: page
                .data
                .into_iter()
                .map(|occurrence| ThreadSearchOccurrence {
                    turn_id: occurrence.turn_id.as_str().to_string(),
                    item_id: occurrence.item_id.as_str().to_string(),
                    snippet: occurrence.snippet,
                    snippet_match_range: ThreadSearchTextRange {
                        start: occurrence.snippet_match_range.start,
                        end: occurrence.snippet_match_range.end,
                    },
                    turn_cursor: occurrence.turn_cursor.into_string(),
                })
                .collect(),
            next_cursor: page.next_cursor.map(StoreCursor::into_string),
        })
    }

    fn canonical_thread_store(&self) -> Result<&dyn ThreadStore, RuntimeCoreError> {
        self.projection_store
            .as_deref()
            .map(|store| store as &dyn ThreadStore)
            .ok_or_else(|| {
                RuntimeCoreError::Backend("canonical thread store is unavailable".to_string())
            })
    }
}

fn merge_thread_product_projection(metadata: &mut serde_json::Value, product: serde_json::Value) {
    let Some(product) = product.as_object() else {
        return;
    };
    if !metadata.is_object() {
        *metadata = serde_json::Value::Object(serde_json::Map::new());
    }
    let Some(metadata) = metadata.as_object_mut() else {
        return;
    };
    metadata.extend(product.clone());
}

fn metadata_object(
    metadata: &mut serde_json::Value,
) -> &mut serde_json::Map<String, serde_json::Value> {
    if !metadata.is_object() {
        *metadata = serde_json::Value::Object(serde_json::Map::new());
    }
    metadata
        .as_object_mut()
        .expect("metadata was normalized to an object")
}

fn apply_git_info_patch(
    metadata: &mut serde_json::Map<String, serde_json::Value>,
    patch: ThreadMetadataGitInfoUpdateParams,
) {
    let legacy = metadata.remove("git_info");
    let current = metadata.remove("gitInfo").or(legacy);
    let mut git_info = current
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    apply_clearable_string(&mut git_info, "sha", patch.sha);
    apply_clearable_string(&mut git_info, "branch", patch.branch);
    apply_clearable_string(&mut git_info, "originUrl", patch.origin_url);
    if !git_info.is_empty() {
        metadata.insert("gitInfo".to_string(), serde_json::Value::Object(git_info));
    }
}

fn apply_clearable_string(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<Option<String>>,
) {
    match value {
        Some(Some(value)) => {
            object.insert(key.to_string(), serde_json::Value::String(value));
        }
        Some(None) => {
            object.remove(key);
        }
        None => {}
    }
}

fn store_page(page: PageCursor) -> Result<PageRequest, RuntimeCoreError> {
    let cursor = page
        .cursor
        .map(StoreCursor::new)
        .transpose()
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    Ok(PageRequest {
        cursor,
        limit: page.limit.unwrap_or(DEFAULT_PAGE_LIMIT),
        sort_direction: page.sort_direction,
    })
}

fn store_error(error: thread_store::ThreadStoreError) -> RuntimeCoreError {
    match error.kind() {
        ThreadStoreErrorKind::InvalidRequest | ThreadStoreErrorKind::ThreadNotFound => {
            RuntimeCoreError::InvalidRequest(error.to_string())
        }
        ThreadStoreErrorKind::Unsupported => RuntimeCoreError::MethodNotFound(error.to_string()),
        ThreadStoreErrorKind::Internal => RuntimeCoreError::Backend(error.to_string()),
    }
}

fn project_section(section: StoredThreadSection) -> ThreadSection {
    ThreadSection {
        id: section.id,
        name: section.name,
    }
}

fn search_store_error(error: thread_store::ThreadStoreError) -> RuntimeCoreError {
    match error.kind() {
        ThreadStoreErrorKind::InvalidRequest | ThreadStoreErrorKind::ThreadNotFound => {
            RuntimeCoreError::InvalidRequest(error.to_string())
        }
        ThreadStoreErrorKind::Unsupported => RuntimeCoreError::MethodNotFound(error.to_string()),
        ThreadStoreErrorKind::Internal => {
            RuntimeCoreError::Backend(format!("failed to search thread occurrences: {error}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectionStore;
    use agent_protocol::{
        ItemId, ItemKind, ItemStatus, SessionId, Thread, ThreadHistoryChangeSet, ThreadId,
        ThreadItem, ThreadItemPayload, ThreadStatus, ThreadTurnsView, Turn, TurnAdmissionState,
        TurnApprovalState, TurnId, TurnItemsView, TurnQueueState, TurnStatus,
    };
    use app_server_protocol::{AgentEvent, AgentSessionStartParams, AgentTurn, AgentTurnStatus};
    use serde_json::json;
    use std::sync::Arc;
    use thread_store::{
        ApplyThreadHistoryParams, ArchiveThreadParams, CreateThreadParams, ThreadStore,
    };

    #[tokio::test]
    async fn idle_unload_refuses_active_turn_then_removes_loaded_session() {
        let runtime = RuntimeCore::default();
        runtime
            .start_session(AgentSessionStartParams {
                session_id: Some("session-unload".to_string()),
                thread_id: Some("thread-unload".to_string()),
                app_id: "test".to_string(),
                workspace_id: None,
                business_object_ref: None,
                locale: None,
            })
            .expect("start unload test session");
        {
            let mut state = runtime
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .get_mut("session-unload")
                .expect("unload test session")
                .turns
                .push(AgentTurn {
                    turn_id: "turn-unload".to_string(),
                    session_id: "session-unload".to_string(),
                    thread_id: "thread-unload".to_string(),
                    status: AgentTurnStatus::Running,
                    started_at: None,
                    completed_at: None,
                });
        }

        assert_eq!(
            runtime
                .unload_thread_if_idle("thread-unload")
                .await
                .expect("active unload check"),
            ThreadUnloadResult::Active
        );
        assert_eq!(
            runtime.loaded_session_id_for_thread("thread-unload"),
            Some("session-unload".to_string())
        );

        runtime
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get_mut("session-unload")
            .expect("unload test session")
            .turns[0]
            .status = AgentTurnStatus::Completed;
        assert_eq!(
            runtime
                .unload_thread_if_idle("thread-unload")
                .await
                .expect("idle unload"),
            ThreadUnloadResult::Unloaded
        );
        assert_eq!(runtime.loaded_session_id_for_thread("thread-unload"), None);
    }

    #[tokio::test]
    async fn canonical_thread_reads_use_the_projection_store_without_session_fallback() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(
            ProjectionStore::initialize(temp.path().join("projection.sqlite")).expect("store"),
        );
        let thread = make_thread("thread-current", 10);
        store
            .create_thread(CreateThreadParams {
                thread: thread.clone(),
            })
            .await
            .expect("create current thread");
        let archived = make_thread("thread-archived", 20);
        store
            .create_thread(CreateThreadParams {
                thread: archived.clone(),
            })
            .await
            .expect("create archived thread");
        store
            .archive_thread(ArchiveThreadParams {
                thread_id: archived.thread_id.clone(),
            })
            .await
            .expect("archive thread");

        let turn = Turn {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            turn_id: TurnId::new("turn-current"),
            status: TurnStatus::Completed,
            admission: TurnAdmissionState::Accepted,
            queue: TurnQueueState::Running,
            approval: TurnApprovalState::NotRequired,
            items: Vec::new(),
            items_view: TurnItemsView::NotLoaded,
            error: None,
            created_at_ms: 10,
            updated_at_ms: 12,
            started_at_ms: Some(10),
            completed_at_ms: Some(12),
            duration_ms: Some(2),
        };
        let item = ThreadItem {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            item_id: ItemId::new("item-current"),
            sequence: 1,
            ordinal: 1,
            created_at_ms: 11,
            updated_at_ms: 12,
            completed_at_ms: Some(12),
            kind: ItemKind::AgentMessage,
            status: ItemStatus::Completed,
            payload: ThreadItemPayload::AgentMessage {
                text: "canonical".to_string(),
                phase: None,
                content_parts: Vec::new(),
            },
            metadata: json!({}),
        };
        store
            .apply_history(ApplyThreadHistoryParams {
                session_id: thread.session_id.clone(),
                thread_id: thread.thread_id.clone(),
                changes: ThreadHistoryChangeSet {
                    sequence: 1,
                    changed_turns: vec![turn],
                    changed_items: vec![item],
                    ..Default::default()
                },
            })
            .await
            .expect("apply history");
        store
            .apply_event(&AgentEvent {
                event_id: "artifact-current".to_string(),
                sequence: 1,
                session_id: thread.session_id.to_string(),
                thread_id: Some(thread.thread_id.to_string()),
                turn_id: None,
                event_type: "artifact.snapshot".to_string(),
                timestamp: "2026-07-21T00:00:00Z".to_string(),
                payload: json!({
                    "session": {
                        "createdAt": "2026-07-21T00:00:00Z",
                        "updatedAt": "2026-07-21T00:00:00Z",
                        "workspaceId": "workspace-current"
                    },
                    "artifact": {
                        "artifactId": "workspace-patch-current",
                        "kind": "content_factory.workspace_patch",
                        "metadata": {
                            "contentFactoryWorkspacePatch": {
                                "schemaVersion": "article-workspace.v1",
                                "appId": "content-factory-app",
                                "sessionId": thread.session_id,
                                "workspaceId": "workspace-current",
                                "objects": [{
                                    "ref": {
                                        "appId": "content-factory-app",
                                        "kind": "articleDraft",
                                        "id": "article-current",
                                        "sessionId": thread.session_id,
                                        "artifactIds": ["artifact-current"]
                                    },
                                    "title": "Current article",
                                    "status": "ready",
                                    "previewArtifactId": "artifact-current",
                                    "source": {"markdown": "# Current article"}
                                }]
                            }
                        }
                    }
                }),
            })
            .expect("apply product projection");

        let runtime = RuntimeCore::default().with_projection_store(store);
        let visible = runtime
            .list_threads(ThreadListParams {
                page: page(),
                include_archived: false,
                turns_view: ThreadTurnsView::NotLoaded,
                section: None,
                sort_by_section_position: false,
            })
            .await
            .expect("list visible");
        assert_eq!(visible.data.len(), 1);
        assert_eq!(visible.data[0].thread_id.as_str(), "thread-current");

        let all = runtime
            .list_threads(ThreadListParams {
                page: page(),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
                section: None,
                sort_by_section_position: false,
            })
            .await
            .expect("list all");
        assert_eq!(all.data.len(), 2);
        assert_eq!(
            all.data
                .iter()
                .find(|item| item.thread_id.as_str() == "thread-current")
                .expect("current thread")
                .turns
                .len(),
            1
        );

        let read = runtime
            .read_thread(ThreadReadParams {
                thread_id: thread.thread_id.clone(),
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .expect("read thread");
        assert_eq!(
            read.thread.turns[0].items[0].item_id.as_str(),
            "item_item-current"
        );
        assert_eq!(
            read.thread.metadata["articleWorkspace"]["objects"][0]["ref"]["id"],
            "article-current"
        );
        assert!(read.thread.metadata["artifacts"]
            .as_array()
            .expect("thread artifacts")
            .iter()
            .any(|artifact| artifact["artifactRef"] == "artifact-current"));

        let turns = runtime
            .list_thread_turns(ThreadTurnsListParams {
                thread_id: thread.thread_id.clone(),
                page: page(),
                items_view: TurnItemsView::Full,
            })
            .await
            .expect("list turns");
        assert_eq!(turns.data[0].items[0].item_id.as_str(), "item_item-current");

        let items = runtime
            .list_thread_items(ThreadItemsListParams {
                thread_id: thread.thread_id,
                turn_id: None,
                page: page(),
            })
            .await
            .expect("list items");
        assert_eq!(items.data[0].item_id.as_str(), "item_item-current");
    }

    fn make_thread(thread_id: &str, timestamp: i64) -> Thread {
        Thread {
            session_id: SessionId::new(format!("session-{thread_id}")),
            thread_id: ThreadId::new(thread_id),
            status: ThreadStatus::Idle,
            created_at_ms: timestamp,
            updated_at_ms: timestamp,
            archived: false,
            recency_at_ms: Some(timestamp),
            parent_thread_id: None,
            agent_path: None,
            agent_nickname: None,
            agent_role: None,
            last_task_message: None,
            agent_state: None,
            forked_from_id: None,
            preview: String::new(),
            model_provider: "test".to_string(),
            product: None,
            name: None,
            metadata: json!({}),
            turns: Vec::new(),
            turns_view: ThreadTurnsView::NotLoaded,
        }
    }

    fn page() -> PageCursor {
        PageCursor {
            cursor: None,
            limit: Some(20),
            sort_direction: agent_protocol::SortDirection::Asc,
        }
    }
}
