# CLI 与 Core 原始 Snapshot Inventory

源文件数：`52`

路径相对于 `/Users/coso/Documents/dev/rust/codex`。迁移 scenario/disposition 见
[../03-source-to-scenario-map.md](../03-source-to-scenario-map.md)。

- `codex-rs/cli/src/doctor/snapshots/codex__doctor__output__tests__doctor_human_report_environment_rows.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__agents_md__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__apps_instructions__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__collaboration_mode__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__environment__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__environments_instructions__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__multi_agent_mode__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__permissions__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__plugins_instructions__tests__snapshots.snap`
- `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__realtime__tests__snapshots.snap`
- `codex-rs/core/src/guardian/snapshots/codex_core__guardian__tests__guardian_followup_review_request_layout.snap`
- `codex-rs/core/src/guardian/snapshots/codex_core__guardian__tests__guardian_review_request_layout.snap`
- `codex-rs/core/src/guardian/snapshots/codex_core__guardian__tests__network_access_guardian_prompt_layout.snap`
- `codex-rs/core/src/session/snapshots/codex_core__codex_tests__fork_startup_context_then_first_turn_diff.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__additional_context__additional_context_simple_input.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__manual_compact_with_history_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__manual_compact_without_prev_user_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__mid_turn_compaction_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_sampling_model_switch_compaction_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_turn_compaction_context_window_exceeded_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_turn_compaction_including_incoming_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_turn_compaction_strips_incoming_model_switch_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_api_auth_prompt_cache_key_request_diff.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_chatgpt_auth_service_tier_prompt_cache_key_request_diff.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_restates_realtime_start_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_with_history_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_without_prev_user_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_does_not_restate_realtime_end_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_multi_summary_reinjects_above_last_summary_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_summary_only_reinjects_context_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_context_window_exceeded_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_failure_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_including_incoming_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_restates_realtime_start_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_strips_incoming_model_switch_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_resume_fork__rollback_followup_turn_trims_context_updates.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__compact_resume_fork__rollback_past_compaction_shapes.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__mcp_tool_exposure__deferred_tools_initial_unchanged_and_removed.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__mcp_tool_exposure__deferred_tools_recover_during_sampling.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__mcp_tool_exposure__deferred_tools_resume_without_duplicate_update.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_cwd_change_refreshes_agents.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_environment_context_includes_one_subagent.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_environment_context_includes_two_subagents.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_resume_override_matches_rollout_model.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_resume_with_personality_change.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_turn_overrides.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__pending_input__pending_input_queued_mail_after_commentary.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__pending_input__pending_input_queued_mail_after_reasoning.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__pending_input__pending_input_user_input_no_preempt_after_reasoning.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__realtime_conversation__conversation_startup_context_current_thread_selects_many_turns_by_budget.snap`
- `codex-rs/core/tests/suite/snapshots/all__suite__token_budget__token_budget_new_context_window_tool_full_context.snap`
