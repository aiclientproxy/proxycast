# Codex Source -> Lime Scenario Map

扫描基线：`/Users/coso/Documents/dev/rust/codex` @ `4c43465133428898aa84f0bfc02c306ed65fb66a`。

每行覆盖一个 `.snap` 源文件。`scenario` 的详细断言在 `01-frontend-test-plan.md` 或
`02-runtime-contract-test-plan.md`；`disposition` 表示迁移处置，不等同于已实现或已验证。

| Source snapshot | Scenario | Disposition |
| --- | --- | --- |
| `codex-rs/cli/src/doctor/snapshots/codex__doctor__output__tests__doctor_human_report_environment_rows.snap` | `cli-doctor` | defer |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__agents_md__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__apps_instructions__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__collaboration_mode__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__environment__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__environments_instructions__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__multi_agent_mode__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__permissions__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__plugins_instructions__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/context/world_state/snapshots/codex_core__context__world_state__realtime__tests__snapshots.snap` | `runtime-environment-context` | contract |
| `codex-rs/core/src/guardian/snapshots/codex_core__guardian__tests__guardian_followup_review_request_layout.snap` | `approval-four-states` | contract + Gate B |
| `codex-rs/core/src/guardian/snapshots/codex_core__guardian__tests__guardian_review_request_layout.snap` | `approval-four-states` | contract + Gate B |
| `codex-rs/core/src/guardian/snapshots/codex_core__guardian__tests__network_access_guardian_prompt_layout.snap` | `approval-four-states` | contract + Gate B |
| `codex-rs/core/src/session/snapshots/codex_core__codex_tests__fork_startup_context_then_first_turn_diff.snap` | `history-fork-lineage` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__additional_context__additional_context_simple_input.snap` | `runtime-context-budget` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__manual_compact_with_history_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__manual_compact_without_prev_user_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__mid_turn_compaction_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_sampling_model_switch_compaction_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_turn_compaction_context_window_exceeded_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_turn_compaction_including_incoming_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact__pre_turn_compaction_strips_incoming_model_switch_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_api_auth_prompt_cache_key_request_diff.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_chatgpt_auth_service_tier_prompt_cache_key_request_diff.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_restates_realtime_start_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_with_history_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_manual_compact_without_prev_user_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_does_not_restate_realtime_end_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_multi_summary_reinjects_above_last_summary_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_mid_turn_compaction_summary_only_reinjects_context_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_context_window_exceeded_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_failure_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_including_incoming_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_restates_realtime_start_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_remote__remote_pre_turn_compaction_strips_incoming_model_switch_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_resume_fork__rollback_followup_turn_trims_context_updates.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__compact_resume_fork__rollback_past_compaction_shapes.snap` | `runtime-compaction` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__mcp_tool_exposure__deferred_tools_initial_unchanged_and_removed.snap` | `runtime-mcp-exposure` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__mcp_tool_exposure__deferred_tools_recover_during_sampling.snap` | `runtime-mcp-exposure` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__mcp_tool_exposure__deferred_tools_resume_without_duplicate_update.snap` | `runtime-mcp-exposure` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_cwd_change_refreshes_agents.snap` | `runtime-model-layout` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_environment_context_includes_one_subagent.snap` | `runtime-model-layout` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_environment_context_includes_two_subagents.snap` | `runtime-model-layout` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_resume_override_matches_rollout_model.snap` | `runtime-model-layout` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_resume_with_personality_change.snap` | `runtime-model-layout` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__model_visible_layout__model_visible_layout_turn_overrides.snap` | `runtime-model-layout` | contract |
| `codex-rs/core/tests/suite/snapshots/all__suite__pending_input__pending_input_queued_mail_after_commentary.snap` | `runtime-pending-input` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__pending_input__pending_input_queued_mail_after_reasoning.snap` | `runtime-pending-input` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__pending_input__pending_input_user_input_no_preempt_after_reasoning.snap` | `runtime-pending-input` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__realtime_conversation__conversation_startup_context_current_thread_selects_many_turns_by_budget.snap` | `turn-stream-complete` | contract + Gate B |
| `codex-rs/core/tests/suite/snapshots/all__suite__token_budget__token_budget_new_context_window_tool_full_context.snap` | `runtime-context-budget` | contract |
| `codex-rs/tui/src/app/snapshots/codex_tui__app__history_ui__tests__desktop_thread_open_error_history.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/snapshots/codex_tui__app__history_ui__tests__desktop_thread_opened_history.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/snapshots/codex_tui__app__tests__directive_only_completion_removes_streamed_directive.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/snapshots/codex_tui__app__tests__required_stream_reflow_during_capped_initial_replay.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/snapshots/codex_tui__app__tests__required_stream_reflow_during_capped_initial_replay_survives_transcript_overlay.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/snapshots/codex_tui__app__thread_goal_actions__tests__thread_goal_ephemeral_error_message_renders_snapshot.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/tests/snapshots/codex_tui__app__tests__model_catalog__model_migration_prompt_shows_for_hidden_model.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/app/tests/snapshots/codex_tui__app__tests__safety_buffering__safety_retry_committed_steer_history.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_auto_resolution_countdown.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_footer_wrap.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_freeform.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_freeform_remapped_interrupt.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_freeform_remapped_submit.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_hidden_options_footer.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_long_option_text.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_multi_question_first.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_multi_question_last.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_options.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_options_notes_visible.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_scrolling_options.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_tight_height.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_unanswered_confirmation.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots/codex_tui__bottom_pane__request_user_input__tests__request_user_input_wrapped_options.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__app_link_view__tests__app_link_view_auth_suggestion_with_reason.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__app_link_view__tests__app_link_view_enable_suggestion_with_reason.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__app_link_view__tests__app_link_view_generic_url_elicitation.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__app_link_view__tests__app_link_view_generic_url_elicitation_confirmation.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__app_link_view__tests__app_link_view_install_suggestion_with_reason.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__approval_overlay__tests__approval_overlay_additional_permissions_prompt.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__approval_overlay__tests__approval_overlay_cross_thread_prompt.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__approval_overlay__tests__approval_overlay_permissions_prompt.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__approval_overlay__tests__network_exec_prompt.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__effort_tests__effort_transition_keeps_the_full_footer_row.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__adjacent_plugin_completion_inserts_separator.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__backspace_after_pastes.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__default_unified_mention_popup.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__empty.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__file_popup_ignores_bare_shell_parameter_with_matching_skill.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__file_popup_ignores_shell_positional_parameter.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__file_popup_ignores_unbindable_qualified_skill.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_empty_full.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_empty_mode_cycle_with_context.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_empty_mode_cycle_without_context.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_empty_mode_only.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_empty_full.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_empty_mode_cycle_with_context.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_empty_mode_cycle_without_context.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_empty_mode_only.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_queue_full.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_queue_message_without_context.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_queue_mode_only.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_queue_short_with_context.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_plan_queue_short_without_context.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_queue_full.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_queue_message_without_context.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_queue_mode_only.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_queue_short_with_context.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_collapse_queue_short_without_context.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_ctrl_c_interrupt.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_ctrl_c_quit.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_ctrl_c_then_esc_hint.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_esc_hint_backtrack.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_esc_hint_from_overlay.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_hidden_while_typing.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_history_search.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_history_search_unavailable.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_overlay_then_external_esc_hint.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_shell_command_absorbs_bang.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_shell_command_escape_exits_empty_mode.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_shortcut_overlay.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__footer_mode_shortcut_overlay_queue_submissions.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__image_placeholder_multiple.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__image_placeholder_single.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__large.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__mention_popup_type_prefixes.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__multiple_pastes.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__parent_owned_thread_placeholder.snap` | `composer-parent-owned` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__plugin_at_mentions_render_with_plugin_accent.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__plugin_mention_popup.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__remote_image_rows.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__remote_image_rows_after_delete_first.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__remote_image_rows_selected.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__shutdown_in_progress.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_accepts_digit_leading_skill.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_closes_after_trailing_space.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_closes_between_spaces_before_plain_text.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_does_not_fuzzy_match_shell_variable.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_falls_back_from_bound_skill_with_path_suffix.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_preserves_normal_target_after_ambiguous_probe.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_targets_unbound_mention_left_of_bound_mention.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__skill_popup_targets_unbound_mention_right_of_adjacent_bound_mention.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__slash_popup_ar.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__slash_popup_bt.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__slash_popup_mo.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__slash_popup_pet.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__slash_popup_res.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__slash_popup_si.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__small.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__ultra_accent_upgrades_prompt_glyph.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__chat_composer__tests__unified_mention_popup_falls_back_from_bound_plugin_on_right.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__command_popup__tests__command_popup_app.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__command_popup__tests__command_popup_default_items.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__command_popup__tests__command_popup_filter_reset_after_scroll.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__effort_ignition__tests__effort_ignition_animation_gallery.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__effort_status_line__tests__effort_status_line_transition_frames.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_upload_consent_lists_doctor_report.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_upload_consent_lists_windows_sandbox_log_when_included.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_bad_result.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_bug.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_good_result.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_other.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_render.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_safety_check.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__feedback_view__tests__feedback_view_with_connectivity_diagnostics.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_active_agent_label.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_composer_has_draft_queue_hint_enabled.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_context_tokens_used.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_ctrl_c_quit_idle.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_ctrl_c_quit_running.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_esc_hint_idle.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_esc_hint_primed.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_mode_indicator_narrow_overlap_hides.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_mode_indicator_running_hides_hint.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_mode_indicator_wide.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_shortcuts_collaboration_modes_enabled.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_shortcuts_context_running.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_shortcuts_default.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_shortcuts_running.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_shortcuts_shift_and_esc.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_disabled_context_right.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_enabled_mode_and_ide_context_right.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_enabled_mode_right.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_enabled_no_mode_right.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_overrides_context.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_overrides_draft_idle.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_overrides_shortcuts.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_truncated_with_gap.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_with_active_agent_label.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__footer__tests__footer_status_line_yields_to_queue_hint.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_additional_context_limit.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_capped_command_details.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_empty_handlers.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_events.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_events_with_issues.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_events_with_review_column.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_handlers.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_managed_handler.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_review_needed_handler.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_scrolled_handlers.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_selected_managed_handler.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__hooks_browser_view__tests__hooks_browser_untrusted_enabled_handler.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_col_width_mode_auto_all_rows_scroll.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_col_width_mode_auto_visible_scroll.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_col_width_mode_fixed_scroll.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_empty_searchable.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_footer_note_wraps.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_model_picker_width_80.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_narrow_width_preserves_rows.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_spacing_with_subtitle.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__list_selection_view__tests__list_selection_spacing_without_subtitle.snap` | `model-picker` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__mcp_server_elicitation__tests__mcp_server_elicitation_approval_form_with_param_summary.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__mcp_server_elicitation__tests__mcp_server_elicitation_approval_form_with_session_persist.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__mcp_server_elicitation__tests__mcp_server_elicitation_approval_form_without_schema.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__mcp_server_elicitation__tests__mcp_server_elicitation_boolean_form.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__mcp_server_elicitation__tests__mcp_server_elicitation_message_only_form.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__mcp_server_elicitation__tests__mcp_server_elicitation_message_only_form_with_persist_options.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__message_queue__tests__render_many_line_message.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__message_queue__tests__render_one_message.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__message_queue__tests__render_two_messages.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__message_queue__tests__render_wrapped_message.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_many_line_message.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_more_than_three_messages.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_multiline_pending_steer_uses_single_prefix_and_truncates.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_one_message.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_one_message_with_shift_left_binding.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_one_pending_steer.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_one_pending_steer_with_remapped_interrupt_binding.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_pending_steers_above_queued_messages.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_two_messages.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__pending_input_preview__tests__render_wrapped_message.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__skill_popup__tests__skill_popup_scrolled.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__skills_toggle_view__tests__skills_toggle_basic.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__skills_toggle_view__tests__skills_toggle_long_names_at_narrow_width.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__skills_toggle_view__tests__skills_toggle_long_names_use_available_width.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__status_line_setup__tests__setup_view_snapshot_uses_runtime_preview_values.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__queued_messages_visible_when_status_hidden_snapshot.snap` | `composer-submit` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__slash_command_popup_dismissed.snap` | `composer-mention-slash` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__status_and_composer_fill_height_without_bottom_padding.snap` | `composer-submit` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__status_and_queued_messages_snapshot.snap` | `composer-submit` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__status_hidden_when_height_too_small_height_1.snap` | `composer-submit` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__status_only_snapshot.snap` | `composer-submit` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__tests__status_with_details_and_queued_messages_snapshot.snap` | `composer-submit` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__textarea__tests__textarea_tabs_render_as_spaces_and_align_with_cursor.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__textarea__tests__vim_e_advances_from_each_word_end.snap` | `composer-queue-steer` | direct / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__title_setup__tests__terminal_title_setup_basic.snap` | `cli-doctor` | defer |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__unified_exec_footer__tests__render_many_sessions.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/bottom_pane/snapshots/codex_tui__bottom_pane__unified_exec_footer__tests__render_more_sessions.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__advanced_only_reasoning_selection_popup.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_collab_spawn_completed_renders_requested_model_and_effort.snap` | `multi-agent-roster` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_collab_wait_items_render_history.snap` | `multi-agent-roster` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_forked_thread_history_line.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_forked_thread_history_line_without_app_server_name.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_guardian_review_denied_renders_denied_request.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_guardian_review_timed_out_renders_timed_out_request.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__app_server_mcp_startup_failure_renders_warning_history.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__apply_patch_manual_flow_history_approved.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approval_modal_exec.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approval_modal_exec_multiline_prefix_no_execpolicy.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approval_modal_exec_no_reason.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approval_modal_patch.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approvals_selection_popup.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approvals_selection_popup@windows.snap` | `approval-four-states` | direct / Gate B + platform |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__approvals_selection_popup@windows_degraded.snap` | `approval-four-states` | direct / Gate B + platform |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__apps_popup_loading_state.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__auto_review_denials_popup.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__binary_size_ideal_response.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chained_config_error_wraps_in_history_snapshot.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chat_small_idle_h1.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chat_small_idle_h2.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chat_small_idle_h3.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chat_small_running_h1.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chat_small_running_h2.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chat_small_running_h3.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chatwidget_exec_and_status_layout_vt100_snapshot.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chatwidget_markdown_code_blocks_vt100_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__chatwidget_tall.snap` | `layout-resize-reflow` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__compact_queues_user_messages_snapshot.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__completed_hook_output_precedes_following_assistant_message_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__completed_hook_with_output_flushes_immediately_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__completed_same_id_hook_output_survives_restart_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__completed_turn_clears_visible_running_hook.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__deltas_then_same_final_message_are_rendered_snapshot.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__direct_budget_limited_turn_message.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__disabled_slash_command_while_task_running_snapshot.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__esc_interrupt_goal_paused_footer.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__esc_interrupt_goal_paused_footer@windows.snap` | `turn-interrupt-resume` | direct / Gate B + platform |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exec_approval_history_decision_aborted_long.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exec_approval_history_decision_aborted_multiline.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exec_approval_history_decision_approved_short.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exec_approval_modal_exec.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__experimental_features_popup.snap` | `settings-memory-personality` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exploring_step1_start_ls.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exploring_step2_finish_ls.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exploring_step3_start_cat_foo.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exploring_step4_finish_cat_foo.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exploring_step5_finish_sed_range.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__exploring_step6_finish_cat_bar.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__failed_image_generation_call_history_snapshot.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__feedback_good_result_consent_popup.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__feedback_selection_popup.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__feedback_upload_consent_popup.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__final_reasoning_then_message_without_deltas_are_rendered.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__final_worked_for_uses_cumulative_turn_duration.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__foreign_image_attachment_history_snapshot.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__forked_thread_history_line.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__forked_thread_history_line_without_name.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__full_access_confirmation_popup.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__goal_edit_prompt.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__goal_menu_active.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__goal_menu_blocked.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__goal_menu_budget_limited.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__goal_menu_paused.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__goal_menu_usage_limited.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__guardian_approved_exec_renders_approved_request.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__guardian_approved_request_permissions_renders_request_summary.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__guardian_denied_exec_renders_warning_and_denied_request.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__guardian_goal_continuation_drops_stale_reviews.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__guardian_parallel_reviews_render_aggregate_status.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__guardian_timed_out_exec_renders_warning_and_timed_out_request.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__hook_blocked_failed_feedback_history_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__hook_completed_before_reveal_renders_completed_without_running_flash_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__hook_live_running_then_quiet_completed_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__hook_runs_while_exec_active_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__hooks_popup_shows_list_diagnostics.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__identical_parallel_running_hooks_collapse_to_count_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__image_generation_begin_restores_working_status.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__image_generation_call_history_snapshot.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupt_clears_unified_exec_wait_streak.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupt_exec_marks_failed.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupt_preserves_unified_exec_wait_streak.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupted_turn_clears_visible_running_hook.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupted_turn_error_message.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupted_turn_goal_budget_limited_message.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__interrupted_turn_pending_steers_message.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__live_app_server_command_execution_strips_shell_wrapper.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__live_app_server_command_output_delta_active.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__live_app_server_command_output_delta_interrupted.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__live_app_server_command_output_delta_transcript.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__live_app_server_turn_completion_repairs_dropped_message_deltas.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__live_app_server_user_message_omits_unsupported_media.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__local_image_attachment_history_snapshot.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__long_hook_context_is_truncated_with_transcript_hint.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__mcp_startup_header_booting.snap` | `mcp-inventory-elicitation` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__memories_enable_prompt.snap` | `settings-memory-personality` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__memories_reset_confirmation.snap` | `settings-memory-personality` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__memories_settings_popup.snap` | `settings-memory-personality` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__model_advanced_reasoning_selection_popup.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__model_picker_filters_hidden_models.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__model_reasoning_selection_popup.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__model_reasoning_selection_popup_extra_high_warning.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__model_selection_popup.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__multi_agent_enable_prompt.snap` | `multi-agent-roster` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__overlapping_hook_live_cell_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__parent_owned_thread_rejects_input.snap` | `composer-parent-owned` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__parent_owned_thread_rejects_settings_shortcuts.snap` | `composer-parent-owned` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__pending_token_activity_refresh_renders_above_composer_snapshot.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__permissions_selection_history_after_mode_switch.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__permissions_selection_history_full_access_to_default.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__permissions_selection_history_full_access_to_default@windows.snap` | `approval-four-states` | direct / Gate B + platform |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__personality_selection_popup.snap` | `settings-memory-personality` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plan_implementation_popup.snap` | `plan-goal` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plan_implementation_popup_context_usage.snap` | `plan-goal` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plan_implementation_popup_no_selected.snap` | `plan-goal` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plan_mode_nudge.snap` | `plan-goal` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plan_mode_nudge_narrow.snap` | `plan-goal` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugin_detail_error_popup.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugin_detail_popup_installable.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugin_detail_popup_installed.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugin_detail_popup_local_share_read_only.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugin_detail_popup_npm_source.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_admin_disabled_installed.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_curated_marketplace.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_empty_shared_section_hidden.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_loading_state.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_marketplace_remove_confirmation.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_newly_installed_marketplace.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__plugins_popup_search_filtered.snap` | `plugin-marketplace` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__post_tool_use_hook_events_render_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__pre_tool_use_hook_events_render_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__preamble_keeps_working_status.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__profile_permissions_selection_popup.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__profile_permissions_selection_popup_with_custom_profiles.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__profile_permissions_selection_popup_with_disallowed_full_access.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__prompt_edit_thread_history_line.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__rate_limit_reset_available_hint.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__rate_limit_reset_confirmation.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__rate_limit_reset_hint_waits_for_active_output.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__rate_limit_reset_picker_narrow.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__rate_limit_reset_popup_states.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__rate_limit_switch_prompt_popup.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__reasoning_delta_restores_recreated_status_indicator.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__renamed_thread_footer_title.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__replayed_interrupted_reconnect_footer_row.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__resume_paused_goal_prompt.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__review_queues_user_messages_snapshot.snap` | `composer-queue-steer` | defer / TUI-only |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__review_submission_warning_snapshot.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__safety_buffering_retry_prompt.snap` | `error-config-provider` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__safety_buffering_side_conversation_without_retry.snap` | `error-config-provider` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__safety_buffering_status_without_retry.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__session_start_hook_events_render_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__side_context_label_preserves_status_line.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__side_context_label_shows_hidden_side.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__side_context_label_shows_parent_status.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__signed_out_usage_command_reports_chatgpt_login_requirement.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__single_line_final_answer_hides_working_status.snap` | `history-replay-isomorphic` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__skills_menu_default_mentions_shortcut.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_app_without_thread_id_shows_starting_error.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_archive_confirmation_popup.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_copy_no_output_info_message.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_delete_confirmation_popup.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_pets_picker.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_rename_prefilled_prompt.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__slash_side_requests_forked_side_question_while_task_running.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_fast_mode_footer.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_goal_active_token_budget_footer.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_goal_complete_elapsed_footer.snap` | `turn-budget-limit` | direct / Gate A/B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_model_with_reasoning_context_remaining_footer.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_model_with_reasoning_fast_footer.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_model_with_reasoning_plan_mode_footer.snap` | `model-picker` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_setup_popup_hardcoded_only.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_setup_popup_live_only.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_setup_popup_mixed.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_setup_popup_rate_limits.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_line_setup_popup_workspace_headline.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_surface_previews_hardcoded_only.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_surface_previews_live_only.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_surface_previews_mixed.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_surface_previews_rate_limits.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_widget_active.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__status_widget_and_approval_modal.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__terminal_title_setup_popup_hardcoded_only.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__terminal_title_setup_popup_live_only.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__terminal_title_setup_popup_mixed.snap` | `turn-stream-complete` | merge / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__terminal_title_setup_popup_rate_limits.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__thread_name_update_resume_hint.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__ultra_reasoning_selection_high_multi_agent_concurrency_warning.snap` | `multi-agent-roster` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_begin_restores_working_status.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_empty_then_non_empty_after.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_non_empty_then_empty_active.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_non_empty_then_empty_after.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_unknown_end_with_active_exploring_cell.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_wait_after_final_agent_message.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_wait_before_streamed_agent_message.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_wait_status_renders_command_in_single_details_row.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__unified_exec_waiting_multiple_empty_after.snap` | `tool-unified-exec-wait` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__update_popup.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__usage_command_menu.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__usage_command_menu_before_reset_refresh.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__usage_command_menu_without_resets.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__usage_command_with_invalid_view_reports_usage.snap` | `apps-and-capabilities` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__user_prompt_submit_app_server_hook_notifications_render_snapshot.snap` | `hooks-lifecycle` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__user_shell_ls_output.snap` | `tool-running-terminal` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__windows_sandbox_required_enable_prompt.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__windows_sandbox_required_fallback_prompt.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__workspace_member_credits_depleted_prompt.snap` | `error-config-provider` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__workspace_member_usage_limit_prompt.snap` | `error-config-provider` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__workspace_owner_credits_nudge_completion_feedback.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__workspace_owner_limit_state_messages.snap` | `error-config-provider` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/snapshots/codex_tui__chatwidget__tests__workspace_owner_usage_limit_nudge_completion_feedback.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__app_server__app_server_bio_policy_error_renders_dedicated_notice.snap` | `error-config-provider` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__exec_approval_history_decision_aborted_long.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__exec_approval_history_decision_aborted_multiline.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__exec_approval_history_decision_approved_short.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__exec_approval_modal_exec.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__network_exec_approval_history_canceled_host_request.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__network_exec_approval_history_one_time_host_allowance.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__approval_requests__network_exec_approval_history_session_host_allowance.snap` | `approval-four-states` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__composer_submission__output_free_ctrl_c_interrupt_keeps_prompt_and_blank_composer.snap` | `turn-interrupt-resume` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__history_replay__replayed_nested_review_prompts.snap` | `history-fork-lineage` | direct / Gate B |
| `codex-rs/tui/src/chatwidget/tests/snapshots/codex_tui__chatwidget__tests__status_and_layout__unsupported_code_mode_warning.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/exec_cell/snapshots/codex_tui__exec_cell__render__tests__truncated_live_output_preview_and_transcript.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__active_mcp_tool_call_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__coalesced_reads_dedupe_names.snap` | `history-replay-isomorphic` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__coalesces_reads_across_multiple_calls.snap` | `history-replay-isomorphic` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__coalesces_sequential_reads_within_one_call.snap` | `history-replay-isomorphic` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__completed_mcp_tool_call_error_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__completed_mcp_tool_call_multiple_outputs_inline_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__completed_mcp_tool_call_multiple_outputs_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__completed_mcp_tool_call_success_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__completed_mcp_tool_call_wrapped_outputs_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__cyber_policy_error_event_narrow_snapshot.snap` | `error-safety-sandbox` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__cyber_policy_error_event_snapshot.snap` | `error-safety-sandbox` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__error_event_bedrock_expired_signature_snapshot.snap` | `error-safety-sandbox` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__error_event_oversized_input_snapshot.snap` | `error-safety-sandbox` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__mcp_inventory_loading_snapshot.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__mcp_tools_output_from_statuses_renders_status_only_servers.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__mcp_tools_output_from_statuses_renders_verbose_inventory.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__mcp_tools_output_lists_tools_for_hyphenated_server_names.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__mcp_tools_output_masks_sensitive_values.snap` | `mcp-inventory-elicitation` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__multiline_command_both_lines_wrap_with_correct_prefixes.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__multiline_command_without_wrap_uses_branch_then_eight_spaces.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__multiline_command_wraps_with_extra_indent_on_subsequent_lines.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__plan_update_with_note_and_wrapping_snapshot.snap` | `plan-goal` | direct / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__plan_update_without_note_snapshot.snap` | `plan-goal` | direct / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__pnpm_update_available_history_cell_snapshot.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__ps_output_chunk_leading_whitespace_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__ps_output_empty_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__ps_output_long_command_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__ps_output_many_sessions_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__ps_output_multiline_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__ran_cell_multiline_with_stderr_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__raw_mode_toggle_transcript.snap` | `history-replay-isomorphic` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__safety_access_block_event_snapshot.snap` | `error-safety-sandbox` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__session_header_clamps_to_narrow_width.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__session_header_indicates_yolo_mode.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__session_info_availability_nux_tooltip_snapshot.snap` | `status-surface-matrix` | merge / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__single_line_command_compact_when_fits.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__single_line_command_over_highlight_limit_uses_plain_text_fallback.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__single_line_command_wraps_with_four_space_continuation.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__standalone_unix_update_available_history_cell_snapshot.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__standalone_windows_update_available_history_cell_snapshot.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__stderr_tail_more_than_five_lines_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__streamed_agent_list_paragraph_preserves_item_indent_when_wrapped.snap` | `history-replay-isomorphic` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__user_history_cell_numbers_multiple_remote_images.snap` | `media-image-attachment` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__user_history_cell_renders_remote_image_urls.snap` | `media-image-attachment` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__user_history_cell_wraps_and_prefixes_each_line_snapshot.snap` | `history-replay-isomorphic` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__web_search_history_cell_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__web_search_history_cell_transcript_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/history_cell/snapshots/codex_tui__history_cell__tests__web_search_history_cell_without_detail_snapshot.snap` | `tool-running-terminal` | direct / Gate A/B |
| `codex-rs/tui/src/onboarding/snapshots/codex_tui__onboarding__auth__tests__continue_in_browser_narrow_long_url.snap` | `onboarding-setup` | direct / Gate A |
| `codex-rs/tui/src/onboarding/snapshots/codex_tui__onboarding__trust_directory__tests__renders_snapshot_for_git_repo.snap` | `onboarding-setup` | direct / Gate A |
| `codex-rs/tui/src/onboarding/snapshots/codex_tui__onboarding__trust_directory__tests__renders_snapshot_for_trust_error.snap` | `onboarding-setup` | direct / Gate A |
| `codex-rs/tui/src/render/snapshots/codex_tui__render__highlight__tests__ansi_family_foreground_palette.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__agent_picker_item_name.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__backtrack_branch_failure_restores_selected_prompt.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__bypass_hook_trust_startup_warning.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__clear_ui_after_long_transcript_fresh_header_only.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__clear_ui_header_fast_status_fast_capable_models.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__in_app_resume_session_cwd_without_metadata.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__path_backed_agent_picker.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__remote_resume_current_cwd_rejected.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__replace_goal_confirmation.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app__tests__side_backtrack_rejection_reports_unavailable_message.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__app_backtrack__tests__backtrack_unavailable_info_message_snapshot.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__cwd_prompt__tests__cwd_prompt_fork_modal.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__cwd_prompt__tests__cwd_prompt_modal.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__cwd_prompt__tests__cwd_prompt_persistence_failure.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__cwd_prompt__tests__cwd_prompt_remembered_current_modal.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__cwd_prompt__tests__cwd_prompt_remote_exec_modal.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__debug_config__tests__debug_config_effective_sandbox_modes_with_deny_read.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__debug_config__tests__debug_config_output_lists_agents_fields.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__debug_config__tests__debug_config_requirement_sources.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__add_details.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__ansi16_insert_delete_no_background.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_add_block.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_delete_block.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_multiple_files_block.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_update_block.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_update_block_line_numbers_three_digits_text.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_update_block_relativizes_path.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_update_block_wraps_long_lines.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_update_block_wraps_long_lines_text.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__apply_update_with_rename_block.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__blank_context_line.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__cpp_module_extension_highlighting.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__diff_gallery_120x40.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__diff_gallery_80x24.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__diff_gallery_94x35.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__single_line_replacement_counts.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__syntax_highlighted_insert_wraps.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__syntax_highlighted_insert_wraps_text.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__theme_scope_background_resolution.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__update_details_with_rename.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__vertical_ellipsis_between_hunks.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__diff_render__tests__wrap_behavior_insert.snap` | `artifact-diff-preview` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_customize.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_customize_action.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_customize_action_windows.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_customize_windows.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_prompt.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_prompt_windows.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration__tests__external_agent_config_migration_secondary_source_customize.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration_flow__tests__external_agent_config_migration_messages.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__external_agent_config_migration_source__tests__external_agent_config_source_prompt.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__git_action_directives__tests__code_comment_directive_fallback.snap` | `history-replay-isomorphic` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__inline_visualization__tests__agent_code_blocks_preserve_visualization_directive_literals.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__inline_visualization__tests__finalized_agent_cell_visualization_link.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__inline_visualization__tests__transcript_overlay_visualization_becomes_available.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__inline_visualization__tests__viewer_document_contract.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__insert_history__tests__zellij_raw_terminal_wrap_above_viewport.snap` | `cli-doctor` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__insert_history__tests__zellij_raw_terminal_wrap_overflow_above_viewport.snap` | `cli-doctor` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_action_menu.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_capture_view.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_debug_view_delayed_hint.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_debug_view_initial.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_debug_view_match.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_picker_all_tab_search.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_picker_custom.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_picker_fast_mode_enabled.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_picker_first_actions.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_picker_narrow.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__keymap_setup__tests__keymap_picker_wide.snap` | `settings-memory-personality` | defer |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__bare_url_with_tilde_keeps_complete_hyperlink.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__list_item_after_code_block_keeps_blank_separator.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__markdown_render_complex_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__markdown_render_file_link_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__mixed_url_markdown_wraps_prose_without_splitting_words_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__multiline_finding_items_are_separated_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__table_keeps_grid_when_only_one_compact_record_fragments_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__table_renders_key_value_records_when_compact_fragmentation_is_systemic_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__table_renders_records_when_multiple_prose_columns_are_starved_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__table_renders_stacked_key_value_records_when_path_column_becomes_too_narrow_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__markdown_render__markdown_render_tests__table_wraps_file_paths_before_collapsing_narrative_columns_snapshot.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__model_migration__tests__model_migration_prompt.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__model_migration__tests__model_migration_prompt_gpt5_codex.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__model_migration__tests__model_migration_prompt_gpt5_codex_mini.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__model_migration__tests__model_migration_prompt_gpt5_family.snap` | `onboarding-setup` | merge / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__multi_agents__tests__collab_agent_transcript.snap` | `multi-agent-roster` | direct / Gate B |
| `codex-rs/tui/src/snapshots/codex_tui__multi_agents__tests__collab_resume_interrupted.snap` | `multi-agent-roster` | direct / Gate B |
| `codex-rs/tui/src/snapshots/codex_tui__pager_overlay__tests__static_overlay_snapshot_basic.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__pager_overlay__tests__static_overlay_wraps_long_lines.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__pager_overlay__tests__transcript_overlay_apply_patch_scroll_vt100.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__pager_overlay__tests__transcript_overlay_renders_live_tail.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__pager_overlay__tests__transcript_overlay_snapshot_basic.snap` | `layout-markdown-rich` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_dense_all.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_dense_all_auto_hidden_cwd.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_dense_all_forced_cwd.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_dense_cwd.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_dense_narrow.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_dense_no_blank_lines.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_expanded_session.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_footer_compact.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_footer_wide.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_more_indicators.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_narrow_session.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_screen.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_search_error.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_search_line_sort_filter_tabs.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_table.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_thread_names.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__resume_picker__tests__resume_picker_transcript_loading_overlay.snap` | `search-resume-picker` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__startup_hooks_review__tests__startup_hooks_review_prompt.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__startup_hooks_review__tests__startup_hooks_review_prompt_with_trust_error.snap` | `hooks-lifecycle` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__status_indicator_widget__tests__renders_remapped_interrupt_hint.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__status_indicator_widget__tests__renders_truncated.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__status_indicator_widget__tests__renders_with_queued_messages.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__status_indicator_widget__tests__renders_with_queued_messages@macos.snap` | `status-running-completed` | direct / Gate A + platform |
| `codex-rs/tui/src/snapshots/codex_tui__status_indicator_widget__tests__renders_with_working_header.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__status_indicator_widget__tests__renders_wrapped_details_panama_two_lines.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/snapshots/codex_tui__update_prompt__tests__update_prompt_modal.snap` | `feedback-update-usage` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_cached_limits_hide_credits_without_flag.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_includes_credits_and_limits.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_includes_enterprise_monthly_credit_limit.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_includes_forked_from.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_includes_monthly_limit.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_includes_reasoning_details.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_active_user_defined_profile.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_auto_review_permissions.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_chatgpt_plan_without_email.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_missing_limits_message.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_refreshing_limits_notice.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_stale_limits_message.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_shows_unavailable_limits_message.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_treats_refreshing_empty_limits_as_unavailable.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_truncates_in_narrow_terminal.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_uses_default_reasoning_when_config_empty.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_uses_generic_limit_labels_for_unsupported_windows.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__status_snapshot_wraps_enterprise_monthly_credit_details_in_narrow_terminal.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/status/snapshots/codex_tui__status__tests__transcript_overlay_status_rate_limit_refresh.snap` | `status-running-completed` | direct / Gate A |
| `codex-rs/tui/src/streaming/snapshots/codex_tui__streaming__render__tests__incremental_render_representative_stream.snap` | `turn-stream-complete` | direct / Gate A/B |
| `codex-rs/tui/src/streaming/snapshots/codex_tui__streaming__render__tests__inline_visualizations_without_context_use_canonical_full_render.snap` | `turn-stream-complete` | direct / Gate A/B |

## 完整性

- 映射行数：`663`
- 逻辑场景数（去平台后缀）：`658`
- 平台后缀仅增加 platform matrix，不创建平行业务 owner。
