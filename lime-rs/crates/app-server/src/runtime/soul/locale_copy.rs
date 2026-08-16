use super::boundary::FORMAL_ARTIFACT_VOICE_SOURCE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Locale {
    ZhCn,
    ZhTw,
    EnUs,
    JaJp,
    KoKr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HandoffCopy {
    pub(crate) plan_title: &'static str,
    pub(crate) progress_title: &'static str,
    pub(crate) handoff_title: &'static str,
    pub(crate) review_summary_title: &'static str,
    pub(crate) session_label: &'static str,
    pub(crate) thread_label: &'static str,
    pub(crate) status_label: &'static str,
    pub(crate) exported_at_label: &'static str,
    pub(crate) todo_summary_title: &'static str,
    pub(crate) recent_artifacts_title: &'static str,
    pub(crate) no_recent_artifacts: &'static str,
    pub(crate) next_step_title: &'static str,
    pub(crate) next_step_body: &'static str,
    pub(crate) review_note: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReplayExportCopy {
    pub(crate) input_title: &'static str,
    pub(crate) expected_title: &'static str,
    pub(crate) grader_title: &'static str,
    pub(crate) evidence_links_title: &'static str,
    pub(crate) grader_heading: &'static str,
    pub(crate) checks_heading: &'static str,
    pub(crate) pending_request_check: &'static str,
    pub(crate) queued_turn_check: &'static str,
    pub(crate) read_model_check: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AnalysisExportCopy {
    pub(crate) response_title: &'static str,
    pub(crate) brief_artifact_title: &'static str,
    pub(crate) context_artifact_title: &'static str,
    pub(crate) markdown_heading: &'static str,
    pub(crate) focus_heading: &'static str,
    pub(crate) focus_current_read_model: &'static str,
    pub(crate) focus_no_legacy: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReviewExportCopy {
    pub(crate) response_title: &'static str,
    pub(crate) markdown_artifact_title: &'static str,
    pub(crate) json_artifact_title: &'static str,
    pub(crate) markdown_heading: &'static str,
    pub(crate) summary_heading: &'static str,
    pub(crate) pending_summary: &'static str,
    pub(crate) followup_heading: &'static str,
    pub(crate) no_followup: &'static str,
    pub(crate) regression_heading: &'static str,
    pub(crate) default_regression_requirement: &'static str,
    pub(crate) notes_heading: &'static str,
    pub(crate) default_decision_regression_requirement: &'static str,
    pub(crate) checklist: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RolloutSummaryCopy {
    pub(crate) runtime_facts_heading: &'static str,
    pub(crate) candidate_memory_heading: &'static str,
    pub(crate) review_before_promoting: &'static str,
    pub(crate) pending_work_label: &'static str,
    pub(crate) pending_label: &'static str,
    pub(crate) in_progress_label: &'static str,
    pub(crate) referenced_artifacts_heading: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GenerationBriefCopy {
    pub(crate) heading: &'static str,
    pub(crate) voice_source_label: &'static str,
    pub(crate) default_voice: &'static str,
    pub(crate) explicit_voice: &'static str,
    pub(crate) fidelity_rule: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeExportCopy {
    locale: Locale,
    pub(crate) handoff: HandoffCopy,
    pub(crate) replay: ReplayExportCopy,
    pub(crate) analysis: AnalysisExportCopy,
    pub(crate) review: ReviewExportCopy,
    pub(crate) rollout: RolloutSummaryCopy,
    pub(crate) generation_brief: GenerationBriefCopy,
}

impl RuntimeExportCopy {
    pub(crate) fn formal_artifact_voice_source(&self) -> &'static str {
        FORMAL_ARTIFACT_VOICE_SOURCE
    }

    pub(crate) fn analysis_copy_prompt(
        &self,
        session_id: &str,
        analysis_relative_root: &str,
        replay_relative_root: &str,
    ) -> String {
        match self.locale {
            Locale::ZhTw => format!(
                "請基於 App Server current 匯出的 `{analysis_relative_root}` 和 `{replay_relative_root}` 分析會話 `{session_id}` 的下一步風險、缺口和回歸驗證；正式產物聲線只接受 Generation Brief，不依賴 legacy agent_runtime_* 輸出。"
            ),
            Locale::EnUs => format!(
                "Use the App Server current exports `{analysis_relative_root}` and `{replay_relative_root}` to review session `{session_id}` for the next risks, gaps, and regressions. Formal artifact voice must come only from Generation Brief; do not rely on legacy agent_runtime_* output."
            ),
            Locale::JaJp => format!(
                "App Server current のエクスポート `{analysis_relative_root}` と `{replay_relative_root}` を基に、セッション `{session_id}` の次のリスク、欠落、回帰確認を分析してください。正式な成果物の声線は Generation Brief のみを使用し、legacy agent_runtime_* 出力には依存しないでください。"
            ),
            Locale::KoKr => format!(
                "App Server current 내보내기 `{analysis_relative_root}` 및 `{replay_relative_root}`를 기준으로 세션 `{session_id}`의 다음 위험, 누락, 회귀 검증을 분석하세요. 정식 산출물의 보이스는 Generation Brief만 사용하며 legacy agent_runtime_* 출력에 의존하지 마세요."
            ),
            Locale::ZhCn => format!(
                "请基于 App Server current 导出的 `{analysis_relative_root}` 和 `{replay_relative_root}` 分析会话 `{session_id}` 的下一步风险、缺口和回归验证；正式产物声线只接受 Generation Brief，不依赖 legacy agent_runtime_* 输出。"
            ),
        }
    }

    pub(crate) fn rollout_opening(&self, session_id: &str, export_kind: &str) -> String {
        match self.locale {
            Locale::ZhTw => {
                format!("會話 `{session_id}` 已匯出 `{export_kind}` 證據，供後續彙整。")
            }
            Locale::EnUs => format!(
                "Session `{session_id}` exported `{export_kind}` evidence for follow-up consolidation."
            ),
            Locale::JaJp => {
                format!(
                    "セッション `{session_id}` は後続整理用に `{export_kind}` 証跡をエクスポートしました。"
                )
            }
            Locale::KoKr => {
                format!(
                    "세션 `{session_id}`가 후속 정리를 위해 `{export_kind}` 증거를 내보냈습니다."
                )
            }
            Locale::ZhCn => {
                format!("会话 `{session_id}` 已导出 `{export_kind}` 证据，供后续汇总。")
            }
        }
    }
}

pub(crate) fn runtime_export_copy(locale: Option<&str>) -> RuntimeExportCopy {
    match resolve_locale(locale) {
        Locale::ZhTw => zh_tw_copy(),
        Locale::EnUs => en_us_copy(),
        Locale::JaJp => ja_jp_copy(),
        Locale::KoKr => ko_kr_copy(),
        Locale::ZhCn => zh_cn_copy(),
    }
}

fn resolve_locale(locale: Option<&str>) -> Locale {
    let Some(locale) = locale else {
        return Locale::ZhCn;
    };
    let locale = locale.trim().replace('_', "-").to_ascii_lowercase();
    if locale == "zh-tw" || locale == "zh-hk" || locale == "zh-mo" || locale.starts_with("zh-hant")
    {
        return Locale::ZhTw;
    }
    if locale == "en" || locale.starts_with("en-") {
        return Locale::EnUs;
    }
    if locale == "ja" || locale.starts_with("ja-") {
        return Locale::JaJp;
    }
    if locale == "ko" || locale.starts_with("ko-") {
        return Locale::KoKr;
    }
    Locale::ZhCn
}

fn zh_cn_copy() -> RuntimeExportCopy {
    RuntimeExportCopy {
        locale: Locale::ZhCn,
        handoff: HandoffCopy {
            plan_title: "计划摘要",
            progress_title: "结构化进度",
            handoff_title: "交接摘要",
            review_summary_title: "审查摘要",
            session_label: "会话",
            thread_label: "线程",
            status_label: "状态",
            exported_at_label: "导出时间",
            todo_summary_title: "Todo 摘要",
            recent_artifacts_title: "最近产物",
            no_recent_artifacts: "当前没有可引用的最近产物。",
            next_step_title: "推荐接手顺序",
            next_step_body: "先读 progress.json 确认结构化状态，再读 handoff.md 决定下一刀。",
            review_note:
                "此摘要来自 App Server current read model；不要把 legacy command 输出当成交付证据。",
        },
        replay: ReplayExportCopy {
            input_title: "回放输入",
            expected_title: "回放预期结果",
            grader_title: "回放评分器",
            evidence_links_title: "回放证据链接",
            grader_heading: "回放评分器",
            checks_heading: "检查项",
            pending_request_check: "pendingRequestCount 应保持为 {count}，除非本次变更有意调整。",
            queued_turn_check: "queuedTurnCount 应保持为 {count}，除非本次变更有意调整。",
            read_model_check: "回放必须保持 App Server current read model 结构。",
        },
        analysis: AnalysisExportCopy {
            response_title: "外部分析交接",
            brief_artifact_title: "外部分析简报",
            context_artifact_title: "外部分析上下文",
            markdown_heading: "外部分析交接",
            focus_heading: "分析重点",
            focus_current_read_model: "审查当前 App Server read model，并决定下一刀实现切片。",
            focus_no_legacy: "不要把 legacy agent_runtime_* command 输出当作生产证据。",
        },
        review: ReviewExportCopy {
            response_title: "人工审核决策",
            markdown_artifact_title: "人工审核记录",
            json_artifact_title: "人工审核记录 JSON",
            markdown_heading: "人工审核决策",
            summary_heading: "摘要",
            pending_summary: "等待人工审核。",
            followup_heading: "后续动作",
            no_followup: "未记录。",
            regression_heading: "回归要求",
            default_regression_requirement: "运行 current 路径定向回归。",
            notes_heading: "备注",
            default_decision_regression_requirement: "标记 accepted 前运行 current 路径定向回归。",
            checklist: &[
                "确认 App Server current 路径证据。",
                "确认生产路径不需要 legacy agent_runtime_* fallback。",
                "接受前运行定向回归。",
            ],
        },
        rollout: RolloutSummaryCopy {
            runtime_facts_heading: "运行时事实",
            candidate_memory_heading: "候选记忆",
            review_before_promoting: "提升为长期记忆前，先审查该导出目录。",
            pending_work_label: "仍有未完成工作",
            pending_label: "待处理",
            in_progress_label: "进行中",
            referenced_artifacts_heading: "引用产物",
        },
        generation_brief: GenerationBriefCopy {
            heading: "Generation Brief 边界",
            voice_source_label: "正式产物声线来源",
            default_voice: "默认使用中性、可审计的导出语气。",
            explicit_voice:
                "只有显式 generation_brief / creator voice / brand voice 可以影响正式产物正文。",
            fidelity_rule:
                "Product Soul 只影响交互表达，不会自动改写正式 artifact、导出正文或报告正文。",
        },
    }
}

fn zh_tw_copy() -> RuntimeExportCopy {
    RuntimeExportCopy {
        locale: Locale::ZhTw,
        handoff: HandoffCopy {
            plan_title: "計畫摘要",
            progress_title: "結構化進度",
            handoff_title: "交接摘要",
            review_summary_title: "審查摘要",
            session_label: "會話",
            thread_label: "執行緒",
            status_label: "狀態",
            exported_at_label: "匯出時間",
            todo_summary_title: "Todo 摘要",
            recent_artifacts_title: "最近產物",
            no_recent_artifacts: "目前沒有可引用的最近產物。",
            next_step_title: "建議接手順序",
            next_step_body: "先讀 progress.json 確認結構化狀態，再讀 handoff.md 決定下一刀。",
            review_note:
                "此摘要來自 App Server current read model；不要把 legacy command 輸出當成交付證據。",
        },
        replay: ReplayExportCopy {
            input_title: "回放輸入",
            expected_title: "回放預期結果",
            grader_title: "回放評分器",
            evidence_links_title: "回放證據連結",
            grader_heading: "回放評分器",
            checks_heading: "檢查項",
            pending_request_check: "pendingRequestCount 應保持為 {count}，除非本次變更有意調整。",
            queued_turn_check: "queuedTurnCount 應保持為 {count}，除非本次變更有意調整。",
            read_model_check: "回放必須保持 App Server current read model 結構。",
        },
        analysis: AnalysisExportCopy {
            response_title: "外部分析交接",
            brief_artifact_title: "外部分析簡報",
            context_artifact_title: "外部分析上下文",
            markdown_heading: "外部分析交接",
            focus_heading: "分析重點",
            focus_current_read_model: "審查目前 App Server read model，並決定下一刀實作切片。",
            focus_no_legacy: "不要把 legacy agent_runtime_* command 輸出當作生產證據。",
        },
        review: ReviewExportCopy {
            response_title: "人工審核決策",
            markdown_artifact_title: "人工審核記錄",
            json_artifact_title: "人工審核記錄 JSON",
            markdown_heading: "人工審核決策",
            summary_heading: "摘要",
            pending_summary: "等待人工審核。",
            followup_heading: "後續動作",
            no_followup: "未記錄。",
            regression_heading: "回歸要求",
            default_regression_requirement: "執行 current 路徑定向回歸。",
            notes_heading: "備註",
            default_decision_regression_requirement: "標記 accepted 前執行 current 路徑定向回歸。",
            checklist: &[
                "確認 App Server current 路徑證據。",
                "確認生產路徑不需要 legacy agent_runtime_* fallback。",
                "接受前執行定向回歸。",
            ],
        },
        rollout: RolloutSummaryCopy {
            runtime_facts_heading: "執行時事實",
            candidate_memory_heading: "候選記憶",
            review_before_promoting: "提升為長期記憶前，先審查該匯出目錄。",
            pending_work_label: "仍有未完成工作",
            pending_label: "待處理",
            in_progress_label: "進行中",
            referenced_artifacts_heading: "引用產物",
        },
        generation_brief: GenerationBriefCopy {
            heading: "Generation Brief 邊界",
            voice_source_label: "正式產物聲線來源",
            default_voice: "預設使用中性、可審計的匯出語氣。",
            explicit_voice:
                "只有明確的 generation_brief / creator voice / brand voice 可以影響正式產物正文。",
            fidelity_rule:
                "Product Soul 只影響互動表達，不會自動改寫正式 artifact、匯出正文或報告正文。",
        },
    }
}

fn en_us_copy() -> RuntimeExportCopy {
    RuntimeExportCopy {
        locale: Locale::EnUs,
        handoff: HandoffCopy {
            plan_title: "Plan Summary",
            progress_title: "Structured Progress",
            handoff_title: "Handoff Summary",
            review_summary_title: "Review Summary",
            session_label: "Session",
            thread_label: "Thread",
            status_label: "Status",
            exported_at_label: "Exported At",
            todo_summary_title: "Todo Summary",
            recent_artifacts_title: "Recent Artifacts",
            no_recent_artifacts: "No recent artifacts are available.",
            next_step_title: "Recommended Handoff Order",
            next_step_body: "Read progress.json for structured state first, then use handoff.md to choose the next implementation slice.",
            review_note: "This summary is generated from the App Server current read model; do not treat legacy command output as delivery evidence.",
        },
        replay: ReplayExportCopy {
            input_title: "Replay Input",
            expected_title: "Replay Expected Result",
            grader_title: "Replay Grader",
            evidence_links_title: "Replay Evidence Links",
            grader_heading: "Replay Grader",
            checks_heading: "Checks",
            pending_request_check: "pendingRequestCount should remain {count} unless intentionally changed.",
            queued_turn_check: "queuedTurnCount should remain {count} unless intentionally changed.",
            read_model_check: "Replay must preserve the App Server current read model shape.",
        },
        analysis: AnalysisExportCopy {
            response_title: "External Analysis Handoff",
            brief_artifact_title: "External Analysis Brief",
            context_artifact_title: "External Analysis Context",
            markdown_heading: "External Analysis Handoff",
            focus_heading: "Focus",
            focus_current_read_model: "Review the current App Server read model and choose the next implementation slice.",
            focus_no_legacy: "Do not use legacy agent_runtime_* command output as production evidence.",
        },
        review: ReviewExportCopy {
            response_title: "Review Decision",
            markdown_artifact_title: "Review Decision",
            json_artifact_title: "Review Decision JSON",
            markdown_heading: "Review Decision",
            summary_heading: "Summary",
            pending_summary: "Pending human review.",
            followup_heading: "Follow-up Actions",
            no_followup: "None recorded.",
            regression_heading: "Regression Requirements",
            default_regression_requirement: "Run current-path targeted regression.",
            notes_heading: "Notes",
            default_decision_regression_requirement: "Run targeted current-path regression before marking accepted.",
            checklist: &[
                "Confirm current App Server path evidence.",
                "Confirm no legacy agent_runtime_* production fallback is required.",
                "Run targeted regression before accepting.",
            ],
        },
        rollout: RolloutSummaryCopy {
            runtime_facts_heading: "Runtime Facts",
            candidate_memory_heading: "Candidate Memory",
            review_before_promoting: "Review this export root before promoting it into long-term memory.",
            pending_work_label: "Pending work remains",
            pending_label: "pending",
            in_progress_label: "in progress",
            referenced_artifacts_heading: "Referenced Artifacts",
        },
        generation_brief: GenerationBriefCopy {
            heading: "Generation Brief Boundary",
            voice_source_label: "Formal artifact voice source",
            default_voice: "Default to neutral, auditable export wording.",
            explicit_voice: "Only explicit generation_brief, creator voice, or brand voice may affect formal artifact body copy.",
            fidelity_rule: "Product Soul affects interaction wording only; it must not automatically rewrite formal artifacts, export bodies, or reports.",
        },
    }
}

fn ja_jp_copy() -> RuntimeExportCopy {
    RuntimeExportCopy {
        locale: Locale::JaJp,
        handoff: HandoffCopy {
            plan_title: "計画サマリー",
            progress_title: "構造化された進捗",
            handoff_title: "引き継ぎサマリー",
            review_summary_title: "レビューサマリー",
            session_label: "セッション",
            thread_label: "スレッド",
            status_label: "状態",
            exported_at_label: "エクスポート時刻",
            todo_summary_title: "Todo サマリー",
            recent_artifacts_title: "最近の成果物",
            no_recent_artifacts: "参照できる最近の成果物はありません。",
            next_step_title: "推奨される引き継ぎ順序",
            next_step_body: "まず progress.json で構造化された状態を確認し、次に handoff.md で次の作業を決めてください。",
            review_note: "このサマリーは App Server current read model から生成されています。legacy command の出力を納品証跡として扱わないでください。",
        },
        replay: ReplayExportCopy {
            input_title: "リプレイ入力",
            expected_title: "リプレイ期待結果",
            grader_title: "リプレイ評価器",
            evidence_links_title: "リプレイ証跡リンク",
            grader_heading: "リプレイ評価器",
            checks_heading: "チェック",
            pending_request_check: "pendingRequestCount は、意図した変更でない限り {count} のままにしてください。",
            queued_turn_check: "queuedTurnCount は、意図した変更でない限り {count} のままにしてください。",
            read_model_check: "リプレイは App Server current read model の形を保持する必要があります。",
        },
        analysis: AnalysisExportCopy {
            response_title: "外部分析引き継ぎ",
            brief_artifact_title: "外部分析ブリーフ",
            context_artifact_title: "外部分析コンテキスト",
            markdown_heading: "外部分析引き継ぎ",
            focus_heading: "焦点",
            focus_current_read_model: "現在の App Server read model を確認し、次の実装単位を決めてください。",
            focus_no_legacy: "legacy agent_runtime_* command 出力を本番証跡として使わないでください。",
        },
        review: ReviewExportCopy {
            response_title: "レビュー判断",
            markdown_artifact_title: "レビュー判断",
            json_artifact_title: "レビュー判断 JSON",
            markdown_heading: "レビュー判断",
            summary_heading: "サマリー",
            pending_summary: "人によるレビュー待ちです。",
            followup_heading: "フォローアップ",
            no_followup: "記録はありません。",
            regression_heading: "回帰要件",
            default_regression_requirement: "current path の対象回帰を実行してください。",
            notes_heading: "メモ",
            default_decision_regression_requirement: "accepted にする前に current path の対象回帰を実行してください。",
            checklist: &[
                "App Server current パスの証跡を確認する。",
                "本番パスに legacy agent_runtime_* fallback が不要であることを確認する。",
                "受け入れ前に対象回帰を実行する。",
            ],
        },
        rollout: RolloutSummaryCopy {
            runtime_facts_heading: "ランタイム事実",
            candidate_memory_heading: "候補メモリー",
            review_before_promoting: "長期メモリーに昇格する前に、このエクスポートルートを確認してください。",
            pending_work_label: "未完了の作業があります",
            pending_label: "未着手",
            in_progress_label: "進行中",
            referenced_artifacts_heading: "参照された成果物",
        },
        generation_brief: GenerationBriefCopy {
            heading: "Generation Brief 境界",
            voice_source_label: "正式成果物の声線ソース",
            default_voice: "デフォルトでは中立で監査可能なエクスポート文体を使用します。",
            explicit_voice: "正式成果物本文に影響できるのは、明示的な generation_brief / creator voice / brand voice のみです。",
            fidelity_rule: "Product Soul は対話表現のみに影響し、正式 artifact、エクスポート本文、レポート本文を自動で書き換えません。",
        },
    }
}

fn ko_kr_copy() -> RuntimeExportCopy {
    RuntimeExportCopy {
        locale: Locale::KoKr,
        handoff: HandoffCopy {
            plan_title: "계획 요약",
            progress_title: "구조화된 진행 상황",
            handoff_title: "인수인계 요약",
            review_summary_title: "리뷰 요약",
            session_label: "세션",
            thread_label: "스레드",
            status_label: "상태",
            exported_at_label: "내보낸 시간",
            todo_summary_title: "Todo 요약",
            recent_artifacts_title: "최근 산출물",
            no_recent_artifacts: "참조할 최근 산출물이 없습니다.",
            next_step_title: "권장 인수인계 순서",
            next_step_body: "먼저 progress.json에서 구조화된 상태를 확인한 뒤 handoff.md에서 다음 작업을 결정하세요.",
            review_note: "이 요약은 App Server current read model에서 생성되었습니다. legacy command 출력을 납품 증거로 사용하지 마세요.",
        },
        replay: ReplayExportCopy {
            input_title: "리플레이 입력",
            expected_title: "리플레이 예상 결과",
            grader_title: "리플레이 채점기",
            evidence_links_title: "리플레이 증거 링크",
            grader_heading: "리플레이 채점기",
            checks_heading: "검사 항목",
            pending_request_check: "의도한 변경이 아니라면 pendingRequestCount는 {count}로 유지되어야 합니다.",
            queued_turn_check: "의도한 변경이 아니라면 queuedTurnCount는 {count}로 유지되어야 합니다.",
            read_model_check: "리플레이는 App Server current read model 형태를 보존해야 합니다.",
        },
        analysis: AnalysisExportCopy {
            response_title: "외부 분석 인수인계",
            brief_artifact_title: "외부 분석 브리프",
            context_artifact_title: "외부 분석 컨텍스트",
            markdown_heading: "외부 분석 인수인계",
            focus_heading: "중점",
            focus_current_read_model: "현재 App Server read model을 검토하고 다음 구현 단위를 결정하세요.",
            focus_no_legacy: "legacy agent_runtime_* command 출력을 프로덕션 증거로 사용하지 마세요.",
        },
        review: ReviewExportCopy {
            response_title: "리뷰 결정",
            markdown_artifact_title: "리뷰 결정",
            json_artifact_title: "리뷰 결정 JSON",
            markdown_heading: "리뷰 결정",
            summary_heading: "요약",
            pending_summary: "사람의 리뷰를 기다리는 중입니다.",
            followup_heading: "후속 작업",
            no_followup: "기록 없음.",
            regression_heading: "회귀 요구 사항",
            default_regression_requirement: "current path 대상 회귀를 실행하세요.",
            notes_heading: "메모",
            default_decision_regression_requirement: "accepted로 표시하기 전에 current path 대상 회귀를 실행하세요.",
            checklist: &[
                "App Server current 경로 증거를 확인하세요.",
                "프로덕션 경로에 legacy agent_runtime_* fallback이 필요 없는지 확인하세요.",
                "수락 전에 대상 회귀를 실행하세요.",
            ],
        },
        rollout: RolloutSummaryCopy {
            runtime_facts_heading: "런타임 사실",
            candidate_memory_heading: "후보 메모리",
            review_before_promoting: "장기 메모리로 승격하기 전에 이 내보내기 루트를 검토하세요.",
            pending_work_label: "남은 작업",
            pending_label: "대기",
            in_progress_label: "진행 중",
            referenced_artifacts_heading: "참조된 산출물",
        },
        generation_brief: GenerationBriefCopy {
            heading: "Generation Brief 경계",
            voice_source_label: "정식 산출물 보이스 출처",
            default_voice: "기본값은 중립적이고 감사 가능한 내보내기 문체입니다.",
            explicit_voice: "정식 산출물 본문에는 명시적인 generation_brief / creator voice / brand voice만 영향을 줄 수 있습니다.",
            fidelity_rule: "Product Soul은 상호작용 표현에만 영향을 주며 정식 artifact, 내보내기 본문 또는 보고서 본문을 자동으로 다시 쓰지 않습니다.",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_export_copy_covers_current_locales() {
        for locale in ["zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"] {
            let copy = runtime_export_copy(Some(locale));
            assert!(!copy.handoff.handoff_title.is_empty());
            assert!(!copy.analysis.response_title.is_empty());
            assert!(!copy.review.response_title.is_empty());
            assert_eq!(copy.formal_artifact_voice_source(), "generation_brief_only");
        }
    }

    #[test]
    fn runtime_export_copy_keeps_formal_artifacts_out_of_product_soul() {
        let copy = runtime_export_copy(Some("en-US"));

        assert!(copy.generation_brief.fidelity_rule.contains("Product Soul"));
        assert!(copy
            .analysis_copy_prompt("session-1", "analysis", "replay")
            .contains("Generation Brief"));
        assert!(!copy
            .analysis_copy_prompt("session-1", "analysis", "replay")
            .contains("cheeky_sassy_executor"));
    }
}
