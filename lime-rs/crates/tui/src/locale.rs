use std::borrow::Cow;

use crate::slash_command::SlashCommand;
use std::env;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Locale {
    ZhCn,
    ZhTw,
    #[default]
    EnUs,
    JaJp,
    KoKr,
}

impl Locale {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().replace('_', "-").to_ascii_lowercase();
        if normalized.is_empty() {
            return None;
        }
        if normalized == "zh-tw"
            || normalized.starts_with("zh-hant")
            || normalized.starts_with("zh-tw-")
        {
            return Some(Self::ZhTw);
        }
        if normalized == "zh"
            || normalized == "zh-cn"
            || normalized.starts_with("zh-hans")
            || normalized.starts_with("zh-cn-")
        {
            return Some(Self::ZhCn);
        }
        if normalized == "ja" || normalized.starts_with("ja-") {
            return Some(Self::JaJp);
        }
        if normalized == "ko" || normalized.starts_with("ko-") {
            return Some(Self::KoKr);
        }
        if normalized == "en" || normalized.starts_with("en-") {
            return Some(Self::EnUs);
        }
        None
    }

    pub(crate) fn resolve(explicit: Option<&str>) -> Self {
        explicit
            .and_then(Self::parse)
            .or_else(|| {
                env::var("LIME_LOCALE")
                    .ok()
                    .and_then(|value| Self::parse(&value))
            })
            .or_else(|| {
                env::var("LC_ALL")
                    .ok()
                    .and_then(|value| Self::parse(&value))
            })
            .or_else(|| env::var("LANG").ok().and_then(|value| Self::parse(&value)))
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Self::ZhCn => "zh-CN",
            Self::ZhTw => "zh-TW",
            Self::EnUs => "en-US",
            Self::JaJp => "ja-JP",
            Self::KoKr => "ko-KR",
        }
    }

    pub(crate) fn model_label(self) -> &'static str {
        match self {
            Self::ZhCn | Self::ZhTw => "模型",
            Self::EnUs => "model",
            Self::JaJp => "モデル",
            Self::KoKr => "모델",
        }
    }

    pub(crate) fn effort_label(self) -> &'static str {
        match self {
            Self::ZhCn | Self::ZhTw => "推理",
            Self::EnUs => "effort",
            Self::JaJp => "推論",
            Self::KoKr => "추론",
        }
    }

    pub(crate) fn permissions_label(self) -> &'static str {
        match self {
            Self::ZhCn => "权限",
            Self::ZhTw => "權限",
            Self::EnUs => "permissions",
            Self::JaJp => "権限",
            Self::KoKr => "권한",
        }
    }

    pub(crate) fn ready_label(self) -> &'static str {
        match self {
            Self::ZhCn => "就绪",
            Self::ZhTw => "就緒",
            Self::EnUs => "ready",
            Self::JaJp => "準備完了",
            Self::KoKr => "준비됨",
        }
    }

    pub(crate) fn working_label(self) -> &'static str {
        match self {
            Self::ZhCn => "处理中",
            Self::ZhTw => "處理中",
            Self::EnUs => "Working",
            Self::JaJp => "処理中",
            Self::KoKr => "작업 중",
        }
    }

    pub(crate) fn interrupt_hint(self) -> &'static str {
        match self {
            Self::ZhCn => "Esc 中断",
            Self::ZhTw => "Esc 中斷",
            Self::EnUs => "esc to interrupt",
            Self::JaJp => "Esc で中断",
            Self::KoKr => "Esc로 중단",
        }
    }

    pub(crate) fn turn_label(self) -> &'static str {
        match self {
            Self::ZhCn | Self::ZhTw => "回合",
            Self::EnUs => "turn",
            Self::JaJp => "ターン",
            Self::KoKr => "턴",
        }
    }

    pub(crate) fn history_search_label(self) -> &'static str {
        match self {
            Self::ZhCn => "反向搜索：",
            Self::ZhTw => "反向搜尋：",
            Self::EnUs => "reverse-i-search: ",
            Self::JaJp => "履歴逆検索: ",
            Self::KoKr => "기록 역검색: ",
        }
    }

    pub(crate) fn slash_command_description(self, command: SlashCommand) -> &'static str {
        match (self, command) {
            (Self::ZhCn, SlashCommand::Model) => "选择模型",
            (Self::ZhTw, SlashCommand::Model) => "選擇模型",
            (Self::EnUs, SlashCommand::Model) => "choose a model",
            (Self::JaJp, SlashCommand::Model) => "モデルを選択",
            (Self::KoKr, SlashCommand::Model) => "모델 선택",
            (Self::ZhCn, SlashCommand::Effort) => "设置推理强度",
            (Self::ZhTw, SlashCommand::Effort) => "設定推理強度",
            (Self::EnUs, SlashCommand::Effort) => "set the reasoning effort",
            (Self::JaJp, SlashCommand::Effort) => "推論強度を設定",
            (Self::KoKr, SlashCommand::Effort) => "추론 강도 설정",
            (Self::ZhCn, SlashCommand::Permissions) => "设置权限配置",
            (Self::ZhTw, SlashCommand::Permissions) => "設定權限設定檔",
            (Self::EnUs, SlashCommand::Permissions) => "set the permission profile",
            (Self::JaJp, SlashCommand::Permissions) => "権限プロファイルを設定",
            (Self::KoKr, SlashCommand::Permissions) => "권한 프로필 설정",
            (Self::ZhCn, SlashCommand::Status) => "查看当前会话状态",
            (Self::ZhTw, SlashCommand::Status) => "檢視目前工作階段狀態",
            (Self::EnUs, SlashCommand::Status) => "show the current session status",
            (Self::JaJp, SlashCommand::Status) => "現在のセッション状態を表示",
            (Self::KoKr, SlashCommand::Status) => "현재 세션 상태 보기",
            (Self::ZhCn, SlashCommand::Copy) => "复制最后一条回复",
            (Self::ZhTw, SlashCommand::Copy) => "複製最後一則回覆",
            (Self::EnUs, SlashCommand::Copy) => "copy the last response",
            (Self::JaJp, SlashCommand::Copy) => "最後の応答をコピー",
            (Self::KoKr, SlashCommand::Copy) => "마지막 응답 복사",
        }
    }

    pub(crate) fn status_title(self) -> &'static str {
        match self {
            Self::ZhCn => "状态",
            Self::ZhTw => "狀態",
            Self::EnUs => "STATUS",
            Self::JaJp => "状態",
            Self::KoKr => "상태",
        }
    }

    pub(crate) fn transcript_title(self) -> &'static str {
        match self {
            Self::ZhCn => "对话记录",
            Self::ZhTw => "對話記錄",
            Self::EnUs => "T R A N S C R I P T",
            Self::JaJp => "会話履歴",
            Self::KoKr => "대화 기록",
        }
    }

    pub(crate) fn thread_label(self) -> &'static str {
        match self {
            Self::ZhCn => "会话",
            Self::ZhTw => "工作階段",
            Self::EnUs => "thread",
            Self::JaJp => "スレッド",
            Self::KoKr => "스레드",
        }
    }

    pub(crate) fn provider_label(self) -> &'static str {
        match self {
            Self::ZhCn | Self::ZhTw => "Provider",
            Self::EnUs => "provider",
            Self::JaJp => "プロバイダー",
            Self::KoKr => "프로바이더",
        }
    }

    pub(crate) fn cwd_label(self) -> &'static str {
        match self {
            Self::ZhCn => "工作目录",
            Self::ZhTw => "工作目錄",
            Self::EnUs => "working directory",
            Self::JaJp => "作業ディレクトリ",
            Self::KoKr => "작업 디렉터리",
        }
    }

    pub(crate) fn state_label(self) -> &'static str {
        match self {
            Self::ZhCn => "状态",
            Self::ZhTw => "狀態",
            Self::EnUs => "status",
            Self::JaJp => "状態",
            Self::KoKr => "상태",
        }
    }

    pub(crate) fn not_set_label(self) -> &'static str {
        match self {
            Self::ZhCn => "未设置",
            Self::ZhTw => "未設定",
            Self::EnUs => "not set",
            Self::JaJp => "未設定",
            Self::KoKr => "설정되지 않음",
        }
    }

    pub(crate) fn image_label(self) -> &'static str {
        match self {
            Self::ZhCn => "图片",
            Self::ZhTw => "圖片",
            Self::EnUs => "image",
            Self::JaJp => "画像",
            Self::KoKr => "이미지",
        }
    }

    pub(crate) fn edit_queued_input_hint(self) -> &'static str {
        match self {
            Self::ZhCn => "Alt+Up 编辑最后一条排队输入",
            Self::ZhTw => "Alt+Up 編輯最後一則排隊輸入",
            Self::EnUs => "Alt+Up edit last queued input",
            Self::JaJp => "Alt+Up 最後のキュー入力を編集",
            Self::KoKr => "Alt+Up 마지막 대기 입력 편집",
        }
    }

    pub(crate) fn pager_footer(self) -> &'static str {
        match self {
            Self::ZhCn => "上下滚动  PgUp/PgDn 翻页  Home/End 跳转  Esc/Q 关闭",
            Self::ZhTw => "上下捲動  PgUp/PgDn 翻頁  Home/End 跳轉  Esc/Q 關閉",
            Self::EnUs => "Up/Down scroll  PgUp/PgDn page  Home/End jump  Esc/Q close",
            Self::JaJp => "上下スクロール  PgUp/PgDn ページ  Home/End 移動  Esc/Q 閉じる",
            Self::KoKr => "위/아래 스크롤  PgUp/PgDn 페이지  Home/End 이동  Esc/Q 닫기",
        }
    }

    pub(crate) fn transcript_pager_footer(self) -> &'static str {
        match self {
            Self::ZhCn => "上下滚动  PgUp/PgDn 翻页  Home/End 跳转  Ctrl+T/Esc/Q 关闭",
            Self::ZhTw => "上下捲動  PgUp/PgDn 翻頁  Home/End 跳轉  Ctrl+T/Esc/Q 關閉",
            Self::EnUs => "Up/Down scroll  PgUp/PgDn page  Home/End jump  Ctrl+T/Esc/Q close",
            Self::JaJp => "上下スクロール  PgUp/PgDn ページ  Home/End 移動  Ctrl+T/Esc/Q 閉じる",
            Self::KoKr => "위/아래 스크롤  PgUp/PgDn 페이지  Home/End 이동  Ctrl+T/Esc/Q 닫기",
        }
    }

    pub(crate) fn status(self, status: &str) -> String {
        let (prefix, rest) = status
            .split_once(':')
            .map_or((status, None), |(prefix, rest)| {
                (prefix, Some(rest.trim_start()))
            });
        let label = match prefix {
            "" => return String::new(),
            "ready" => self.ready_label(),
            "running" => match self {
                Self::ZhCn => "运行中",
                Self::ZhTw => "執行中",
                Self::EnUs => "running",
                Self::JaJp => "実行中",
                Self::KoKr => "실행 중",
            },
            "completed" => match self {
                Self::ZhCn | Self::ZhTw => "已完成",
                Self::EnUs => "completed",
                Self::JaJp => "完了",
                Self::KoKr => "완료",
            },
            "failed" => match self {
                Self::ZhCn => "失败",
                Self::ZhTw => "失敗",
                Self::EnUs => "failed",
                Self::JaJp => "失敗",
                Self::KoKr => "실패",
            },
            "interrupted" => match self {
                Self::ZhCn => "已中断",
                Self::ZhTw => "已中斷",
                Self::EnUs => "interrupted",
                Self::JaJp => "中断",
                Self::KoKr => "중단됨",
            },
            "declined" => match self {
                Self::ZhCn => "已拒绝",
                Self::ZhTw => "已拒絕",
                Self::EnUs => "declined",
                Self::JaJp => "拒否",
                Self::KoKr => "거부됨",
            },
            "settings updated" => match self {
                Self::ZhCn => "设置已更新",
                Self::ZhTw => "設定已更新",
                Self::EnUs => "settings updated",
                Self::JaJp => "設定を更新しました",
                Self::KoKr => "설정이 업데이트됨",
            },
            "choose model" => match self {
                Self::ZhCn => "请选择模型",
                Self::ZhTw => "請選擇模型",
                Self::EnUs => "choose model",
                Self::JaJp => "モデルを選択",
                Self::KoKr => "모델 선택",
            },
            "steering" => match self {
                Self::ZhCn => "正在引导",
                Self::ZhTw => "正在引導",
                Self::EnUs => "steering",
                Self::JaJp => "ステアリング中",
                Self::KoKr => "조정 중",
            },
            "queued" => match self {
                Self::ZhCn => "已排队",
                Self::ZhTw => "已排隊",
                Self::EnUs => "queued",
                Self::JaJp => "キューに追加済み",
                Self::KoKr => "대기열에 추가됨",
            },
            "queue unavailable" => match self {
                Self::ZhCn => "队列不可用",
                Self::ZhTw => "佇列無法使用",
                Self::EnUs => "queue unavailable",
                Self::JaJp => "キューを利用できません",
                Self::KoKr => "대기열을 사용할 수 없음",
            },
            "queued input editing" => match self {
                Self::ZhCn => "正在编辑排队输入",
                Self::ZhTw => "正在編輯排隊輸入",
                Self::EnUs => "editing queued input",
                Self::JaJp => "キュー入力を編集中",
                Self::KoKr => "대기 입력 편집 중",
            },
            "queued input unavailable" => match self {
                Self::ZhCn => "排队输入已不可用",
                Self::ZhTw => "排隊輸入已無法使用",
                Self::EnUs => "queued input unavailable",
                Self::JaJp => "キュー入力を利用できません",
                Self::KoKr => "대기 입력을 사용할 수 없음",
            },
            "queue edit failed" => match self {
                Self::ZhCn => "编辑排队输入失败",
                Self::ZhTw => "編輯排隊輸入失敗",
                Self::EnUs => "queue edit failed",
                Self::JaJp => "キュー入力の編集に失敗しました",
                Self::KoKr => "대기 입력 편집 실패",
            },
            "interrupting" => match self {
                Self::ZhCn => "正在中断",
                Self::ZhTw => "正在中斷",
                Self::EnUs => "interrupting",
                Self::JaJp => "中断中",
                Self::KoKr => "중단 중",
            },
            "editor draft empty" => match self {
                Self::ZhCn => "编辑器草稿为空",
                Self::ZhTw => "編輯器草稿為空",
                Self::EnUs => "editor draft empty",
                Self::JaJp => "エディターの下書きが空です",
                Self::KoKr => "편집기 초안이 비어 있음",
            },
            "copied last response" => match self {
                Self::ZhCn => "已复制上一条回复",
                Self::ZhTw => "已複製上一則回覆",
                Self::EnUs => "copied last response",
                Self::JaJp => "前回の応答をコピーしました",
                Self::KoKr => "마지막 응답을 복사함",
            },
            "no agent response to copy" => match self {
                Self::ZhCn => "没有可复制的 Agent 回复",
                Self::ZhTw => "沒有可複製的 Agent 回覆",
                Self::EnUs => "no agent response to copy",
                Self::JaJp => "コピーできる Agent 応答がありません",
                Self::KoKr => "복사할 Agent 응답이 없음",
            },
            "copy failed" => match self {
                Self::ZhCn => "复制失败",
                Self::ZhTw => "複製失敗",
                Self::EnUs => "copy failed",
                Self::JaJp => "コピーに失敗しました",
                Self::KoKr => "복사 실패",
            },
            "image attached" => match self {
                Self::ZhCn => "已附加图片",
                Self::ZhTw => "已附加圖片",
                Self::EnUs => "image attached",
                Self::JaJp => "画像を添付しました",
                Self::KoKr => "이미지 첨부됨",
            },
            "image paste failed" => match self {
                Self::ZhCn => "粘贴图片失败",
                Self::ZhTw => "貼上圖片失敗",
                Self::EnUs => "image paste failed",
                Self::JaJp => "画像の貼り付けに失敗しました",
                Self::KoKr => "이미지 붙여넣기 실패",
            },
            "reconnecting" => match self {
                Self::ZhCn => "正在重连",
                Self::ZhTw => "正在重新連線",
                Self::EnUs => "reconnecting",
                Self::JaJp => "再接続中",
                Self::KoKr => "재연결 중",
            },
            "effort" => self.effort_label(),
            "permissions" => self.permissions_label(),
            "prompt history unavailable" => match self {
                Self::ZhCn => "提示历史不可用",
                Self::ZhTw => "提示歷史不可用",
                Self::EnUs => "prompt history unavailable",
                Self::JaJp => "プロンプト履歴を利用できません",
                Self::KoKr => "프롬프트 기록을 사용할 수 없음",
            },
            "editor unavailable" => match self {
                Self::ZhCn => "编辑器不可用",
                Self::ZhTw => "編輯器不可用",
                Self::EnUs => "editor unavailable",
                Self::JaJp => "エディターを利用できません",
                Self::KoKr => "편집기를 사용할 수 없음",
            },
            _ => return status.to_string(),
        };
        rest.map_or_else(|| label.to_string(), |rest| format!("{label}: {rest}"))
    }

    pub(crate) fn detail(self, detail: &str) -> String {
        const PREFIXES: &[(&str, &str, &str, &str, &str)] = &[
            ("exit ", "退出 ", "退出 ", "exit ", "종료 "),
            ("duration ", "耗时 ", "耗時 ", "所要時間 ", "소요 시간 "),
            (
                "files: ",
                "文件数：",
                "檔案數：",
                "ファイル数: ",
                "파일 수: ",
            ),
            ("added: ", "新增：", "新增：", "追加: ", "추가: "),
            ("deleted: ", "删除：", "刪除：", "削除: ", "삭제: "),
            ("updated: ", "更新：", "更新：", "更新: ", "업데이트: "),
            (
                "result items: ",
                "结果项：",
                "結果項：",
                "結果項目: ",
                "결과 항목: ",
            ),
            ("error: ", "错误：", "錯誤：", "エラー: ", "오류: "),
            ("success: ", "成功：", "成功：", "成功: ", "성공: "),
            (
                "content items: ",
                "内容项：",
                "內容項：",
                "内容項目: ",
                "콘텐츠 항목: ",
            ),
            (
                "agents: ",
                "Agent 数：",
                "Agent 數：",
                "Agent 数: ",
                "에이전트 수: ",
            ),
            ("model: ", "模型：", "模型：", "モデル: ", "모델: "),
            (
                "effort: ",
                "思考强度：",
                "思考強度：",
                "推論強度: ",
                "추론 강도: ",
            ),
            (
                "prompt: ",
                "提示词：",
                "提示詞：",
                "プロンプト: ",
                "프롬프트: ",
            ),
            (
                "web search: ",
                "网页搜索：",
                "網頁搜尋：",
                "ウェブ検索: ",
                "웹 검색: ",
            ),
            (
                "view image: ",
                "查看图片：",
                "檢視圖片：",
                "画像を表示: ",
                "이미지 보기: ",
            ),
            ("sleep: ", "休眠：", "休眠：", "スリープ: ", "대기: "),
            ("result: ", "结果：", "結果：", "結果: ", "결과: "),
            ("saved: ", "已保存：", "已儲存：", "保存済み: ", "저장됨: "),
            (
                "revised prompt: ",
                "修订提示：",
                "修訂提示：",
                "修正プロンプト: ",
                "수정된 프롬프트: ",
            ),
            (
                "review started: ",
                "审查开始：",
                "審查開始：",
                "レビュー開始: ",
                "검토 시작: ",
            ),
            (
                "review completed: ",
                "审查完成：",
                "審查完成：",
                "レビュー完成: ",
                "검토 완료: ",
            ),
            (
                "context compacted",
                "上下文已压缩",
                "內容已壓縮",
                "コンテキストを圧縮しました",
                "컨텍스트가 압축됨",
            ),
            (
                "image generation",
                "图片生成",
                "圖片生成",
                "画像生成",
                "이미지 생성",
            ),
            (
                "unsupported item: ",
                "不支持的条目：",
                "不支援的項目：",
                "未対応項目: ",
                "지원되지 않는 항목: ",
            ),
            ("pending: ", "待处理：", "待處理：", "保留中: ", "대기 중: "),
            ("running: ", "运行中：", "執行中：", "実行中: ", "실행 중: "),
            (
                "interrupted: ",
                "已中断：",
                "已中斷：",
                "中断: ",
                "중단됨: ",
            ),
            ("completed: ", "已完成：", "已完成：", "完了: ", "완료: "),
            ("errored: ", "出错：", "發生錯誤：", "エラー: ", "오류: "),
            (
                "shutdown: ",
                "已关闭：",
                "已關閉：",
                "シャットダウン: ",
                "종료됨: ",
            ),
            (
                "not-found: ",
                "未找到：",
                "找不到：",
                "見つかりません: ",
                "찾을 수 없음: ",
            ),
        ];
        PREFIXES
            .iter()
            .find_map(|(prefix, zh_cn, zh_tw, ja, ko)| {
                detail.strip_prefix(prefix).map(|rest| {
                    let label = match self {
                        Self::ZhCn => zh_cn,
                        Self::ZhTw => zh_tw,
                        Self::EnUs => prefix,
                        Self::JaJp => ja,
                        Self::KoKr => ko,
                    };
                    format!("{label}{rest}")
                })
            })
            .unwrap_or_else(|| detail.to_string())
    }

    pub(crate) fn output_omitted_lines(self, count: usize) -> String {
        match self {
            Self::ZhCn => format!("… 已省略 {count} 行 …"),
            Self::ZhTw => format!("… 已省略 {count} 行 …"),
            Self::EnUs => format!("… {count} lines omitted …"),
            Self::JaJp => format!("… {count} 行を省略 …"),
            Self::KoKr => format!("… {count}줄 생략 …"),
        }
    }

    pub(crate) fn output_omitted_bytes(self, count: usize) -> String {
        match self {
            Self::ZhCn => format!("… 已省略 {count} 字节 …"),
            Self::ZhTw => format!("… 已省略 {count} 位元組 …"),
            Self::EnUs => format!("… {count} bytes omitted …"),
            Self::JaJp => format!("… {count} バイトを省略 …"),
            Self::KoKr => format!("… {count}바이트 생략 …"),
        }
    }

    pub(crate) fn approval_title(self, kind: &str) -> &'static str {
        match (self, kind) {
            (Self::ZhCn, "command") => "批准命令？",
            (Self::ZhTw, "command") => "核准命令？",
            (Self::JaJp, "command") => "コマンドを承認しますか？",
            (Self::KoKr, "command") => "명령을 승인할까요?",
            (Self::ZhCn, "file") => "批准文件更改？",
            (Self::ZhTw, "file") => "核准檔案變更？",
            (Self::JaJp, "file") => "ファイル変更を承認しますか？",
            (Self::KoKr, "file") => "파일 변경을 승인할까요?",
            (Self::ZhCn, "permissions") => "授予额外权限？",
            (Self::ZhTw, "permissions") => "授予額外權限？",
            (Self::JaJp, "permissions") => "追加権限を付与しますか？",
            (Self::KoKr, "permissions") => "추가 권한을 부여할까요?",
            (_, "command") => "Approve command?",
            (_, "file") => "Approve file changes?",
            _ => "Grant additional permissions?",
        }
    }

    pub(crate) fn approval_option(self, label: &str) -> String {
        match (self, label) {
            (Self::ZhCn, "Accept") => "批准".to_string(),
            (Self::ZhTw, "Accept") => "核准".to_string(),
            (Self::JaJp, "Accept") => "承認".to_string(),
            (Self::KoKr, "Accept") => "승인".to_string(),
            (Self::ZhCn, "Decline") => "拒绝".to_string(),
            (Self::ZhTw, "Decline") => "拒絕".to_string(),
            (Self::JaJp, "Decline") => "拒否".to_string(),
            (Self::KoKr, "Decline") => "거부".to_string(),
            (Self::ZhCn, "Cancel") => "取消".to_string(),
            (Self::ZhTw, "Cancel") => "取消".to_string(),
            (Self::JaJp, "Cancel") => "キャンセル".to_string(),
            (Self::KoKr, "Cancel") => "취소".to_string(),
            (Self::ZhCn, "Allow once") => "允许一次".to_string(),
            (Self::ZhTw, "Allow once") => "允許一次".to_string(),
            (Self::JaJp, "Allow once") => "一度だけ許可".to_string(),
            (Self::KoKr, "Allow once") => "한 번 허용".to_string(),
            (Self::ZhCn, "Allow for this session") => "允许本会话".to_string(),
            (Self::ZhTw, "Allow for this session") => "允許本工作階段".to_string(),
            (Self::JaJp, "Allow for this session") => "このセッションで許可".to_string(),
            (Self::KoKr, "Allow for this session") => "이 세션에 허용".to_string(),
            (Self::ZhCn, "Cancel turn") => "取消本回合".to_string(),
            (Self::ZhTw, "Cancel turn") => "取消本回合".to_string(),
            (Self::JaJp, "Cancel turn") => "ターンをキャンセル".to_string(),
            (Self::KoKr, "Cancel turn") => "턴 취소".to_string(),
            (Self::ZhCn, "Grant for this turn") => "仅授予本回合".to_string(),
            (Self::ZhTw, "Grant for this turn") => "僅授予本回合".to_string(),
            (Self::JaJp, "Grant for this turn") => "このターンのみ許可".to_string(),
            (Self::KoKr, "Grant for this turn") => "이 턴에만 허용".to_string(),
            (Self::ZhCn, "Grant for this session") => "授予本会话".to_string(),
            (Self::ZhTw, "Grant for this session") => "授予本工作階段".to_string(),
            (Self::JaJp, "Grant for this session") => "このセッションで許可".to_string(),
            (Self::KoKr, "Grant for this session") => "이 세션에 허용".to_string(),
            _ => label.to_string(),
        }
    }

    pub(crate) fn picker_title(self) -> &'static str {
        match self {
            Self::ZhCn => "选择模型",
            Self::ZhTw => "選擇模型",
            Self::EnUs => "Choose a model",
            Self::JaJp => "モデルを選択",
            Self::KoKr => "모델 선택",
        }
    }

    pub(crate) fn picker_empty(self) -> &'static str {
        match self {
            Self::ZhCn => "没有匹配的模型。按 Esc 取消",
            Self::ZhTw => "沒有符合的模型。按 Esc 取消",
            Self::EnUs => "No matching models. Esc cancel",
            Self::JaJp => "一致するモデルがありません。Esc でキャンセル",
            Self::KoKr => "일치하는 모델이 없습니다. Esc로 취소",
        }
    }

    pub(crate) fn picker_footer(self) -> &'static str {
        match self {
            Self::ZhCn => "上下移动  Enter 选择  Esc 取消",
            Self::ZhTw => "上下移動  Enter 選擇  Esc 取消",
            Self::EnUs => "Up/Down move  Enter select  Esc cancel",
            Self::JaJp => "上下移動  Enter 選択  Esc キャンセル",
            Self::KoKr => "위/아래 이동  Enter 선택  Esc 취소",
        }
    }

    pub(crate) fn resume_title(self) -> &'static str {
        match self {
            Self::ZhCn => "选择会话",
            Self::ZhTw => "選擇工作階段",
            Self::EnUs => "Select a conversation",
            Self::JaJp => "会話を選択",
            Self::KoKr => "대화 선택",
        }
    }

    pub(crate) fn untitled_conversation(self) -> &'static str {
        match self {
            Self::ZhCn => "未命名会话",
            Self::ZhTw => "未命名工作階段",
            Self::EnUs => "Untitled conversation",
            Self::JaJp => "無題の会話",
            Self::KoKr => "제목 없는 대화",
        }
    }

    pub(crate) fn resume_label(self) -> &'static str {
        match self {
            Self::ZhCn => "恢复",
            Self::ZhTw => "恢復",
            Self::EnUs => "Resume",
            Self::JaJp => "再開",
            Self::KoKr => "재개",
        }
    }

    pub(crate) fn thread_state(self, state: &str) -> Cow<'static, str> {
        match (self, state) {
            (Self::ZhCn, "active") => Cow::Borrowed("活动"),
            (Self::ZhTw, "active") => Cow::Borrowed("使用中"),
            (Self::JaJp, "active") => Cow::Borrowed("アクティブ"),
            (Self::KoKr, "active") => Cow::Borrowed("활성"),
            (Self::ZhCn, "idle") => Cow::Borrowed("空闲"),
            (Self::ZhTw, "idle") => Cow::Borrowed("閒置"),
            (Self::JaJp, "idle") => Cow::Borrowed("アイドル"),
            (Self::KoKr, "idle") => Cow::Borrowed("유휴"),
            (Self::ZhCn, "not loaded") => Cow::Borrowed("未加载"),
            (Self::ZhTw, "not loaded") => Cow::Borrowed("未載入"),
            (Self::JaJp, "not loaded") => Cow::Borrowed("未読込"),
            (Self::KoKr, "not loaded") => Cow::Borrowed("로드되지 않음"),
            (Self::ZhCn, "error") => Cow::Borrowed("错误"),
            (Self::ZhTw, "error") => Cow::Borrowed("錯誤"),
            (Self::JaJp, "error") => Cow::Borrowed("エラー"),
            (Self::KoKr, "error") => Cow::Borrowed("오류"),
            (_, state) => Cow::Owned(state.to_string()),
        }
    }

    pub(crate) fn resume_empty(self) -> &'static str {
        match self {
            Self::ZhCn => "没有可恢复的会话。按 Esc 取消。",
            Self::ZhTw => "沒有可恢復的工作階段。按 Esc 取消。",
            Self::EnUs => "No resumable conversations. Press Esc to cancel.",
            Self::JaJp => "再開できる会話がありません。Esc でキャンセル。",
            Self::KoKr => "재개할 수 있는 대화가 없습니다. Esc로 취소.",
        }
    }

    pub(crate) fn resume_loading(self) -> &'static str {
        match self {
            Self::ZhCn => "正在加载会话…",
            Self::ZhTw => "正在載入工作階段…",
            Self::EnUs => "Loading conversations…",
            Self::JaJp => "会話を読み込み中…",
            Self::KoKr => "대화 로드 중…",
        }
    }

    pub(crate) fn resume_status_label(self, archived: bool) -> &'static str {
        match (self, archived) {
            (Self::ZhCn, false) => "活动",
            (Self::ZhCn, true) => "已归档",
            (Self::ZhTw, false) => "使用中",
            (Self::ZhTw, true) => "已封存",
            (Self::EnUs, false) => "Active",
            (Self::EnUs, true) => "Archived",
            (Self::JaJp, false) => "アクティブ",
            (Self::JaJp, true) => "アーカイブ済み",
            (Self::KoKr, false) => "활성",
            (Self::KoKr, true) => "보관됨",
        }
    }

    pub(crate) fn resume_filter_label(self, show_all: bool) -> &'static str {
        match (self, show_all) {
            (Self::ZhCn, false) => "当前目录",
            (Self::ZhCn, true) => "全部目录",
            (Self::ZhTw, false) => "目前目錄",
            (Self::ZhTw, true) => "所有目錄",
            (Self::EnUs, false) => "Current cwd",
            (Self::EnUs, true) => "All directories",
            (Self::JaJp, false) => "現在のディレクトリ",
            (Self::JaJp, true) => "すべてのディレクトリ",
            (Self::KoKr, false) => "현재 디렉터리",
            (Self::KoKr, true) => "모든 디렉터리",
        }
    }

    pub(crate) fn resume_sort_label(self, created: bool) -> &'static str {
        match (self, created) {
            (Self::ZhCn, false) => "按更新时间",
            (Self::ZhCn, true) => "按创建时间",
            (Self::ZhTw, false) => "依更新時間",
            (Self::ZhTw, true) => "依建立時間",
            (Self::EnUs, false) => "Updated",
            (Self::EnUs, true) => "Created",
            (Self::JaJp, false) => "更新日時順",
            (Self::JaJp, true) => "作成日時順",
            (Self::KoKr, false) => "업데이트순",
            (Self::KoKr, true) => "생성순",
        }
    }

    pub(crate) fn resume_density_label(self, dense: bool) -> &'static str {
        match (self, dense) {
            (Self::ZhCn, false) => "紧凑视图",
            (Self::ZhCn, true) => "舒适视图",
            (Self::ZhTw, false) => "緊湊檢視",
            (Self::ZhTw, true) => "舒適檢視",
            (Self::EnUs, false) => "Dense view",
            (Self::EnUs, true) => "Comfortable view",
            (Self::JaJp, false) => "コンパクト表示",
            (Self::JaJp, true) => "標準表示",
            (Self::KoKr, false) => "밀집 보기",
            (Self::KoKr, true) => "편안한 보기",
        }
    }

    pub(crate) fn resume_search_placeholder(self) -> &'static str {
        match self {
            Self::ZhCn => "输入以搜索会话",
            Self::ZhTw => "輸入以搜尋工作階段",
            Self::EnUs => "Type to search conversations",
            Self::JaJp => "入力して会話を検索",
            Self::KoKr => "입력하여 대화 검색",
        }
    }

    pub(crate) fn resume_picker_title(self, fork: bool) -> &'static str {
        match (self, fork) {
            (Self::ZhCn, false) => "恢复之前的会话",
            (Self::ZhCn, true) => "从之前的会话分叉",
            (Self::ZhTw, false) => "恢復先前的工作階段",
            (Self::ZhTw, true) => "從先前的工作階段分支",
            (Self::EnUs, false) => "Resume a previous session",
            (Self::EnUs, true) => "Fork a previous session",
            (Self::JaJp, false) => "以前のセッションを再開",
            (Self::JaJp, true) => "以前のセッションを分岐",
            (Self::KoKr, false) => "이전 세션 재개",
            (Self::KoKr, true) => "이전 세션에서 분기",
        }
    }

    pub(crate) fn resume_enter_hint(self, fork: bool, archived: bool) -> &'static str {
        match (self, fork, archived) {
            (Self::ZhCn, _, true) => "Enter 恢复",
            (Self::ZhCn, false, false) => "Enter 恢复",
            (Self::ZhCn, true, false) => "Enter 分叉",
            (Self::ZhTw, _, true) => "Enter 恢復",
            (Self::ZhTw, false, false) => "Enter 恢復",
            (Self::ZhTw, true, false) => "Enter 分支",
            (Self::EnUs, _, true) => "Enter restore",
            (Self::EnUs, false, false) => "Enter resume",
            (Self::EnUs, true, false) => "Enter fork",
            (Self::JaJp, _, true) => "Enter 再開",
            (Self::JaJp, false, false) => "Enter 再開",
            (Self::JaJp, true, false) => "Enter 分岐",
            (Self::KoKr, _, true) => "Enter 복원",
            (Self::KoKr, false, false) => "Enter 재개",
            (Self::KoKr, true, false) => "Enter 분기",
        }
    }

    pub(crate) fn resume_controls_hint(self) -> &'static str {
        match self {
            Self::ZhCn => {
                "Esc 新建  Ctrl+C 退出  Ctrl+S 状态  Ctrl+F 目录  Ctrl+R 排序  Ctrl+O 密度"
            }
            Self::ZhTw => {
                "Esc 新建  Ctrl+C 離開  Ctrl+S 狀態  Ctrl+F 目錄  Ctrl+R 排序  Ctrl+O 密度"
            }
            Self::EnUs => {
                "Esc new  Ctrl+C quit  Ctrl+S status  Ctrl+F directory  Ctrl+R sort  Ctrl+O density"
            }
            Self::JaJp => {
                "Esc 新規  Ctrl+C 終了  Ctrl+S 状態  Ctrl+F ディレクトリ  Ctrl+R 並び順  Ctrl+O 密度"
            }
            Self::KoKr => {
                "Esc 새로 만들기  Ctrl+C 종료  Ctrl+S 상태  Ctrl+F 디렉터리  Ctrl+R 정렬  Ctrl+O 밀도"
            }
        }
    }

    pub(crate) fn resume_expand_hint(self) -> &'static str {
        match self {
            Self::ZhCn => "Ctrl+E 展开记录",
            Self::ZhTw => "Ctrl+E 展開記錄",
            Self::EnUs => "Ctrl+E expand transcript",
            Self::JaJp => "Ctrl+E 履歴を展開",
            Self::KoKr => "Ctrl+E 대화 기록 펼치기",
        }
    }

    pub(crate) fn created_label(self) -> &'static str {
        match self {
            Self::ZhCn => "创建时间",
            Self::ZhTw => "建立時間",
            Self::EnUs => "created",
            Self::JaJp => "作成日時",
            Self::KoKr => "생성 시간",
        }
    }

    pub(crate) fn updated_label(self) -> &'static str {
        match self {
            Self::ZhCn => "更新时间",
            Self::ZhTw => "更新時間",
            Self::EnUs => "updated",
            Self::JaJp => "更新日時",
            Self::KoKr => "업데이트 시간",
        }
    }

    pub(crate) fn resume_transcript_loading(self) -> &'static str {
        match self {
            Self::ZhCn => "正在加载对话记录…",
            Self::ZhTw => "正在載入對話記錄…",
            Self::EnUs => "Loading transcript…",
            Self::JaJp => "会話履歴を読み込み中…",
            Self::KoKr => "대화 기록 로드 중…",
        }
    }

    pub(crate) fn resume_transcript_failed(self) -> &'static str {
        match self {
            Self::ZhCn => "无法加载对话记录",
            Self::ZhTw => "無法載入對話記錄",
            Self::EnUs => "Could not load transcript",
            Self::JaJp => "会話履歴を読み込めません",
            Self::KoKr => "대화 기록을 불러올 수 없음",
        }
    }

    pub(crate) fn resume_transcript_empty(self) -> &'static str {
        match self {
            Self::ZhCn => "没有对话内容",
            Self::ZhTw => "沒有對話內容",
            Self::EnUs => "No transcript content",
            Self::JaJp => "会話内容がありません",
            Self::KoKr => "대화 내용 없음",
        }
    }

    pub(crate) fn no_questions(self) -> &'static str {
        match self {
            Self::ZhCn => "没有问题",
            Self::ZhTw => "沒有問題",
            Self::EnUs => "No questions",
            Self::JaJp => "質問はありません",
            Self::KoKr => "질문 없음",
        }
    }

    pub(crate) fn other_option(self) -> &'static str {
        match self {
            Self::ZhCn => "其他  输入自定义答案",
            Self::ZhTw => "其他  輸入自訂答案",
            Self::EnUs => "Other  Type a custom answer",
            Self::JaJp => "その他  カスタム回答を入力",
            Self::KoKr => "기타  사용자 지정 답변 입력",
        }
    }

    pub(crate) fn add_notes(self) -> &'static str {
        match self {
            Self::ZhCn => "按 Tab 添加备注",
            Self::ZhTw => "按 Tab 新增備註",
            Self::EnUs => "Tab to add notes",
            Self::JaJp => "Tab でメモを追加",
            Self::KoKr => "Tab으로 메모 추가",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Locale;
    use crate::slash_command::SlashCommand;

    #[test]
    fn parses_supported_locale_families_and_falls_back_to_english() {
        assert_eq!(Locale::parse("zh_CN"), Some(Locale::ZhCn));
        assert_eq!(Locale::parse("zh-Hant-TW"), Some(Locale::ZhTw));
        assert_eq!(Locale::parse("ja"), Some(Locale::JaJp));
        assert_eq!(Locale::parse("ko-KR"), Some(Locale::KoKr));
        assert_eq!(Locale::parse("en-GB"), Some(Locale::EnUs));
        assert_eq!(Locale::parse("fr-FR"), None);
    }

    #[test]
    fn translates_statuses_and_dynamic_detail_labels_without_changing_unknown_text() {
        assert_eq!(Locale::ZhCn.status("ready"), "就绪");
        assert_eq!(Locale::JaJp.status("effort: high"), "推論: high");
        assert_eq!(
            Locale::ZhTw.status("image attached: 20x10"),
            "已附加圖片: 20x10"
        );
        assert_eq!(
            Locale::KoKr.status("image paste failed: unavailable"),
            "이미지 붙여넣기 실패: unavailable"
        );
        assert_eq!(Locale::KoKr.detail("files: 2"), "파일 수: 2");
        assert_eq!(Locale::ZhCn.detail("model: fixture"), "模型：fixture");
        assert_eq!(Locale::ZhTw.detail("effort: high"), "思考強度：high");
        assert_eq!(Locale::EnUs.detail("prompt: inspect"), "prompt: inspect");
        assert_eq!(Locale::JaJp.detail("model: fixture"), "モデル: fixture");
        assert_eq!(Locale::KoKr.detail("effort: high"), "추론 강도: high");
        assert_eq!(Locale::ZhTw.detail("custom: value"), "custom: value");
        assert_eq!(Locale::ZhCn.approval_option("Allow once"), "允许一次");
        assert_eq!(
            Locale::JaJp.approval_option("Cancel turn"),
            "ターンをキャンセル"
        );
    }

    #[test]
    fn command_output_omission_markers_cover_all_product_locales() {
        let line_markers = [
            (Locale::ZhCn, "已省略 2 行"),
            (Locale::ZhTw, "已省略 2 行"),
            (Locale::EnUs, "2 lines omitted"),
            (Locale::JaJp, "2 行を省略"),
            (Locale::KoKr, "2줄 생략"),
        ];
        let byte_markers = [
            (Locale::ZhCn, "已省略 3 字节"),
            (Locale::ZhTw, "已省略 3 位元組"),
            (Locale::EnUs, "3 bytes omitted"),
            (Locale::JaJp, "3 バイトを省略"),
            (Locale::KoKr, "3바이트 생략"),
        ];

        for (locale, marker) in line_markers {
            assert!(locale.output_omitted_lines(2).contains(marker));
        }
        for (locale, marker) in byte_markers {
            assert!(locale.output_omitted_bytes(3).contains(marker));
        }
    }

    #[test]
    fn image_clipboard_statuses_cover_all_product_locales() {
        let cases = [
            (
                Locale::ZhCn,
                "已附加图片: 20x10",
                "粘贴图片失败: unavailable",
            ),
            (
                Locale::ZhTw,
                "已附加圖片: 20x10",
                "貼上圖片失敗: unavailable",
            ),
            (
                Locale::EnUs,
                "image attached: 20x10",
                "image paste failed: unavailable",
            ),
            (
                Locale::JaJp,
                "画像を添付しました: 20x10",
                "画像の貼り付けに失敗しました: unavailable",
            ),
            (
                Locale::KoKr,
                "이미지 첨부됨: 20x10",
                "이미지 붙여넣기 실패: unavailable",
            ),
        ];

        for (locale, attached, failed) in cases {
            assert_eq!(locale.status("image attached: 20x10"), attached);
            assert_eq!(locale.status("image paste failed: unavailable"), failed);
        }
    }

    #[test]
    fn queue_unavailable_status_covers_all_product_locales() {
        let cases = [
            (
                Locale::ZhCn,
                "队列不可用: offline",
                "正在编辑排队输入",
                "排队输入已不可用",
                "编辑排队输入失败: offline",
                "Alt+Up 编辑最后一条排队输入",
            ),
            (
                Locale::ZhTw,
                "佇列無法使用: offline",
                "正在編輯排隊輸入",
                "排隊輸入已無法使用",
                "編輯排隊輸入失敗: offline",
                "Alt+Up 編輯最後一則排隊輸入",
            ),
            (
                Locale::EnUs,
                "queue unavailable: offline",
                "editing queued input",
                "queued input unavailable",
                "queue edit failed: offline",
                "Alt+Up edit last queued input",
            ),
            (
                Locale::JaJp,
                "キューを利用できません: offline",
                "キュー入力を編集中",
                "キュー入力を利用できません",
                "キュー入力の編集に失敗しました: offline",
                "Alt+Up 最後のキュー入力を編集",
            ),
            (
                Locale::KoKr,
                "대기열을 사용할 수 없음: offline",
                "대기 입력 편집 중",
                "대기 입력을 사용할 수 없음",
                "대기 입력 편집 실패: offline",
                "Alt+Up 마지막 대기 입력 편집",
            ),
        ];
        for (locale, unavailable, editing, input_unavailable, failed, hint) in cases {
            assert_eq!(locale.status("queue unavailable: offline"), unavailable);
            assert_eq!(locale.status("queued input editing"), editing);
            assert_eq!(locale.status("queued input unavailable"), input_unavailable);
            assert_eq!(locale.status("queue edit failed: offline"), failed);
            assert_eq!(locale.edit_queued_input_hint(), hint);
        }
    }

    #[test]
    fn exposes_all_product_locale_tags() {
        let locales = [
            Locale::ZhCn,
            Locale::ZhTw,
            Locale::EnUs,
            Locale::JaJp,
            Locale::KoKr,
        ];
        assert_eq!(
            locales
                .iter()
                .map(|locale| locale.tag())
                .collect::<Vec<_>>(),
            vec!["zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"]
        );
    }

    #[test]
    fn slash_command_descriptions_cover_all_product_locales() {
        let cases = [
            (Locale::ZhCn, "选择模型"),
            (Locale::ZhTw, "選擇模型"),
            (Locale::EnUs, "choose a model"),
            (Locale::JaJp, "モデルを選択"),
            (Locale::KoKr, "모델 선택"),
        ];
        for (locale, expected) in cases {
            assert_eq!(
                locale.slash_command_description(SlashCommand::Model),
                expected
            );
            for command in SlashCommand::ALL {
                assert!(!locale.slash_command_description(command).is_empty());
            }
        }
    }

    #[test]
    fn status_pager_labels_cover_all_product_locales() {
        for locale in [
            Locale::ZhCn,
            Locale::ZhTw,
            Locale::EnUs,
            Locale::JaJp,
            Locale::KoKr,
        ] {
            for label in [
                locale.status_title(),
                locale.transcript_title(),
                locale.thread_label(),
                locale.provider_label(),
                locale.cwd_label(),
                locale.state_label(),
                locale.not_set_label(),
                locale.history_search_label(),
                locale.pager_footer(),
                locale.transcript_pager_footer(),
            ] {
                assert!(!label.is_empty(), "{locale:?}");
            }
        }
        assert_eq!(Locale::ZhCn.status_title(), "状态");
        assert_eq!(Locale::EnUs.transcript_title(), "T R A N S C R I P T");
        assert_eq!(Locale::ZhTw.state_label(), "狀態");
        assert_eq!(Locale::JaJp.not_set_label(), "未設定");
        assert!(Locale::KoKr.pager_footer().contains("Esc/Q"));
        assert!(Locale::ZhCn.transcript_pager_footer().contains("Ctrl+T"));
    }

    #[test]
    fn resume_picker_controls_cover_all_product_locales() {
        for locale in [
            Locale::ZhCn,
            Locale::ZhTw,
            Locale::EnUs,
            Locale::JaJp,
            Locale::KoKr,
        ] {
            assert!(!locale.resume_picker_title(false).is_empty());
            assert!(!locale.resume_picker_title(true).is_empty());
            assert!(!locale.resume_enter_hint(false, false).is_empty());
            assert!(!locale.resume_enter_hint(true, false).is_empty());
            assert!(!locale.resume_enter_hint(false, true).is_empty());
            assert!(!locale.resume_controls_hint().is_empty());
            assert!(!locale.resume_search_placeholder().is_empty());
            assert!(!locale.resume_loading().is_empty());
            assert!(!locale.resume_status_label(false).is_empty());
            assert!(!locale.resume_status_label(true).is_empty());
            assert!(!locale.resume_filter_label(false).is_empty());
            assert!(!locale.resume_filter_label(true).is_empty());
            assert!(!locale.resume_sort_label(false).is_empty());
            assert!(!locale.resume_sort_label(true).is_empty());
            assert!(!locale.resume_density_label(false).is_empty());
            assert!(!locale.resume_density_label(true).is_empty());
            assert!(!locale.resume_expand_hint().is_empty());
            assert!(!locale.created_label().is_empty());
            assert!(!locale.updated_label().is_empty());
            assert!(!locale.resume_transcript_loading().is_empty());
            assert!(!locale.resume_transcript_failed().is_empty());
            assert!(!locale.resume_transcript_empty().is_empty());
        }
    }
}
