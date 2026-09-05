import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const inventory = JSON.parse(
  readFileSync(
    path.resolve(
      process.cwd(),
      "internal/exec-plans/tui-structure-inventory.json",
    ),
    "utf8",
  ),
);

describe("Codex TUI structure inventory", () => {
  it("records both TUI source trees", () => {
    expect(inventory.schemaVersion).toBe(1);
    expect(inventory.trees["codex-rs/tui/src"].fileCount).toBeGreaterThan(0);
    expect(inventory.trees["lime-rs/crates/tui/src"].fileCount).toBeGreaterThan(0);
  });

  it("locks Codex-shaped current TUI module and symbol names", () => {
    const files = new Set(inventory.trees["lime-rs/crates/tui/src"].files);
    for (const file of [
      "markdown_render.rs",
      "status_indicator_widget.rs",
      "resume_picker.rs",
      "resume_picker/archive.rs",
      "resume_picker/archive_tests.rs",
      "resume_picker/page_loading.rs",
      "resume_picker_transcript_preview.rs",
      "resume_picker_transcript_preview_tests.rs",
      "bottom_pane/chat_composer.rs",
      "bottom_pane/approval_overlay.rs",
      "bottom_pane/request_user_input/mod.rs",
      "bottom_pane/request_user_input/render.rs",
      "clipboard_copy.rs",
      "clipboard_paste.rs",
      "command_popup.rs",
      "pending_input_preview.rs",
      "terminal_hyperlinks.rs",
      "terminal_palette.rs",
      "table_detect.rs",
      "wrapping.rs",
      "tui.rs",
      "tui/event_stream.rs",
      "tui/frame_rate_limiter.rs",
      "tui/frame_requester.rs",
      "selection_list.rs",
      "thread_transcript.rs",
      "transcript_reflow.rs",
    ]) {
      expect(files.has(file), file).toBe(true);
    }
    const symbols = new Set(
      inventory.trees["lime-rs/crates/tui/src"].symbols.map(
        (symbol) => symbol.name,
      ),
    );
    for (const name of [
      "render_markdown_text",
      "render_markdown_lines_with_width",
      "fmt_elapsed_compact",
      "ChatComposer",
      "InputResult",
      "ApprovalOverlay",
      "TerminalGuard",
      "run_resume_picker_with_app_server",
      "run_fork_picker_with_app_server",
      "run_session_picker_with_app_server",
      "PickerState",
      "SessionTarget",
      "SessionSelection",
      "SessionPickerAction",
      "SessionPickerLaunchContext",
      "ArchiveState",
      "PaginationState",
      "load_transcript_preview",
      "load_session_transcript_with_handle",
      "thread_to_transcript_entries",
      "selection_option_row",
      "selection_option_row_with_dim",
      "TranscriptReflowState",
      "TranscriptWidthChange",
      "SessionTranscriptState",
      "render_expanded_session_details",
      "render_transcript_content_lines",
      "EventBroker",
      "TuiEventStream",
      "TuiEvent",
      "FrameRequester",
      "RtOptions",
      "adaptive_wrap_line",
      "adaptive_wrap_lines",
      "word_wrap_line",
      "word_wrap_lines",
      "wrap_ranges",
      "wrap_ranges_trim",
      "url_preserving_wrap_options",
      "line_has_mixed_url_and_non_url_tokens",
      "parse_table_segments",
      "FenceTracker",
      "StdoutColorLevel",
      "best_color",
      "effective_stdout_color_level",
    ]) {
      expect(symbols.has(name), name).toBe(true);
    }
    expect(files.has("composer.rs")).toBe(false);
    expect(files.has("bottom_pane/request_user_input.rs")).toBe(false);
    expect(files.has("terminal.rs")).toBe(false);
  });

  it("keeps upstream product-only differences explicit", () => {
    expect(inventory.comparisons.filesMissingInLime).toContain(
      "onboarding/mod.rs",
    );
    expect(inventory.comparisons.symbolNamesMissingInLime.length).toBeGreaterThan(0);
    expect(inventory.comparisons.filesOnlyInLime).not.toContain("session_picker.rs");
  });
});
