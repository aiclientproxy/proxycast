import type { Dispatch, SetStateAction } from "react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";

export type CanvasOpenReason =
  | "user_toggle_canvas"
  | "user_open_artifact"
  | "user_open_file"
  | "user_open_message_preview"
  | "runtime_write"
  | "runtime_image_workbench"
  | "runtime_team_workbench";

const MANUAL_CANVAS_OPEN_REASONS = new Set<CanvasOpenReason>([
  "user_toggle_canvas",
  "user_open_artifact",
  "user_open_file",
  "user_open_message_preview",
  "runtime_write",
]);

export function shouldOpenCanvasForReason(reason: CanvasOpenReason): boolean {
  return MANUAL_CANVAS_OPEN_REASONS.has(reason);
}

export function openCanvasForReason(
  reason: CanvasOpenReason,
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>,
): void {
  if (!shouldOpenCanvasForReason(reason)) {
    return;
  }

  setLayoutMode("chat-canvas");
}
