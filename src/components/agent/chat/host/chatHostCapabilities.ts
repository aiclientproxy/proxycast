import * as chatHostDialog from "@/lib/desktop-host/dialog";
import type {
  OpenDialogOptions,
  SaveDialogOptions,
} from "@/lib/desktop-host/dialog";

export async function requestChatHostOpenPath(
  options?: OpenDialogOptions & { multiple?: false },
): Promise<string | null> {
  return chatHostDialog.open(options);
}

export function requestChatHostSavePath(
  options?: SaveDialogOptions,
): Promise<string | null> {
  return chatHostDialog.save(options);
}
