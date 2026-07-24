import * as chatHostDialog from "@/lib/desktop-host/plugin-dialog";
import type {
  OpenDialogOptions,
  SaveDialogOptions,
} from "@/lib/desktop-host/plugin-dialog";
import { selectPluginDirectory } from "@/lib/api/plugins";
import { getElectronHostBridge } from "@/lib/electron-host";

export async function requestChatHostOpenPath(
  options?: OpenDialogOptions & { multiple?: false },
): Promise<string | null> {
  if (!options?.directory || getElectronHostBridge()?.dialog) {
    return chatHostDialog.open(options);
  }

  const result = await selectPluginDirectory({
    ...(options.title ? { title: options.title } : {}),
  });
  return result.cancelled ? null : result.path;
}

export function requestChatHostSavePath(
  options?: SaveDialogOptions,
): Promise<string | null> {
  return chatHostDialog.save(options);
}
