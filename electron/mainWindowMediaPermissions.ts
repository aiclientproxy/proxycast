import type { Session, WebContents } from "electron";

type MediaPermissionDetails = {
  mediaTypes?: string[];
};

interface InstallMainWindowMediaPermissionHandlerArgs {
  session: Pick<Session, "setPermissionRequestHandler">;
  getMainWindow: () => { webContents: Pick<WebContents, "id"> } | null;
}

export function shouldAllowMainWindowMediaPermission({
  mainWebContents,
  requestWebContents,
  permission,
  details,
}: {
  mainWebContents: Pick<WebContents, "id"> | null;
  requestWebContents: Pick<WebContents, "id">;
  permission: string;
  details?: MediaPermissionDetails;
}): boolean {
  if (!mainWebContents || requestWebContents.id !== mainWebContents.id) {
    return false;
  }

  if (
    permission === "microphone" ||
    permission === "audioCapture" ||
    permission === "camera" ||
    permission === "videoCapture"
  ) {
    return true;
  }
  if (permission !== "media") {
    return false;
  }

  const mediaTypes = details?.mediaTypes ?? [];
  return (
    mediaTypes.length > 0 &&
    mediaTypes.every(
      (mediaType) => mediaType === "audio" || mediaType === "video",
    )
  );
}

export function installMainWindowMediaPermissionHandler({
  session,
  getMainWindow,
}: InstallMainWindowMediaPermissionHandlerArgs): void {
  session.setPermissionRequestHandler(
    (webContents, permission, callback, details) => {
      callback(
        shouldAllowMainWindowMediaPermission({
          mainWebContents: getMainWindow()?.webContents ?? null,
          requestWebContents: webContents,
          permission,
          details: details as MediaPermissionDetails,
        }),
      );
    },
  );
}
