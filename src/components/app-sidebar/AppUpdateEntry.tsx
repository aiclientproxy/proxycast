import { useCallback, useEffect, useState, type MouseEvent } from "react";
import styled from "styled-components";
import { Bell, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  checkForUpdates,
  getUpdateInstallSession,
  isUpdateInstallSessionActive,
  listenUpdateInstallSession,
  openUpdateWindow,
  type UpdateNotificationAnchorRect,
  type UpdateInstallSession,
  type VersionInfo,
} from "@/lib/api/appUpdate";

interface AppUpdateEntryProps {
  collapsed?: boolean;
  onOpenPanel?: () => void;
}

function shouldShowInstallSession(
  session: UpdateInstallSession | null,
): boolean {
  if (!session) {
    return false;
  }

  if (session.stage === "installing" || session.stage === "restarting") {
    return true;
  }

  return Boolean(
    session.latestVersion &&
    (session.stage === "completed" || session.stage === "failed"),
  );
}

function readUpdateNotificationAnchorRect(
  element: HTMLElement | null,
): UpdateNotificationAnchorRect | null {
  if (!element) {
    return null;
  }

  const rect = element.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return null;
  }

  return {
    x: rect.x,
    y: rect.y,
    width: rect.width,
    height: rect.height,
  };
}

const EntryRoot = styled.div<{ $collapsed?: boolean }>`
  position: relative;
  display: flex;
  justify-content: ${({ $collapsed }) =>
    $collapsed ? "center" : "flex-start"};
  padding: 0 2px;
`;

const EntryButton = styled.button<{ $active?: boolean }>`
  position: relative;
  width: 30px;
  height: 30px;
  border: 1px solid
    ${({ $active }) =>
      $active
        ? "var(--lime-brand-soft-border, #bbf7d0)"
        : "var(--sidebar-card-border, #e2f0e2)"};
  border-radius: 10px;
  background: ${({ $active }) =>
    $active
      ? "var(--lime-brand-soft, #ecfdf5)"
      : "var(--lime-surface, #ffffff)"};
  color: ${({ $active }) =>
    $active ? "var(--lime-brand-strong, #166534)" : "var(--sidebar-muted)"};
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.56);
  transition:
    background-color 0.16s ease,
    border-color 0.16s ease,
    color 0.16s ease;

  &:hover {
    border-color: var(--lime-brand-soft-border, #bbf7d0);
    background: var(--lime-brand-soft, #ecfdf5);
    color: var(--lime-brand-strong, #166534);
  }

  svg {
    width: 15px;
    height: 15px;
  }

  &::after {
    content: "";
    position: absolute;
    right: 5px;
    top: 5px;
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: var(--lime-warning, #f59e0b);
    box-shadow: 0 0 0 2px var(--lime-surface, #ffffff);
  }
`;

const SpinningIcon = styled(RefreshCw)`
  animation: appSidebarUpdateSpin 0.8s linear infinite;

  @keyframes appSidebarUpdateSpin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
`;

export function AppUpdateEntry({
  collapsed,
  onOpenPanel,
}: AppUpdateEntryProps) {
  const { t } = useTranslation("navigation");
  const [versionInfo, setVersionInfo] = useState<VersionInfo | null>(null);
  const [installSession, setInstallSession] =
    useState<UpdateInstallSession | null>(null);

  useEffect(() => {
    let disposed = false;

    void checkForUpdates({ automatic: true })
      .then((result) => {
        if (!disposed) {
          setVersionInfo(result);
        }
      })
      .catch((error) => {
        console.error("检查应用更新失败:", error);
      });

    void getUpdateInstallSession()
      .then((session) => {
        if (!disposed && shouldShowInstallSession(session)) {
          setInstallSession(session);
        }
      })
      .catch((error) => {
        console.error("读取更新安装会话失败:", error);
      });

    const unlistenPromise = listenUpdateInstallSession((session) => {
      setInstallSession(session);
    });

    return () => {
      disposed = true;
      void unlistenPromise
        .then((unlisten) => unlisten())
        .catch((error) => {
          console.error("取消更新安装会话监听失败:", error);
        });
    };
  }, []);

  const installActive = isUpdateInstallSessionActive(installSession);
  const sessionVisible = shouldShowInstallSession(installSession);
  const updateAvailable = Boolean(versionInfo?.hasUpdate && !versionInfo.error);
  const shouldShowEntry = updateAvailable || sessionVisible;
  const openLabel = t("navigation.sidebar.update.open");

  const handleOpenUpdateWindow = useCallback(
    async (event: MouseEvent<HTMLButtonElement>) => {
      const anchorRect = readUpdateNotificationAnchorRect(event.currentTarget);

      onOpenPanel?.();
      try {
        await openUpdateWindow(anchorRect);
      } catch (error) {
        console.error("打开更新窗口失败:", error);
      }
    },
    [onOpenPanel],
  );

  if (!shouldShowEntry) {
    return null;
  }

  return (
    <EntryRoot
      $collapsed={collapsed}
      data-testid="app-sidebar-update-entry"
      onMouseDown={(event) => event.stopPropagation()}
    >
      <EntryButton
        type="button"
        $active={installActive}
        title={openLabel}
        aria-label={openLabel}
        data-testid="app-sidebar-update-button"
        data-update-notification-anchor="true"
        onClick={handleOpenUpdateWindow}
      >
        {installActive ? <SpinningIcon /> : <Bell />}
      </EntryButton>
    </EntryRoot>
  );
}
