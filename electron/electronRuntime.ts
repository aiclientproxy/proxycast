/* eslint-disable no-redeclare */
import { createRequire } from "node:module";
import type * as Electron from "electron";

const requireElectron = createRequire(import.meta.url);
const electron = requireElectron("electron") as typeof Electron;

export const {
  app,
  autoUpdater,
  BrowserWindow,
  clipboard,
  dialog,
  globalShortcut,
  ipcMain,
  Menu,
  nativeImage,
  Notification,
  net,
  powerMonitor,
  screen,
  session,
  shell,
  systemPreferences,
  protocol,
  Tray,
  WebContentsView,
} = electron;

export type BrowserWindow = Electron.BrowserWindow;
export type Menu = Electron.Menu;
export type Rectangle = Electron.Rectangle;
export type Tray = Electron.Tray;
export type WebContentsView = Electron.WebContentsView;

export type {
  IpcMainInvokeEvent,
  MenuItemConstructorOptions,
  NotificationConstructorOptions,
  OpenDialogOptions,
  SaveDialogOptions,
} from "electron";
