import type { ReactNode } from "react";

export type WorkspaceRightSurfaceKind =
  | "workbench"
  | "appSurface"
  | "articleWorkspace"
  | "expertInfo"
  | "browser"
  | "files"
  | "shell"
  | "harness"
  | "trace"
  | "activity";

export type WorkspaceRightSurfaceSource = "user" | "route" | "runtime";

export type WorkspaceRightSurfaceLayoutVariant =
  | "docked"
  | "expanded"
  | "canvasFirst";

export interface WorkspaceRightSurfaceState {
  activeSurface: WorkspaceRightSurfaceKind | null;
  previousSurface: WorkspaceRightSurfaceKind | null;
  openSurfaces: readonly WorkspaceRightSurfaceKind[];
  source: WorkspaceRightSurfaceSource;
  layoutVariant: WorkspaceRightSurfaceLayoutVariant;
}

export interface RightSurfaceRenderInput {
  activeSurface: WorkspaceRightSurfaceKind | null;
}

export interface RightSurfaceDefinition {
  kind: WorkspaceRightSurfaceKind;
  label?: string | null;
  render(input: RightSurfaceRenderInput): ReactNode;
}

export function normalizeWorkspaceRightSurfaceKind(
  value?: string | null,
): WorkspaceRightSurfaceKind | null {
  const normalized = value?.trim();
  switch (normalized) {
    case "workbench":
    case "appSurface":
    case "articleWorkspace":
    case "expertInfo":
    case "browser":
    case "files":
    case "shell":
    case "harness":
    case "trace":
    case "activity":
      return normalized;
    default:
      return null;
  }
}
