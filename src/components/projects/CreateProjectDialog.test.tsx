import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanupMountedRoots,
  clickElement,
  fillTextInput,
  findButtonByText,
  findInputById,
  flushEffects,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "@/components/workspace/hooks/testUtils";
import { changeLimeLocale } from "@/i18n/createI18n";
import { CreateProjectDialog } from "./CreateProjectDialog";

const {
  mockExtractErrorMessage,
  mockGetCreateProjectErrorMessage,
  mockGetProjectByRootPath,
  mockGetWorkspaceProjectsRoot,
  mockOpenDialog,
  mockResolveProjectRootPath,
} = vi.hoisted(() => ({
  mockExtractErrorMessage: vi.fn((error: unknown) =>
    error instanceof Error ? error.message : String(error),
  ),
  mockGetCreateProjectErrorMessage: vi.fn((message: string) => message),
  mockGetProjectByRootPath: vi.fn(),
  mockGetWorkspaceProjectsRoot: vi.fn(),
  mockOpenDialog: vi.fn(),
  mockResolveProjectRootPath: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: mockOpenDialog,
}));

vi.mock("@/lib/api/project", () => ({
  USER_PROJECT_TYPES: ["general"],
  extractErrorMessage: mockExtractErrorMessage,
  getCreateProjectErrorMessage: mockGetCreateProjectErrorMessage,
  getProjectTypeIcon: vi.fn(() => "📁"),
  getProjectByRootPath: mockGetProjectByRootPath,
  getWorkspaceProjectsRoot: mockGetWorkspaceProjectsRoot,
  resolveProjectRootPath: mockResolveProjectRootPath,
}));

setupReactActEnvironment();

const mountedRoots: MountedRoot[] = [];

describe("CreateProjectDialog", () => {
  beforeEach(async () => {
    await changeLimeLocale("en-US");
    mockGetWorkspaceProjectsRoot.mockResolvedValue("/tmp/workspace");
    mockResolveProjectRootPath.mockImplementation(
      async (name: string, parentRootPath = "/tmp/workspace") =>
        `${parentRootPath}/${name}`,
    );
    mockGetProjectByRootPath.mockResolvedValue(null);
  });

  afterEach(async () => {
    cleanupMountedRoots(mountedRoots);
    vi.clearAllMocks();
    await changeLimeLocale("zh-CN");
  });

  it("uses a compact project creation form", async () => {
    mountHarness(
      CreateProjectDialog,
      {
        open: true,
        onOpenChange: vi.fn(),
        onSubmit: vi.fn(async () => undefined),
        defaultType: "general",
        defaultName: "Research Notes",
      },
      mountedRoots,
    );

    await flushEffects(3);

    const text = document.body.textContent ?? "";
    expect(text).toContain("New project");
    expect(text).toContain("Project name");
    expect(text).toContain("Location");
    expect(text).toContain("Project folder");
    expect(text).toContain("Browse…");
    expect(text).toContain("/tmp/workspace/Research Notes");
    expect(text).not.toContain("Create a new project workspace");
    expect(text).not.toContain("Choose project type");
    expect(text).not.toContain("Creation tips");
    expect(text).not.toContain("General chat");
  });

  it("localizes path conflict feedback and disables creation", async () => {
    mockGetProjectByRootPath.mockResolvedValue({
      id: "project-exists",
      name: "Existing Project",
    });

    mountHarness(
      CreateProjectDialog,
      {
        open: true,
        onOpenChange: vi.fn(),
        onSubmit: vi.fn(async () => undefined),
        defaultType: "general",
      },
      mountedRoots,
    );

    await flushEffects(2);

    const nameInput = findInputById(document.body, "name");
    expect(nameInput).not.toBeNull();
    fillTextInput(nameInput, "Conflicting Project");

    await flushEffects(3);

    expect(document.body.textContent ?? "").toContain(
      "Path already belongs to project: Existing Project",
    );
    expect(document.body.textContent ?? "").not.toContain("路径已存在项目");
    const createButton = findButtonByText(document.body, "Create", {
      exact: true,
    });
    expect(createButton).toBeDefined();
    expect(createButton?.disabled).toBe(true);
  });

  it("lets the user choose a parent folder and submits the resolved project path", async () => {
    mockOpenDialog.mockResolvedValueOnce("/Users/test/Documents");
    const onSubmit = vi.fn(async () => undefined);

    mountHarness(
      CreateProjectDialog,
      {
        open: true,
        onOpenChange: vi.fn(),
        onSubmit,
        defaultType: "general",
        defaultName: "Research Notes",
      },
      mountedRoots,
    );

    await flushEffects(3);

    const browseButton = findButtonByText(document.body, "Browse…", {
      exact: true,
    });
    clickElement(browseButton ?? null);

    await flushEffects(4);

    expect(mockOpenDialog).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
      defaultPath: "/tmp/workspace",
    });
    expect(mockResolveProjectRootPath).toHaveBeenLastCalledWith(
      "Research Notes",
      "/Users/test/Documents",
    );
    expect(document.body.textContent ?? "").toContain(
      "/Users/test/Documents/Research Notes",
    );

    const createButton = findButtonByText(document.body, "Create", {
      exact: true,
    });
    expect(createButton?.disabled).toBe(false);
    clickElement(createButton ?? null);

    await flushEffects(2);

    expect(onSubmit).toHaveBeenCalledWith(
      "Research Notes",
      "general",
      "/Users/test/Documents/Research Notes",
    );
  });
});
