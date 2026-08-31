import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_PERMISSION_PROFILE_LIST,
  type PermissionProfileListParams,
  type PermissionProfileListResponse,
  type PermissionProfileSummary,
} from "@limecloud/app-server-client";

type PermissionProfileAppServerClient = Pick<AppServerClient, "request">;

export async function listPermissionProfiles(
  params: PermissionProfileListParams = {},
  appServerClient: PermissionProfileAppServerClient = new AppServerClient(),
): Promise<PermissionProfileSummary[]> {
  const response = await appServerClient.request<PermissionProfileListResponse>(
    METHOD_PERMISSION_PROFILE_LIST,
    params,
  );
  if (!Array.isArray(response.result.data)) {
    throw new Error("App Server permissionProfile/list did not return data");
  }
  return response.result.data.map(assertPermissionProfile);
}

export async function resolveAllowedPermissionProfile(
  id: string,
  cwd?: string,
  appServerClient?: PermissionProfileAppServerClient,
): Promise<PermissionProfileSummary> {
  const normalizedId = id.trim();
  if (!normalizedId) {
    throw new Error("permission profile id must not be empty");
  }
  const normalizedCwd = cwd?.trim();
  const matches = (
    await listPermissionProfiles(
      normalizedCwd ? { cwd: normalizedCwd } : {},
      appServerClient,
    )
  ).filter((profile) => profile.id === normalizedId);
  if (matches.length !== 1) {
    throw new Error(
      `App Server permissionProfile/list must return exactly one ${normalizedId} profile`,
    );
  }
  if (!matches[0].allowed) {
    throw new Error(`Permission profile ${normalizedId} is not allowed`);
  }
  return matches[0];
}

function assertPermissionProfile(
  value: PermissionProfileSummary,
  index: number,
): PermissionProfileSummary {
  if (!value || typeof value !== "object") {
    throw new Error(
      `App Server permissionProfile/list returned invalid profile at index ${index}`,
    );
  }
  const id = value.id?.trim();
  if (!id || typeof value.allowed !== "boolean") {
    throw new Error(
      `App Server permissionProfile/list returned invalid profile at index ${index}`,
    );
  }
  const description = value.description;
  if (description != null && typeof description !== "string") {
    throw new Error(
      `App Server permissionProfile/list returned invalid description at index ${index}`,
    );
  }
  return {
    id,
    description: description?.trim() || undefined,
    allowed: value.allowed,
  };
}
