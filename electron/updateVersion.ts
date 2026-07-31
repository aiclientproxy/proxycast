interface ParsedUpdateVersion {
  core: [string, string, string];
  normalized: string;
  prerelease: string[];
}

export type UpdateCandidateStatus = "invalid" | "newer" | "not_newer";

export interface UpdateCandidateAssessment {
  status: UpdateCandidateStatus;
  version: string | null;
}

const UPDATE_VERSION_PATTERN =
  /^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$/i;

function parseUpdateVersion(
  value: string | null | undefined,
): ParsedUpdateVersion | null {
  const match = UPDATE_VERSION_PATTERN.exec(value?.trim() ?? "");
  if (!match) {
    return null;
  }

  const prerelease = match[4]?.split(".") ?? [];
  if (
    prerelease.some(
      (identifier) =>
        /^\d+$/.test(identifier) &&
        identifier.length > 1 &&
        identifier.startsWith("0"),
    )
  ) {
    return null;
  }

  const core: [string, string, string] = [match[1], match[2], match[3]];
  const prereleaseSuffix =
    prerelease.length > 0 ? `-${prerelease.join(".")}` : "";
  const buildSuffix = match[5] ? `+${match[5]}` : "";
  return {
    core,
    normalized: `${core.join(".")}${prereleaseSuffix}${buildSuffix}`,
    prerelease,
  };
}

function compareNumericIdentifiers(left: string, right: string): number {
  if (left.length !== right.length) {
    return left.length > right.length ? 1 : -1;
  }
  if (left === right) {
    return 0;
  }
  return left > right ? 1 : -1;
}

function comparePrerelease(left: string[], right: string[]): number {
  if (left.length === 0 || right.length === 0) {
    if (left.length === right.length) {
      return 0;
    }
    return left.length === 0 ? 1 : -1;
  }

  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const leftIdentifier = left[index];
    const rightIdentifier = right[index];
    if (leftIdentifier === undefined || rightIdentifier === undefined) {
      return leftIdentifier === undefined ? -1 : 1;
    }
    if (leftIdentifier === rightIdentifier) {
      continue;
    }

    const leftNumeric = /^\d+$/.test(leftIdentifier);
    const rightNumeric = /^\d+$/.test(rightIdentifier);
    if (leftNumeric && rightNumeric) {
      return compareNumericIdentifiers(leftIdentifier, rightIdentifier);
    }
    if (leftNumeric !== rightNumeric) {
      return leftNumeric ? -1 : 1;
    }
    return leftIdentifier > rightIdentifier ? 1 : -1;
  }
  return 0;
}

function compareParsedVersions(
  left: ParsedUpdateVersion,
  right: ParsedUpdateVersion,
): number {
  for (let index = 0; index < left.core.length; index += 1) {
    const result = compareNumericIdentifiers(
      left.core[index],
      right.core[index],
    );
    if (result !== 0) {
      return result;
    }
  }
  return comparePrerelease(left.prerelease, right.prerelease);
}

export function normalizeUpdateVersion(
  value: string | null | undefined,
): string | null {
  return parseUpdateVersion(value)?.normalized ?? null;
}

export function assessUpdateCandidate(
  candidateVersion: string | null | undefined,
  currentVersion: string | null | undefined,
): UpdateCandidateAssessment {
  const candidate = parseUpdateVersion(candidateVersion);
  const current = parseUpdateVersion(currentVersion);
  if (!candidate || !current) {
    return { status: "invalid", version: null };
  }

  return {
    status:
      compareParsedVersions(candidate, current) > 0 ? "newer" : "not_newer",
    version: candidate.normalized,
  };
}
