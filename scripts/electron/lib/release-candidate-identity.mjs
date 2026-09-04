const CANDIDATE_SHA_PATTERN = /^[0-9a-f]{40}$/u;
const CANDIDATE_RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;

export function normalizeCandidateSha(value) {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  if (!CANDIDATE_SHA_PATTERN.test(normalized)) {
    throw new Error("candidate SHA must be a 40-character Git commit SHA");
  }
  return normalized;
}

export function normalizeCandidateRunId(value) {
  const normalized = String(value || "").trim();
  if (!CANDIDATE_RUN_ID_PATTERN.test(normalized)) {
    throw new Error(
      "candidate run id contains unsupported characters or is too long",
    );
  }
  return normalized;
}
