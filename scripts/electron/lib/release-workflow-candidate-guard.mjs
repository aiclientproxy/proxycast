function stepByName(steps, name) {
  return steps.find((step) => step?.name === name);
}

function assertIncludes(haystack, needle, label) {
  if (!String(haystack || "").includes(needle)) {
    throw new Error(`${label} must include ${needle}`);
  }
}

function assertCandidatePermissions(workflow) {
  if (
    workflow?.permissions?.["id-token"] !== "write" ||
    workflow?.permissions?.attestations !== "write"
  ) {
    throw new Error(
      "release workflow must grant id-token: write and attestations: write for provenance",
    );
  }
}

function assertPrepareReleaseSteps(workflow) {
  const steps = workflow?.jobs?.prepare_release?.steps;
  if (!Array.isArray(steps)) {
    throw new Error("prepare_release job must define steps");
  }
  const sourceIdentityStep = stepByName(
    steps,
    "Validate release source identity",
  );
  for (const required of [
    "git rev-parse 'HEAD^{commit}'",
    'git rev-parse "${GITHUB_SHA}^{commit}"',
    'CANDIDATE_SHA" != "$WORKFLOW_SHA',
  ]) {
    assertIncludes(
      sourceIdentityStep?.run,
      required,
      "release source identity",
    );
  }
  const releaseStep = stepByName(steps, "Ensure GitHub release exists");
  assertIncludes(releaseStep?.run, "gh release create", "release preparation");
  assertIncludes(releaseStep?.run, "--draft", "release preparation");
  assertIncludes(
    releaseStep?.run,
    'TARGET_REF="${{ github.sha }}"',
    "release preparation",
  );
}

function assertImmutableReleaseCheckouts(workflow) {
  for (const jobName of [
    "build",
    "publish_release_assets",
    "publish_updater_assets_r2",
    "publish_cli_assets",
  ]) {
    const checkout = stepByName(
      workflow?.jobs?.[jobName]?.steps || [],
      "Checkout",
    );
    if (checkout?.with?.ref !== "${{ github.sha }}") {
      throw new Error(
        `release job ${jobName} must checkout immutable github.sha`,
      );
    }
  }
}

function assertBuildCandidateIdentity(workflow) {
  const steps = workflow?.jobs?.build?.steps || [];
  const candidateIdentityStep = stepByName(
    steps,
    "Capture release candidate identity",
  );
  for (const required of [
    "git rev-parse 'HEAD^{commit}'",
    'git rev-parse "${GITHUB_SHA}^{commit}"',
    'CANDIDATE_SHA" != "$WORKFLOW_SHA',
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "matrix.host_platform",
    "matrix.arch",
    "LIME_CANDIDATE_SHA",
    "LIME_GATE_RUN_ID",
    '>> "$GITHUB_ENV"',
  ]) {
    assertIncludes(
      candidateIdentityStep?.run,
      required,
      "release candidate identity",
    );
  }
}

function assertReleaseProvenance(workflow) {
  const steps = workflow?.jobs?.publish_release_assets?.steps || [];
  const provenanceStep = stepByName(
    steps,
    "Attest Electron release provenance",
  );
  if (
    provenanceStep?.uses !==
    "actions/attest-build-provenance@a2bbfa25375fe432b6a289bc6b6cd05ecd0c4c32"
  ) {
    throw new Error(
      "Electron release provenance must use the pinned attest-build-provenance v4.1.0 action",
    );
  }
  assertIncludes(
    provenanceStep?.with?.["subject-path"],
    "release-github-assets/*",
    "Electron release provenance",
  );
}

export function validateReleaseCandidateWorkflow(workflow) {
  assertCandidatePermissions(workflow);
  assertPrepareReleaseSteps(workflow);
  assertImmutableReleaseCheckouts(workflow);
  assertBuildCandidateIdentity(workflow);
  assertReleaseProvenance(workflow);
}
