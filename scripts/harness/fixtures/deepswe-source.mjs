import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function createDeepSweSourceFixture({
  repoRoot,
  manifestPath = "internal/test/deepswe-coding-slice-v2.json",
} = {}) {
  const manifest = JSON.parse(
    fs.readFileSync(path.resolve(repoRoot, manifestPath), "utf8"),
  );
  const sourceRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "lime-deepswe-source-test-"),
  );
  const tasksRoot = path.join(sourceRoot, "tasks");
  const tasks = new Map();

  fs.mkdirSync(tasksRoot, { recursive: true });
  fs.writeFileSync(path.join(sourceRoot, "LICENSE"), "Apache-2.0\n", "utf8");
  fs.writeFileSync(
    path.join(sourceRoot, "PROVENANCE.md"),
    "# Test fixture provenance\n",
    "utf8",
  );

  for (const task of manifest.tasks) {
    const taskDir = path.join(tasksRoot, task.id);
    const baseCommit = createHash("sha1").update(task.id).digest("hex");
    const repositoryUrl = `https://github.com/${task.repository}.git`;
    const instruction = `Fix the ${task.id} regression.`;

    fs.mkdirSync(taskDir, { recursive: true });
    fs.writeFileSync(path.join(taskDir, "instruction.md"), instruction, "utf8");
    fs.writeFileSync(
      path.join(taskDir, "task.toml"),
      `schema_version = "${manifest.source.taskSchemaVersion}"
artifacts = ["/logs/artifacts/model.patch"]

[metadata]
repository_url = "${repositoryUrl}"
base_commit_hash = "${baseCommit}"

[agent]
network_mode = "no-network"
timeout_sec = 5400

[environment]
docker_image = "lime/deepswe-test:latest"
cpus = 2
memory_mb = 4096
storage_mb = 8192

[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 600

[[verifier.collect]]
command = "git diff --binary ${baseCommit} HEAD > /logs/artifacts/model.patch"
timeout_sec = 300
`,
      "utf8",
    );
    tasks.set(task.id, { baseCommit, instruction, repositoryUrl });
  }

  return {
    sourceRoot,
    sourceCommit: manifest.source.commit,
    tasks,
  };
}
