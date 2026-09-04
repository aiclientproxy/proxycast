const EXPECTED_RELEASE_MATRIX = new Map([
  [
    "macOS-arm64",
    {
      arch: "arm64",
      feed: "darwin-arm64",
      forge_targets: "dmg,zip",
      host_platform: "darwin",
      platform: "macos-15",
      target: "aarch64-apple-darwin",
    },
  ],
  [
    "macOS-x64",
    {
      arch: "x64",
      feed: "darwin-x64",
      forge_targets: "dmg,zip",
      host_platform: "darwin",
      platform: "macos-15-intel",
      target: "x86_64-apple-darwin",
    },
  ],
  [
    "Windows-x64",
    {
      arch: "x64",
      feed: "win32-x64",
      forge_targets: "squirrel",
      host_platform: "win32",
      platform: "windows-2022",
      target: "x86_64-pc-windows-msvc",
    },
  ],
]);

export function validateReleaseMatrix(buildJob) {
  const matrix = buildJob?.strategy?.matrix?.include;
  if (!Array.isArray(matrix)) {
    throw new Error("release build job must define strategy.matrix.include");
  }
  for (const [name, fields] of EXPECTED_RELEASE_MATRIX) {
    const row = matrix.find((item) => item?.name === name);
    if (!row) {
      throw new Error(`release build matrix missing ${name}`);
    }
    for (const [key, value] of Object.entries(fields)) {
      if (row[key] !== value) {
        throw new Error(
          `release build matrix ${name}.${key} expected ${value}, got ${row[key]}`,
        );
      }
    }
  }
}
