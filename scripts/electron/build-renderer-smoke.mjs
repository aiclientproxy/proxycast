process.env.LIME_ELECTRON_RENDERER = "1";
// Gate B fixtures share dist while other fixture processes may rebuild it.
process.env.LIME_VITE_EMPTY_OUT_DIR = "0";

const { spawn } = await import("node:child_process");
const {
  acquireRendererBuildLock,
  rendererBuildEnv,
  startRendererBuildHeartbeat,
} =
  await import("./renderer-build-env.mjs");

const releaseBuildLock = acquireRendererBuildLock();
const stopHeartbeat = startRendererBuildHeartbeat();
try {
  await run("npx", ["vite", "build", "--base", "./"]);
} finally {
  stopHeartbeat();
  releaseBuildLock();
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env: rendererBuildEnv(),
      stdio: "inherit",
      shell: process.platform === "win32",
    });
    child.once("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
    });
    child.once("error", reject);
  });
}
