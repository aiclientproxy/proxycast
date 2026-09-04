const fs = require("fs");
const os = require("os");
const path = require("path");
const { execSync } = require("child_process");
const {
  resolveReleaseAssetMeta,
  isSupportedReleaseAssetMeta,
  supportedReleaseAssetLabels,
  NAME,
} = require("./release-meta");

const meta = resolveReleaseAssetMeta();
const isWindows = meta.isWindows;
const archiveName = meta.archiveName;
const githubUrl = meta.githubUrl;

const binDir = path.join(__dirname, "..", "bin");
const dest = path.join(binDir, meta.binaryName);

fs.mkdirSync(binDir, { recursive: true });

function download(url, destPath) {
  const sslFlag = isWindows ? "--ssl-revoke-best-effort " : "";
  execSync(
    `curl ${sslFlag}--fail --location --silent --show-error --connect-timeout 10 --max-time 120 --output "${destPath}" "${url}"`,
    {
      stdio: ["ignore", "ignore", "pipe"],
    },
  );
}

function install() {
  if (!isSupportedReleaseAssetMeta(meta)) {
    console.warn(
      `[${NAME}] 当前平台 ${meta.platform}/${meta.arch} 暂无预编译资产。` +
        `已跳过下载；可通过 LIME_CLI_BINARY_PATH 指向本地二进制，` +
        `或在源码仓库中通过 cargo run/cargo build 使用。`,
    );
    return;
  }

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "lime-"));
  const archivePath = path.join(tmpDir, archiveName);

  try {
    download(githubUrl, archivePath);

    if (isWindows) {
      execSync(
        `powershell -Command "Expand-Archive -Path '${archivePath}' -DestinationPath '${tmpDir}'"`,
        { stdio: "ignore" },
      );
    } else {
      execSync(`tar -xzf "${archivePath}" -C "${tmpDir}"`, {
        stdio: "ignore",
      });
    }

    const extractedBinary = path.join(tmpDir, meta.binaryName);
    fs.copyFileSync(extractedBinary, dest);
    fs.chmodSync(dest, 0o755);
    console.log(`${NAME} v${meta.version} installed successfully`);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

try {
  install();
} catch (error) {
  console.warn(`Failed to download prebuilt ${NAME}: ${error.message}`);
  console.warn(
    `\n已继续完成 npm 安装，但当前不会自动提供预编译二进制。` +
      `你可以：\n` +
      `1. 设置 LIME_CLI_BINARY_PATH 指向本地构建好的 lime 二进制\n` +
      `2. 在 Lime 源码仓库中运行该命令，由 wrapper 自动回退到 cargo run\n` +
      `3. 后续补发 GitHub Release asset 后重新安装`,
  );
}
