#!/usr/bin/env python3
"""Stage and optionally package the @limecloud/lime npm module."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CLI_ROOT = SCRIPT_DIR.parent
NPM_NAME = "@limecloud/lime"

# Alias names are resolved by bin/lime.js. Every platform tarball is published
# under NPM_NAME with a unique version suffix, matching the Codex npm layout.
PLATFORM_PACKAGES: dict[str, dict[str, str]] = {
    "lime-linux-x64": {
        "npm_name": "@limecloud/lime-linux-x64",
        "npm_tag": "linux-x64",
        "target_triple": "x86_64-unknown-linux-gnu",
        "os": "linux",
        "cpu": "x64",
    },
    "lime-darwin-x64": {
        "npm_name": "@limecloud/lime-darwin-x64",
        "npm_tag": "darwin-x64",
        "target_triple": "x86_64-apple-darwin",
        "os": "darwin",
        "cpu": "x64",
    },
    "lime-darwin-arm64": {
        "npm_name": "@limecloud/lime-darwin-arm64",
        "npm_tag": "darwin-arm64",
        "target_triple": "aarch64-apple-darwin",
        "os": "darwin",
        "cpu": "arm64",
    },
    "lime-win32-x64": {
        "npm_name": "@limecloud/lime-win32-x64",
        "npm_tag": "win32-x64",
        "target_triple": "x86_64-pc-windows-msvc",
        "os": "win32",
        "cpu": "x64",
    },
}

PACKAGE_CHOICES = ("lime", *PLATFORM_PACKAGES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or stage the Lime CLI npm package.")
    parser.add_argument("--package", choices=PACKAGE_CHOICES, default="lime")
    parser.add_argument("--version")
    parser.add_argument("--release-version")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--pack-output", type=Path)
    parser.add_argument(
        "--vendor-src",
        type=Path,
        help="Vendor root containing <target-triple>/bin runtime payloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = args.version
    if args.release_version:
        if version and version != args.release_version:
            raise RuntimeError(
                "--version and --release-version must match when both are provided."
            )
        version = args.release_version
    if not version:
        raise RuntimeError("Must specify --version or --release-version.")

    staging_dir = prepare_staging_dir(args.staging_dir)
    stage_sources(staging_dir, version, args.package)

    platform = PLATFORM_PACKAGES.get(args.package)
    if platform:
        if not args.vendor_src:
            raise RuntimeError(
                f"Native runtime payload required for package '{args.package}'. "
                "Provide --vendor-src."
            )
        copy_native_binaries(args.vendor_src.resolve(), staging_dir, platform)

    print(f"Staged package {args.package} in {staging_dir}")
    if args.pack_output:
        output_path = run_npm_pack(staging_dir, args.pack_output)
        print(f"npm pack output written to {output_path}")
    return 0


def prepare_staging_dir(staging_dir: Path | None) -> Path:
    if staging_dir is None:
        return Path(tempfile.mkdtemp(prefix="lime-npm-stage-"))
    resolved = staging_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if any(resolved.iterdir()):
        raise RuntimeError(f"Staging directory {resolved} is not empty.")
    return resolved


def stage_sources(staging_dir: Path, version: str, package: str) -> None:
    readme_src = CLI_ROOT / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, staging_dir / "README.md")

    with open(CLI_ROOT / "package.json", "r", encoding="utf-8") as source:
        root_package_json = json.load(source)

    if package == "lime":
        bin_dir = staging_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CLI_ROOT / "bin" / "lime.js", bin_dir / "lime.js")
        package_json = dict(root_package_json)
        package_json["version"] = version
        package_json["files"] = ["bin/lime.js"]
        package_json.pop("scripts", None)
        package_json["optionalDependencies"] = {
            config["npm_name"]: (
                f"npm:{NPM_NAME}@{compute_platform_package_version(version, config['npm_tag'])}"
            )
            for config in PLATFORM_PACKAGES.values()
        }
    else:
        platform = PLATFORM_PACKAGES[package]
        package_json = {
            "name": NPM_NAME,
            "version": compute_platform_package_version(version, platform["npm_tag"]),
            "description": root_package_json.get("description"),
            "license": root_package_json.get("license", "MIT"),
            "os": [platform["os"]],
            "cpu": [platform["cpu"]],
            "files": ["vendor"],
            "repository": root_package_json.get("repository"),
        }
        if isinstance(root_package_json.get("engines"), dict):
            package_json["engines"] = root_package_json["engines"]
        if isinstance(root_package_json.get("packageManager"), str):
            package_json["packageManager"] = root_package_json["packageManager"]

    with open(staging_dir / "package.json", "w", encoding="utf-8") as output:
        json.dump(package_json, output, indent=2)
        output.write("\n")


def compute_platform_package_version(version: str, npm_tag: str) -> str:
    return f"{version}-{npm_tag}"


def copy_native_binaries(
    vendor_src: Path, staging_dir: Path, platform: dict[str, str]
) -> None:
    target_triple = platform["target_triple"]
    source_target = vendor_src / target_triple
    validate_native_payload(source_target, platform)
    destination_target = staging_dir / "vendor" / target_triple
    destination_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_target, destination_target)


def validate_native_payload(target_root: Path, platform: dict[str, str]) -> None:
    if not target_root.is_dir():
        raise RuntimeError(f"Missing target directory in vendor source: {target_root}")

    bin_dir = target_root / "bin"
    suffix = ".exe" if platform["os"] == "win32" else ""
    required = [f"lime{suffix}", f"app-server{suffix}", f"code-mode-host{suffix}"]
    if platform["os"] == "win32":
        required.extend(["windows-sandbox-setup.exe", "windows-sandbox-runner.exe"])

    missing = [name for name in required if not (bin_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete native runtime payload for {platform['target_triple']}: "
            f"missing {', '.join(missing)}"
        )

    names = [entry.name for entry in bin_dir.iterdir() if entry.is_file()]
    runtime_markers = {
        "darwin": ("libsherpa-onnx-c-api", "libonnxruntime"),
        "linux": ("libsherpa-onnx-c-api", "libonnxruntime"),
        "win32": ("sherpa-onnx-c-api.dll", "onnxruntime.dll"),
    }[platform["os"]]
    missing_runtime = [
        marker for marker in runtime_markers if not any(marker in name for name in names)
    ]
    if missing_runtime:
        raise RuntimeError(
            f"Incomplete native runtime libraries for {platform['target_triple']}: "
            f"missing {', '.join(missing_runtime)}"
        )


def run_npm_pack(staging_dir: Path, output_path: Path) -> Path:
    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lime-npm-pack-") as pack_dir_name:
        pack_dir = Path(pack_dir_name)
        env = os.environ.copy()
        env["NPM_CONFIG_CACHE"] = str(pack_dir / "npm-cache")
        env["NPM_CONFIG_LOGS_DIR"] = str(pack_dir / "npm-logs")
        stdout = subprocess.check_output(
            ["npm", "pack", "--json", "--pack-destination", str(pack_dir)],
            cwd=staging_dir,
            env=env,
            text=True,
        )
        pack_output = json.loads(stdout)
        if not pack_output:
            raise RuntimeError("npm pack did not produce an output tarball.")
        tarball_name = pack_output[0].get("filename") or pack_output[0].get("name")
        if not tarball_name:
            raise RuntimeError("Unable to determine npm pack output filename.")
        tarball_path = pack_dir / tarball_name
        if not tarball_path.is_file():
            raise RuntimeError(f"Expected npm tarball not found: {tarball_path}")
        shutil.move(str(tarball_path), resolved_output)
    return resolved_output


if __name__ == "__main__":
    raise SystemExit(main())
