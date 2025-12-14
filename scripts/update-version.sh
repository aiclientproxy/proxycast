#!/bin/bash
# 统一版本管理脚本
# 用法: ./scripts/update-version.sh <new_version>
# 示例: ./scripts/update-version.sh 0.5.0

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <new_version>"
    echo "示例: $0 0.5.0"
    exit 1
fi

NEW_VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "更新版本到 $NEW_VERSION..."

# 更新 package.json
sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/package.json"
echo "✓ package.json"

# 更新 Cargo.toml
sed -i '' "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" "$PROJECT_ROOT/src-tauri/Cargo.toml"
echo "✓ src-tauri/Cargo.toml"

# 更新 tauri.conf.json
sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/src-tauri/tauri.conf.json"
echo "✓ src-tauri/tauri.conf.json"

echo ""
echo "版本已更新到 $NEW_VERSION"
echo ""
echo "下一步:"
echo "  git add -A"
echo "  git commit -m 'chore: bump version to $NEW_VERSION'"
echo "  git tag v$NEW_VERSION"
echo "  git push origin main --tags"
