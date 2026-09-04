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

# 更新 CLI npm package
sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/packages/cli/package.json"
echo "✓ packages/cli/package.json"

# 更新 Cargo.toml
sed -i '' "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" "$PROJECT_ROOT/lime-rs/Cargo.toml"
echo "✓ lime-rs/Cargo.toml"

echo ""
echo "版本已更新到 $NEW_VERSION"
echo ""
echo "下一步:"
echo "  npm run verify:app-version"
echo "  如需发布，再走 Electron release workflow"
