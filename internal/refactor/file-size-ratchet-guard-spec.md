# 文件体量棘轮守卫规格

> 状态：proposed（实现 R-60 时参考）
> 创建时间：2026-06-11（定位为护栏/仪表，不是重构主线）
> 对应计划：`v2/IMPLEMENTATION-PLAN.md`
> 参考实现：`scripts/check-scripts-governance.mjs` + `script-root-governance-baseline.json`
> 注意：守卫脚本落点遵守 scripts 根目录冻结，放 `scripts/governance/`（下文示例路径按此理解）；R-10 的 `protocol.generated.ts` 等生成代码必须豁免

本文件定义「文件体量棘轮守卫」的接口契约。它复用项目已有的 baseline + 守卫模式，不发明新机制。

---

## 1. 目标

建立机械约束，防止代码体量债务继续增长：

- **已有超线文件只许变小**（允许 ±5% 容差，避免格式化/注释导致误报）。
- **新文件不许超 800 行**（`AGENTS.md` 基础约束 3 的预警线）。

---

## 2. 基线文件规格

### 2.1 路径

```
governance/file-size-baseline.json
```

新建 `governance/` 目录（与 `scripts/` 平级），因为它是治理数据，不是可执行脚本。

### 2.2 格式

```json
{
  "generatedAt": "2026-06-11",
  "policy": "Prevents code bloat. Existing oversized files may only shrink (±5% tolerance). New files must not exceed 800 lines. See AGENTS.md § 基础约束 3 and internal/refactor/README.md.",
  "frontend": {
    "scanPaths": ["src/**/*.ts", "src/**/*.tsx"],
    "excludePatterns": ["**/*.test.ts", "**/*.test.tsx", "**/*.d.ts"],
    "oversizedFrozen": [
      {
        "path": "src/components/agent/chat/AgentChatWorkspace.tsx",
        "lines": 7029,
        "comment": "待拆分（R-03）"
      },
      {
        "path": "src/components/agent/chat/workspace/useWorkspaceSendActions.ts",
        "lines": 5117,
        "comment": "待拆分（R-06）"
      }
      // ... 共 107 个前端超 1000 行文件
    ]
  },
  "rust": {
    "scanPaths": ["lime-rs/crates/**/*.rs"],
    "excludePatterns": ["**/*_test.rs", "**/tests/**"],
    "oversizedFrozen": [
      {
        "path": "lime-rs/crates/agent-rust/crates/agent/src/agents/agent.rs",
        "lines": 8206,
        "comment": "待拆分（R-05）"
      },
      {
        "path": "lime-rs/crates/app-server/src/runtime.rs",
        "lines": 8010,
        "comment": "待拆分（R-04）"
      }
      // ... 共 126 个 Rust 超 1000 行文件
    ]
  },
  "thresholds": {
    "newFileMax": 800,
    "frozenTolerance": 0.05
  }
}
```

### 2.3 字段说明

- `generatedAt`：基线生成日期（`YYYY-MM-DD`）。
- `policy`：策略说明（会被守卫脚本输出）。
- `frontend` / `rust`：分语言配置。
  - `scanPaths`：glob 模式，扫描范围。
  - `excludePatterns`：glob 模式，排除测试文件。
  - `oversizedFrozen`：超线文件列表。
    - `path`：相对仓库根的路径（POSIX 风格，`/` 分隔）。
    - `lines`：当前行数（基线快照）。
    - `comment`：可选，说明何时拆分（方便追踪）。
- `thresholds`：
  - `newFileMax`：新文件行数上限（800）。
  - `frozenTolerance`：冻结文件容差（0.05 = ±5%）。

---

## 3. 守卫脚本规格

### 3.1 路径

```
scripts/check-file-size-governance.mjs
```

### 3.2 依赖

```javascript
import fs from "node:fs";
import path from "node:path";
import { glob } from "glob";  // 需在 package.json 已有（如果没有，用 fast-glob）
```

### 3.3 核心逻辑（伪代码）

```javascript
function main() {
  const baseline = readBaseline("governance/file-size-baseline.json");
  const violations = [];

  // 检查前端
  const frontendFiles = scanFiles(baseline.frontend.scanPaths, baseline.frontend.excludePatterns);
  for (const file of frontendFiles) {
    const currentLines = countLines(file.path);
    const frozen = baseline.frontend.oversizedFrozen.find(f => f.path === file.path);
    
    if (frozen) {
      // 已在基线：只能变小或在容差内
      const baselineLines = frozen.lines;
      const tolerance = Math.ceil(baselineLines * baseline.thresholds.frozenTolerance);
      if (currentLines > baselineLines + tolerance) {
        violations.push({
          path: file.path,
          issue: "oversized file grew",
          baseline: baselineLines,
          current: currentLines,
          allowed: baselineLines + tolerance,
        });
      }
    } else {
      // 新文件：不许超 800 行
      if (currentLines > baseline.thresholds.newFileMax) {
        violations.push({
          path: file.path,
          issue: "new file exceeds threshold",
          current: currentLines,
          threshold: baseline.thresholds.newFileMax,
        });
      }
    }
  }

  // 检查 Rust（逻辑同上，扫 baseline.rust）
  // ...

  // 输出结果
  if (violations.length > 0) {
    console.error(`\n❌ File size governance failed (${violations.length} violations):\n`);
    console.error(baseline.policy);
    console.error("");
    for (const v of violations) {
      if (v.issue === "oversized file grew") {
        console.error(`  ${v.path}`);
        console.error(`    Baseline: ${v.baseline} lines, Current: ${v.current} lines`);
        console.error(`    Allowed max: ${v.allowed} lines (baseline + 5% tolerance)`);
        console.error(`    ❌ File grew by ${v.current - v.baseline} lines. Oversized files may only shrink.`);
      } else if (v.issue === "new file exceeds threshold") {
        console.error(`  ${v.path}`);
        console.error(`    Current: ${v.current} lines, Threshold: ${v.threshold} lines`);
        console.error(`    ❌ New files must not exceed ${v.threshold} lines. Split into smaller modules.`);
      }
      console.error("");
    }
    console.error(`Hint: If you intentionally split a file, update governance/file-size-baseline.json.`);
    process.exit(1);
  } else {
    console.log("✅ File size governance passed.");
  }
}

function readBaseline(path) {
  const raw = fs.readFileSync(path, "utf8");
  return JSON.parse(raw);
}

function scanFiles(scanPaths, excludePatterns) {
  // 使用 glob 扫描，返回 {path: string, lines: number}[]
  // ...
}

function countLines(filePath) {
  const content = fs.readFileSync(filePath, "utf8");
  return content.split(/\r?\n/).length;
}
```

### 3.4 输出格式（示例）

**通过时**：

```
✅ File size governance passed.
```

**失败时**：

```
❌ File size governance failed (3 violations):

Prevents code bloat. Existing oversized files may only shrink (±5% tolerance). New files must not exceed 800 lines. See AGENTS.md § 基础约束 3 and internal/refactor/README.md.

  src/components/agent/chat/AgentChatWorkspace.tsx
    Baseline: 7029 lines, Current: 7150 lines
    Allowed max: 7380 lines (baseline + 5% tolerance)
    ❌ File grew by 121 lines. Oversized files may only shrink.

  src/components/workspace/NewLargeComponent.tsx
    Current: 950 lines, Threshold: 800 lines
    ❌ New files must not exceed 800 lines. Split into smaller modules.

  lime-rs/crates/app-server/src/runtime.rs
    Baseline: 8010 lines, Current: 8200 lines
    Allowed max: 8410 lines (baseline + 5% tolerance)
    ❌ File grew by 190 lines. Oversized files may only shrink.

Hint: If you intentionally split a file, update governance/file-size-baseline.json.
```

---

## 4. npm scripts 集成

### 4.1 新增 script

在 `package.json` 的 `scripts` 字段补：

```json
{
  "scripts": {
    "governance:file-size": "node scripts/check-file-size-governance.mjs"
  }
}
```

### 4.2 挂入验证链

修改 `verify:local`（或 `verify:local:full`）：

```json
{
  "scripts": {
    "verify:local": "npm run typecheck && npm run lint && npm run governance:file-size && npm run test:contracts",
    "verify:local:full": "npm run verify:local && npm run test -- --run"
  }
}
```

顺序建议：typecheck → lint → **governance:file-size** → test:contracts → test（越轻量的检查越靠前）。

---

## 5. CI 集成

### 5.1 GitHub Actions 示例

在 `.github/workflows/verify.yml`（或对应 CI 配置）补一步：

```yaml
- name: Check file size governance
  run: npm run governance:file-size
```

插入位置建议：在 `npm run lint` 之后、`npm test` 之前。

---

## 6. 基线更新流程

### 6.1 何时更新

- **拆分完成后**：P2 每完成一个 R-xx 条目（如 R-03 拆完 `AgentChatWorkspace.tsx`），更新基线。
- **定期巡检**：每 2 周扫一次，把已降到 <800 行的文件从 `oversizedFrozen` 移除。

### 6.2 更新步骤（手动）

1. 找到基线文件 `governance/file-size-baseline.json`。
2. 找到对应条目（如 `src/components/agent/chat/AgentChatWorkspace.tsx`）。
3. 如果文件已拆分且新行数 <800：从 `oversizedFrozen` 数组删除该条目。
4. 如果文件仍 >800 但已缩小：更新 `lines` 字段为新行数，保留 `comment`。
5. Commit message：`chore: update file-size baseline after R-03 split`。

### 6.3 自动更新脚本（可选，P2 后期补）

路径：`scripts/governance/update-file-size-baseline.mjs`

功能：
- 扫描当前代码，重新计算所有 `oversizedFrozen` 文件的行数。
- 自动移除已降到 <800 行的文件。
- 输出 diff，供人工审查后 commit。

执行：`npm run governance:file-size:update`（手动触发，不入 CI）。

---

## 7. 边界情况处理

### 7.1 格式化导致行数微变

**场景**：Prettier / Rust fmt 改了换行，文件从 7029 → 7035 行（+6 行，<5% 容差）。  
**处理**：守卫通过（在容差内）。  
**最佳实践**：提交前先跑 `npm run format`，避免格式化与业务改动混在一起。

### 7.2 注释/文档增加

**场景**：给超线文件补 JSDoc，行数从 7029 → 7080（+51 行，<5% 容差 351）。  
**处理**：守卫通过。  
**最佳实践**：如果注释导致超容差，说明文件确实该拆了。

### 7.3 生成代码

**场景**：`protocol.ts` 是从 Rust schema 生成的，可能很大。  
**处理**：
- 方案 A：生成代码不入基线（在 `excludePatterns` 加 `**/generated/**` 或特定路径）。
- 方案 B：生成代码入基线，但容差放宽到 10%（在基线 `comment` 标注 `generated`）。

当前推荐 **方案 A**（生成代码不受棘轮约束，但生成逻辑本身要优化）。

### 7.4 测试文件

**场景**：`AgentChatWorkspace.test.tsx` 可能很大（测试用例多）。  
**处理**：测试文件已被 `excludePatterns` 排除，不受棘轮约束。  
**原因**：测试文件大小与业务复杂度正相关，不应机械限制；但如果 >2000 行，建议拆 test suite。

---

## 8. 实现检查清单（给 Codex）

R-02 执行时，按此清单逐项验证：

- [ ] 创建 `governance/` 目录（如果不存在）。
- [ ] 生成 `governance/file-size-baseline.json`：
  - [ ] 扫描 `src/**/*.{ts,tsx}`，排除 `*.test.*` 和 `*.d.ts`。
  - [ ] 扫描 `lime-rs/crates/**/*.rs`，排除 `*_test.rs` 和 `tests/`。
  - [ ] 对 >800 行文件，记录 `{path, lines, comment}`。
  - [ ] 前端预期 ~148 条（>800 行），Rust 预期 ~126 条（>1000 行）。
- [ ] 实现 `scripts/check-file-size-governance.mjs`：
  - [ ] 逻辑：已有文件 ±5% 容差、新文件 <800 行。
  - [ ] 输出：失败时清晰 diff，通过时简短确认。
  - [ ] 错误码：失败时 `process.exit(1)`。
- [ ] 补 `package.json` scripts：
  - [ ] `"governance:file-size": "node scripts/check-file-size-governance.mjs"`
  - [ ] 挂入 `verify:local` 依赖链。
- [ ] 补 CI 检查（`.github/workflows/verify.yml` 或对应文件）。
- [ ] 补文档：
  - [ ] `scripts/README.md` 补「文件体量棘轮」一节。
  - [ ] `AGENTS.md` 基础约束 3 补一句「守卫见 `governance:file-size`」。
- [ ] 验证：
  - [ ] 当前代码运行 `npm run governance:file-size` 应通过。
  - [ ] 手动改一个基线文件 +100 行 → 应报错。
  - [ ] 手动新增一个 900 行文件 → 应报错。
  - [ ] 手动改一个基线文件 -50 行 → 应通过。
  - [ ] Revert 上述手动改动。
- [ ] Commit：
  - [ ] Message：`feat(governance): add file-size ratchet guard (R-02)`
  - [ ] 包含：`governance/file-size-baseline.json`、`scripts/check-file-size-governance.mjs`、`package.json`、`scripts/README.md`、`AGENTS.md`、CI 配置。

---

## 9. 性能考量

### 9.1 扫描性能

- 当前仓库 ~175 万行，扫描 + 计数预计 <5s（Node.js `fs.readFileSync` + 简单 split）。
- 如果 >10s，考虑：
  - 用 `wc -l` 调系统命令（更快，但跨平台兼容性差）。
  - 缓存行数（在基线文件加 `lastModified` 字段，只扫改过的文件）。

### 9.2 CI 影响

- 本地 `verify:local` 增加 <5s，可接受。
- CI parallel jobs 可与其他检查并行（如 lint 一个 job，governance 另一个 job）。

---

## 10. 后续优化方向（P2 完成后）

1. **自动基线更新脚本**（§ 6.3）。
2. **行数趋势图**：定期（每周）扫描，生成「超线文件数 vs 时间」曲线，可视化债务偿还进度。
3. **热点文件识别**：结合 `git log --numstat` 统计改动频率，输出「高频改动 × 大文件」Top 10，指导 P2 优先级。
4. **IDE 集成**：VS Code extension，编辑超线文件时在状态栏提示「此文件已冻结，请缩小」。

---

## 附录：参考实现片段

完整实现由 Codex 完成，这里只给关键函数骨架。

### A.1 countLines（精确版）

```javascript
function countLines(filePath) {
  const content = fs.readFileSync(filePath, "utf8");
  // 排除空行和只有空白的行（可选，看团队约定）
  // const lines = content.split(/\r?\n/).filter(line => line.trim().length > 0);
  const lines = content.split(/\r?\n/);
  return lines.length;
}
```

### A.2 scanFiles（使用 glob）

```javascript
import { glob } from "glob";

function scanFiles(scanPaths, excludePatterns) {
  const allFiles = [];
  for (const pattern of scanPaths) {
    const files = glob.sync(pattern, {
      ignore: excludePatterns,
      nodir: true,
      posix: true,  // 统一用 / 分隔（跨平台）
    });
    allFiles.push(...files);
  }
  return allFiles.map(path => ({
    path: path.replace(/\\/g, "/"),  // Windows 兼容
    lines: countLines(path),
  }));
}
```

### A.3 比对逻辑（核心）

```javascript
function checkFile(file, baseline, thresholds, isRust = false) {
  const bucket = isRust ? baseline.rust : baseline.frontend;
  const frozen = bucket.oversizedFrozen.find(f => f.path === file.path);
  
  if (frozen) {
    // 已在基线：只能变小或在容差内
    const baselineLines = frozen.lines;
    const tolerance = Math.ceil(baselineLines * thresholds.frozenTolerance);
    if (file.lines > baselineLines + tolerance) {
      return {
        path: file.path,
        issue: "oversized file grew",
        baseline: baselineLines,
        current: file.lines,
        allowed: baselineLines + tolerance,
      };
    }
  } else {
    // 新文件：不许超阈值
    if (file.lines > thresholds.newFileMax) {
      return {
        path: file.path,
        issue: "new file exceeds threshold",
        current: file.lines,
        threshold: thresholds.newFileMax,
      };
    }
  }
  
  return null;  // 通过
}
```

---

**结束**。Codex 实现 R-02 时，按本规格实现即可，细节可根据实际情况调整（如 glob 库选型、输出格式润色），但核心逻辑（棘轮 + 容差）不变。
