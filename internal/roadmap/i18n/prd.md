# Lime 全球本地化 PRD

> 状态：current
> 更新时间：2026-09-02
> Owner：`src/i18n/`、各 locale namespace 与 i18n 治理脚本

## 1. 目标

所有用户可见界面和错误提示统一使用 key-based i18n，稳定覆盖：

- `zh-CN`
- `zh-TW`
- `en-US`
- `ja-JP`
- `ko-KR`

UI locale、Agent 输出语言、浏览器环境语言和用户创作目标语言是不同概念，不共用一个无语义的 `language` 状态。

## 2. 唯一事实源

```text
src/i18n/resources/<locale>/<namespace>.json
  -> typed resource keys
  -> namespace loader
  -> react-i18next consumer
```

规则：

1. 组件不得维护第二份翻译字典或中文 `defaultValue`。
2. 运行时数据、协议枚举、文件名和用户输入不做资源化。
3. 插值、复数和上下文变化使用 i18next 标准能力，不拼接半句翻译。
4. 新增或删除 key 必须同步五种 locale、typed key 测试和 unused-key 检查。
5. Electron、App Server 和 Rust 错误先归一为稳定错误码，再由 Renderer 翻译用户文案。
6. `document.documentElement.lang` 必须跟随 UI locale；方向支持通过统一 locale metadata 决定。

## 3. Namespace

namespace 按稳定产品领域拆分。新增 namespace 需要有独立加载收益和清晰 owner，不能仅为单个组件创建。

核心 namespace 随桌面首屏加载；低频设置或领域页面可以延迟加载，但加载失败必须显示稳定 fallback 状态，不能静默回退硬编码文本。

已删除功能的 namespace/key 直接移除，不保留空壳或兼容别名。

## 4. Locale 与持久化

- UI locale 使用 BCP 47 tag，并通过统一 normalizer 处理系统值。
- 未支持的 locale 回退到 `en-US` 或产品明确配置的默认 locale。
- locale 设置走 App Server/config current 边界；Electron 只提供系统 locale 事实。
- Browser Workspace 的 locale、timezone 和 `Accept-Language` 是站点环境设置，不自动覆盖 UI locale。
- Agent 输出语言由 Turn 输入或用户偏好决定，不根据按钮语言猜测。

## 5. 迁移与删除

`current`：typed resource、namespace loader、locale registry、`Intl` formatter、五语言回归。

`deprecated`：DOM 文本替换层，只允许迁出，不接受新 key 或新页面。

`dead`：组件内翻译 map、旧功能专属 namespace、测试内完整资源镜像和生产 fallback 文案。

迁移一个 surface 时，同一变更集必须：

1. 替换所有用户可见 literal。
2. 补齐五语言 key。
3. 删除旧 patch/map/key。
4. 更新 typed key 和 namespace loader 测试。
5. 验证布局不会因长文本溢出或遮挡。

## 6. 验收

- 五种 locale 的 namespace/key 集合一致。
- 无未使用 key、缺失 key和组件级 `defaultValue` fallback。
- 日期、数字、相对时间和列表连接使用 locale-aware formatter。
- 切换 locale 后当前页面、弹窗和后续异步状态一致更新。
- GUI smoke 至少覆盖设置切换、Agent Workspace 与一个长文本页面。

最低验证：

```bash
npm run i18n:check
npm run typecheck
npm run test:contracts
npm run verify:gui-smoke
```

## 7. 非目标

- 不在首期支持所有语言或自动翻译所有用户内容。
- 不通过翻译资源改变 protocol、model id、tool name 或持久化 schema。
- 不保留已删除产品对象的文案键作为所谓兼容资产。
