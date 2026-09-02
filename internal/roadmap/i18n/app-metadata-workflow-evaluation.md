# Lime installer / app metadata i18n 评估

> 关联 PRD：`internal/roadmap/i18n/prd.md`
> 关联 inventory：`internal/roadmap/i18n/evidence/app-metadata-workflow-inventory.json`
> 关联 scope：`internal/roadmap/i18n/app-metadata-translation-scope.json`
> 评估时间：2026-05-27

## 评估目标

判断 Lime 的 installer、应用元数据和壳层描述，是否已经具备独立的多语言工作流。

## 当前事实

- 根级 `package.json`、`lime-rs/Cargo.toml` 与 `forge.config.mjs` 是当前 installer / app metadata 审阅事实源。
- 这些字段的真实配置目前仍是单语事实源；多语言发布前的审阅链路现在由 metadata translation scope 与 build-time locale manifest 承担。
- `forge.config.mjs` 的 `PRODUCT_NAME`、`APP_ID`、deep-link scheme、macOS / Windows icon 与 Forge maker target 是当前 Electron 发布元数据事实源。
- `package.json` 与 `lime-rs/Cargo.toml` 的 description 仍是单一英文描述，不是分 locale 的 metadata bundle。
- `internal/roadmap/i18n/app-metadata-translation-scope.json` 已定义 installer / app metadata 的最小 ownership 与字段分类：`forge.config.mjs#productName`、`appId` 与 deep-link scheme 属于稳定品牌 / 标识字段，`package.json#description` 属于多语言发布前需要处理的 translatable 字段。
- `internal/roadmap/i18n/evidence/app-metadata-locale-build-manifest.json` 已把 scope 转成 build-time locale manifest；当前 `workflowStatus=ready`，9 个 metadata entry 中 1 个 localized entry、3 个 stable entry、5 个 source-only entry，missing field 与 required localized missing 均为 `0`。
- `internal/roadmap/i18n/evidence/app-metadata-workflow-inventory.json` 已增加 metadata field coverage 与 manifest readiness：当前审计到 `9` 个真实 app / installer metadata 字段，全部纳入 scope，`metadataUnscopedFieldCount=0`、`metadataMissingScopedFieldCount=0`，且 `appMetadataLocaleBuildManifestReady=true`、`hasInstallerLocalizationWorkflow=true`。
- 旧 Tauri 宿主配置文件已按 `dead` release / metadata surface 下线；它们不是 current app metadata、installer、release、updater、签名或版本同步事实源，也不能作为 i18n evidence 输入回流。

## 结论

当前已经具备独立的 installer / app metadata build-time locale manifest workflow：它能按 scope 审阅 source locale、`en-US` localized values、stable brand / identifier 与 source-only 字段，并在发布前暴露缺失字段或 required localized value 缺口。真实 `package.json`、`forge.config.mjs` 与平台 installer 配置仍保持单语 source，不由本轮 workflow 自动改写。

## 现状评价

1. app / installer 元数据确实已经有若干文本字段，但它们不是可按 locale 切换的资源。
2. 当前仓库是“单份真实配置 + build-time locale manifest”形态；这已经满足发布前审阅 workflow，但不是平台 installer metadata 生成器。
3. Metadata translation scope 已经定义 source locale、owner、哪些字段允许本地化、哪些字段必须保持稳定，以及当前是否允许生成 metadata；`manifestGenerationAllowed=true` 只允许生成审阅 manifest，`generatedMetadataAllowed=false` 继续禁止改写真实安装器配置。
4. Metadata field coverage 已经能发现两类 drift：真实配置里新增 app / installer metadata 字段但未进入 scope，以及 scope 引用了已经不存在的字段。
5. App metadata locale build manifest 已经能发现两类发布前缺口：scope 字段缺失，以及 translatable 字段缺少目标 locale localized value。
6. 若要做多语言 installer / metadata 生成，下一步必须先设计 Electron Forge / 平台发布链路如何消费这些值，而不是手工复制多份平台配置。

## 建议工作流

- 新增或改动 app / installer metadata 字段时，同步更新 `app-metadata-translation-scope.json`，并刷新 inventory；`metadataUnscopedFieldCount` 与 `metadataMissingScopedFieldCount` 应保持为 `0`。
- 新增或改动 translatable metadata 字段时，同步维护 `localizedValues`，并刷新 `app-metadata-locale-build-manifest.json`；`missingFieldCount` 与 `requiredLocalizedMissingCount` 应保持为 `0`。
- 继续维持单份真实元数据配置；如果后续要把字段抽成生成式资源，必须先设计平台 installer / package registry 的消费链路。
- 若未来接入多语言发布链路，应优先产出 build-time inventory / generator，而不是手工维护多份配置。
- 在 `generatedMetadataAllowed=false` 期间，不允许新增平行的 locale 配置文件或手工派生 installer metadata；只能更新 scope、locale manifest 和 inventory evidence。

## 重新评估条件

满足以下任一条件时，再推进独立 workflow：

1. 安装包和 App Store / Release 发布需要稳定的 `zh-CN / en-US` 元数据对照。
2. Electron Forge、Windows installer 或 macOS bundle 需要按 locale 生成不同 title / description。
3. 公开发布材料开始要求和桌面 App 保持同一套术语和语言覆盖。
4. installer / app metadata 进入 CI 校验或发布前审核门禁。

## 证据链接

- [app-metadata-translation-scope.json](/Users/coso/Documents/dev/ai/aiclientproxy/lime/internal/roadmap/i18n/app-metadata-translation-scope.json)
- [app-metadata-locale-build-manifest.json](/Users/coso/Documents/dev/ai/aiclientproxy/lime/internal/roadmap/i18n/evidence/app-metadata-locale-build-manifest.json)
- [app-metadata-workflow-inventory.json](/Users/coso/Documents/dev/ai/aiclientproxy/lime/internal/roadmap/i18n/evidence/app-metadata-workflow-inventory.json)
- [package.json](/Users/coso/Documents/dev/ai/aiclientproxy/lime/package.json)
- [lime-rs/Cargo.toml](/Users/coso/Documents/dev/ai/aiclientproxy/lime/lime-rs/Cargo.toml)
- [forge.config.mjs](/Users/coso/Documents/dev/ai/aiclientproxy/lime/forge.config.mjs)
