<div align="center"><a name="readme-top"></a>

<img src="./docs/images/readme-hero.png" alt="Lime README 主视觉：青柠一下，灵感即来" width="100%" />

# Lime

### 让 Agent 真正把事情做完

**开源、全栈、桌面端 AI Agent**

Full-stack AI agent for coding, files, terminals, tools, research, content, multimodal work, and multi-agent workflows.

**简体中文** · [English](./README.en.md) · [文档](./docs/README.md) · [发布记录](./RELEASE_NOTES.md) · [问题反馈](https://github.com/limecloud/lime/issues)

<p>
  <a href="https://github.com/limecloud/lime/releases"><img src="https://img.shields.io/github/v/release/limecloud/lime?label=release" alt="Lime GitHub Release" /></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-246B45" alt="Lime supports macOS and Windows" />
  <img src="https://img.shields.io/badge/desktop-Electron-24C8DB" alt="Lime is an Electron desktop app" />
  <img src="https://img.shields.io/badge/license-GPLv3-2F4F4F" alt="Lime GPLv3 license" />
</p>

Lime 不只是回答问题的聊天框，而是可以理解上下文、调用工具、修改文件、运行命令、整理资料、生成交付物并持续推进任务的桌面 AI Agent。

</div>

---

<details>
<summary><kbd>目录</kbd></summary>

- [Lime 是什么](#lime-是什么)
- [全栈 Agent 能做什么](#全栈-agent-能做什么)
- [一次任务如何推进](#一次任务如何推进)
- [核心能力](#核心能力)
- [适合谁](#适合谁)
- [产品定位](#产品定位)
- [快速开始](#快速开始)
- [技术栈与平台](#技术栈与平台)
- [常见问题](#常见问题)
- [开源协议](#开源协议)
- [免责声明](#免责声明)

</details>

---

## Lime 是什么

Lime 是一个开源的全栈 AI Agent 桌面应用。它把 Agent loop、文件系统、终端进程、代码修改、工具调用、MCP、Skills、多模态输入输出、多模型路由和多 Agent 协作放进同一条可追踪的任务链。

它和 Claude Code、WorkBuddy、Codex 属于同一类“能动手完成任务”的 Agent 产品，但 Lime 更强调桌面 GUI、可视化工作区、可配置 Provider，以及面向中文用户的研究、创作和工程混合场景。

Lime 的基本工作方式是：

- 先理解目标、仓库、文件、历史会话和约束，再给出可执行计划
- 根据权限执行读写文件、搜索、补丁、终端命令、测试和其他工具调用
- 将过程投影为 Thread、Turn、Item 和可复用 artifact，方便中断、恢复、审阅和继续
- 让你选择自己的 Provider 和模型，在不同任务中切换能力而不改变工作上下文

你可以把 Lime 当作一个坐在桌面上的工程搭档、研究助理、内容合作者和自动化执行器：你给出目标和边界，它负责把目标拆成动作并留下可检查的结果。

---

## 全栈 Agent 能做什么

- **代码理解与修改**：浏览仓库、定位问题、跨文件实现功能、重构、补测试、生成 patch 并解释变更
- **终端与进程操作**：在受控权限下运行脚本、构建、测试、安装依赖、观察输出和管理长任务
- **文件与工作区协作**：读写文本和结构化文件，整理目录，生成文档、报告、网页草稿和其他 artifact
- **工具、MCP 与 Skills**：发现可用能力，调用外部工具或本地 Skill，把重复流程变成可复用的执行单元
- **研究与内容交付**：处理资料、截图、图片和多轮对话，完成从分析、写作到发布准备的连续工作
- **多 Agent 与长任务**：拆分子任务、并行推进、保留状态，在同一 Thread 中恢复和继续复杂工作
- **模型与 Provider 控制**：按任务选择模型，管理能力目录、凭证、路由、重试和故障边界

---

## 一次任务如何推进

### 1. 修复一个真实 Bug

你把仓库和报错交给 Lime。Agent 会先读取相关文件和配置，定位调用链，说明假设，再修改实现、运行相关测试并展示 diff。

你可以在同一个 Thread 里继续追问“为什么这样改”“还有哪些边界”“把修复同步到文档”，每一步都有状态和结果可回看。

### 2. 从需求交付一个全栈功能

需求进入后，Lime 可以先拆分前端、App Server、Rust runtime、协议和测试，再按依赖顺序执行。你可以批准每个高风险动作，也可以随时暂停、修改计划或回滚未提交改动。

### 3. 把资料变成可交付内容

将网页、笔记、截图、会议记录和历史结果放入任务，Agent 可以整理结构、指出缺口、生成报告、脚本、方案或发布稿，并保留引用上下文供你复核。

### 4. 把重复流程变成 Skill

把常用的检查、发布、研究或团队规范写成 Skill，Agent 在需要时发现并执行它。工具通过 MCP 或受控能力接入，不需要把全部步骤重新写进每次提示词。

### 5. 多 Agent 协作处理大任务

将研究、实现、测试、文档等子任务交给不同 Agent，主 Thread 汇总结果并保持统一上下文、权限和审阅边界。

---

## 核心能力

### 从一个目标开始

<img src="./docs/images/readme-feature-start.png" alt="Lime 从一个任务开始功能图" />

输入一句目标、一个仓库、一个目录或一组资料。Agent 会先建立上下文，再提出计划和需要你确认的边界。

### 在同一个工作区里执行和审阅

<img src="./docs/images/readme-feature-workspace.png" alt="Lime 同一空间持续打磨功能图" />

对话、计划、文件变更、命令输出、工具结果和生成物都围绕同一个 Thread 展示。你能逐步批准、拒绝、重试、继续或恢复任务。

### 连接自己的模型和工具

<img src="./docs/images/readme-feature-provider.png" alt="Lime 使用自己的 AI 服务功能图" />

Lime 不绑定单一模型服务。你可以配置 Provider、模型和凭证，并通过 MCP、Skills 和受控工具扩展 Agent 的执行范围。

---

## 适合谁

- 需要读代码、改代码、跑测试和交付功能的开发者
- 需要同时处理产品、设计、数据、文档和自动化的全栈团队
- 需要本地资料、终端工具和多轮推理的研究者与内容创作者
- 希望把团队规范、检查流程和工具能力沉淀为 Skills 的团队
- 想在桌面 GUI 中获得类似 Claude Code、WorkBuddy、Codex 工作方式的用户

---

## 产品定位

Lime 属于全栈 AI Agent / coding agent / desktop AI agent / terminal agent 这一产品类别。它可以承担代码任务，也可以承担研究、写作、资料整理、自动化和多模态交付；重点不是生成一段文本，而是围绕目标完成一组可验证的动作。

---

## 快速开始

### 下载安装

从 [Releases](https://github.com/limecloud/lime/releases) 下载对应平台安装包。

- macOS 用户下载 `.dmg` 或使用 Homebrew 安装
- Windows 用户下载 `Lime_*_x64-setup.exe`
- 当前仅提供 macOS 与 Windows 发布包，Linux 桌面端已暂停支持
- 如果 Windows 出现 SmartScreen 提示，通常是未签名或签名信誉不足导致，不代表安装包一定损坏

会使用 Homebrew 的 macOS 用户也可以运行：

```bash
brew tap aiclientproxy/tap
brew install --cask lime
```

### 第一次使用

1. 打开 Lime，进入 Provider 配置页并测试模型连接
2. 选择一个工作区或项目目录，确认文件和终端权限
3. 新建 Agent Thread，写下目标、约束和验收标准
4. 先让 Agent 给出计划，再按需批准文件、命令和外部工具调用
5. 检查 diff、测试结果和生成物，继续追问或结束任务

---

## 技术栈与平台

- 桌面框架：Electron、Rust App Server、App Server JSON-RPC
- Agent runtime：Thread / Turn / Item projection、工具生命周期、Skills、MCP、多 Agent 和历史恢复
- 前端技术：React、TypeScript、Vite
- 本地能力：文件系统、进程、工作区、artifact 和持久化状态
- 支持平台：macOS、Windows
- 开源协议：GPLv3

---

## 常见问题

### Lime 会提供 AI 模型吗？

不会。Lime 是 Agent 宿主和工作区，不直接销售模型服务。你需要配置可用的 Provider、模型和凭证。

### Agent 能修改代码和运行命令吗？

可以。在你授予的权限范围内，Agent 可以读取和修改文件、执行终端命令、运行测试并调用工具；高风险动作应先审阅或批准。

### 我的资料会全部上传吗？

项目资料、会话历史和配置优先保存在本机。调用模型或外部工具时，相关输入会发送到你配置的 Provider 或目标服务；敏感资料请按对应服务商政策判断。

### 它和普通聊天工具有什么不同？

普通聊天工具主要返回文本。Lime 会理解工作区上下文，实际操作文件、进程和工具，保留执行状态与结果，并允许你在同一任务中继续推进。

### 需要会写复杂提示词吗？

不需要。直接描述目标、约束、上下文和验收方式即可；可复用的做法可以沉淀为 Skill，复杂任务也可以让 Agent 先规划再执行。

---

## 开源协议

[GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0)

## 免责声明

本项目仅供学习研究使用，用户需自行承担使用风险。
本项目不直接提供 AI 模型服务，模型能力由第三方服务商提供。

---

<div align="center">

### 微信交流

<img src="./docs/images/coso.jpg" alt="Lime 微信交流群二维码" width="180" />

扫码加微信，备注 `Lime`，拉你进群讨论。

</div>
