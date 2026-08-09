<div align="center"><a name="readme-top"></a>

<img src="./docs/images/readme-hero.png" alt="Lime README 主视觉：青柠一下，灵感即来" width="100%" />

# Lime

### 一个真正能把事情推进下去的 Agent

**面向全球用户的开源全栈桌面 AI Agent**

代码、文件、终端、工具、研究、内容、多模态和多 Agent 协作，都在同一个可恢复的任务空间里完成。

[English](./README.md) · **简体中文** · [文档](./docs/README.md) · [发布记录](./RELEASE_NOTES.md) · [问题反馈](https://github.com/limecloud/lime/issues)

<p>
  <a href="https://github.com/limecloud/lime/releases"><img src="https://img.shields.io/github/v/release/limecloud/lime?label=release" alt="Lime GitHub Release" /></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-246B45" alt="Lime supports macOS and Windows" />
  <img src="https://img.shields.io/badge/desktop-Electron-24C8DB" alt="Lime is an Electron desktop app" />
  <img src="https://img.shields.io/badge/license-GPLv3-2F4F4F" alt="Lime GPLv3 license" />
</p>

</div>

---

## Lime 是什么

Lime 是一个全栈桌面 AI Agent。它能理解目标和工作区，读取和修改文件，运行终端命令，调用工具、MCP 和 Skills，处理多模态输入，生成可交付 artifact，并把整个过程保留在 Thread / Turn / Item 中。

它和 Claude Code、WorkBuddy、Codex 属于同一类“能动手完成任务”的 Agent 产品，同时提供桌面 GUI、可视化工作区、Provider 选择和跨工程/研究/内容的统一工作流。

## 能力一览

| 能力 | 可以完成的工作 |
| --- | --- |
| 代码与工程 | 理解仓库、定位问题、跨文件修改、运行测试、解释 diff |
| 文件与终端 | 读写文件、搜索目录、启动进程、查看输出、管理长任务 |
| 工具与扩展 | 使用 MCP、Skills、浏览器和受控工具扩展执行范围 |
| 多模态 | 理解文本、代码、图片、截图、音频、视频、PDF、表格和结构化数据 |
| 生成与交付 | 生成文档、图片、音频、视频、图表、网页草稿和结构化 artifact |
| 协作与恢复 | 多 Agent 分工、权限审批、取消/重试、历史恢复和持续执行 |

## 一次任务如何工作

1. 写下目标、约束和验收标准，选择工作区或项目目录。
2. Agent 读取必要上下文，先给出计划和需要确认的边界。
3. 在授权范围内修改文件、运行命令、调用工具和生成结果。
4. 检查 diff、命令输出、测试结果和 artifact，继续追问或结束任务。

## 核心工作区

### 从一个目标开始

<img src="./docs/images/readme-feature-start.png" alt="Lime 从一个任务开始功能图" />

从一句话、一个仓库、一组资料或一张截图开始，Agent 会先建立上下文，而不是直接猜答案。

### 执行与审阅在同一个 Thread

<img src="./docs/images/readme-feature-workspace.png" alt="Lime 同一空间持续打磨功能图" />

对话、计划、文件变更、终端输出、工具结果和生成物都可回看。高风险动作可以逐项批准、拒绝、重试或暂停。

### 连接自己的模型与工具

<img src="./docs/images/readme-feature-provider.png" alt="Lime 使用自己的 AI 服务功能图" />

Lime 不绑定单一模型服务。配置 Provider、模型和凭证后，可以按任务切换能力，并通过 MCP 与 Skills 扩展 Agent。

## 快速开始

从 [Releases](https://github.com/limecloud/lime/releases) 下载 macOS 或 Windows 安装包。

1. 打开 Lime，配置 Provider 并测试模型连接。
2. 选择工作区，确认文件和终端权限。
3. 新建 Agent Thread，写下目标和验收标准。
4. 让 Agent 先规划，再批准需要执行的动作。

macOS 也可以使用 Homebrew：

```bash
brew tap aiclientproxy/tap
brew install --cask lime
```

## 数据与权限

项目资料、会话历史和配置默认保存在本机。调用模型或外部工具时，相关输入会发送到你配置的 Provider 或目标服务。文件修改、终端命令和外部工具调用遵循权限与审批边界。

## 文档与社区

- [文档](./docs/README.md)
- [发布记录](./RELEASE_NOTES.md)
- [GitHub Issues](https://github.com/limecloud/lime/issues)

## 开源协议

[GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0)

本项目仅供学习研究使用，用户需自行承担使用风险。模型能力由用户配置的第三方服务提供。

---

<div align="center">

### 微信交流

<img src="./docs/images/coso.jpg" alt="Lime 微信交流群二维码" width="180" />

扫码加微信，备注 `Lime`，拉你进群讨论。

</div>
