<div align="center"><a name="readme-top"></a>

<img src="./docs/images/readme-hero-en.png" alt="Lime README hero banner: An agent that can actually finish the work" width="100%" />

# Lime

### An agent that can actually finish the work

**Open-source full-stack desktop AI agent**

Full-stack AI agent for coding, files, terminals, tools, research, content, multimodal work, and multi-agent workflows worldwide.

[简体中文](./README.zh-CN.md) · **English** · [Feature Map](./FEATURE-MAP.md) · [Docs](./docs/README.md) · [Release Notes](./RELEASE_NOTES.en.md) · [Issues](https://github.com/limecloud/lime/issues)

<p>
  <a href="https://github.com/limecloud/lime/releases"><img src="https://img.shields.io/github/v/release/limecloud/lime?label=release" alt="Lime GitHub Release" /></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-246B45" alt="Lime supports macOS and Windows" />
  <img src="https://img.shields.io/badge/desktop-Electron-24C8DB" alt="Lime is an Electron desktop app" />
  <img src="https://img.shields.io/badge/license-GPLv3-2F4F4F" alt="Lime GPLv3 license" />
</p>

Lime is more than a chat box: it understands context, calls tools, edits files, runs commands, organizes material, creates deliverables, and keeps moving a task forward from one desktop workspace.

</div>

---

<details>
<summary><kbd>Table of Contents</kbd></summary>

- [What is Lime?](#what-is-lime)
- [What the full-stack agent can do](#what-the-full-stack-agent-can-do)
- [How a task moves forward](#how-a-task-moves-forward)
- [Core capabilities](#core-capabilities)
- [Who Lime is for](#who-lime-is-for)
- [Product positioning](#product-positioning)
- [Quick Start](#quick-start)
- [Tech Stack and Platforms](#tech-stack-and-platforms)
- [FAQ](#faq)
- [License](#license)
- [Disclaimer](#disclaimer)

</details>

---

## What is Lime?

Lime is an open-source full-stack desktop AI agent for users and teams worldwide. It brings the Agent loop, filesystem, terminal processes, code changes, tool calls, MCP, Skills, multimodal input and output, model routing, and multi-agent collaboration into one traceable task chain.

It belongs to the same category of hands-on agents as Claude Code, WorkBuddy, and Codex, while emphasizing a desktop GUI, visual workspace, configurable providers, and mixed engineering, research, and content workflows for global users and teams.

Lime works by:

- Understanding the goal, repository, files, history, and constraints before proposing an executable plan.
- Reading and writing files, searching, applying patches, running terminal commands, testing, and calling tools within granted permissions.
- Projecting work as Thread, Turn, Item, and reusable artifacts so tasks can be paused, reviewed, restored, and continued.
- Letting you choose providers and models without losing the surrounding task context.

Use Lime as an engineering partner, research assistant, content collaborator, or automation executor: you provide the goal and boundaries, and it turns them into inspectable actions and results.

---

## What the full-stack agent can do

- **Code understanding and changes**: inspect repositories, locate bugs, implement cross-file features, refactor, add tests, produce patches, and explain diffs.
- **Terminal and process operations**: run scripts, builds, tests, dependency commands, long-running processes, and inspect their output within controlled permissions.
- **Files and workspaces**: read and write text or structured files, organize directories, and create documents, reports, web drafts, and other artifacts.
- **Tools, MCP, and Skills**: discover capabilities, call external tools or local Skills, and turn repeatable procedures into reusable execution units.
- **Multimodal understanding and generation**: work with text, code, images, screenshots, audio, video, PDFs, tables, and structured data in one task, then create images, audio, video, documents, charts, and other artifacts.
- **Research and content delivery**: work with web pages, references, screenshots, images, audio, video, and multi-turn context from analysis through publish-ready output.
- **Multi-agent and long-running work**: split subtasks, make progress in parallel, preserve state, and resume complex work in the same Thread.
- **Provider and model control**: choose models per task and manage capability catalogs, credentials, routing, retries, and failure boundaries.

---

## How a task moves forward

### 1. Fix a real bug

Give Lime a repository and an error. The Agent reads the relevant files and configuration, traces the call path, states its assumptions, changes the implementation, runs focused tests, and shows the diff.

Continue in the same Thread with questions such as "why this change?" or "what are the edge cases?" Every step remains visible and reviewable.

### 2. Ship a full-stack feature

Lime can split a requirement across the frontend, App Server, Rust runtime, protocol, and tests, then execute in dependency order. Approve risky actions, pause, revise the plan, or inspect uncommitted changes at any point.

### 3. Turn material into a deliverable

Add web pages, notes, screenshots, meeting records, and prior results. The Agent can structure them, identify gaps, produce a report, script, plan, or launch draft, and keep context for review.

### 4. Turn a repeated procedure into a Skill

Encode a recurring check, release step, research method, or team rule as a Skill. The Agent can discover and run it through MCP or controlled capabilities instead of repeating every instruction in every prompt.

### 5. Coordinate multiple agents

Delegate research, implementation, testing, and documentation to different agents. The main Thread keeps shared context, permissions, and review boundaries while it collects the results.

---

## Core capabilities

### Start from one goal

<img src="./docs/images/readme-feature-start-en.png" alt="Lime start from one task feature image" />

Enter a goal, repository, directory, or set of materials. The Agent establishes context first, then proposes a plan and the boundaries that need your approval.

### Execute and review in one workspace

<img src="./docs/images/readme-feature-workspace-en.png" alt="Lime refine in one workspace feature image" />

Conversation, plans, file changes, command output, tool results, and artifacts stay around the same Thread. Approve, reject, retry, continue, or restore work step by step.

### Connect your models and tools

<img src="./docs/images/readme-feature-provider-en.png" alt="Lime use your own AI services feature image" />

Lime does not lock you to one model service. Configure providers, models, and credentials, then extend the Agent through MCP, Skills, and controlled tools.

---

## Who Lime is for

- Developers who need to read code, change code, run tests, and ship features.
- Full-stack teams handling product, design, data, documentation, and automation together.
- Researchers and creators who need local material, terminal tools, and long-running reasoning.
- Teams that want to encode rules, checks, and tools as reusable Skills.
- Users who want a desktop GUI with the working style of Claude Code, WorkBuddy, or Codex.

---

## Product positioning

Lime belongs to the full-stack AI agent, coding agent, desktop AI agent, and terminal agent category for users and teams worldwide. It handles code tasks as well as research, writing, material organization, automation, and multimodal delivery; the goal is a set of verifiable actions, not just a generated paragraph.

Multimodality is a core Agent workflow, not a separate attachment feature. One Thread can combine a written brief, a codebase, images and screenshots, voice or video material, PDFs, and spreadsheets so the Agent can cross-understand them, call tools, and produce delivery-ready results.

---

## Quick Start

### Download and install

Download the installer for your platform from [Releases](https://github.com/limecloud/lime/releases).

- macOS users can download the `.dmg` package or install with Homebrew.
- Windows users can download `Lime_*_x64-setup.exe`.
- Lime currently publishes macOS and Windows builds only. Linux desktop builds are paused.
- If Windows SmartScreen appears, it usually means the installer is unsigned or has not built enough signing reputation. It does not necessarily mean the installer is broken.

macOS users who use Homebrew can run:

```bash
brew tap aiclientproxy/tap
brew install --cask lime
```

### First run

1. Open Lime, configure a Provider, and test a model connection.
2. Choose a workspace or project directory and confirm file and terminal permissions.
3. Create an Agent Thread with a goal, constraints, and acceptance criteria.
4. Ask for a plan first, then approve file changes, commands, or external tool calls as needed.
5. Inspect the diff, test results, and artifacts; continue the task or close it.

---

## Tech Stack and Platforms

- Desktop framework: Electron, Rust App Server, App Server JSON-RPC.
- Agent runtime: Thread / Turn / Item projection, tool lifecycle, Skills, MCP, multi-agent work, and history restoration.
- Frontend: React, TypeScript, Vite.
- Local capabilities: filesystem, processes, workspaces, artifacts, and persisted state.
- Supported platforms: macOS, Windows.
- License: GPLv3.

---

## FAQ

### Does Lime provide AI models?

No. Lime is an Agent host and workspace, not a model vendor. Configure an available Provider, model, and credential.

### Can the Agent edit code and run commands?

Yes. Within the permissions you grant, the Agent can read and edit files, run terminal commands, run tests, and call tools. Review or approve high-risk actions.

### Will all my materials be uploaded?

Project materials, conversation history, and configuration are kept locally by default. Inputs needed for model or external-tool calls are sent to the Provider or target service you configured; follow the relevant service policy for sensitive material.

### How is Lime different from a normal chat tool?

Normal chat mainly returns text. Lime understands workspace context, operates files, processes, and tools, retains execution state and results, and lets you keep moving in the same task.

### Do I need advanced prompt-writing skills?

No. Describe the goal, constraints, context, and acceptance criteria. Reusable procedures can become Skills, and complex tasks can start with an Agent-generated plan.

---

## License

[GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0)

## Disclaimer

This project is provided for learning and research purposes only. Users are responsible for their own use and risk.

Lime does not directly provide AI model services. Model capabilities are provided by third-party AI service providers configured by the user.
