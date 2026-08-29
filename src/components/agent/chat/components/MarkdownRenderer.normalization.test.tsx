import { describe, expect, it } from "vitest";

import { renderMarkdown as render } from "./MarkdownRenderer.testHarness";

describe("MarkdownRenderer normalization", () => {
  it("开发分析正文应渲染标题、表格、粗体和行内代码，而不是露出原始 Markdown", () => {
    const content = [
      "## BADOUCMS 架构分析",
      "",
      "| 发现 | 说明 |",
      "| --- | --- |",
      "| **底层框架** | `ThinkPHP` |",
    ].join("\n");

    const container = render(content);

    expect(
      container.querySelector('h2[data-markdown-heading-level="2"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-table-scroll"] table'),
    ).not.toBeNull();
    expect(container.querySelector("strong")?.textContent).toBe("底层框架");
    expect(
      container.querySelector('code[data-inline-code="true"]')?.textContent,
    ).toBe("ThinkPHP");
    expect(container.textContent).not.toContain("## BADOUCMS");
    expect(container.textContent).not.toContain("| 发现 | 说明 |");
  });

  it("应修复模型输出的松散 Markdown，避免标题、粗体和表格语法原样露出", () => {
    const content = [
      "五年级选购指南###",
      "####如果孩子基础一般，优先选护眼和内容稳定的机型",
      "**推荐 型号 **：学习机 S30",
      "**理由 **：系统足够简单，家长管理清楚。",
      "选购对比：",
      "| 品牌 | 型号 | 适合场景 |",
      "| --- | --- | --- |",
      "| 星火 | S30 | 五年级基础巩固 |",
    ].join("\n");

    const container = render(content);

    expect(
      container.querySelector('h3[data-markdown-heading-level="3"]')
        ?.textContent,
    ).toBe("五年级选购指南");
    expect(container.querySelector("h4")?.textContent).toBe(
      "如果孩子基础一般，优先选护眼和内容稳定的机型",
    );
    expect(
      Array.from(container.querySelectorAll("strong")).map(
        (node) => node.textContent,
      ),
    ).toEqual(["推荐 型号", "理由"]);
    expect(
      container.querySelector('[data-testid="markdown-table-scroll"] table'),
    ).not.toBeNull();
    expect(container.textContent).not.toContain("五年级选购指南###");
    expect(container.textContent).not.toContain("####如果孩子");
    expect(container.textContent).not.toContain("**推荐 型号 **");
    expect(container.textContent).not.toContain("| 品牌 | 型号 |");
  });

  it("压缩回答中标题紧跟粗体正文时应恢复为独立标题和段落", () => {
    const content =
      '---###项目定位**"Agents remember. Humans innovate."** ——解决 Agent 重复劳动的问题。';

    const container = render(content);

    expect(container.querySelector("h3")?.textContent).toBe("项目定位");
    expect(container.querySelector("strong")?.textContent).toBe(
      '"Agents remember. Humans innovate."',
    );
    expect(container.querySelectorAll("h3")).toHaveLength(1);
    expect(container.textContent).not.toContain("项目定位**");
  });

  it("压缩回答中标题紧跟紧凑表格时应恢复为独立标题和 GFM 表格", () => {
    const content =
      "---###四大核心模块|模块 |角色 |技术栈 ||------|--------|------------|| **MemoryCore** |记忆引擎 | TypeScript, SQLite || **MemoryKnowledge** |知识库服务 | Hono.js";

    const container = render(content);

    expect(container.querySelector("h3")?.textContent).toBe("四大核心模块");
    const table = container.querySelector(
      '[data-testid="markdown-table-scroll"] table',
    );
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll("th")).toHaveLength(3);
    expect(table?.querySelectorAll("tbody tr")).toHaveLength(2);
    expect(container.textContent).not.toContain("四大核心模块|模块");
  });

  it("压缩回答中的编号粗体和表格换行不应破坏列表与单元格", () => {
    const content =
      "---###核心设计理念**1.四层记忆分层**-**L0 Conversation** —原始对话。**2.四类 Memory Asset**-**Chat Memory** —对话记忆\n| 模块 | 技术栈 ||------|| MemoryCore | TypeScript，SQLite + vector search |";

    const container = render(content);

    expect(container.textContent).toContain("1.四层记忆分层");
    expect(container.textContent).toContain("L0 Conversation");
    expect(container.textContent).toContain("2.四类 Memory Asset");
    expect(container.querySelectorAll("li").length).toBeGreaterThanOrEqual(2);
    expect(
      container.querySelector('[data-testid="markdown-table-scroll"] table'),
    ).not.toBeNull();
    expect(container.textContent).toContain("SQLite + vector search");
  });

  it("压缩的目录树代码块应恢复分支换行并按纯文本代码显示", () => {
    const content =
      "---###部署架构```MemoryHub (整体)├── MemoryCore ←记忆引擎└── MemoryProxy ←代理```";

    const container = render(content);
    const codeBlock = container.querySelector(
      '[data-testid="markdown-plain-code-block"]',
    );

    expect(codeBlock).not.toBeNull();
    expect(codeBlock?.textContent).toContain("MemoryHub");
    expect(codeBlock?.textContent).toContain("\n├── MemoryCore");
    expect(codeBlock?.textContent).toContain("\n└── MemoryProxy");
  });

  it("真实 MemoryHub 回答应按 Codex 块级语义完整渲染", () => {
    const content =
      '好，这个项目的分析来了。整体是个挺完整的 **Agent记忆系统**，腾讯出品的。---##🧠 TencentDB-Agent-Memory项目分析###项目定位**"Agents remember. Humans innovate."** ——解决的核心问题是：Agent每次会话都从零开始，重复劳动。这个系统让 Agent的经验可以沉淀、共享、复用，而不是每次都要"重新学习"你的项目。---###四大核心模块|模块 |角色 |技术栈 ||------|------|--------|| **MemoryCore** |记忆引擎 | TypeScript，SQLite + vector search，支持 OpenClaw/Hermes插件 || **MemoryKnowledge** |知识库服务 | Hono.js + SQLite + Drizzle ORM，提供 CodeGraph + Wiki || **MemoryPanel** |管控面板（前后端） |后端 Hono.js，前端 React，管理 team/user/agent/asset || **MemoryProxy** | LLM请求代理 | Hono.js + Redis会话，JSONL日志 + Opik/Langfuse追踪 |---###核心设计理念**1.四层记忆分层（L0→L1→L2→L3）**- **L0 Conversation** —原始对话，完整上下文，用于追溯原文- **L1 Atom** —从对话中提取的事实、偏好、约束（精确检索）- **L2 Scenario** —按项目/场景组织的知识块（快速恢复上下文）- **L3 Core/Persona** —长期用户画像和团队认知（冷启动必备）检索时用 BM25 +向量 + RRF混合召回，同时用 item count、字符预算、超时限制防止记忆撑爆 context window。**2.四类 Memory Asset**- **Chat Memory** —对话中的偏好、决策、历史- **Skill** —可复用的工作流程（版本管理、触发边界、执行步骤、验证规则）- **Wiki** —文档的结构化知识图谱（受 Karpathy LLM Wiki启发）- **CodeGraph** —代码符号、文件、调用关系、影响路径（借用 codegraph开源项目）**3.权限模型**`private` → `team` → `restricted` → `agent`四级可见性，配合 User/Role/Agent ACL，实现"经验共享但不暴露隐私"。---###技术亮点- **向量检索**：用 `sqlite-vec`做本地向量搜索，不依赖外部向量数据库- **中文分词**：集成 `@node-rs/jieba`做中文 NLP- **可插拔适配器**：同时支持 OpenClaw和 Hermes两大 Agent框架- **异步 Pipeline**：Wiki和 CodeGraph构建是异步的，状态可监控- **MCP协议支持**：MemoryKnowledge同时暴露 MCP接口，方便其他工具调用- **Telemetry**：全链路 OpenTelemetry + Langfuse/Opik追踪---###部署架构```MemoryHub (整体)├── MemoryCore ←记忆引擎（本地 SQLite）├── MemoryKnowledge ←知识库服务（CodeGraph + Wiki）├── MemoryPanel ← Web管控面板（port8125）└── MemoryProxy ← LLM请求代理（转发 +注入记忆）```一键部署通过 `deploy/global-images/start-all.sh`启动全部服务，支持 Docker。---###整体评价这个项目做得挺扎实的——不是玩具级 demo，有完整的分层架构、权限模型、部署方案。L0→L3的分层设计是核心创新点，把"记忆"从简单的 RAG向量检索提升到了可管理的资产体系。支持 OpenClaw/Hermes/Claude Code/CodeBuddy多框架适配也是加分项。技术上主要是 TypeScript + Hono + SQLite的组合，不算复杂但够用。';

    const container = render(content);
    const headings = Array.from(
      container.querySelectorAll("[data-markdown-heading-level]"),
    ).map((heading) => heading.textContent);
    const listItems = Array.from(container.querySelectorAll("li")).map(
      (item) => item.textContent || "",
    );
    const table = container.querySelector(
      '[data-testid="markdown-table-scroll"] table',
    );
    const codeBlock = container.querySelector(
      '[data-testid="markdown-plain-code-block"]',
    );

    expect(headings).toEqual([
      "🧠 TencentDB-Agent-Memory项目分析",
      "项目定位",
      "四大核心模块",
      "核心设计理念",
      "技术亮点",
      "部署架构",
      "整体评价",
    ]);
    expect(table?.querySelectorAll("tbody tr")).toHaveLength(4);
    expect(listItems).toHaveLength(14);
    expect(listItems).toEqual(
      expect.arrayContaining([
        expect.stringContaining("BM25 +向量 + RRF混合召回"),
        expect.stringContaining("OpenTelemetry + Langfuse/Opik追踪"),
      ]),
    );
    expect(
      Array.from(container.querySelectorAll("li code")).some(
        (code) => code.textContent === "sqlite-vec",
      ),
    ).toBe(true);
    expect(codeBlock?.textContent).toContain("\n├── MemoryCore");
    expect(codeBlock?.textContent).toContain("\n└── MemoryProxy");
    expect(container.textContent).toContain("TypeScript + Hono + SQLite");
    expect(container.textContent).not.toContain("核心设计理念**");
    expect(container.textContent).not.toContain("四大核心模块|模块");
  });

  it("markdown 围栏里确实是表格时应拆掉围栏并渲染为表格", () => {
    const content = [
      "```markdown",
      "| 文件 | 作用 |",
      "| --- | --- |",
      "| build.bat | Windows 构建入口 |",
      "```",
    ].join("\n");

    const container = render(content);

    expect(
      container.querySelector('[data-testid="markdown-table-scroll"] table'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-syntax-code-block"]'),
    ).toBeNull();
    expect(container.textContent).toContain("build.bat");
    expect(container.textContent).not.toContain("```");
  });

  it("markdown 围栏里不是表格时应继续作为代码块显示", () => {
    const content = ["```markdown", "**强调示例**", "```"].join("\n");

    const container = render(content);

    expect(
      container.querySelector('[data-testid="markdown-syntax-code-block"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-table-scroll"]'),
    ).toBeNull();
  });

  it("Markdown 表格应包裹在横向滚动容器中，避免窄列压缩", () => {
    const content = [
      "| 模块 | 输入 | 输出 | 备注 |",
      "| --- | --- | --- | --- |",
      "| Browser Runtime | 页面信息 | 结构化摘要 | 主链 |",
    ].join("\n");

    const container = render(content);
    const tableScroll = container.querySelector(
      '[data-testid="markdown-table-scroll"]',
    );

    expect(tableScroll).not.toBeNull();
    const table = tableScroll?.querySelector("table");
    const headerCell = table?.querySelector("th");
    expect(table).not.toBeNull();
    expect(headerCell).not.toBeNull();
    const headerBackground = getComputedStyle(
      headerCell as HTMLElement,
    ).backgroundColor;
    const rgbMatch = /rgb\((\d+), (\d+), (\d+)\)/.exec(headerBackground);
    expect(rgbMatch).not.toBeNull();
    const [, red = "0", green = "0", blue = "0"] = rgbMatch ?? [];
    expect(Number(red)).toBeGreaterThanOrEqual(240);
    expect(Number(green)).toBeGreaterThanOrEqual(240);
    expect(Number(blue)).toBeGreaterThanOrEqual(240);
    expect(container.textContent).toContain("Browser Runtime");
  });

  it("应把模型压成单行的紧凑竖线表格恢复为 GFM 表格", () => {
    const content =
      "| 事件 | 要点 ||------|| 美伊霍尔木兹海峡交火 | 美军空袭伊朗油轮及境内发射场 || 停火谈判 | 美国要求伊朗在周五前答复止战方案 || 特朗普威胁 | 若伊朗拒绝，将发动更猛烈打击 |";

    const container = render(content);
    const tableScroll = container.querySelector(
      '[data-testid="markdown-table-scroll"]',
    );
    const table = tableScroll?.querySelector("table");

    expect(table).not.toBeNull();
    expect(table?.querySelectorAll("th")).toHaveLength(2);
    expect(table?.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(table?.textContent).toContain("美伊霍尔木兹海峡交火");
    expect(table?.textContent).toContain("美国要求伊朗在周五前答复止战方案");
  });

  it("旧历史压缩 Markdown 应恢复块级结构，避免表格吞掉后续正文", () => {
    const content =
      "##结论`/Users/coso/.yansu-agent` 是 **Yansu Agent 桌面/本地代理应用的数据目录**，不是普通项目目录。它包含：- 本地可执行依赖：`bin/`、`git/`、`opencli/`- 本地 AI/识别模型：`models/`、`sherpa/`- 活动记录与截图：`activity/`、`activity.db`---##目录体积分布主要占用如下：| 路径 | 大小 | 判断 ||---|---:|---|| `activity/` | **974M** | 最大头，主要是截图快照 || `models/` | **729M** | 本地 ONNX 模型 || `sherpa/` | **229M** | 语音识别/音频相关模型 |最值得关注的是：```text/Users/coso/.yansu-agent/activity/snapshots/2026-05-26973M/Users/coso/.yansu-agent/models/gliner-pii-base/model.onnx634M```---##关键发现###1. `activity/` 是最大空间来源";

    const container = render(content);
    const headings = container.querySelectorAll(
      "[data-markdown-heading-level]",
    );
    const table = container.querySelector(
      '[data-testid="markdown-table-scroll"] table',
    );
    const codeBlock = container.querySelector(
      '[data-testid="markdown-plain-code-block"]',
    );

    expect(headings).toHaveLength(4);
    expect(headings[0]?.textContent).toContain("结论");
    expect(headings[1]?.textContent).toContain("目录体积分布");
    expect(headings[2]?.textContent).toContain("关键发现");
    expect(headings[3]?.textContent).toContain("activity/");
    expect(container.querySelectorAll("li")).toHaveLength(3);
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(table?.textContent).not.toContain("关键发现");
    expect(codeBlock).not.toBeNull();
    expect(codeBlock?.textContent).toContain(
      "/Users/coso/.yansu-agent/activity/snapshots/2026-05-26",
    );
  });

  it("压成单段的编号建议与 Markdown 模板应恢复为列表和标题", () => {
    const content =
      "抱歉，我这边尝试调用联网检索，但当前工具面只返回了 WebSearch/WebFetch 的注册信息，没有实际返回新闻搜索结果。为了避免编造“今日新闻”，我不能直接给出未经核实的国际新闻摘要。你可以任选一种方式继续： 1. 你发我几个新闻链接或截图，我帮你整理成： - 今日国际要闻摘要 - 按地区/主题分类 - 每条一句话版 - 适合朋友圈/日报/会议简报的版本 2. 你复制一段新闻列表过来，我可以快速压缩成一页简报。 3. 如果联网工具恢复，我可以按这个结构帮你整理： ## 今日国际新闻简报模板### 一、地缘政治与冲突-事件：-关键进展：-影响：### 二、国际外交-事件：-相关国家/组织：-后续看点：### 三、经济与市场-事件：-对全球市场/能源/贸易的影响：### 四、科技与产业-事件：-影响范围：";

    const container = render(content);
    const orderedItems = container.querySelectorAll("ol > li");
    const headings = container.querySelectorAll(
      "[data-markdown-heading-level]",
    );
    const bulletItems = container.querySelectorAll("ul > li");

    expect(orderedItems).toHaveLength(3);
    expect(orderedItems[0]?.textContent).toContain("你发我几个新闻链接或截图");
    expect(orderedItems[1]?.textContent).toContain("你复制一段新闻列表过来");
    expect(orderedItems[2]?.textContent).toContain("如果联网工具恢复");
    expect(headings).toHaveLength(5);
    expect(headings[0]?.textContent).toContain("今日国际新闻简报模板");
    expect(headings[1]?.textContent).toContain("地缘政治与冲突");
    expect(bulletItems.length).toBeGreaterThanOrEqual(8);
    expect(container.textContent).not.toContain("模板### 一");
    expect(container.textContent).not.toContain("继续： 1.");
  });

  it("压成单段的简报应恢复标题、时间口径和分节列表", () => {
    const content =
      "## 今日简报**时间口径：2026 年 6 月 2 日；主要依据可核实来源。---## 一、地缘政治- 第一条事件 来源：[Source A](https://example.com/a)- 第二条事件**观察重点：*局势变化仍需继续关注。---## 任意小节1. 第一项2. 第二项3. 第三项";

    const container = render(content);
    const headings = container.querySelectorAll(
      "[data-markdown-heading-level]",
    );
    const paragraphs = container.querySelectorAll("p");
    const links = container.querySelectorAll("a");
    const orderedItems = container.querySelectorAll("ol > li");
    const bulletItems = container.querySelectorAll("ul > li");

    expect(headings).toHaveLength(3);
    expect(headings[0]?.textContent).toBe("今日简报");
    expect(paragraphs[0]?.textContent).toContain("时间口径");
    expect(links[0]?.getAttribute("href")).toBe("https://example.com/a");
    expect(bulletItems.length).toBeGreaterThanOrEqual(2);
    expect(orderedItems).toHaveLength(3);
    expect(container.textContent).not.toContain("简报**时间口径");
    expect(container.textContent).not.toContain("小节1.");
  });

  it("真实压缩国际新闻简报应恢复来源、标题和影响判断边界", () => {
    const content =
      "## 今日国际新闻简报｜2026年6月2日>口径：根据已检索到的 **NPR、AP News、Al Jazeera** 等公开页面整理；Reuters/BBC 部分页面抓取受限，因此以下以可核验页面内容为主。### 一句话总览今天国际新闻的主线集中在 **中东冲突升级、美伊/伊以相关紧张、刚果 Ebola 疫情、东欧俄乌外溢风险、非洲与拉美政治动态，以及 AI/科技资本市场动向**。---##1. 中东：以色列、黎巴嫩、伊朗、美国相关局势升温- **以色列在黎巴嫩南部和加沙的军事行动继续引发地区紧张。**- NPR 报道称，伊朗因以色列在黎巴嫩、加沙的行动，**暂停与美国的相关谈判**。- Al Jazeera 报道称，伊朗警告以色列在黎巴嫩和加沙的攻击可能威胁美国推动的停火谈判。- AP News 页面头条显示，美国轰炸伊朗军事设施，并拦截伊朗向驻科威特美军发射的导弹。**影响判断：**中东局势正从局部冲突向更广泛的美伊、伊以、以黎关系扩散，短期内会继续影响能源、航运与地区安全预期。---##2.以色列控制周边土地问题引发争议- NPR关注以色列近年来在 **加沙、黎巴嫩、叙利亚邻近区域** 控制土地的问题。-以方称这些区域是安全缓冲区，但以色列国内也有人主张更永久性地扩大边界。**影响判断：**这类“临时安全区”是否长期化，将影响未来停火安排、边境谈判和地区政治格局。---## 今日值得继续关注的3 条主线1. **中东是否进一步升级** 特别是美国、伊朗、以色列、黎巴嫩真主党之间是否出现新一轮军事行动或谈判破裂。2. **刚果 Ebola 疫情是否外溢**重点看 WHO 后续评估、周边国家防控措施，以及疫苗/治疗资源调配。---##主要信息来源- [NPR World News](https://www.npr.org/sections/world/)- [AP News World](https://apnews.com/world-news)- [Al Jazeera News](https://www.aljazeera.com/news/)";

    const container = render(content);
    const headings = Array.from(
      container.querySelectorAll("[data-markdown-heading-level]"),
    ).map((heading) => heading.textContent);
    const bulletItems = Array.from(container.querySelectorAll("ul > li")).map(
      (item) => item.textContent,
    );
    const orderedItems = container.querySelectorAll("ol > li");
    const links = Array.from(container.querySelectorAll("a")).map((link) =>
      link.getAttribute("href"),
    );

    expect(headings).toEqual([
      "今日国际新闻简报｜2026年6月2日",
      "一句话总览",
      "1. 中东：以色列、黎巴嫩、伊朗、美国相关局势升温",
      "2. 以色列控制周边土地问题引发争议",
      "今日值得继续关注的3 条主线",
      "主要信息来源",
    ]);
    expect(container.querySelector("blockquote")?.textContent).toContain(
      "NPR、AP News、Al Jazeera",
    );
    expect(bulletItems).toEqual(
      expect.arrayContaining([
        expect.stringContaining("以色列在黎巴嫩南部和加沙"),
        expect.stringContaining("NPR 报道称"),
        expect.stringContaining("以方称这些区域是安全缓冲区"),
        expect.stringContaining("NPR World News"),
      ]),
    );
    expect(orderedItems).toHaveLength(2);
    expect(links).toEqual([
      "https://www.npr.org/sections/world/",
      "https://apnews.com/world-news",
      "https://www.aljazeera.com/news/",
    ]);
    expect(container.textContent).not.toContain("总览今天");
    expect(container.textContent).not.toContain("升温-");
    expect(container.textContent).not.toContain("导弹。影响判断");
    expect(container.textContent).not.toContain("##主要信息来源");
  });

  it("局部压缩的列表项应在来源、观察重点和后续关注处恢复块级边界", () => {
    const content = [
      "## 二、俄乌战争：联合国呼吁降温",
      "- 联合国强调乌克兰战争需要降级，近期袭击增加 联合国方面警告，乌克兰战事中的袭击活动上升，呼吁各方避免进一步升级。 来源：UN News**观察重点：**俄乌战争仍是欧洲安全核心风险，近期袭击增加意味着谈判空间可能继续收窄，民用基础设施、人道援助与能源供应仍面临压力。",
      "## 三、Gaza、西岸与阿富汗：人道问题持续",
      "- 联合国发布“世界新闻简报”，涉及 Gaza、西岸、阿富汗等地动态 当日联合国简报提及 Gaza、西岸与阿富汗局势。 来源：UN News- Gaza 难民营中以足球活动提供短暂喘息 联合国报道提到，前职业球员组织足球比赛。",
      "## 六、气候与社会议题",
      "- 全球气温预计仍将接近纪录高位 联合国相关机构警告，全球温度仍可能维持在接近历史纪录的水平。- **联合国提醒：禁止儿童使用社交媒体不是唯一答案，平台应“安全设计”** 联合国人权相关报道强调，保护儿童线上安全不能只靠简单禁令。",
      "## 任意后续小节",
      "黎巴嫩—以色列边境是否进一步升级2. Gaza 停火与人道物资准入是否改善3. 俄乌双方袭击频率是否继续上升",
    ].join("\n\n");

    const container = render(content);
    const listItems = Array.from(container.querySelectorAll("li")).map(
      (item) => item.textContent,
    );
    const paragraphs = Array.from(container.querySelectorAll("p")).map(
      (paragraph) => paragraph.textContent,
    );

    expect(listItems).toEqual(
      expect.arrayContaining([
        expect.stringContaining("联合国强调乌克兰战争需要降级"),
        expect.stringContaining("Gaza 难民营中以足球活动"),
        expect.stringContaining("联合国提醒："),
        expect.stringContaining("禁止儿童使用社交媒体"),
        expect.stringContaining("Gaza 停火与人道物资准入是否改善"),
        expect.stringContaining("俄乌双方袭击频率是否继续上升"),
      ]),
    );
    expect(container.textContent).toContain("黎巴嫩—以色列边境是否进一步升级");
    expect(paragraphs).toEqual(
      expect.arrayContaining([
        "观察重点：",
        expect.stringContaining("俄乌战争仍是欧洲安全核心风险"),
      ]),
    );
    expect(container.textContent).not.toContain("UN News观察重点");
    expect(container.textContent).not.toContain("UN News**观察重点");
    expect(container.textContent).not.toContain("UN News- Gaza");
    expect(container.textContent).not.toContain("水平。- 联合国提醒");
    expect(container.textContent).not.toContain("安全设计”**");
    expect(container.textContent).not.toContain("升级2.");
  });

  it("任意标题后的无编号首项和后续编号应按结构恢复", () => {
    const content = "## 任意标题\n\n第一项内容2. 第二项内容3. 第三项内容";

    const container = render(content);
    const listItems = Array.from(container.querySelectorAll("li")).map(
      (item) => item.textContent,
    );

    expect(listItems).toEqual(["第一项内容", "第二项内容", "第三项内容"]);
    expect(container.textContent).not.toContain("内容2.");
    expect(container.textContent).not.toContain("内容3.");
  });

  it("不应改写代码块里的紧凑竖线文本", () => {
    const content = [
      "```text",
      "| 事件 | 要点 ||------|| 示例 | 保持原文 |",
      "```",
    ].join("\n");

    const container = render(content);

    expect(
      container.querySelector('[data-testid="markdown-table-scroll"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-plain-code-block"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("||------||");
  });

  it("长文报告块应渲染标题层级、引用卡与分隔线", () => {
    const content = [
      "# Hermes Engine 选型建议",
      "",
      "这是导语段，用来概括结论与适用范围。",
      "",
      "## 为什么优先考虑它",
      "",
      "> 结论先行：优先保证稳定交付，再谈极限性能。",
      "",
      "---",
      "",
      "### 对比表",
      "",
      "| 方案 | 优势 |",
      "| --- | --- |",
      "| A | 稳定 |",
    ].join("\n");

    const container = render(content);

    expect(
      container.querySelector('h1[data-markdown-heading-level="1"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('h2[data-markdown-heading-level="2"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-blockquote-card"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-divider"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-table-scroll"]'),
    ).not.toBeNull();
  });
});
