# App Server 流程图

> 状态：current planning source
> 更新时间：2026-06-08
> 作用：用流程图固定 App Server 的用户路径、技术主链、服务抽取、渐进替换和多 App 复用。

## 1. 用户路径：独立 App 内完成 Agent 任务

```mermaid
flowchart TD
    A["用户在独立 App 中操作业务对象"] --> B["App 收集业务上下文"]
    B --> C["App Main 调用 App Server"]
    C --> D["创建 Agent Session"]
    D --> E["提交 Turn"]
    E --> F["Lime Runtime 执行"]
    F --> G{"需要人工确认?"}
    G -- 是 --> H["App 展示 action.required"]
    H --> I["用户确认 / 拒绝"]
    I --> F
    G -- 否 --> J["持续接收事件"]
    J --> K["Artifact / Evidence 更新"]
    K --> L["App 写回业务投影"]
```

## 2. 技术主链：Request 到 Runtime Facts

```mermaid
flowchart TD
    A["JSON-RPC Request"] --> B["Initialize Gate"]
    B --> C["Method Router"]
    C --> D["DTO Validation"]
    D --> E["RuntimeCore"]
    E --> F["Session / Thread / Turn / Run"]
    F --> G["ExecutionBackend"]
    G --> H["RuntimeBackend / FutureBackend"]
    H --> I["Tool / Skill / Workspace / Memory / Policy"]
    I --> J["Runtime Events"]
    J --> K["Snapshots / Read Models"]
    K --> L["JSON-RPC Response / Notification"]
    L --> M["App Projection"]
```

## 3. 服务抽取流程

```mermaid
flowchart TD
    A["legacy desktop command runtime glue"] --> B["盘点 legacy desktop-only 依赖"]
    B --> C{"是否依赖壳层对象?"}
    C -- 是 --> D["抽 HostBridge / context / event sink"]
    C -- 否 --> E["判断公共 core 还是 backend"]
    D --> E
    E --> F{"是否 Agent 私有逻辑?"}
    F -- 是 --> G["收进 RuntimeBackend"]
    F -- 否 --> H["下沉 RuntimeCore"]
    G --> I["legacy desktop facade 改为委托 RuntimeCore"]
    H --> I
    I --> J["App Server 调用同一 RuntimeCore"]
    J --> K["合同测试对比输出"]
    K --> L{"行为一致?"}
    L -- 否 --> M["修 core/backend 边界"]
    M --> K
    L -- 是 --> N["旧 wrapper 撤注册并删除<br/>删不动登记 blocker"]
```

说明：服务抽取完成后，`lime-rs/src/commands/**` 已物理删除，不得恢复 thin facade、fail-closed stub、compat wrapper 或 tombstone。需要保留兼容命令名时，兼容入口也必须在 Electron Desktop Host / App Server current 边界内说明退出条件。

## 4. 渐进式替换流程

```mermaid
flowchart LR
    P0["P0 公共边界冻结"] --> P1["P1 app-server crate 家族"]
    P1 --> P2["P2 RuntimeCore / ExecutionBackend"]
    P2 --> P3["P3 RuntimeBackend"]
    P3 --> P4["P4 App Server 接入 core"]
    P4 --> P5["P5 Desktop Host bridge"]
    P5 --> P6["P6 content-studio client"]
    P6 --> P7["P7 tool/action/artifact/evidence"]
    P7 --> P8["P8 多 App capability discovery"]
    P8 --> P9["P9 compat glue 退场审计"]
```

## 5. 多 App 复用图

```mermaid
flowchart TB
    subgraph AppLayer["独立 App"]
        A["Lime Desktop"]
        B["content-studio"]
        C["Plugin Shell"]
        D["未来垂直 App"]
    end

    subgraph ClientLayer["Desktop Host Bridge / Client Layer"]
        E["Electron Desktop Host Bridge<br/>current"]
        F["Legacy Desktop Facade<br/>compat"]
        G["Generic App Client"]
    end

    subgraph ServerLayer["App Server"]
        H["JSON-RPC Router"]
        I["Capability Discovery"]
        J["RuntimeCore"]
    end

    subgraph RuntimeLayer["Runtime Owner"]
        K["Session / Turn"]
        L["ExecutionBackend"]
        M["Tool / Skill"]
        N["Workspace / Memory"]
        O["Artifact / Evidence"]
        P["Policy / Action"]
    end

    A --> E
    B --> E
    C --> G
    D --> G
    E --> H
    F -. "compat only" .-> H
    G --> H
    H --> I
    H --> J
    J --> K
    K --> L
    L --> M
    J --> N
    J --> O
    J --> P
```

## 6. Capability 调用流程

```mermaid
flowchart TD
    A["App 请求 capability/list"] --> B["CapabilityService"]
    B --> C["读取 Skill Catalog"]
    B --> D["读取 Tool Inventory"]
    B --> E["读取 Workspace Policy"]
    C --> F["过滤 App 可见能力"]
    D --> F
    E --> F
    F --> G["返回 capability descriptors"]
    G --> H["App 选择 capabilityId"]
    H --> I["turn/start 携带 capabilityId"]
    I --> J["Runtime 按 capability 执行"]
```

## 7. Action / Permission 流程

```mermaid
flowchart TD
    A["Runtime 准备执行工具"] --> B["PolicyService evaluate"]
    B --> C{"结果"}
    C -- allow --> D["执行工具"]
    C -- deny --> E["tool.failed denied"]
    C -- ask --> F["action.required"]
    F --> G["App 展示确认 UI"]
    G --> H{"用户选择"}
    H -- approve --> I["action.resolved approved"]
    I --> D
    H -- deny --> J["action.resolved denied"]
    J --> E
    D --> K["tool.result"]
    E --> L["read model 更新"]
    K --> L
```

## 8. Artifact / Evidence 写回流程

```mermaid
flowchart TD
    A["Execution Loop"] --> B["生成或修改 Artifact"]
    B --> C["ArtifactService 写入"]
    C --> D["artifact.changed event"]
    D --> E["App Projection 更新"]
    A --> F["Runtime Facts / Timeline"]
    F --> G["EvidenceService"]
    G --> H["evidence pack / summary"]
    H --> I["evidence.changed event"]
    I --> E
```

## 9. 禁止路径

```mermaid
flowchart TD
    A["独立 App"] --> B{"需要 Agent 能力"}
    B --> C["调用 App Server"]
    B -.禁止.-> D["复制 runtime_turn"]
    B -.禁止.-> E["自建 tool runtime"]
    B -.禁止.-> F["直接写 runtime DB"]
    B -.禁止.-> G["用 UI state 标记执行成功"]
    B -.禁止.-> I["新增 / 复活 lime-rs/src/commands wrapper"]
    C --> H["Runtime facts"]
```
