import React, { useState } from "react";
import styled, { keyframes, css } from "styled-components";
import {
  Sparkles,
  ArrowRight,
  ImageIcon,
  Video,
  FileText,
  PenTool,
  BrainCircuit,
  CalendarRange,
  ChevronDown,
  Search,
  Globe,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Badge } from "@/components/ui/badge";

// Import Assets
import iconXhs from "@/assets/platforms/xhs.png";
import iconGzh from "@/assets/platforms/gzh.png";
import iconZhihu from "@/assets/platforms/zhihu.png";
import iconToutiao from "@/assets/platforms/toutiao.png";
import iconJuejin from "@/assets/platforms/juejin.png";
import iconCsdn from "@/assets/platforms/csdn.png";

import modelGemini from "@/assets/models/gemini.png";
import modelClaude from "@/assets/models/claude.png";
import modelDeepseek from "@/assets/models/deepseek.png";

// --- Animations ---
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
`;

// --- Styled Components ---

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 40px 20px;
  background-color: hsl(var(--background));
  overflow-y: auto;
  position: relative;

  // Subtle mesh background effect
  &::before {
    content: "";
    position: absolute;
    top: -10%;
    left: 20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(
      circle,
      hsl(var(--primary) / 0.05) 0%,
      transparent 70%
    );
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
  }
`;

const ContentWrapper = styled.div`
  max-width: 900px;
  width: 100%;
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 36px;
  animation: ${fadeIn} 0.5s ease-out;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 8px;
`;

const shimmer = keyframes`
  0% { background-position: 0% 50%; filter: brightness(100%); }
  50% { background-position: 100% 50%; filter: brightness(120%); }
  100% { background-position: 0% 50%; filter: brightness(100%); }
`;

const MainTitle = styled.h1`
  font-size: 42px;
  font-weight: 800;
  color: hsl(var(--foreground));
  margin-bottom: 16px;
  letter-spacing: -1px;
  line-height: 1.15;

  // Advanced Light & Shadow Gradient
  background: linear-gradient(
    135deg,
    hsl(var(--foreground)) 0%,
    #8b5cf6 25%,
    #ec4899 50%,
    #8b5cf6 75%,
    hsl(var(--foreground)) 100%
  );
  background-size: 300% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;

  // Animation
  animation: ${shimmer} 5s ease-in-out infinite;

  // Optical Glow
  filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.3));

  span {
    display: block; // Force new line for the second part naturally if needed, or keep inline
    background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
`;

// --- Custom Tabs ---
const TabsContainer = styled.div`
  display: flex;
  gap: 8px;
  padding: 6px;
  background-color: hsl(var(--muted) / 0.4);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid hsl(var(--border) / 0.5);
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.01),
    0 2px 4px -1px rgba(0, 0, 0, 0.01);
  overflow-x: auto;
  max-width: 100%;
  scrollbar-width: none; // hide scrollbar
`;

const TabItem = styled.button<{ $active?: boolean }>`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.25s cubic-bezier(0.25, 1, 0.5, 1);
  white-space: nowrap;

  ${(props) =>
    props.$active
      ? css`
          background-color: hsl(var(--background));
          color: hsl(var(--foreground));
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
          transform: scale(1.02);
        `
      : css`
          color: hsl(var(--muted-foreground));
          &:hover {
            background-color: hsl(var(--muted) / 0.5);
            color: hsl(var(--foreground));
          }
        `}
`;

// --- Input Card ---
const InputCard = styled.div`
  width: 100%;
  position: relative;
  background-color: hsl(var(--card));
  border: 1px solid hsl(var(--border) / 0.6);
  border-radius: 20px;
  box-shadow:
    0 20px 40px -5px rgba(0, 0, 0, 0.03),
    0 8px 16px -4px rgba(0, 0, 0, 0.03);
  overflow: visible; // Allow dropdowns to overflow
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    box-shadow:
      0 25px 50px -12px rgba(0, 0, 0, 0.06),
      0 12px 24px -6px rgba(0, 0, 0, 0.04);
    border-color: hsl(var(--primary) / 0.3);
  }

  &:focus-within {
    border-color: hsl(var(--primary));
    box-shadow:
      0 0 0 4px hsl(var(--primary) / 0.1),
      0 25px 50px -12px rgba(0, 0, 0, 0.08);
  }
`;

const StyledTextarea = styled(Textarea)`
  min-height: 150px;
  padding: 24px 28px;
  border: none;
  font-size: 16px;
  line-height: 1.6;
  resize: none;
  background: transparent;
  color: hsl(var(--foreground));

  &::placeholder {
    color: hsl(var(--muted-foreground) / 0.7);
    font-weight: 300;
  }

  &:focus-visible {
    ring: 0;
    outline: none;
    box-shadow: none;
  }
`;

const Toolbar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px 16px 20px;
  background: linear-gradient(to bottom, transparent, hsl(var(--muted) / 0.2));
  border-bottom-left-radius: 20px;
  border-bottom-right-radius: 20px;
`;

const ToolLoginLeft = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
`;

// --- Styles for Selectors ---
const ColorDot = styled.div<{ $color: string }>`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: ${(props) => props.$color};
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1) inset;
`;

const GridSelect = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  padding: 8px;
`;

const GridItem = styled.div<{ $active?: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid
    ${(props) => (props.$active ? "hsl(var(--primary))" : "transparent")};
  background-color: ${(props) =>
    props.$active ? "hsl(var(--primary)/0.08)" : "hsl(var(--muted)/0.3)"};
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background-color: hsl(var(--primary) / 0.05);
  }
`;

interface EmptyStateProps {
  input: string;
  setInput: (value: string) => void;
  onSend: (value: string) => void;
}

// Scenarios Configuration
const CATEGORIES = [
  {
    id: "knowledge",
    label: "知识探索",
    icon: <BrainCircuit className="w-4 h-4" />,
  },
  {
    id: "planning",
    label: "计划规划",
    icon: <CalendarRange className="w-4 h-4" />,
  },
  { id: "social", label: "社媒内容", icon: <PenTool className="w-4 h-4" /> },
  { id: "image", label: "图文海报", icon: <ImageIcon className="w-4 h-4" /> },
  { id: "office", label: "办公文档", icon: <FileText className="w-4 h-4" /> },
  { id: "video", label: "短视频", icon: <Video className="w-4 h-4" /> },
];

export const EmptyState: React.FC<EmptyStateProps> = ({
  input,
  setInput,
  onSend,
}) => {
  const [activeTab, setActiveTab] = useState("knowledge");

  // Local state for parameters (Mocking visual state)
  const [platform, setPlatform] = useState("xiaohongshu");
  const [model, setModel] = useState("gemini");
  const [ratio, setRatio] = useState("3:4");
  const [style, setStyle] = useState("minimal");
  const [depth, setDepth] = useState("deep");

  const handleSend = () => {
    if (!input.trim()) return;
    let prefix = "";
    if (activeTab === "social")
      prefix = `[社媒创作: ${platform}, Model: ${model}] `;
    if (activeTab === "image") prefix = `[图文生成: ${ratio}, ${style}] `;
    if (activeTab === "video") prefix = `[视频脚本] `;
    if (activeTab === "office") prefix = `[办公文档] `;
    if (activeTab === "knowledge")
      prefix = `[知识探索: ${depth === "deep" ? "深度" : "快速"}, Model: ${model}] `;
    if (activeTab === "planning") prefix = `[计划规划] `;

    onSend(prefix + input);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Dynamic Placeholder
  const getPlaceholder = () => {
    switch (activeTab) {
      case "knowledge":
        return "想了解什么？我可以帮你深度搜索、解析概念或总结长文...";
      case "planning":
        return "告诉我你的目标，无论是旅行计划、职业规划还是活动筹备...";
      case "social":
        return "输入主题，帮你创作小红书爆款文案、公众号文章...";
      case "image":
        return "描述画面主体、风格、构图，生成精美海报或插画...";
      case "video":
        return "输入视频主题，生成分镜脚本和口播文案...";
      case "office":
        return "输入需求，生成周报、汇报PPT大纲或商务邮件...";
      default:
        return "输入你的想法...";
    }
  };

  // Helper to get platform icon
  const getPlatformIcon = (val: string) => {
    if (val === "xiaohongshu") return iconXhs;
    if (val === "wechat") return iconGzh;
    if (val === "zhihu") return iconZhihu;
    if (val === "toutiao") return iconToutiao;
    if (val === "juejin") return iconJuejin;
    if (val === "csdn") return iconCsdn;
    return undefined;
  };

  // Helper to get platform label
  const getPlatformLabel = (val: string) => {
    if (val === "xiaohongshu") return "小红书";
    if (val === "wechat") return "公众号";
    if (val === "zhihu") return "知乎";
    if (val === "toutiao") return "头条";
    if (val === "juejin") return "掘金";
    if (val === "csdn") return "CSDN";
    return val;
  };

  // Helper to get model icon
  const getModelIcon = (val: string) => {
    if (val === "gemini") return modelGemini;
    if (val === "claude") return modelClaude;
    if (val === "deepseek") return modelDeepseek;
    return undefined;
  };

  // Helper to get model label
  const getModelLabel = (val: string) => {
    if (val === "gemini") return "Gemini 3.0 Pro";
    if (val === "claude") return "Claude 3.5 Sonnet";
    if (val === "deepseek") return "DeepSeek V3";
    return val;
  };

  return (
    <Container>
      <ContentWrapper>
        <Header>
          <MainTitle>
            你想在这个平台 <br />
            <span>完成什么？</span>
          </MainTitle>
        </Header>

        <TabsContainer>
          {CATEGORIES.map((cat) => (
            <TabItem
              key={cat.id}
              $active={activeTab === cat.id}
              onClick={() => setActiveTab(cat.id)}
            >
              <span
                className={activeTab === cat.id ? "text-primary" : "opacity-70"}
              >
                {cat.icon}
              </span>
              {cat.label}
            </TabItem>
          ))}
        </TabsContainer>

        <InputCard>
          <StyledTextarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={getPlaceholder()}
          />

          <Toolbar>
            <ToolLoginLeft>
              {activeTab === "social" && (
                <>
                  <Select
                    value={platform}
                    onValueChange={setPlatform}
                    closeOnMouseLeave
                  >
                    <SelectTrigger className="h-8 text-xs bg-background border shadow-sm min-w-[120px]">
                      <div className="flex items-center gap-2">
                        {getPlatformIcon(platform) && (
                          <img
                            src={getPlatformIcon(platform)}
                            className="w-4 h-4 rounded-full"
                          />
                        )}
                        <span>{getPlatformLabel(platform)}</span>
                      </div>
                    </SelectTrigger>
                    <SelectContent className="p-1">
                      <div className="px-2 py-1.5 text-xs text-muted-foreground font-medium">
                        选择要创作的内容平台
                      </div>
                      <SelectItem value="xiaohongshu">
                        <div className="flex items-center gap-2">
                          <img src={iconXhs} className="w-4 h-4 rounded-full" />{" "}
                          小红书
                        </div>
                      </SelectItem>
                      <SelectItem value="wechat">
                        <div className="flex items-center gap-2">
                          <img src={iconGzh} className="w-4 h-4 rounded-full" />{" "}
                          公众号
                        </div>
                      </SelectItem>
                      <SelectItem value="toutiao">
                        <div className="flex items-center gap-2">
                          <img
                            src={iconToutiao}
                            className="w-4 h-4 rounded-full"
                          />{" "}
                          今日头条
                        </div>
                      </SelectItem>
                      <SelectItem value="zhihu">
                        <div className="flex items-center gap-2">
                          <img
                            src={iconZhihu}
                            className="w-4 h-4 rounded-full"
                          />{" "}
                          知乎
                        </div>
                      </SelectItem>
                      <SelectItem value="juejin">
                        <div className="flex items-center gap-2">
                          <img
                            src={iconJuejin}
                            className="w-4 h-4 rounded-full"
                          />{" "}
                          掘金
                        </div>
                      </SelectItem>
                      <SelectItem value="csdn">
                        <div className="flex items-center gap-2">
                          <img
                            src={iconCsdn}
                            className="w-4 h-4 rounded-full"
                          />{" "}
                          CSDN
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </>
              )}

              {activeTab === "knowledge" && (
                <>
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-muted font-normal h-8 px-3 gap-1"
                  >
                    <Search className="w-3.5 h-3.5 mr-1" />
                    联网搜索
                  </Badge>
                  <Select value={depth} onValueChange={setDepth}>
                    <SelectTrigger className="h-8 text-xs bg-background border-input shadow-sm w-[110px]">
                      <BrainCircuit className="w-3.5 h-3.5 mr-2 text-muted-foreground" />
                      <SelectValue placeholder="深度" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="deep">深度解析</SelectItem>
                      <SelectItem value="quick">快速概览</SelectItem>
                    </SelectContent>
                  </Select>
                </>
              )}

              {activeTab === "planning" && (
                <Badge
                  variant="outline"
                  className="h-8 font-normal text-muted-foreground gap-1"
                >
                  <Globe className="w-3.5 h-3.5 mr-1" />
                  旅行/职业/活动
                </Badge>
              )}

              {activeTab === "image" && (
                <>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 text-xs font-normal"
                      >
                        <div className="w-3.5 h-3.5 border border-current rounded-[2px] mr-2 flex items-center justify-center text-[6px]">
                          3:4
                        </div>
                        {ratio}
                        <ChevronDown className="w-3 h-3 ml-1 opacity-50" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-64 p-2" align="start">
                      <div className="text-xs font-medium mb-2 px-2 text-muted-foreground">
                        宽高比
                      </div>
                      <GridSelect>
                        {["1:1", "3:4", "4:3", "9:16", "16:9", "21:9"].map(
                          (r) => (
                            <GridItem
                              key={r}
                              $active={ratio === r}
                              onClick={() => setRatio(r)}
                            >
                              <div className="w-5 h-5 border-2 border-current rounded-sm mb-1 opacity-50"></div>
                              <span className="text-xs">{r}</span>
                            </GridItem>
                          ),
                        )}
                      </GridSelect>
                    </PopoverContent>
                  </Popover>

                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 text-xs font-normal"
                      >
                        <ColorDot $color="#3b82f6" className="mr-2" />
                        {style === "minimal"
                          ? "极简风格"
                          : style === "tech"
                            ? "科技质感"
                            : "温暖治愈"}
                        <ChevronDown className="w-3 h-3 ml-1 opacity-50" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-48 p-1" align="start">
                      <div className="p-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full justify-start h-8"
                          onClick={() => setStyle("minimal")}
                        >
                          <ColorDot $color="#e2e8f0" className="mr-2" />{" "}
                          极简风格
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full justify-start h-8"
                          onClick={() => setStyle("tech")}
                        >
                          <ColorDot $color="#3b82f6" className="mr-2" />{" "}
                          科技质感
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full justify-start h-8"
                          onClick={() => setStyle("warm")}
                        >
                          <ColorDot $color="#f59e0b" className="mr-2" />{" "}
                          温暖治愈
                        </Button>
                      </div>
                    </PopoverContent>
                  </Popover>
                </>
              )}

              {/* Model Selector using Popover for better control or just a Select */}
              <Select value={model} onValueChange={setModel} closeOnMouseLeave>
                <SelectTrigger className="h-8 text-xs bg-background border shadow-sm min-w-[200px] px-2">
                  <div className="flex items-center gap-1.5 text-muted-foreground">
                    {getModelIcon(model) ? (
                      <img src={getModelIcon(model)} className="w-3.5 h-3.5" />
                    ) : (
                      <Sparkles className="w-3.5 h-3.5" />
                    )}
                    <span>{getModelLabel(model)}</span>
                  </div>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gemini">
                    <div className="flex items-center gap-2">
                      <img src={modelGemini} className="w-4 h-4" /> Gemini 3.0
                      Pro
                    </div>
                  </SelectItem>
                  <SelectItem value="claude">
                    <div className="flex items-center gap-2">
                      <img src={modelClaude} className="w-4 h-4" /> Claude 3.5
                      Sonnet
                    </div>
                  </SelectItem>
                  <SelectItem value="deepseek">
                    <div className="flex items-center gap-2">
                      <img src={modelDeepseek} className="w-4 h-4" /> DeepSeek
                      V3
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 rounded-full ml-1 bg-background shadow-sm hover:bg-muted"
              >
                <Globe className="w-4 h-4 opacity-70" />
              </Button>
            </ToolLoginLeft>

            <Button
              size="sm"
              onClick={handleSend}
              disabled={!input.trim()}
              className="bg-primary hover:bg-primary/90 text-primary-foreground h-9 px-5 rounded-xl shadow-lg shadow-primary/20 transition-all hover:scale-105 active:scale-95"
            >
              开始生成
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </Toolbar>
        </InputCard>

        {/* Dynamic Inspiration/Tips based on Tab - Styled nicely */}
        <div className="w-full max-w-[800px] flex flex-wrap gap-3 justify-center">
          {activeTab === "social" &&
            ["爆款标题生成", "小红书文案", "公众号排版", "评论区回复"].map(
              (item) => (
                <Badge
                  key={item}
                  variant="secondary"
                  className="px-4 py-2 text-xs font-normal cursor-pointer hover:bg-muted-foreground/10 transition-colors"
                >
                  ✨ {item}
                </Badge>
              ),
            )}
          {activeTab === "image" &&
            ["海报设计", "插画生成", "UI 界面", "Logo 设计", "摄影修图"].map(
              (item) => (
                <Badge
                  key={item}
                  variant="secondary"
                  className="px-4 py-2 text-xs font-normal cursor-pointer hover:bg-muted-foreground/10 transition-colors"
                >
                  🎨 {item}
                </Badge>
              ),
            )}
          {activeTab === "knowledge" &&
            ["解释量子计算", "总结这篇论文", "如何制定OKR", "分析行业趋势"].map(
              (item) => (
                <Badge
                  key={item}
                  variant="secondary"
                  className="px-4 py-2 text-xs font-normal cursor-pointer hover:bg-muted-foreground/10 transition-colors"
                >
                  🔍 {item}
                </Badge>
              ),
            )}
          {activeTab === "planning" &&
            ["日本旅行计划", "年度职业规划", "婚礼流程表", "健身计划"].map(
              (item) => (
                <Badge
                  key={item}
                  variant="secondary"
                  className="px-4 py-2 text-xs font-normal cursor-pointer hover:bg-muted-foreground/10 transition-colors"
                >
                  📅 {item}
                </Badge>
              ),
            )}
        </div>
      </ContentWrapper>
    </Container>
  );
};
