import React, { useState } from "react";
import { Bot, ChevronDown, Check, Box, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Navbar } from "../styles";
import { PROVIDER_CONFIG } from "../types";
import { cn } from "@/lib/utils";

interface ChatNavbarProps {
  providerType: string;
  setProviderType: (type: string) => void;
  model: string;
  setModel: (model: string) => void;
  isRunning: boolean;
  onToggleHistory: () => void;
  onToggleFullscreen: () => void;
  onToggleSettings?: () => void;
}

export const ChatNavbar: React.FC<ChatNavbarProps> = ({
  providerType,
  setProviderType,
  model,
  setModel,
  isRunning: _isRunning,
  onToggleHistory,
  onToggleFullscreen: _onToggleFullscreen,
  onToggleSettings,
}) => {
  const [open, setOpen] = useState(false);

  const selectedProviderLabel =
    PROVIDER_CONFIG[providerType]?.label || providerType;
  const currentModels = PROVIDER_CONFIG[providerType]?.models || [];

  return (
    <Navbar>
      <div className="flex items-center gap-2">
        {/* History Toggle (Left) */}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground"
          onClick={onToggleHistory}
        >
          <Box size={18} />
        </Button>
      </div>

      {/* Center: Model Selector */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              role="combobox"
              aria-expanded={open}
              className="h-9 px-3 gap-2 font-normal hover:bg-muted text-foreground"
            >
              <Bot size={16} className="text-primary" />
              <span className="font-medium">{selectedProviderLabel}</span>
              <span className="text-muted-foreground">/</span>
              <span className="text-sm">{model || "Select Model"}</span>
              <ChevronDown className="ml-1 h-3 w-3 text-muted-foreground opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            className="w-[400px] p-0 bg-background/95 backdrop-blur-sm border-border shadow-lg"
            align="center"
          >
            <div className="flex h-[300px]">
              {/* Left Column: Providers */}
              <div className="w-[140px] border-r bg-muted/30 p-2 flex flex-col gap-1 overflow-y-auto">
                <div className="text-xs font-semibold text-muted-foreground px-2 py-1.5 mb-1">
                  Providers
                </div>
                {Object.entries(PROVIDER_CONFIG).map(([key, config]) => (
                  <button
                    key={key}
                    onClick={() => {
                      setProviderType(key);
                      // Auto-select first model if available
                      if (config.models.length > 0) {
                        setModel(config.models[0]);
                      } else {
                        setModel("");
                      }
                    }}
                    className={cn(
                      "flex items-center justify-between w-full px-2 py-1.5 text-sm rounded-md transition-colors text-left",
                      providerType === key
                        ? "bg-primary/10 text-primary font-medium"
                        : "hover:bg-muted text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {config.label}
                    {providerType === key && (
                      <div className="w-1 h-1 rounded-full bg-primary" />
                    )}
                  </button>
                ))}
              </div>

              {/* Right Column: Models */}
              <div className="flex-1 p-2 flex flex-col overflow-hidden">
                <div className="text-xs font-semibold text-muted-foreground px-2 py-1.5 mb-1">
                  Models
                </div>
                <ScrollArea className="flex-1">
                  <div className="space-y-1 p-1">
                    {currentModels.length === 0 ? (
                      <div className="text-xs text-muted-foreground p-2">
                        No models available
                      </div>
                    ) : (
                      currentModels.map((m) => (
                        <button
                          key={m}
                          onClick={() => {
                            setModel(m);
                            setOpen(false);
                          }}
                          className={cn(
                            "flex items-center justify-between w-full px-2 py-1.5 text-sm rounded-md transition-colors text-left group",
                            model === m
                              ? "bg-accent text-accent-foreground"
                              : "hover:bg-muted text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {m}
                          {model === m && (
                            <Check size={14} className="text-primary" />
                          )}
                        </button>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {/* Right: Status & Settings */}
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground"
          onClick={onToggleSettings}
        >
          <Settings2 size={18} />
        </Button>
      </div>
    </Navbar>
  );
};
