import type { AutoContinueRequestPayload } from "@/lib/api/agentRuntime/sessionTypes";
import type { HandleSendOptions } from "../../hooks/handleSendTypes";
import type { MessageImage } from "../../types";
import type {
  ComposerDraftSnapshot,
  ComposerIntent,
  ComposerSubmitTarget,
} from "@/components/input-kit";

export type InputbarSendTriggerSource = "button" | "enter" | "ime" | "adapter";

export interface InputbarSendPayload {
  images?: MessageImage[];
  textOverride?: string;
  autoContinuePayload?: AutoContinueRequestPayload;
  sendOptions?: HandleSendOptions;
  triggeredAt?: number;
  triggerSource?: InputbarSendTriggerSource;
  composerIntent?: ComposerIntent;
  composerTarget?: ComposerSubmitTarget;
  composerDraft?: ComposerDraftSnapshot;
}

export type InputbarSendHandler = (
  payload?: InputbarSendPayload,
) => void | Promise<boolean> | boolean;
