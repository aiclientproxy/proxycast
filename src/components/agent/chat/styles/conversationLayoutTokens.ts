// The outer workspace caps the column at 720px; nested surfaces must fill it.
export const CONVERSATION_CONTENT_TARGET_WIDTH = "clamp(640px, 100%, 720px)";

export const CONVERSATION_CONTENT_MAX_WIDTH = `min(100%, ${CONVERSATION_CONTENT_TARGET_WIDTH})`;

export const INLINE_CONVERSATION_CONTENT_WIDTH = `min(calc(100% - 20px), ${CONVERSATION_CONTENT_MAX_WIDTH})`;

export const FLOATING_CONVERSATION_CONTENT_WIDTH = `min(calc(100% - 16px), ${CONVERSATION_CONTENT_MAX_WIDTH})`;
