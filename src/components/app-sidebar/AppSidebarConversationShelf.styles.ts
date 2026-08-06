import styled from "styled-components";

export const ConversationShelf = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 2px 0 12px;
`;

export const ConversationSection = styled.section<{ $compact?: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 7px;
  min-height: ${({ $compact }) => ($compact ? "auto" : "116px")};
  max-height: ${({ $compact }) => ($compact ? "180px" : "248px")};
  padding: 8px;
  border-radius: 14px;
  border: 1px solid var(--sidebar-card-border, var(--sidebar-border));
  background: color-mix(
    in srgb,
    var(--sidebar-search-bg, #ffffff) 88%,
    transparent
  );
  box-shadow: inset 0 1px 0 var(--sidebar-card-highlight);
  overflow: hidden;
`;

export const ConversationSectionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 3px;
  color: var(--sidebar-muted);
`;

export const ConversationSectionActions = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 2px;
  flex-shrink: 0;
`;

export const ConversationSectionTitle = styled.h2`
  display: inline-flex;
  align-items: center;
  padding: 0;
  margin: 0;
  color: inherit;
  font-size: 12px;
  font-weight: 760;
`;

export const ConversationActionButton = styled.button`
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 9px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--sidebar-muted);
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    color 0.18s ease;

  &:hover {
    background: var(--sidebar-hover);
    color: var(--sidebar-foreground);
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.48;
  }

  svg {
    width: 16px;
    height: 16px;
  }
`;

export const ConversationList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 2px;

  &::-webkit-scrollbar {
    width: 4px;
  }

  &::-webkit-scrollbar-track {
    background: transparent;
  }

  &::-webkit-scrollbar-thumb {
    background: var(--sidebar-border);
    border-radius: 9999px;
  }
`;

export const ConversationListMoreButton = styled.button`
  width: 100%;
  min-height: 32px;
  border: 1px solid var(--sidebar-card-border, var(--sidebar-border));
  border-radius: 11px;
  background: var(--sidebar-search-bg);
  color: var(--sidebar-muted);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    border-color 0.18s ease,
    color 0.18s ease;

  &:hover {
    background: var(--sidebar-hover);
    border-color: var(--sidebar-search-border-hover);
    color: var(--sidebar-foreground);
  }
`;
