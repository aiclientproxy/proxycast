import React, { useEffect, useRef } from "react";
import styled, { keyframes } from "styled-components";
import { HomeSkillGallery } from "../home/HomeSkillGallery";
import type {
  HomeSurfaceChromeCopy,
  HomeSurfaceHeroCopy,
} from "../home/homeSurfaceCopy";
import type { HomeSkillSurfaceItem } from "../home/homeSurfaceTypes";
import { EmptyStateHero } from "./EmptyStateHero";
import {
  EMPTY_STATE_CONTENT_WRAPPER_CLASSNAME,
  EMPTY_STATE_PAGE_CONTAINER_CLASSNAME,
} from "./emptyStateSurfaceTokens";

const contentReveal = keyframes`
  from {
    opacity: 0;
    transform: translateY(18px) scale(0.992);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
`;

const PageContainer = styled.div.attrs({
  className: EMPTY_STATE_PAGE_CONTAINER_CLASSNAME,
})`
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  height: 100%;
  flex-direction: column;
  overflow-y: auto;
  overscroll-behavior: contain;
  scroll-behavior: smooth;
  scroll-snap-type: y mandatory;
  isolation: isolate;
  background: var(--lime-stage-surface, #fdf3f7);
`;

const ContentWrapper = styled.div.attrs({
  className: EMPTY_STATE_CONTENT_WRAPPER_CLASSNAME,
})`
  display: flex;
  flex: 0 0 auto;
  min-height: 100%;
  height: auto;
  width: 100%;
  box-sizing: border-box;
  position: relative;
  overflow: visible;
  scroll-snap-align: start;
  scroll-snap-stop: always;
  animation: ${contentReveal} 560ms cubic-bezier(0.22, 1, 0.36, 1) both;
  padding: 0.25rem 0.25rem 3.5rem;

  @media (prefers-reduced-motion: reduce) {
    animation: none;
  }

  @media (max-height: 860px) {
    padding-top: 0.15rem;
  }
`;

const ComposerGlowFrame = styled.div`
  position: relative;
  isolation: isolate;

  @media (max-height: 780px) {
    margin-top: clamp(-112px, calc(100vh - 720px), 0px);
    z-index: 60;
  }

  &::after {
    content: "";
    position: absolute;
    left: clamp(1.5rem, 9vw, 7rem);
    right: clamp(1.5rem, 9vw, 7rem);
    bottom: -1.1rem;
    z-index: 0;
    height: clamp(24px, 4vw, 44px);
    border-radius: 999px;
    background: var(
      --lime-composer-glow,
      linear-gradient(
        90deg,
        transparent 0%,
        rgba(217, 95, 135, 0.05) 16%,
        rgba(217, 95, 135, 0.16) 42%,
        rgba(238, 130, 164, 0.14) 58%,
        rgba(250, 196, 214, 0.08) 84%,
        transparent 100%
      )
    );
    filter: blur(20px);
    opacity: 0.46;
    pointer-events: none;
  }

  > * {
    position: relative;
    z-index: 1;
  }
`;

const PrimaryStackFrame = styled.div`
  display: flex;
  width: 100%;
  min-width: 0;
  flex-direction: column;
  gap: 0.75rem;
`;

const SecondScreenSection = styled.section`
  display: flex;
  flex: 0 0 auto;
  min-height: 100%;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  align-items: flex-start;
  justify-content: center;
  overflow-y: auto;
  overscroll-behavior: auto;
  scroll-snap-align: start;
  scroll-snap-stop: always;
  padding: clamp(3.5rem, 8vh, 6rem) 0.25rem clamp(3rem, 8vh, 5.5rem);
`;

const SecondScreenInner = styled.div`
  width: min(1180px, 100%);
  min-width: 0;
`;

interface UseSecondScreenWheelInput {
  enabled: boolean;
  pageContainerRef: React.RefObject<HTMLDivElement | null>;
  secondScreenSectionRef: React.RefObject<HTMLElement | null>;
}

function useSecondScreenWheelReturnToFirstScreen({
  enabled,
  pageContainerRef,
  secondScreenSectionRef,
}: UseSecondScreenWheelInput) {
  useEffect(() => {
    if (!enabled) {
      return;
    }

    const secondScreenSection = secondScreenSectionRef.current;
    if (!secondScreenSection) {
      return;
    }

    const handleSecondScreenWheel = (event: WheelEvent) => {
      if (event.deltaY >= 0 || secondScreenSection.scrollTop > 1) {
        return;
      }

      event.preventDefault();
      pageContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    };

    secondScreenSection.addEventListener("wheel", handleSecondScreenWheel, {
      passive: false,
    });

    return () => {
      secondScreenSection.removeEventListener("wheel", handleSecondScreenWheel);
    };
  }, [enabled, pageContainerRef, secondScreenSectionRef]);
}

interface EmptyStateComposerFrameProps {
  children: React.ReactNode;
}

export function EmptyStateComposerFrame({
  children,
}: EmptyStateComposerFrameProps) {
  return (
    <ComposerGlowFrame data-testid="empty-state-composer-frame">
      {children}
    </ComposerGlowFrame>
  );
}

interface EmptyStatePrimaryStackProps {
  children: React.ReactNode;
}

export function EmptyStatePrimaryStack({
  children,
}: EmptyStatePrimaryStackProps) {
  return (
    <PrimaryStackFrame data-testid="empty-state-primary-stack">
      {children}
    </PrimaryStackFrame>
  );
}

interface EmptyStateLayoutProps {
  heroCopy: HomeSurfaceHeroCopy;
  chromeCopy: HomeSurfaceChromeCopy;
  prioritySlot: React.ReactNode;
  supportingSlot?: React.ReactNode;
  isGeneralTheme: boolean;
  galleryItems: HomeSkillSurfaceItem[];
  onSelectGalleryItem: (item: HomeSkillSurfaceItem) => void;
  children?: React.ReactNode;
}

export function EmptyStateLayout({
  heroCopy,
  chromeCopy,
  prioritySlot,
  supportingSlot,
  isGeneralTheme,
  galleryItems,
  onSelectGalleryItem,
  children,
}: EmptyStateLayoutProps) {
  const pageContainerRef = useRef<HTMLDivElement | null>(null);
  const secondScreenSectionRef = useRef<HTMLElement | null>(null);
  const shouldShowSecondScreen = isGeneralTheme && galleryItems.length > 0;

  useSecondScreenWheelReturnToFirstScreen({
    enabled: shouldShowSecondScreen,
    pageContainerRef,
    secondScreenSectionRef,
  });

  return (
    <PageContainer ref={pageContainerRef}>
      <ContentWrapper data-testid="empty-state-first-screen">
        <EmptyStateHero
          eyebrow={heroCopy.eyebrow}
          slogan={heroCopy.slogan}
          description={heroCopy.description}
          skinBrandSubtitle={heroCopy.skinBrandSubtitle}
          skinTagline={heroCopy.skinTagline}
          skinStatus={heroCopy.skinStatus}
          prioritySlot={prioritySlot}
          supportingSlot={supportingSlot}
        />
      </ContentWrapper>
      {shouldShowSecondScreen ? (
        <SecondScreenSection
          ref={secondScreenSectionRef}
          id="home-skill-gallery-screen"
          aria-label={chromeCopy.secondScreenLabel}
          data-testid="home-second-screen"
        >
          <SecondScreenInner>
            <HomeSkillGallery
              items={galleryItems}
              copy={chromeCopy}
              onSelectItem={onSelectGalleryItem}
            />
          </SecondScreenInner>
        </SecondScreenSection>
      ) : null}
      {children}
    </PageContainer>
  );
}
