import {
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 64;
const RESIZE_FOLLOW_FRAME_COUNT = 4;
const USER_SCROLL_IDLE_MS = 500;

interface UseMessageListAutoScrollOptions {
  isRestoringSession: boolean;
  isSending: boolean;
  renderedMessageCount: number;
  scrollRef: RefObject<HTMLDivElement | null>;
  shouldAutoScroll: boolean;
}

export function useMessageListAutoScroll({
  isRestoringSession,
  isSending,
  renderedMessageCount,
  scrollRef,
  shouldAutoScroll,
}: UseMessageListAutoScrollOptions) {
  const previousVisibleMessageCountRef = useRef<number | null>(null);

  useLayoutEffect(() => {
    const previousVisibleMessageCount = previousVisibleMessageCountRef.current;
    previousVisibleMessageCountRef.current = renderedMessageCount;

    if (!shouldAutoScroll || !scrollRef.current) {
      return;
    }

    const shouldAnimateScroll =
      !isRestoringSession &&
      previousVisibleMessageCount !== null &&
      previousVisibleMessageCount > 0 &&
      renderedMessageCount <= previousVisibleMessageCount + 1;

    scrollRef.current.scrollIntoView({
      behavior: shouldAnimateScroll ? "smooth" : "auto",
      block: "end",
    });
  }, [
    isRestoringSession,
    isSending,
    renderedMessageCount,
    scrollRef,
    shouldAutoScroll,
  ]);
}

function isNearScrollBottom(container: HTMLDivElement): boolean {
  const { scrollTop, scrollHeight, clientHeight } = container;
  return (
    scrollHeight - scrollTop - clientHeight <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX
  );
}

export function useMessageListScrollController() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const shouldAutoScrollRef = useRef(true);
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setAutoScrollEnabled = useCallback((value: boolean) => {
    shouldAutoScrollRef.current = value;
    setShouldAutoScroll(value);
  }, []);

  const markUserScrolling = useCallback(() => {
    setIsUserScrolling(true);

    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }

    scrollTimeoutRef.current = setTimeout(() => {
      setIsUserScrolling(false);
      scrollTimeoutRef.current = null;
    }, USER_SCROLL_IDLE_MS);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const handleScroll = () => {
      markUserScrolling();
      if (isNearScrollBottom(container)) {
        setAutoScrollEnabled(true);
      }
    };

    const handleWheel = (event: WheelEvent) => {
      markUserScrolling();
      if (event.deltaY < 0) {
        setAutoScrollEnabled(false);
      }
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    container.addEventListener("wheel", handleWheel, { passive: true });

    return () => {
      container.removeEventListener("scroll", handleScroll);
      container.removeEventListener("wheel", handleWheel);

      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
        scrollTimeoutRef.current = null;
      }
    };
  }, [markUserScrolling, setAutoScrollEnabled]);

  const scrollToTail = useCallback((behavior: "auto" | "smooth") => {
    const container = containerRef.current;
    if (container && typeof container.scrollTo === "function") {
      container.scrollTo({
        behavior,
        top: container.scrollHeight,
      });
      return;
    }

    scrollRef.current?.scrollIntoView({
      behavior,
      block: "end",
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") {
      return;
    }

    let animationFrame: number | null = null;
    let followFramesRemaining = 0;
    const runResizeFollow = () => {
      animationFrame = null;
      if (!shouldAutoScrollRef.current) {
        followFramesRemaining = 0;
        return;
      }

      scrollToTail("auto");
      followFramesRemaining -= 1;
      if (followFramesRemaining > 0) {
        animationFrame = window.requestAnimationFrame(runResizeFollow);
      }
    };
    const scheduleResizeFollow = () => {
      if (!shouldAutoScrollRef.current) {
        return;
      }
      if (animationFrame !== null && typeof window !== "undefined") {
        window.cancelAnimationFrame(animationFrame);
        animationFrame = null;
      }
      followFramesRemaining = RESIZE_FOLLOW_FRAME_COUNT;
      if (typeof window !== "undefined" && window.requestAnimationFrame) {
        animationFrame = window.requestAnimationFrame(runResizeFollow);
        return;
      }
      scrollToTail("auto");
      followFramesRemaining = 0;
    };

    const resizeObserver = new ResizeObserver(scheduleResizeFollow);
    resizeObserver.observe(container);
    const messageColumn = container.firstElementChild;
    if (messageColumn instanceof Element) {
      resizeObserver.observe(messageColumn);
    }
    window.addEventListener("resize", scheduleResizeFollow);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", scheduleResizeFollow);
      if (animationFrame !== null && typeof window !== "undefined") {
        window.cancelAnimationFrame(animationFrame);
      }
    };
  }, [scrollToTail]);

  const handleStreamingOverlayUpdate = useCallback(() => {
    if (!scrollRef.current) {
      return;
    }

    if (!shouldAutoScrollRef.current) {
      return;
    }

    setAutoScrollEnabled(true);

    const run = () => scrollToTail("auto");

    if (typeof window !== "undefined" && window.requestAnimationFrame) {
      window.requestAnimationFrame(run);
      return;
    }

    run();
  }, [scrollToTail, setAutoScrollEnabled]);

  return {
    containerRef,
    handleStreamingOverlayUpdate,
    isUserScrolling,
    scrollRef,
    shouldAutoScroll,
  };
}
