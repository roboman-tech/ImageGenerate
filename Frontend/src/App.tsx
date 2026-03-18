import React, { useEffect, useMemo, useRef, useState } from "react";
import HTMLFlipBook from "react-pageflip";
import {
  API_BASE,
  loginUser,
  registerUser,
  sendDiaryChatMessage,
  fetchDiaryDates,
  fetchDiaryDay,
  DiaryEntryOut
} from "./api";

type AuthMode = "login" | "register";

interface ChatItem {
  id: string;
  role: "user" | "assistant";
  content: string;
  date: string;
  imageBase64?: string | null;
  imageUrl?: string | null;
  imagePrompt?: string | null;
}

type FlipBookHandle = {
  pageFlip: () => { flip: (page: number) => void };
};

export const App: React.FC = () => {
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState<string | null>(
    () => window.localStorage.getItem("cm_token")
  );
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  const [chatInput, setChatInput] = useState("");
  const [chatItems, setChatItems] = useState<ChatItem[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [expandedPrompts, setExpandedPrompts] = useState<Record<string, boolean>>({});
  const [selectedDate, setSelectedDate] = useState<string>(() =>
    new Date().toISOString().slice(0, 10)
  );
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [diaryDates, setDiaryDates] = useState<string[]>([]);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [pageSize, setPageSize] = useState({ w: 340, h: 420 });

  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const bookContainerRef = useRef<HTMLDivElement | null>(null);
  const flipBookRef = useRef<FlipBookHandle | null>(null);

  useEffect(() => {
    if (token) {
      window.localStorage.setItem("cm_token", token);
    } else {
      window.localStorage.removeItem("cm_token");
    }
  }, [token]);

  const loadDiaryForDate = async (t: string, d: string) => {
    const day = await fetchDiaryDay(d, t);
    const items: ChatItem[] = day.items.map((it: DiaryEntryOut) => ({
      id: `${it.id}-${it.role}-${it.created_at}`,
      role: it.role,
      content: it.content,
      date: it.entry_date,
      imageBase64: it.image?.base64 ?? null,
      imageUrl: it.image?.url ?? null,
      imagePrompt: it.image_prompt ?? it.image?.prompt ?? null
    }));
    setChatItems(items);
    const firstWithImage = items.find((x) => x.role === "assistant" && (x.imageBase64 || x.imageUrl));
    setSelectedImageId(firstWithImage?.id ?? null);
  };

  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const { dates } = await fetchDiaryDates(token);
        setDiaryDates(dates);
        const effectiveDate = dates[0] ?? selectedDate;
        if (effectiveDate !== selectedDate) setSelectedDate(effectiveDate);
        await loadDiaryForDate(token, effectiveDate);
      } catch (e) {
        // keep UI usable even if history fails
        console.error(e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        await loadDiaryForDate(token, selectedDate);
      } catch (e) {
        console.error(e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate, token]);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [chatItems.length, selectedDate]);

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setAuthLoading(true);
    try {
      if (authMode === "register") {
        await registerUser(email, password);
        const res = await loginUser(email, password);
        setToken(res.access_token);
      } else {
        const res = await loginUser(email, password);
        setToken(res.access_token);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Authentication failed";
      setAuthError(message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setToken(null);
    setChatItems([]);
    setChatInput("");
    setDiaryDates([]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token || !chatInput.trim()) return;

    const content = chatInput.trim();
    const today = selectedDate;
    const userItem: ChatItem = {
      id: `${Date.now()}-user`,
      role: "user",
      content,
      date: today
    };
    setChatItems((prev) => [...prev, userItem]);
    setChatInput("");
    setChatError(null);
    setChatLoading(true);

    try {
      const res = await sendDiaryChatMessage(content, token, today);
      const assistantItem: ChatItem = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: res.reply ?? "",
        date: today,
        imageBase64: res.image?.base64 ?? null,
        imageUrl: res.image?.url ?? null,
        imagePrompt: res.image_prompt ?? res.image?.prompt ?? null
      };
      setChatItems((prev) => [...prev, assistantItem]);
      if (assistantItem.imageBase64) {
        setSelectedImageId(assistantItem.id);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to send message";
      setChatError(message);
    } finally {
      setChatLoading(false);
    }
  };

  const datesInDiary = Array.from(
    new Set(chatItems.map((item) => item.date).filter(Boolean))
  ).sort();

  const itemsForSelectedDate = chatItems.filter(
    (item) => item.date === selectedDate
  );

  const imagesForSelectedDate = itemsForSelectedDate.filter(
    (item) => item.role === "assistant" && (item.imageBase64 || item.imageUrl)
  );
  const selectedImageIndex = Math.max(
    0,
    imagesForSelectedDate.findIndex((img) => img.id === selectedImageId)
  );

  const imageSrc = (it: ChatItem | undefined) => {
    if (!it) return null;
    if (it.imageBase64) return `data:image/png;base64,${it.imageBase64}`;
    if (it.imageUrl) return `${API_BASE}${it.imageUrl}`;
    return null;
  };
  const pages = useMemo(() => imagesForSelectedDate, [imagesForSelectedDate]);

  useEffect(() => {
    if (!bookContainerRef.current) return;
    const el = bookContainerRef.current;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      const bookW = Math.max(520, Math.floor(rect.width));
      const pageW = Math.floor(bookW / 2);
      const pageH = Math.floor(pageW * (3 / 4)); // 4:3 page
      setPageSize({ w: pageW, h: pageH });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    // When user selects a thumbnail, jump flipbook to that page
    if (!flipBookRef.current) return;
    if (selectedImageIndex < 0) return;
    setCurrentPageIndex(selectedImageIndex);
    try {
      flipBookRef.current.pageFlip().flip(selectedImageIndex);
    } catch {
      // ignore if not ready
    }
  }, [selectedImageIndex]);

  return (
    <div className="app-root">
      <div className="app-gradient" />
      <div className="app-shell">
        <header className="app-header">
          <div className="logo-group">
            <div className="logo-mark">DC</div>
            <div>
              <div className="logo-title">Diary Cartoon Books</div>
              <div className="logo-subtitle">
                A daily picture diary for kids
              </div>
            </div>
          </div>
          {token && (
            <button className="ghost-button" onClick={handleLogout}>
              Logout
            </button>
          )}
        </header>

        <main className="app-main">
          {!token ? (
            <section className="auth-layout">
              <div className="auth-hero">
                <h1>Welcome to Diary Cartoon Books</h1>
                <p>
                  Turn your daily thoughts into a cozy cartoon diary. Every chat
                  becomes a new page in your picture book.
                </p>
                <ul>
                  <li>Pick a date and write your diary</li>
                  <li>Every story becomes a colorful illustration</li>
                  <li>Safe, gentle conversations for young minds</li>
                </ul>
              </div>

              <div className="auth-card">
                <div className="auth-toggle">
                  <button
                    className={
                      authMode === "login" ? "auth-toggle-btn active" : "auth-toggle-btn"
                    }
                    onClick={() => {
                      setAuthMode("login");
                      setAuthError(null);
                    }}
                  >
                    Login
                  </button>
                  <button
                    className={
                      authMode === "register"
                        ? "auth-toggle-btn active"
                        : "auth-toggle-btn"
                    }
                    onClick={() => {
                      setAuthMode("register");
                      setAuthError(null);
                    }}
                  >
                    Create account
                  </button>
                </div>

                <form className="auth-form" onSubmit={handleAuthSubmit}>
                  <h2 className="auth-title">
                    {authMode === "login" ? "Sign in to continue" : "Create your account"}
                  </h2>
                  <p className="auth-subtitle">
                    Use a parent or guardian email. Keep it safe and private.
                  </p>

                  <label className="field">
                    <span>Email</span>
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com"
                    />
                  </label>
                  <label className="field">
                    <span>Password</span>
                    <input
                      type="password"
                      required
                      minLength={6}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="At least 6 characters"
                    />
                  </label>
                  {authError && <div className="error-banner">{authError}</div>}
                  <button
                    type="submit"
                    className="primary-button primary-button-wide"
                    disabled={authLoading}
                  >
                    {authLoading
                      ? authMode === "login"
                        ? "Logging in..."
                        : "Creating account..."
                      : authMode === "login"
                      ? "Login"
                      : "Create account"}
                  </button>
                  <p className="auth-note">
                    This space is designed for under‑18 users. Conversations stay
                    kind, safe, and educational.
                  </p>
                </form>
              </div>
            </section>
          ) : (
            <section className="chat-layout">
              <div className="chat-column">
                <div className="panel-header panel-header-row">
                  <div>
                    <h2>Diary chat</h2>
                    <p>Write what happened today, or any day you choose.</p>
                  </div>
                  <label className="date-picker">
                    <span>Date</span>
                    <input
                      type="date"
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                    />
                  </label>
                </div>

                <div className="chat-window">
                  {itemsForSelectedDate.length === 0 ? (
                    <div className="chat-empty">
                      <h3>Start today&apos;s diary</h3>
                      <p>
                        Pick a date and tell a short story about your day. We&apos;ll
                        turn it into a picture.
                      </p>
                    </div>
                  ) : (
                    <div className="chat-messages">
                      {itemsForSelectedDate.map((item, index) => {
                        const showDateHeader =
                          index === 0 ||
                          itemsForSelectedDate[index - 1].date !== item.date;
                        return (
                          <React.Fragment key={item.id}>
                            {showDateHeader && (
                              <div className="chat-date-label">
                                {item.date}
                              </div>
                            )}
                            <div
                              className={
                                item.role === "user"
                                  ? "chat-bubble chat-bubble-user"
                                  : "chat-bubble chat-bubble-assistant"
                              }
                            >
                              <div className="chat-bubble-label">
                                {item.role === "user" ? "You" : "Story helper"}
                              </div>
                              <div className="chat-bubble-body">{item.content}</div>
                            </div>
                          </React.Fragment>
                        );
                      })}
                      <div ref={chatEndRef} />
                    </div>
                  )}
                </div>

                <form className="chat-input-row" onSubmit={handleSendMessage}>
                  <textarea
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Type your message here..."
                    rows={2}
                  />
                  <button
                    type="submit"
                    className="primary-button"
                    disabled={!chatInput.trim() || chatLoading}
                  >
                    {chatLoading ? "Thinking..." : "Send & Create Image"}
                  </button>
                </form>
                {chatError && <div className="error-banner mt-8">{chatError}</div>}
              </div>

              <div className="image-column">
                <div className="panel-header">
                  <h2>Diary book</h2>
                  <p>
                    Flip through today&apos;s cartoon pages. Tap a thumbnail to open
                    it like a story book.
                  </p>
                </div>
                <div className="image-frame">
                  {imagesForSelectedDate.length > 0 ? (
                    <>
                      <div className="book-viewer">
                        {pages.length > 0 ? (
                          <div className="flipbook-shell" ref={bookContainerRef}>
                            <HTMLFlipBook
                              ref={flipBookRef}
                              width={pageSize.w}
                              height={pageSize.h}
                              size="fixed"
                              maxShadowOpacity={0.6}
                              showCover={false}
                              mobileScrollSupport={true}
                              className="flipbook"
                              onFlip={(e: { data: number }) => {
                                const idx = Math.max(0, Math.min(pages.length - 1, e.data));
                                setCurrentPageIndex(idx);
                                setSelectedImageId(pages[idx]?.id ?? null);
                              }}
                            >
                              {pages.map((p) => (
                                <div className="flipbook-page" key={p.id}>
                                  <div className="book-page-inner">
                                    <div className="book-page-half">
                                      <img
                                        src={imageSrc(p) ?? ""}
                                        draggable={false}
                                        alt={p.imagePrompt ?? "Diary illustration"}
                                      />
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </HTMLFlipBook>
                          </div>
                        ) : (
                          <div className="image-empty">
                            <p>Select a thumbnail below to open a page.</p>
                          </div>
                        )}
                      </div>
                      <div className="thumbnail-strip">
                        {imagesForSelectedDate.map((item) => (
                          <button
                            key={item.id}
                            type="button"
                            className={
                              item.id === selectedImageId
                                ? "thumb thumb-active"
                                : "thumb"
                            }
                            onClick={() => setSelectedImageId(item.id)}
                          >
                            <img
                              src={imageSrc(item) ?? ""}
                              draggable={false}
                              alt="Diary thumbnail"
                            />
                          </button>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="image-empty">
                      <p>
                        When you write a diary entry and get an illustration, it will
                        appear here as today&apos;s book page.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
};

