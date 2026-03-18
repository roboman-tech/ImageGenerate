const DEFAULT_API_BASE = "http://localhost:8000";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE;

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface ChatImagePayload {
  description?: string | null;
  prompt?: string | null;
  negative_prompt?: string | null;
  base64?: string | null;
  url?: string | null;
  mime_type?: string | null;
  width?: number | null;
  height?: number | null;
}

export interface ChatResponse {
  reply?: string;
  image?: ChatImagePayload | null;
  image_description?: string | null;
  image_prompt?: string | null;
  negative_prompt?: string | null;
  entry_id?: number | null;
  entry_date?: string | null;
  created_at?: string | null;
}

export interface DiaryDatesResponse {
  dates: string[];
}

export interface DiaryEntryOut {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  entry_date: string;
  image?: ChatImagePayload | null;
  image_prompt?: string | null;
}

export interface DiaryDayResponse {
  date: string;
  items: DiaryEntryOut[];
}

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {})
    },
    ...options
  });

  const text = await res.text();
  let data: unknown;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error("Invalid JSON response from server");
  }

  if (!res.ok) {
    const message =
      (data as { detail?: string })?.detail ??
      `Request failed with status ${res.status}`;
    throw new Error(message);
  }

  return data as T;
}

export async function registerUser(email: string, password: string) {
  return request<{ message: string }>("/register", {
    method: "POST",
    body: JSON.stringify({
      type: "local",
      email,
      password
    })
  });
}

export async function loginUser(
  email: string,
  password: string
): Promise<TokenResponse> {
  return request<TokenResponse>("/login", {
    method: "POST",
    body: JSON.stringify({
      type: "local",
      email,
      password
    })
  });
}

export async function sendChatMessage(
  content: string,
  token: string
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      content,
      token,
      entry_date: null
    })
  });
}

export async function sendDiaryChatMessage(
  content: string,
  token: string,
  entryDate: string
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      content,
      token,
      entry_date: entryDate
    })
  });
}

export async function fetchDiaryDates(token: string): Promise<DiaryDatesResponse> {
  return request<DiaryDatesResponse>(`/diary/dates?token=${encodeURIComponent(token)}`, {
    method: "GET"
  });
}

export async function fetchDiaryDay(
  entryDate: string,
  token: string
): Promise<DiaryDayResponse> {
  return request<DiaryDayResponse>(
    `/diary/${encodeURIComponent(entryDate)}?token=${encodeURIComponent(token)}`,
    { method: "GET" }
  );
}

