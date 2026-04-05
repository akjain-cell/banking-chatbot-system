/**
 * API.JS - Frontend API Client
 * Sends X-API-Key header on all /api/v1/chat requests
 */

import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

console.log("🔌 API Base URL:", API_BASE_URL);

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor - add session ID + API key
api.interceptors.request.use((config) => {
  const sessionId = getOrCreateSessionId();
  config.headers["X-Session-Id"] = sessionId;

  // Add API key for protected endpoints
  if (API_KEY) {
    config.headers["X-API-Key"] = API_KEY;
  }

  console.log("📤 Sending request to:", config.baseURL + config.url);
  return config;
});

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log("📥 Response received:", response.data);
    return response;
  },
  (error) => {
    console.error("❌ API Error:", error.response?.status, error.message);
    throw error;
  }
);

// Session management
function getOrCreateSessionId() {
  const KEY = "chatbot_session_id";
  let id;
  try {
    id = window.localStorage.getItem(KEY);
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem(KEY, id);
      console.log("🆔 Created new session:", id);
    }
  } catch (e) {
    // localStorage blocked in some browsers
    id = crypto.randomUUID();
  }
  return id;
}

/**
 * Main chat query
 */
export async function queryBot({ query, userId }) {
  try {
    const payload = {
      query,
      user_id: userId || "web-client",
      channel: "web"
    };

    console.log("❓ Querying:", payload);
    const res = await api.post("/api/v1/chat", payload);
    console.log("✅ Got answer:", res.data);
    return res.data;

  } catch (error) {
    console.error("❌ Query failed:", error.message);

    // Return a safe error object instead of crashing the UI
    return {
      success: false,
      query,
      answer: null,
      confidence_level: "low",
      confidence_score: 0,
      requires_human_handoff: true,
      fallback_message:
        error.response?.status === 403
          ? "Authentication error. Please contact support."
          : "Something went wrong. Please try again.",
      related_questions: [],
      youtube_links: [],
    };
  }
}

/**
 * Get frequent/popular questions
 */
export async function getFrequentQuestions(limit = 10) {
  try {
    const res = await api.get("/api/v1/frequent-questions", { params: { limit } });
    return res.data;
  } catch (error) {
    console.warn("⚠️ Could not fetch frequent questions:", error.message);
    return { success: false, questions: [], count: 0 };
  }
}

/**
 * Get autocomplete suggestions
 */
export async function getSuggestions(query = "", limit = 5) {
  try {
    const res = await api.get("/api/v1/suggestions", { params: { query, limit } });
    return res.data;
  } catch (error) {
    console.warn("⚠️ Could not fetch suggestions:", error.message);
    return { suggestions: [], count: 0 };
  }
}

/**
 * Health check
 */
export async function checkHealth() {
  try {
    const res = await api.get("/health");
    return res.data;
  } catch (error) {
    console.error("❌ Health check failed:", error.message);
    throw error;
  }
}
