/**
 * API.JS - Frontend API Client
 * Sends X-API-Key header on all protected requests.
 * Timeout raised to 75s to survive Render free-tier cold starts (~45-60s).
 */
import { getEmbedding } from "./services/onnxEmbedder.js";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API_KEY      = import.meta.env.VITE_API_KEY      || "";

console.log("🔌 API Base URL:", API_BASE_URL);

// ─── Axios instance ────────────────────────────────────────────────────────
// 75 000 ms = 75 s — enough for Render free-tier cold boot (usually 45-60 s)
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 75000,
  headers: { "Content-Type": "application/json" },
});

// ─── Request interceptor ───────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  const sessionId = getOrCreateSessionId();
  config.headers["X-Session-Id"] = sessionId;
  if (API_KEY) config.headers["X-API-Key"] = API_KEY;
  console.log("📤 Sending request to:", config.baseURL + config.url);
  return config;
});

// ─── Response interceptor ──────────────────────────────────────────────────
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

// ─── Session management ────────────────────────────────────────────────────
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
  } catch {
    id = crypto.randomUUID();
  }
  return id;
}

// ─── Wake-up ping ──────────────────────────────────────────────────────────
/**
 * Hit /health as soon as the page loads so the Render dyno wakes up
 * BEFORE the user submits their first question.
 * Called once from App.jsx on mount (useEffect).
 */
export async function warmupServer() {
  try {
    // Use a short independent fetch — don't block the UI
    await axios.get(`${API_BASE_URL}/health`, { timeout: 75000 });
    console.log("✅ Server warm — ready for queries");
  } catch {
    console.warn("⚠️ Server warmup ping failed (may still be booting)");
  }
}

// ─── queryBot ─────────────────────────────────────────────────────────────
export async function queryBot({ query, userId }) {
  try {
    console.log("🧠 Generating embedding in browser...");
    const embedding = await getEmbedding(query);
    console.log(`✅ Embedding ready: ${embedding.length} dims`);

    const res = await api.post("/api/v1/search-by-vector", {
      embedding,
      top_k: 5,
      user_id: userId || "web-client",
    });

    return res.data;
  } catch (error) {
    // Friendly message if still cold-starting
    const isTimeout = error.message?.includes("timeout") || error.code === "ECONNABORTED";
    console.error("❌ Query failed:", error.message);
    return {
      success: false,
      answer: null,
      fallback_message: isTimeout
        ? "The server is waking up — please wait a moment and try again."
        : "Something went wrong. Please try again.",
      confidence_level: "low",
      requires_human_handoff: true,
      related_questions: [],
      youtube_links: [],
    };
  }
}

// ─── getFrequentQuestions ─────────────────────────────────────────────────
export async function getFrequentQuestions(limit = 10) {
  try {
    const res = await api.get("/api/v1/frequent-questions", { params: { limit } });
    return res.data;
  } catch (error) {
    console.warn("⚠️ Could not fetch frequent questions:", error.message);
    return { success: false, questions: [], count: 0 };
  }
}

// ─── getSuggestions ───────────────────────────────────────────────────────
export async function getSuggestions(query = "", limit = 5) {
  try {
    const res = await api.get("/api/v1/suggestions", { params: { query, limit } });
    return res.data;
  } catch (error) {
    console.warn("⚠️ Could not fetch suggestions:", error.message);
    return { suggestions: [], count: 0 };
  }
}

// ─── checkHealth ──────────────────────────────────────────────────────────
export async function checkHealth() {
  try {
    const res = await api.get("/health");
    return res.data;
  } catch (error) {
    console.error("❌ Health check failed:", error.message);
    throw error;
  }
}
