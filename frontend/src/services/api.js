// frontend/src/api.js

import axios from "axios";

// Get API base URL from environment or default to localhost:8000
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

console.log("API Base URL:", API_BASE_URL);

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  }
});

// Add session ID to all requests
api.interceptors.request.use((config) => {
  const sessionId = getOrCreateSessionId();
  config.headers["X-Session-Id"] = sessionId;
  return config;
});

function getOrCreateSessionId() {
  const KEY = "chatbot_session_id";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem(KEY, id);
  }
  return id;
}

/**
 * Query the chatbot for an answer
 */
export async function queryBot({ query, userId }) {
  try {
    const payload = {
      query,
      user_id: userId || "web-client",
      channel: "web"
    };
    console.log("Sending query:", payload);
    const res = await api.post("/api/query", payload);
    console.log("Response:", res.data);
    return res.data;
  } catch (error) {
    console.error("Query error:", error);
    throw error;
  }
}

/**
 * Get suggested questions
 */
export async function getSuggestions() {
  try {
    const res = await api.get("/api/suggestions");
    return res.data;
  } catch (error) {
    console.error("Suggestions error:", error);
    throw error;
  }
}

export default api;
