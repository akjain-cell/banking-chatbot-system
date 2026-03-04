/**
 * FIXED API.JS - Frontend API Client
 * 
 * Changes:
 * 1. Added getFrequentQuestions() function
 * 2. Better error handling
 * 3. Session management
 */

import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

console.log("🔌 API Base URL:", API_BASE_URL);

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor - add session ID
api.interceptors.request.use((config) => {
  const sessionId = getOrCreateSessionId();
  config.headers["X-Session-Id"] = sessionId;
  console.log("📤 Sending request to:", config.baseURL + config.url);
  return config;
});

// Response interceptor - log responses
api.interceptors.response.use(
  (response) => {
    console.log("📥 Response received:", response.data);
    return response;
  },
  (error) => {
    console.error("❌ API Error:", error);
    throw error;
  }
);

// Session management
function getOrCreateSessionId() {
  const KEY = "chatbot_session_id";
  let id = window.localStorage.getItem(KEY);
  
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem(KEY, id);
    console.log("🆔 Created new session:", id);
  }
  
  return id;
}

/**
 * Main chat query function
 * @param {Object} params - { query: string, userId: string }
 * @returns {Promise} Response with answer, confidence, related questions
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
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      "Failed to query chatbot"
    );
  }
}

/**
 * Get frequent/popular questions to display on homepage
 * @param {number} limit - Number of questions to fetch (default: 10)
 * @returns {Promise} Array of frequent questions
 */
export async function getFrequentQuestions(limit = 10) {
  try {
    console.log("📋 Fetching frequent questions...");
    
    const res = await api.get("/api/v1/frequent-questions", {
      params: { limit }
    });
    
    console.log("✅ Got frequent questions:", res.data);
    
    return res.data;
  } catch (error) {
    console.warn("⚠️ Could not fetch frequent questions:", error.message);
    
    // Return empty result on error (don't break the app)
    return {
      success: false,
      questions: [],
      count: 0
    };
  }
}

/**
 * Get autocomplete suggestions based on partial query
 * @param {string} query - Partial query string
 * @param {number} limit - Number of suggestions (default: 5)
 * @returns {Promise} Array of suggested questions
 */
export async function getSuggestions(query = "", limit = 5) {
  try {
    console.log("📋 Fetching suggestions for:", query);
    
    const res = await api.get("/api/v1/suggestions", {
      params: { query, limit }
    });
    
    console.log("✅ Got suggestions:", res.data);
    
    return res.data;
  } catch (error) {
    console.warn("⚠️ Could not fetch suggestions:", error.message);
    
    return {
      suggestions: [],
      count: 0
    };
  }
}

/**
 * Check system health
 * @returns {Promise} Health status
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