/**
 * Frontend configuration for API endpoints.
 * 
 * In development: Uses Vite proxy (relative paths)
 * In production: Uses environment variable for backend URL
 */

// Get API base URL from environment or default to relative path
export const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// API endpoints
export const API_ENDPOINTS = {
  query: `${API_BASE_URL}/api/query`,
  queryStream: `${API_BASE_URL}/api/query/stream`,
  upload: `${API_BASE_URL}/api/upload`,
  uploadStatus: (taskId: string) => `${API_BASE_URL}/api/upload/status/${taskId}`,
  documents: `${API_BASE_URL}/api/documents`,
  document: (id: string) => `${API_BASE_URL}/api/documents/${encodeURIComponent(id)}`,
  conversations: `${API_BASE_URL}/api/conversations`,
  conversation: (id: string) => `${API_BASE_URL}/api/conversations/${id}`,
  cacheStats: `${API_BASE_URL}/api/cache/stats`,
  graphSearch: `${API_BASE_URL}/api/graph/search`,
  health: `${API_BASE_URL}/health`,
};

// Feature flags
export const FEATURES = {
  enableCache: import.meta.env.VITE_ENABLE_CACHE !== 'false',
  enableKnowledgeGraph: import.meta.env.VITE_ENABLE_KG !== 'false',
  enableConversations: import.meta.env.VITE_ENABLE_CONVERSATIONS !== 'false',
};

