import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const customError = {
      message: error.response?.data?.detail || 'An unexpected error occurred',
      status: error.response?.status || 500,
      data: error.response?.data || null,
    };

    // Log errors in development
    if (process.env.NEXT_PUBLIC_NODE_ENV === 'development') {
      console.error('API Error:', customError);
    }

    return Promise.reject(customError);
  }
);

// API endpoints
export const recommendationsApi = {
  getRecommendations: async (params: any) => {
    try {
      const response = await api.get('/recommendations', { params });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  updatePreferences: async (data: any) => {
    try {
      const response = await api.post('/preferences', data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getDiningHalls: async () => {
    try {
      const response = await api.get('/dining-halls');
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};

export default api;