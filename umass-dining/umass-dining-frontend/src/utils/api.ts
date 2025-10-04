import axios from 'axios';
import type { RecommendationResponse, ApiError } from '../types/api';

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
    const customError: ApiError = {
      message: error.response?.data?.detail || 'An unexpected error occurred',
      status: error.response?.status || 500,
      data: error.response?.data || null,
    };

    return Promise.reject(customError);
  }
);

interface RecommendationParams {
  dining_hall: string;
  meal_period: string;
}

export const recommendationsApi = {
  getRecommendations: async (params: RecommendationParams): Promise<RecommendationResponse> => {
    try {
      const response = await api.get('/recommendations', { params });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  updatePreferences: async (data: Record<string, unknown>) => {
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