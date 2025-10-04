export interface Recommendation {
  item_id: string;
  item_name: string;
  score: number;
  confidence?: number;
  method?: string;
  dining_hall?: string;
  meal_period?: string;
  nutrition?: {
    calories?: number;
    protein?: number;
    carbs?: number;
    fat?: number;
  };
}

export interface ApiError {
  message: string;
  status: number;
  data?: Record<string, unknown>;
}

export interface RecommendationResponse {
  recommendations: Recommendation[];
  meta?: {
    total: number;
    page: number;
    per_page: number;
  };
}