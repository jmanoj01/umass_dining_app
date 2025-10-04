'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { AlertCircle } from 'lucide-react';
import { LoadingCard } from '@/components/LoadingStates';
import { recommendationsApi } from '@/utils/api';
import type { Recommendation } from '@/types/api';

interface DiningHall {
  id: string;
  name: string;
}

interface MealPeriod {
  id: string;
  label: string;
  emoji: string;
}

const DINING_HALLS: DiningHall[] = [
  { id: 'worcester', name: 'Worcester' },
  { id: 'franklin', name: 'Franklin' },
  { id: 'berkshire', name: 'Berkshire' },
  { id: 'hampshire', name: 'Hampshire' },
];

const MEAL_PERIODS: MealPeriod[] = [
  { id: 'breakfast', label: 'Breakfast', emoji: '🍳' },
  { id: 'lunch', label: 'Lunch', emoji: '🥪' },
  { id: 'dinner', label: 'Dinner', emoji: '🍽️' },
  { id: 'late_night', label: 'Late Night', emoji: '🌙' },
];

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [selectedDiningHall, setSelectedDiningHall] = useState<string>('worcester');
  const [selectedMeal, setSelectedMeal] = useState<string>('lunch');

  const fetchRecommendations = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await recommendationsApi.getRecommendations({
        dining_hall: selectedDiningHall,
        meal_period: selectedMeal,
      });
      setRecommendations(response.recommendations || []);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch recommendations';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [selectedDiningHall, selectedMeal]);

  useEffect(() => {
    fetchRecommendations();
  }, [fetchRecommendations]);

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">UMass Dining Recommendations</h1>
          <nav className="space-x-4">
            <Link href="/explore" className="text-blue-600 hover:text-blue-700">
              Explore
            </Link>
            <Link href="/profile" className="text-blue-600 hover:text-blue-700">
              Profile
            </Link>
          </nav>
        </div>
        
        {/* Dining Hall Selection */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Select Dining Hall</h2>
          <div className="flex flex-wrap gap-4">
            {DINING_HALLS.map((hall) => (
              <button
                key={hall.id}
                onClick={() => setSelectedDiningHall(hall.id)}
                className={`px-4 py-2 rounded-lg ${
                  selectedDiningHall === hall.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 hover:bg-gray-200'
                }`}
              >
                {hall.name}
              </button>
            ))}
          </div>
        </div>

        {/* Meal Period Selection */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Select Meal</h2>
          <div className="flex flex-wrap gap-4">
            {MEAL_PERIODS.map((meal) => (
              <button
                key={meal.id}
                onClick={() => setSelectedMeal(meal.id)}
                className={`px-4 py-2 rounded-lg ${
                  selectedMeal === meal.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 hover:bg-gray-200'
                }`}
              >
                {meal.emoji} {meal.label}
              </button>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div>
          <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
          
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md mb-4">
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="w-5 h-5" />
                <p>{error}</p>
              </div>
            </div>
          )}

          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3].map((n) => (
                <LoadingCard key={n} />
              ))}
            </div>
          ) : recommendations.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map((item) => (
                <div
                  key={item.item_id}
                  className="p-4 border rounded-lg shadow-sm hover:shadow-md transition-shadow bg-white"
                >
                  <h3 className="font-semibold mb-2">{item.item_name}</h3>
                  <p className="text-sm text-gray-600">
                    Score: {(item.score * 100).toFixed(1)}%
                  </p>
                  {item.nutrition && (
                    <div className="mt-2 text-xs text-gray-500">
                      {item.nutrition.calories && (
                        <span className="mr-3">🔥 {item.nutrition.calories} cal</span>
                      )}
                      {item.nutrition.protein && (
                        <span className="mr-3">🥩 {item.nutrition.protein}g</span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">No recommendations available.</p>
          )}
        </div>
      </main>
    </div>
  );
}