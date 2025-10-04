import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { AlertCircle } from 'lucide-react';
import { LoadingCard } from './components/LoadingStates';
import { recommendationsApi } from './utils/api';
import type { Recommendation } from './types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

// Error boundary component
class ErrorBoundary extends React.Component<{ children: React.ReactNode }> {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center gap-2 text-red-600">
            <AlertCircle className="w-5 h-5" />
            <h2>Something went wrong!</h2>
          </div>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-2 text-sm text-red-600 hover:text-red-700"
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

interface DiningHall {
  id: string;
  name: string;
}

interface MealPeriod {
  id: string;
  label: string;
  emoji: string;
}

// Main page component
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
  { id: 'late night', label: 'Late Night', emoji: '🌙' },
];

// Main page component
export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [selectedDiningHall, setSelectedDiningHall] = useState<string>('worcester');
  const [selectedMeal, setSelectedMeal] = useState<string>('lunch');

  useEffect(() => {
    fetchRecommendations();
  }, [selectedDiningHall, selectedMeal]);

  const fetchRecommendations = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await recommendationsApi.getRecommendations({
        dining_hall: selectedDiningHall,
        meal_period: selectedMeal,
      });
      setRecommendations(response.recommendations || []);
    } catch (error: any) {
      setError(error.message || 'Failed to fetch recommendations');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ErrorBoundary>
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">UMass Dining Recommendations</h1>
        
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
                  className="p-4 border rounded-lg shadow-sm hover:shadow-md transition-shadow"
                >
                  <h3 className="font-semibold mb-2">{item.item_name}</h3>
                  <p className="text-sm text-gray-600">
                    Score: {(item.score * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">No recommendations available.</p>
          )}
        </div>
      </main>
    </ErrorBoundary>
  );
}
const DIETARY = [
  { id: 'vegan', label: 'Vegan', color: 'green' },
  { id: 'vegetarian', label: 'Vegetarian', color: 'green' },
  { id: 'gluten-free', label: 'Gluten-Free', color: 'yellow' },
];

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
  station?: string;
  calories?: number;
  protein?: number;
  allergens?: string;
  is_vegan?: boolean;
  is_vegetarian?: boolean;
  explanations?: string[];
};

export default function Home() {
  const [userId, setUserId] = useState('justin');
  const [diningHall, setDiningHall] = useState(DINING_HALLS[0].id);
  const [mealPeriod, setMealPeriod] = useState(MEAL_PERIODS[0].id);
  const [dietary, setDietary] = useState<string[]>([]);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dark, setDark] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  const handleDietaryToggle = (id: string) => {
    setDietary((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  const getRecommendations = async () => {
    setLoading(true);
    setError('');
    setRecs([]);
    try {
      const res = await axios.get(`${API_URL}/recommendations/${userId}`, {
        params: {
          dining_hall: diningHall,
          meal_period: mealPeriod,
          top_k: 3,
        },
      });
      setRecs(res.data.recommendations || []);
    } catch (e: unknown) {
      if (axios.isAxiosError(e)) {
        setError(e.response?.data?.detail || 'Failed to fetch recommendations.');
      } else {
        setError('Failed to fetch recommendations.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-maroon transition">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-4 bg-maroon text-white shadow">
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold">UMass Dining</span>
          <span className="ml-2 text-lg font-semibold hidden sm:inline" aria-label="food">🍽️</span>
        </div>
        <nav className="flex items-center gap-2">
          <Link href="/explore" className="underline text-yellow hover:text-orange focus:outline-none focus:ring-2 focus:ring-orange px-2 py-1 rounded transition">Explore</Link>
          <Link href="/profile" className="underline text-yellow hover:text-orange focus:outline-none focus:ring-2 focus:ring-orange px-2 py-1 rounded transition">Profile</Link>
          <button
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="ml-4 px-3 py-1 rounded bg-white text-maroon font-bold shadow hover:bg-orange focus:outline-none focus:ring-2 focus:ring-orange transition"
            onClick={() => setDark((d) => !d)}
          >
            {dark ? '☀️ Light' : '🌙 Dark'}
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="max-w-2xl mx-auto p-4">
        {/* Greeting */}
        <h1 className="text-2xl sm:text-3xl font-bold mb-2 mt-4 text-maroon dark:text-yellow">
          Hi {userId.charAt(0).toUpperCase() + userId.slice(1)}! Here’s what’s hot at the dining commons today 🍽️
        </h1>
        <div className="mb-6 text-gray-700 dark:text-yellow">
          Get personalized food recommendations for your next meal at UMass Amherst.
        </div>

        {/* Filters */}
        <form className="flex flex-col sm:flex-row gap-4 mb-4" onSubmit={e => { e.preventDefault(); getRecommendations(); }}>
          <input
            className="flex-1"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Your Name or NetID"
            aria-label="User ID"
          />
          <select
            className="flex-1"
            value={diningHall}
            onChange={(e) => setDiningHall(e.target.value)}
            aria-label="Dining Hall"
          >
            {DINING_HALLS.map((hall) => (
              <option key={hall.id} value={hall.id}>{hall.name}</option>
            ))}
          </select>
          <select
            className="flex-1"
            value={mealPeriod}
            onChange={(e) => setMealPeriod(e.target.value)}
            aria-label="Meal Period"
          >
            {MEAL_PERIODS.map((m) => (
              <option key={m.id} value={m.id}>{m.emoji} {m.label}</option>
            ))}
          </select>
        </form>
        {/* Dietary Chips */}
        <div className="flex gap-2 mb-4 flex-wrap" role="group" aria-label="Dietary Preferences">
          {DIETARY.map((d) => (
            <button
              key={d.id}
              className={`px-3 py-1 rounded-full border-2 font-semibold text-sm transition focus:outline-none focus:ring-2 focus:ring-orange ${dietary.includes(d.id)
                ? `bg-${d.color}-100 border-${d.color}-500 text-${d.color}-800`
                : 'bg-white border-gray-300 text-gray-700 dark:bg-gray-800 dark:text-white'}`}
              onClick={() => handleDietaryToggle(d.id)}
              type="button"
              aria-pressed={dietary.includes(d.id)}
            >
              {d.label}
            </button>
          ))}
        </div>
        {/* Get Recommendations Button */}
        <button
          className="w-full btn text-lg mb-6"
          onClick={getRecommendations}
          disabled={loading}
          aria-busy={loading}
        >
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
        {/* Error */}
        {error && <div className="text-red-600 mb-4" role="alert">{error}</div>}
        {/* Recommendations */}
        <div className="grid gap-4">
          {recs.map((rec) => (
            <div key={rec.item_id} className="card flex flex-col sm:flex-row items-center gap-4 border-l-8 border-maroon shadow-lg hover:shadow-xl transition focus-within:ring-2 focus-within:ring-orange" tabIndex={0} aria-label={`Recommendation: ${rec.item_name}`}> 
              <div className="flex-shrink-0 text-4xl" aria-hidden="true">
                {rec.is_vegan ? '🥗' : rec.is_vegetarian ? '🥦' : '🍗'}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xl font-bold text-maroon dark:text-yellow">{rec.item_name}</span>
                  {rec.is_vegan && <span className="ml-1 px-2 py-0.5 rounded bg-green-100 text-green-800 text-xs">Vegan</span>}
                  {rec.is_vegetarian && !rec.is_vegan && <span className="ml-1 px-2 py-0.5 rounded bg-green-100 text-green-800 text-xs">Vegetarian</span>}
                  {rec.score > 0.8 && <span className="ml-2 px-2 py-0.5 rounded bg-orange text-white text-xs font-semibold">You might like this</span>}
                </div>
                <div className="text-gray-700 dark:text-yellow text-sm mb-1">
                  {rec.station && <span className="mr-2">{rec.station}</span>}
                  {rec.calories && <span className="mr-2">{rec.calories} cal</span>}
                  {rec.protein && <span className="mr-2">{rec.protein}g protein</span>}
                </div>
                <div className="flex gap-2 flex-wrap mb-1">
                  {rec.allergens && rec.allergens.split(',').map((a: string) => (
                    <span key={a} className="px-2 py-0.5 rounded bg-yellow text-maroon text-xs">{a.trim()}</span>
                  ))}
                </div>
                <div className="text-xs text-gray-500 dark:text-yellow-200 mb-1">
                  {rec.explanations && rec.explanations.join(' | ')}
                </div>
                {/* Rating and favorite (placeholder, not functional yet) */}
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-yellow text-lg" aria-label="4 out of 5 stars">★ ★ ★ ★ ☆</span>
                  <button className="ml-2 text-maroon dark:text-yellow text-xl hover:scale-110 transition" aria-label="Add to favorites">♡</button>
                </div>
              </div>
            </div>
          ))}
        </div>
        {/* Trending/Popular (placeholder) */}
        <div className="mt-10">
          <h2 className="text-lg font-bold mb-2 text-maroon dark:text-yellow">Trending Today</h2>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {["Pizza", "Salad", "Chicken Tenders", "Vegan Bowl", "Pasta"].map((item) => (
              <div key={item} className="min-w-[140px] card flex flex-col items-center justify-center bg-orange/10 border border-orange/30" tabIndex={0} aria-label={`Trending: ${item}`}>
                <span className="text-3xl mb-1" aria-hidden="true">{item === "Pizza" ? "🍕" : item === "Salad" || item === "Vegan Bowl" ? "🥗" : item === "Chicken Tenders" ? "🍗" : "🍝"}</span>
                <span className="font-semibold text-maroon dark:text-yellow text-sm">{item}</span>
                <span className="text-xs text-gray-500">Popular</span>
              </div>
            ))}
          </div>
        </div>
      </main>
      {/* Footer */}
      <footer className="text-center text-gray-500 py-6 mt-10 text-xs">
        UMass Dining Recommender &copy; {new Date().getFullYear()} | Made with <span className="text-orange">♥</span> for UMass Amherst
      </footer>
    </div>
  );
}
