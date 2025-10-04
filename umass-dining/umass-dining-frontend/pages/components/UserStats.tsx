'use client';

import { User, Star, Clock, Heart, Filter } from 'lucide-react';

interface UserStatsProps {
  userId: string;
  stats: {
    total_ratings: number;
    average_rating: number;
    total_meals_tracked: number;
    dietary_restrictions: string[];
    favorite_items?: Array<{ item_name: string; rating: number }>;
    disliked_items?: Array<{ item_name: string; rating: number }>;
  };
}

export default function UserStats({ userId, stats }: UserStatsProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center space-x-3 mb-6">
        <User className="h-6 w-6 text-red-600" />
        <h2 className="text-xl font-semibold text-gray-900">User Profile</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Ratings */}
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-blue-100 rounded-lg mx-auto mb-2">
            <Star className="h-6 w-6 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">{stats.total_ratings}</div>
          <div className="text-sm text-gray-500">Items Rated</div>
        </div>

        {/* Average Rating */}
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-yellow-100 rounded-lg mx-auto mb-2">
            <Star className="h-6 w-6 text-yellow-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {stats.average_rating ? stats.average_rating.toFixed(1) : 'N/A'}
          </div>
          <div className="text-sm text-gray-500">Average Rating</div>
        </div>

        {/* Meals Tracked */}
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-green-100 rounded-lg mx-auto mb-2">
            <Clock className="h-6 w-6 text-green-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">{stats.total_meals_tracked}</div>
          <div className="text-sm text-gray-500">Meals Tracked</div>
        </div>

        {/* Dietary Restrictions */}
        <div className="text-center">
          <div className="flex items-center justify-center w-12 h-12 bg-purple-100 rounded-lg mx-auto mb-2">
            <Filter className="h-6 w-6 text-purple-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">{stats.dietary_restrictions.length}</div>
          <div className="text-sm text-gray-500">Dietary Restrictions</div>
        </div>
      </div>

      {/* Dietary Restrictions */}
      {stats.dietary_restrictions.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-3">Dietary Preferences</h3>
          <div className="flex flex-wrap gap-2">
            {stats.dietary_restrictions.map((restriction, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-purple-100 text-purple-800 text-sm rounded-full"
              >
                {restriction}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Favorite Items */}
      {stats.favorite_items && stats.favorite_items.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-3">Favorite Items</h3>
          <div className="space-y-2">
            {stats.favorite_items.map((item, index) => (
              <div key={index} className="flex items-center justify-between bg-green-50 p-3 rounded-lg">
                <span className="text-sm font-medium text-gray-900">{item.item_name}</span>
                <div className="flex items-center space-x-1">
                  <Star className="h-4 w-4 text-yellow-400 fill-current" />
                  <span className="text-sm text-gray-600">{item.rating}/5</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disliked Items */}
      {stats.disliked_items && stats.disliked_items.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-3">Disliked Items</h3>
          <div className="space-y-2">
            {stats.disliked_items.map((item, index) => (
              <div key={index} className="flex items-center justify-between bg-red-50 p-3 rounded-lg">
                <span className="text-sm font-medium text-gray-900">{item.item_name}</span>
                <div className="flex items-center space-x-1">
                  <Star className="h-4 w-4 text-gray-400" />
                  <span className="text-sm text-gray-600">{item.rating}/5</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
