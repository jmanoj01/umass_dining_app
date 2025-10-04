'use client';

import { Star, MapPin, Utensils, Heart, Info } from 'lucide-react';
import { useState } from 'react';

interface Recommendation {
  item_id: number;
  item_name: string;
  score: number;
  method: string;
  confidence: number;
  station: string;
  calories?: number;
  protein?: number;
  allergens: string;
  is_vegan: boolean;
  is_vegetarian: boolean;
  explanations?: string[];
}

interface RecommendationCardProps {
  recommendation: Recommendation;
  index: number;
  onRate: (itemId: number, itemName: string, rating: number) => void;
}

export default function RecommendationCard({ recommendation, index, onRate }: RecommendationCardProps) {
  const [showDetails, setShowDetails] = useState(false);

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'collaborative':
        return 'bg-blue-100 text-blue-800';
      case 'content_based':
        return 'bg-purple-100 text-purple-800';
      case 'hybrid':
        return 'bg-indigo-100 text-indigo-800';
      case 'popularity':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-all duration-200 bg-white">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 text-lg leading-tight">
            {recommendation.item_name}
          </h3>
          <div className="flex items-center space-x-2 mt-1">
            <span className="text-sm text-gray-500">#{index + 1}</span>
            <span className={`text-sm font-medium ${getScoreColor(recommendation.score)}`}>
              {Math.round(recommendation.score * 100)}% match
            </span>
          </div>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <Info className="h-4 w-4" />
        </button>
      </div>

      {/* Basic Info */}
      <div className="space-y-2 mb-4">
        <div className="flex items-center text-sm text-gray-600">
          <MapPin className="h-4 w-4 mr-1 flex-shrink-0" />
          <span className="truncate">{recommendation.station || 'Unknown Station'}</span>
        </div>
        
        {recommendation.calories && (
          <div className="flex items-center text-sm text-gray-600">
            <Utensils className="h-4 w-4 mr-1 flex-shrink-0" />
            <span>
              {recommendation.calories} calories
              {recommendation.protein && ` • ${recommendation.protein}g protein`}
            </span>
          </div>
        )}

        {/* Tags */}
        <div className="flex flex-wrap gap-1">
          <span className={`text-xs px-2 py-1 rounded ${getMethodColor(recommendation.method)}`}>
            {recommendation.method.replace('_', ' ').toUpperCase()}
          </span>
          <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
            {Math.round(recommendation.confidence * 100)}% confidence
          </span>
          {(recommendation.is_vegan || recommendation.is_vegetarian) && (
            <>
              {recommendation.is_vegan && (
                <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                  Vegan
                </span>
              )}
              {recommendation.is_vegetarian && (
                <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                  Vegetarian
                </span>
              )}
            </>
          )}
        </div>

        {recommendation.allergens && (
          <div className="text-xs text-gray-500">
            <span className="font-medium">Allergens:</span> {recommendation.allergens}
          </div>
        )}
      </div>

      {/* Rating Section */}
      <div className="border-t pt-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-600">Rate this item:</span>
          <Heart className="h-4 w-4 text-red-500" />
        </div>
        <div className="flex space-x-1">
          {[1, 2, 3, 4, 5].map((rating) => (
            <button
              key={rating}
              onClick={() => onRate(recommendation.item_id, recommendation.item_name, rating)}
              className="text-gray-300 hover:text-yellow-400 transition-colors duration-150 hover:scale-110"
            >
              <Star className="h-4 w-4" />
            </button>
          ))}
        </div>
      </div>

      {/* Detailed Information */}
      {showDetails && (
        <div className="mt-4 pt-3 border-t space-y-3">
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Recommendation Details</h4>
            <div className="space-y-1 text-xs text-gray-600">
              <div className="flex justify-between">
                <span>Item ID:</span>
                <span className="font-mono">{recommendation.item_id}</span>
              </div>
              <div className="flex justify-between">
                <span>Score:</span>
                <span className="font-mono">{recommendation.score.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span>Method:</span>
                <span className="capitalize">{recommendation.method.replace('_', ' ')}</span>
              </div>
            </div>
          </div>

          {/* Explanations */}
          {recommendation.explanations && recommendation.explanations.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Why Recommended</h4>
              <div className="space-y-1">
                {recommendation.explanations.map((explanation, idx) => (
                  <p key={idx} className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                    {explanation}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
