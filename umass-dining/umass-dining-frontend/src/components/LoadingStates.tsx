import { Loader2 } from 'lucide-react';

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center w-full h-32">
      <Loader2 className="w-8 h-8 animate-spin text-gray-500" />
    </div>
  );
}

export function LoadingCard() {
  return (
    <div className="p-4 border rounded-lg shadow-sm animate-pulse">
      <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
      <div className="space-y-3">
        <div className="h-3 bg-gray-200 rounded"></div>
        <div className="h-3 bg-gray-200 rounded w-5/6"></div>
      </div>
    </div>
  );
}