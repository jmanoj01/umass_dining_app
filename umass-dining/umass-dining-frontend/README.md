# UMass Dining Frontend

A modern Next.js frontend for the UMass Dining Recommender System, providing an intuitive interface for personalized dining recommendations.

## Features

- 🍽️ **Personalized Recommendations**: Get AI-powered food recommendations based on your preferences
- ⭐ **Rating System**: Rate items to improve future recommendations
- 🏢 **Dining Hall Filtering**: Filter recommendations by specific dining halls
- 🕐 **Meal Period Selection**: Get recommendations for breakfast, lunch, or dinner
- 🔍 **Search Functionality**: Search through recommendations
- 📊 **User Statistics**: View your rating history and preferences
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon library
- **Axios** - HTTP client for API calls

## Getting Started

### Prerequisites

- Node.js 18+ (recommended: 20+)
- npm or yarn
- UMass Dining Recommender API running on `http://localhost:8000`

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Environment Variables

Create a `.env.local` file in the root directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Main page component
│   ├── layout.tsx            # Root layout
│   └── globals.css           # Global styles
├── components/
│   ├── RecommendationCard.tsx # Individual recommendation card
│   └── UserStats.tsx         # User statistics component
└── types/
    └── index.ts              # TypeScript type definitions
```

## API Integration

The frontend communicates with the UMass Dining Recommender API through the following endpoints:

- `GET /api/v1/recommendations/{user_id}` - Get personalized recommendations
- `POST /api/v1/rate` - Rate a food item
- `GET /api/v1/dining-halls` - Get list of dining halls
- `GET /api/v1/search` - Search for food items
- `GET /api/v1/user/{user_id}/stats` - Get user statistics

## Components

### RecommendationCard

Displays individual food recommendations with:
- Item name and basic information
- Nutritional information (calories, protein)
- Dietary tags (vegan, vegetarian)
- Rating interface
- Detailed explanations
- Expandable details section

### UserStats

Shows user profile information including:
- Total ratings and average rating
- Meals tracked
- Dietary restrictions
- Favorite and disliked items

## Features in Detail

### Recommendation System

1. **Algorithm Selection**: Choose between different recommendation algorithms
2. **Filtering**: Filter by dining hall and meal period
3. **Search**: Search through recommendations
4. **Rating**: Rate items to improve future recommendations

### User Experience

- **Responsive Design**: Optimized for all screen sizes
- **Loading States**: Clear feedback during API calls
- **Error Handling**: User-friendly error messages
- **Interactive Elements**: Hover effects and smooth transitions

## Customization

### Styling

The app uses Tailwind CSS for styling. You can customize the appearance by:

1. Modifying the color scheme in `tailwind.config.js`
2. Updating component styles in individual files
3. Adding custom CSS in `globals.css`

### API Configuration

Update the API URL in the following places:

1. `src/app/page.tsx` - Update `API_URL` constant
2. `.env.local` - Set `NEXT_PUBLIC_API_URL`

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Code Style

The project uses:
- TypeScript for type safety
- ESLint for code linting
- Prettier for code formatting (recommended)

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Deploy automatically

### Other Platforms

The app can be deployed to any platform that supports Next.js:
- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Ensure the backend API is running on the correct port
2. **CORS Issues**: Check that the API has CORS enabled for your frontend domain
3. **Build Errors**: Make sure all dependencies are installed and TypeScript types are correct

### Getting Help

- Check the browser console for error messages
- Verify API endpoints are working with tools like Postman
- Ensure all environment variables are set correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the UMass Dining Recommender System and is intended for educational purposes.