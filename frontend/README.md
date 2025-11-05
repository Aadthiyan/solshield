# Smart Contract Vulnerability Detection - Frontend

A modern React-based frontend application for smart contract vulnerability detection using AI models including CodeBERT and Graph Neural Networks.

## 🚀 Features

- **File Upload Interface**: Drag-and-drop file upload with progress indicators
- **Real-time Analysis**: Live progress tracking during contract analysis
- **Vulnerability Reports**: Detailed vulnerability reports with explanations and recommendations
- **Interactive Dashboard**: Comprehensive dashboard with charts and statistics
- **Responsive Design**: Mobile-first design that works on all devices
- **Real-time Updates**: Live updates during analysis and processing
- **Accessibility**: WCAG compliant with keyboard navigation support

## 🛠️ Technology Stack

- **Framework**: Next.js 14 with React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom components
- **State Management**: Zustand for global state
- **API Client**: Axios with React Query for caching
- **Animations**: Framer Motion for smooth animations
- **Charts**: Recharts for data visualization
- **Testing**: Jest, React Testing Library, Cypress
- **File Handling**: React Dropzone for file uploads

## 📦 Installation

### Prerequisites

- Node.js 18+ 
- npm 8+ or yarn
- Backend API running on http://localhost:8000

### Install Dependencies

```bash
cd frontend
npm install
```

### Environment Setup

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=Smart Contract Vulnerability Detection
NEXT_PUBLIC_APP_VERSION=1.0.0
```

## 🚀 Development

### Start Development Server

```bash
npm run dev
```

The application will be available at http://localhost:3000

### Build for Production

```bash
npm run build
npm start
```

### Run Tests

```bash
# Unit tests
npm test

# Unit tests with coverage
npm run test:coverage

# E2E tests
npm run e2e

# Open Cypress
npm run e2e:open
```

## 📁 Project Structure

```
frontend/
├── components/           # React components
│   ├── FileUpload.tsx   # File upload component
│   ├── ProgressIndicator.tsx
│   ├── VulnerabilityCard.tsx
│   ├── OptimizationCard.tsx
│   ├── ReportDisplay.tsx
│   └── Dashboard.tsx
├── pages/               # Next.js pages
│   ├── _app.tsx        # App wrapper
│   └── index.tsx       # Home page
├── styles/              # Global styles
│   └── globals.css
├── utils/               # Utility functions
│   ├── api.ts          # API client
│   └── store.ts        # State management
├── types/               # TypeScript types
│   └── index.ts
├── tests/               # Unit tests
│   ├── components/
│   └── pages/
├── e2e/                 # E2E tests
│   └── cypress/
└── public/              # Static assets
```

## 🎨 Components

### FileUpload
- Drag-and-drop file upload
- File validation and error handling
- Progress indicators
- Support for .sol and .txt files

### ProgressIndicator
- Real-time progress tracking
- Step-by-step analysis progress
- Estimated time remaining
- Completion notifications

### VulnerabilityCard
- Detailed vulnerability information
- Expandable sections
- Severity indicators
- Copy functionality for code snippets

### OptimizationCard
- Gas optimization suggestions
- Implementation details
- Priority levels
- Potential savings estimates

### ReportDisplay
- Complete vulnerability report
- Interactive charts and graphs
- Export functionality
- Responsive layout

### Dashboard
- Overview statistics
- Vulnerability type distribution
- Risk score trends
- Recent analyses

## 🔧 Configuration

### Tailwind CSS
Custom configuration with:
- Extended color palette
- Custom animations
- Responsive breakpoints
- Dark mode support

### Next.js
- App directory structure
- API rewrites for backend
- Security headers
- Image optimization

### TypeScript
- Strict type checking
- Path mapping
- Custom type definitions
- Interface segregation

## 🧪 Testing

### Unit Tests
- Component testing with React Testing Library
- Mock implementations for external dependencies
- Coverage reporting
- Snapshot testing

### E2E Tests
- Cypress for end-to-end testing
- API mocking
- User interaction testing
- Cross-browser compatibility

### Test Commands
```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run E2E tests
npm run e2e

# Run tests with coverage
npm run test:coverage
```

## 📱 Responsive Design

### Breakpoints
- Mobile: 375px - 767px
- Tablet: 768px - 1023px
- Desktop: 1024px+

### Features
- Mobile-first approach
- Touch-friendly interactions
- Responsive charts and tables
- Adaptive navigation

## ♿ Accessibility

### WCAG Compliance
- Keyboard navigation
- Screen reader support
- Color contrast ratios
- Focus management

### Features
- ARIA labels and roles
- Semantic HTML structure
- Skip links
- High contrast mode

## 🎯 Performance

### Optimization
- Code splitting
- Lazy loading
- Image optimization
- Bundle analysis

### Metrics
- First Contentful Paint < 1.5s
- Largest Contentful Paint < 2.5s
- Cumulative Layout Shift < 0.1
- First Input Delay < 100ms

## 🔒 Security

### Features
- Content Security Policy
- XSS protection
- CSRF protection
- Secure headers

### Best Practices
- Input validation
- Output encoding
- Secure file handling
- API security

## 🚀 Deployment

### Production Build
```bash
npm run build
npm start
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables
```env
NEXT_PUBLIC_API_URL=https://api.example.com
NEXT_PUBLIC_APP_NAME=Smart Contract Vulnerability Detection
NEXT_PUBLIC_APP_VERSION=1.0.0
```

## 📊 Monitoring

### Analytics
- User interactions
- Performance metrics
- Error tracking
- Usage statistics

### Logging
- Console logging
- Error boundaries
- Performance monitoring
- User feedback

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install dependencies
4. Make changes
5. Run tests
6. Submit pull request

### Code Standards
- ESLint configuration
- Prettier formatting
- TypeScript strict mode
- Component documentation

## 📚 API Integration

### Endpoints
- `POST /api/v1/analyze` - Analyze contract
- `GET /api/v1/report/{id}` - Get report
- `GET /api/v1/health` - Health check
- `GET /api/v1/status` - System status
- `GET /api/v1/metrics` - Performance metrics

### Error Handling
- Network errors
- API errors
- Validation errors
- Timeout handling

## 🔄 State Management

### Zustand Store
- Global state management
- Persistence
- DevTools integration
- Type safety

### State Structure
```typescript
interface AppState {
  reports: VulnerabilityReport[];
  currentReport: VulnerabilityReport | null;
  isAnalyzing: boolean;
  analysisProgress: AnalysisState;
  dashboardStats: DashboardStats | null;
  filters: FilterOptions;
  sortOptions: SortOptions;
  error: string | null;
  loading: boolean;
}
```

## 🎨 Styling

### Tailwind CSS
- Utility-first approach
- Custom components
- Responsive design
- Dark mode support

### Custom Components
- Button variants
- Card layouts
- Form elements
- Navigation components

## 📱 Mobile Support

### Features
- Touch gestures
- Mobile navigation
- Responsive charts
- Optimized performance

### Testing
- Device testing
- Touch interactions
- Performance on mobile
- Cross-platform compatibility

## 🔧 Development Tools

### VS Code Extensions
- ES7+ React/Redux/React-Native snippets
- Tailwind CSS IntelliSense
- TypeScript Importer
- Prettier - Code formatter

### Debugging
- React DevTools
- Redux DevTools
- Network monitoring
- Performance profiling

## 📈 Performance Monitoring

### Metrics
- Core Web Vitals
- Bundle size analysis
- Runtime performance
- Memory usage

### Optimization
- Code splitting
- Lazy loading
- Image optimization
- Caching strategies

## 🚀 Future Enhancements

### Planned Features
- Dark mode toggle
- Advanced filtering
- Export functionality
- Batch processing
- Real-time collaboration

### Technical Improvements
- PWA support
- Offline functionality
- Advanced caching
- Micro-frontend architecture

## 📞 Support

### Documentation
- Component documentation
- API documentation
- User guides
- Troubleshooting

### Community
- GitHub issues
- Discussion forums
- Contributing guidelines
- Code of conduct

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- React team for the amazing framework
- Next.js team for the full-stack solution
- Tailwind CSS for the utility-first approach
- Framer Motion for smooth animations
- Recharts for beautiful charts
- Cypress for reliable E2E testing
