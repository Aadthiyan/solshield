import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';

// Configure testing library
configure({
  testIdAttribute: 'data-testid',
});

// Mock Next.js router
jest.mock('next/router', () => ({
  useRouter: () => ({
    route: '/',
    pathname: '/',
    query: {},
    asPath: '/',
    push: jest.fn(),
    pop: jest.fn(),
    reload: jest.fn(),
    back: jest.fn(),
    prefetch: jest.fn(),
    beforePopState: jest.fn(),
    events: {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
    },
    isFallback: false,
  }),
}));

// Mock Next.js Image component
jest.mock('next/image', () => ({
  __esModule: true,
  default: (props: any) => {
    // eslint-disable-next-line @next/next/no-img-element
    return <img {...props} />;
  },
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: 'div',
    span: 'span',
    button: 'button',
    h1: 'h1',
    h2: 'h2',
    h3: 'h3',
    p: 'p',
    section: 'section',
    article: 'article',
    header: 'header',
    footer: 'footer',
    nav: 'nav',
    main: 'main',
    aside: 'aside',
    ul: 'ul',
    ol: 'ol',
    li: 'li',
    a: 'a',
    img: 'img',
    input: 'input',
    textarea: 'textarea',
    select: 'select',
    option: 'option',
    form: 'form',
    label: 'label',
    fieldset: 'fieldset',
    legend: 'legend',
    table: 'table',
    thead: 'thead',
    tbody: 'tbody',
    tfoot: 'tfoot',
    tr: 'tr',
    th: 'th',
    td: 'td',
    caption: 'caption',
    figure: 'figure',
    figcaption: 'figcaption',
    blockquote: 'blockquote',
    cite: 'cite',
    code: 'code',
    pre: 'pre',
    kbd: 'kbd',
    samp: 'samp',
    var: 'var',
    mark: 'mark',
    del: 'del',
    ins: 'ins',
    sub: 'sub',
    sup: 'sup',
    small: 'small',
    strong: 'strong',
    em: 'em',
    b: 'b',
    i: 'i',
    u: 'u',
    s: 's',
    q: 'q',
    abbr: 'abbr',
    acronym: 'acronym',
    address: 'address',
    big: 'big',
    tt: 'tt',
    br: 'br',
    hr: 'hr',
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
  useAnimation: () => ({
    start: jest.fn(),
    stop: jest.fn(),
    set: jest.fn(),
  }),
  useMotionValue: (value: any) => ({ get: () => value, set: jest.fn() }),
  useTransform: (value: any, input: any, output: any) => value,
}));

// Mock react-hot-toast
jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    loading: jest.fn(),
    dismiss: jest.fn(),
  },
  Toaster: () => null,
}));

// Mock react-dropzone
jest.mock('react-dropzone', () => ({
  useDropzone: () => ({
    getRootProps: () => ({}),
    getInputProps: () => ({}),
    isDragActive: false,
  }),
}));

// Mock recharts
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => children,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
}));

// Mock lucide-react
jest.mock('lucide-react', () => ({
  Upload: () => <div data-testid="upload-icon" />,
  Shield: () => <div data-testid="shield-icon" />,
  AlertTriangle: () => <div data-testid="alert-triangle-icon" />,
  CheckCircle: () => <div data-testid="check-circle-icon" />,
  Clock: () => <div data-testid="clock-icon" />,
  RefreshCw: () => <div data-testid="refresh-cw-icon" />,
  Settings: () => <div data-testid="settings-icon" />,
  BarChart3: () => <div data-testid="bar-chart-3-icon" />,
  ChevronDown: () => <div data-testid="chevron-down-icon" />,
  ChevronRight: () => <div data-testid="chevron-right-icon" />,
  ChevronUp: () => <div data-testid="chevron-up-icon" />,
  ExternalLink: () => <div data-testid="external-link-icon" />,
  Copy: () => <div data-testid="copy-icon" />,
  X: () => <div data-testid="x-icon" />,
  Lightbulb: () => <div data-testid="lightbulb-icon" />,
  TrendingUp: () => <div data-testid="trending-up-icon" />,
  Star: () => <div data-testid="star-icon" />,
  FileText: () => <div data-testid="file-text-icon" />,
  Download: () => <div data-testid="download-icon" />,
  Share2: () => <div data-testid="share-2-icon" />,
  Filter: () => <div data-testid="filter-icon" />,
  Eye: () => <div data-testid="eye-icon" />,
  Trash2: () => <div data-testid="trash-2-icon" />,
}));

// Mock API client
jest.mock('@/utils/api', () => ({
  __esModule: true,
  default: {
    analyzeContract: jest.fn(),
    getReport: jest.fn(),
    healthCheck: jest.fn(),
    getSystemStatus: jest.fn(),
    getMetrics: jest.fn(),
    batchAnalyze: jest.fn(),
    getBatchReports: jest.fn(),
    listReports: jest.fn(),
    deleteReport: jest.fn(),
    checkConnection: jest.fn(),
  },
  analyzeContract: jest.fn(),
  getReport: jest.fn(),
  healthCheck: jest.fn(),
  getSystemStatus: jest.fn(),
  getMetrics: jest.fn(),
  batchAnalyze: jest.fn(),
  getBatchReports: jest.fn(),
  listReports: jest.fn(),
  deleteReport: jest.fn(),
  checkConnection: jest.fn(),
}));

// Mock store
jest.mock('@/utils/store', () => ({
  useAppStore: jest.fn(() => ({
    reports: [],
    currentReport: null,
    isAnalyzing: false,
    analysisProgress: {
      isAnalyzing: false,
      progress: 0,
      currentStep: '',
      estimatedTime: 0,
    },
    dashboardStats: null,
    filters: {
      vulnerabilityTypes: [],
      severityLevels: [],
      dateRange: {
        start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        end: new Date(),
      },
      riskScoreRange: {
        min: 0,
        max: 10,
      },
    },
    sortOptions: {
      field: 'timestamp',
      direction: 'desc',
    },
    error: null,
    loading: false,
    setReports: jest.fn(),
    addReport: jest.fn(),
    setCurrentReport: jest.fn(),
    setAnalyzing: jest.fn(),
    setAnalysisProgress: jest.fn(),
    setDashboardStats: jest.fn(),
    setFilters: jest.fn(),
    setSortOptions: jest.fn(),
    setError: jest.fn(),
    setLoading: jest.fn(),
    clearError: jest.fn(),
    reset: jest.fn(),
  })),
  useReports: jest.fn(() => []),
  useCurrentReport: jest.fn(() => null),
  useIsAnalyzing: jest.fn(() => false),
  useAnalysisProgress: jest.fn(() => ({
    isAnalyzing: false,
    progress: 0,
    currentStep: '',
    estimatedTime: 0,
  })),
  useDashboardStats: jest.fn(() => null),
  useFilters: jest.fn(() => ({
    vulnerabilityTypes: [],
    severityLevels: [],
    dateRange: {
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      end: new Date(),
    },
    riskScoreRange: {
      min: 0,
      max: 10,
    },
  })),
  useSortOptions: jest.fn(() => ({
    field: 'timestamp',
    direction: 'desc',
  })),
  useError: jest.fn(() => null),
  useLoading: jest.fn(() => false),
  useFilteredReports: jest.fn(() => []),
  useVulnerabilityStats: jest.fn(() => []),
  useSeverityStats: jest.fn(() => []),
  useRecentReports: jest.fn(() => []),
  useHighRiskReports: jest.fn(() => []),
  useVulnerableReports: jest.fn(() => []),
}));

// Global test utilities
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(),
    readText: jest.fn(),
  },
});

// Mock fetch
global.fetch = jest.fn();

// Mock console methods to reduce noise in tests
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is no longer supported')
    ) {
      return;
    }
    originalConsoleError.call(console, ...args);
  };
  
  console.warn = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is no longer supported')
    ) {
      return;
    }
    originalConsoleWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalConsoleError;
  console.warn = originalConsoleWarn;
});
