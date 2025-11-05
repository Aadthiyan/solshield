import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { AppState, AppActions, VulnerabilityReport, AnalysisState, DashboardStats, FilterOptions, SortOptions } from '@/types';

interface AppStore extends AppState, AppActions {}

export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (set, get) => ({
        // State
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
            start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
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

        // Actions
        setReports: (reports: VulnerabilityReport[]) => {
          set({ reports }, false, 'setReports');
        },

        addReport: (report: VulnerabilityReport) => {
          const { reports } = get();
          const existingIndex = reports.findIndex(r => r.report_id === report.report_id);
          
          if (existingIndex >= 0) {
            // Update existing report
            const updatedReports = [...reports];
            updatedReports[existingIndex] = report;
            set({ reports: updatedReports }, false, 'addReport');
          } else {
            // Add new report
            set({ reports: [report, ...reports] }, false, 'addReport');
          }
        },

        setCurrentReport: (report: VulnerabilityReport | null) => {
          set({ currentReport: report }, false, 'setCurrentReport');
        },

        setAnalyzing: (isAnalyzing: boolean) => {
          set({ isAnalyzing }, false, 'setAnalyzing');
        },

        setAnalysisProgress: (progress: AnalysisState) => {
          set({ analysisProgress: progress }, false, 'setAnalysisProgress');
        },

        setDashboardStats: (stats: DashboardStats) => {
          set({ dashboardStats: stats }, false, 'setDashboardStats');
        },

        setFilters: (filters: FilterOptions) => {
          set({ filters }, false, 'setFilters');
        },

        setSortOptions: (options: SortOptions) => {
          set({ sortOptions: options }, false, 'setSortOptions');
        },

        setError: (error: string | null) => {
          set({ error }, false, 'setError');
        },

        setLoading: (loading: boolean) => {
          set({ loading }, false, 'setLoading');
        },

        clearError: () => {
          set({ error: null }, false, 'clearError');
        },

        reset: () => {
          set({
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
          }, false, 'reset');
        },

        // Computed getters
        getFilteredReports: () => {
          const { reports, filters, sortOptions } = get();
          
          let filteredReports = [...reports];

          // Apply filters
          if (filters.vulnerabilityTypes.length > 0) {
            filteredReports = filteredReports.filter(report =>
              report.vulnerabilities.some(vuln =>
                filters.vulnerabilityTypes.includes(vuln.type)
              )
            );
          }

          if (filters.severityLevels.length > 0) {
            filteredReports = filteredReports.filter(report =>
              report.vulnerabilities.some(vuln =>
                filters.severityLevels.includes(vuln.severity)
              )
            );
          }

          if (filters.dateRange.start && filters.dateRange.end) {
            filteredReports = filteredReports.filter(report => {
              const reportDate = new Date(report.timestamp);
              return reportDate >= filters.dateRange.start && reportDate <= filters.dateRange.end;
            });
          }

          if (filters.riskScoreRange.min !== 0 || filters.riskScoreRange.max !== 10) {
            filteredReports = filteredReports.filter(report =>
              report.risk_score >= filters.riskScoreRange.min &&
              report.risk_score <= filters.riskScoreRange.max
            );
          }

          // Apply sorting
          filteredReports.sort((a, b) => {
            let aValue: any, bValue: any;

            switch (sortOptions.field) {
              case 'timestamp':
                aValue = new Date(a.timestamp).getTime();
                bValue = new Date(b.timestamp).getTime();
                break;
              case 'risk_score':
                aValue = a.risk_score;
                bValue = b.risk_score;
                break;
              case 'vulnerability_count':
                aValue = a.vulnerabilities.length;
                bValue = b.vulnerabilities.length;
                break;
              case 'contract_name':
                aValue = a.contract_name || '';
                bValue = b.contract_name || '';
                break;
              default:
                return 0;
            }

            if (sortOptions.direction === 'asc') {
              return aValue > bValue ? 1 : -1;
            } else {
              return aValue < bValue ? 1 : -1;
            }
          });

          return filteredReports;
        },

        getVulnerabilityStats: () => {
          const { reports } = get();
          const stats: Record<string, number> = {};

          reports.forEach(report => {
            report.vulnerabilities.forEach(vuln => {
              stats[vuln.type] = (stats[vuln.type] || 0) + 1;
            });
          });

          return Object.entries(stats)
            .map(([type, count]) => ({ type, count }))
            .sort((a, b) => b.count - a.count);
        },

        getSeverityStats: () => {
          const { reports } = get();
          const stats: Record<string, number> = {};

          reports.forEach(report => {
            report.vulnerabilities.forEach(vuln => {
              stats[vuln.severity] = (stats[vuln.severity] || 0) + 1;
            });
          });

          return Object.entries(stats)
            .map(([severity, count]) => ({ severity, count }))
            .sort((a, b) => b.count - a.count);
        },

        getRecentReports: (limit: number = 5) => {
          const { reports } = get();
          return reports
            .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
            .slice(0, limit);
        },

        getHighRiskReports: () => {
          const { reports } = get();
          return reports.filter(report => report.risk_score >= 7);
        },

        getVulnerableReports: () => {
          const { reports } = get();
          return reports.filter(report => report.is_vulnerable);
        },
      }),
      {
        name: 'smart-contract-vulnerability-store',
        partialize: (state) => ({
          reports: state.reports,
          filters: state.filters,
          sortOptions: state.sortOptions,
        }),
      }
    ),
    {
      name: 'smart-contract-vulnerability-store',
    }
  )
);

// Selectors for better performance
export const useReports = () => useAppStore((state) => state.reports);
export const useCurrentReport = () => useAppStore((state) => state.currentReport);
export const useIsAnalyzing = () => useAppStore((state) => state.isAnalyzing);
export const useAnalysisProgress = () => useAppStore((state) => state.analysisProgress);
export const useDashboardStats = () => useAppStore((state) => state.dashboardStats);
export const useFilters = () => useAppStore((state) => state.filters);
export const useSortOptions = () => useAppStore((state) => state.sortOptions);
export const useError = () => useAppStore((state) => state.error);
export const useLoading = () => useAppStore((state) => state.loading);

export const useFilteredReports = () => useAppStore((state) => state.getFilteredReports());
export const useVulnerabilityStats = () => useAppStore((state) => state.getVulnerabilityStats());
export const useSeverityStats = () => useAppStore((state) => state.getSeverityStats());
export const useRecentReports = (limit?: number) => useAppStore((state) => state.getRecentReports(limit));
export const useHighRiskReports = () => useAppStore((state) => state.getHighRiskReports());
export const useVulnerableReports = () => useAppStore((state) => state.getVulnerableReports());
