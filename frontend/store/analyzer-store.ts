import { create } from "zustand"

interface AnalysisResult {
  vulnerabilities: Array<{
    type: string
    severity: "critical" | "high" | "medium" | "low"
    description: string
    line?: number
    suggestion?: string
  }>
  optimizations: Array<{
    type: string
    description: string
  }>
  timestamp?: string
}

interface AnalyzerState {
  code: string
  results: AnalysisResult | null
  error: string | null
  lastAnalyzedCode: string | null
  setCode: (code: string) => void
  setResults: (results: AnalysisResult) => void
  setError: (error: string | null) => void
  reset: () => void
  loadFromHistory: (code: string, results: AnalysisResult) => void
}

export const useAnalyzerStore = create<AnalyzerState>((set) => ({
  code: "",
  results: null,
  error: null,
  lastAnalyzedCode: null,
  setCode: (code) => set({ code }),
  setResults: (results) => set({ results, error: null, lastAnalyzedCode: results ? JSON.stringify(results) : null }),
  setError: (error) => set({ error }),
  reset: () => set({ code: "", results: null, error: null, lastAnalyzedCode: null }),
  loadFromHistory: (code, results) =>
    set({
      code,
      results,
      error: null,
      lastAnalyzedCode: JSON.stringify(results),
    }),
}))
