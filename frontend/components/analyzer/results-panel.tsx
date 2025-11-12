"use client"

import { useAnalyzerStore } from "@/store/analyzer-store"
import { AlertCircle, CheckCircle2, AlertTriangle, Loader2, TrendingDown, Copy } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useState } from "react"

export default function ResultsPanel({ isAnalyzing }: { isAnalyzing: boolean }) {
  const { results, error } = useAnalyzerStore()
  const [copied, setCopied] = useState(false)

  const handleCopyResults = () => {
    if (results) {
      const text = JSON.stringify(results, null, 2)
      navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (isAnalyzing) {
    return (
      <div className="glass rounded-xl p-6 h-[600px] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-primary animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Analyzing your code...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass rounded-xl p-6 h-[600px] flex items-center justify-center">
        <div className="text-center space-y-3">
          <AlertCircle className="w-8 h-8 text-destructive mx-auto" />
          <div>
            <p className="text-destructive font-medium">Analysis Error</p>
            <p className="text-sm text-muted-foreground mt-1">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="glass rounded-xl p-6 h-[600px] flex items-center justify-center">
        <div className="text-center">
          <CheckCircle2 className="w-8 h-8 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">Analysis results will appear here</p>
          <p className="text-xs text-muted-foreground/60 mt-2">
            Paste your Solidity code and click Analyze to get started
          </p>
        </div>
      </div>
    )
  }

  const vulnCount = results.vulnerabilities?.length || 0
  const optCount = results.optimizations?.length || 0

  return (
    <div className="glass rounded-xl p-6 h-[600px] flex flex-col">
      <div className="flex items-center justify-between mb-4 pb-4 border-b border-border/50">
        <div className="flex items-center gap-6">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Analysis Results</h2>
          </div>
          <div className="flex gap-6 text-sm">
            {vulnCount > 0 && (
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-destructive" />
                <span className="text-foreground/70">{vulnCount} vulnerabilities</span>
              </div>
            )}
            {optCount > 0 && (
              <div className="flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-accent" />
                <span className="text-foreground/70">{optCount} optimizations</span>
              </div>
            )}
          </div>
        </div>
        <button
          onClick={handleCopyResults}
          className="p-2 hover:bg-card rounded-lg transition-colors text-muted-foreground hover:text-foreground"
          title="Copy results"
        >
          {copied ? "✓" : <Copy className="w-4 h-4" />}
        </button>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-4 pr-4">
          {vulnCount === 0 && optCount === 0 && (
            <div className="text-center py-8">
              <CheckCircle2 className="w-8 h-8 text-accent mx-auto mb-3" />
              <p className="text-foreground font-medium">No vulnerabilities detected!</p>
              <p className="text-sm text-muted-foreground mt-1">Your code looks secure.</p>
            </div>
          )}

          {/* Vulnerabilities */}
          {results.vulnerabilities?.map((vuln: any, idx: number) => (
            <div
              key={`vuln-${idx}`}
              className="p-4 rounded-lg bg-destructive/10 border border-destructive/30 hover:bg-destructive/15 transition-colors"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="font-semibold text-foreground break-words">{vuln.type}</h4>
                    {vuln.severity && (
                      <span
                        className={`text-xs font-medium px-2 py-1 rounded whitespace-nowrap ${
                          vuln.severity === "critical"
                            ? "bg-destructive/30 text-destructive"
                            : vuln.severity === "high"
                              ? "bg-orange-500/30 text-orange-300"
                              : "bg-yellow-500/30 text-yellow-300"
                        }`}
                      >
                        {vuln.severity}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{vuln.description}</p>
                  {vuln.line && <p className="text-xs text-muted-foreground mt-2">Line: {vuln.line}</p>}
                  {vuln.suggestion && (
                    <div className="mt-2 p-2 bg-card/50 rounded text-xs text-foreground/70 border border-border/50">
                      <span className="font-medium">Suggestion:</span> {vuln.suggestion}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Optimizations */}
          {results.optimizations?.map((opt: any, idx: number) => (
            <div
              key={`opt-${idx}`}
              className="p-4 rounded-lg bg-accent/10 border border-accent/30 hover:bg-accent/15 transition-colors"
            >
              <div className="flex items-start gap-3">
                <TrendingDown className="w-5 h-5 text-accent mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-foreground break-words">{opt.type}</h4>
                  <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{opt.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
