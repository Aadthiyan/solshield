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
      <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl p-6 h-[600px] flex items-center justify-center shadow-2xl">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-cyan-500 animate-spin mx-auto mb-4" />
          <p className="text-slate-400 text-lg">Analyzing your contract...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-slate-900/50 border border-red-500/20 backdrop-blur-md rounded-xl p-6 h-[600px] flex items-center justify-center shadow-2xl">
        <div className="text-center space-y-3">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto" />
          <div>
            <p className="text-red-400 font-bold text-lg">Analysis Error</p>
            <p className="text-sm text-slate-400 mt-1 max-w-sm mx-auto">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl p-6 h-[600px] flex items-center justify-center shadow-2xl">
        <div className="text-center">
          <div className="bg-white/5 p-4 rounded-full w-fit mx-auto mb-4 ring-1 ring-white/10">
            <CheckCircle2 className="w-8 h-8 text-slate-500" />
          </div>
          <h3 className="text-white font-medium text-lg mb-1">Ready to Analyze</h3>
          <p className="text-slate-400 max-w-xs mx-auto">
            Paste your Smart Contract code or upload a file to detect vulnerabilities and gas optimizations.
          </p>
        </div>
      </div>
    )
  }

  const vulnCount = results.vulnerabilities?.length || 0
  const optCount = results.optimizations?.length || 0

  return (
    <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl flex flex-col h-[600px] overflow-hidden shadow-2xl">
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/5">
        <div className="flex items-center gap-6">
          <div>
            <h2 className="text-lg font-semibold text-white">Analysis Results</h2>
          </div>
          <div className="flex gap-6 text-sm">
            {vulnCount > 0 && (
              <div className="flex items-center gap-2 bg-red-500/10 px-3 py-1 rounded-full border border-red-500/20">
                <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                <span className="text-red-200 font-medium">{vulnCount} Issues</span>
              </div>
            )}
            {optCount > 0 && (
              <div className="flex items-center gap-2 bg-blue-500/10 px-3 py-1 rounded-full border border-blue-500/20">
                <TrendingDown className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-blue-200 font-medium">{optCount} Optimizations</span>
              </div>
            )}
          </div>
          <div className="text-xs text-slate-500 font-mono hidden md:block">AI-Powered Audit</div>
        </div>
        <button
          onClick={handleCopyResults}
          className="p-2 hover:bg-white/10 rounded-lg transition-colors text-slate-400 hover:text-white"
          title="Copy results"
        >
          {copied ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
        </button>
      </div>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-4">
          {vulnCount === 0 && optCount === 0 && (
            <div className="text-center py-16">
              <div className="bg-green-500/10 p-4 rounded-full w-fit mx-auto mb-4 ring-1 ring-green-500/20">
                <CheckCircle2 className="w-10 h-10 text-green-400" />
              </div>
              <p className="text-white font-bold text-lg">No vulnerabilities detected!</p>
              <p className="text-sm text-slate-400 mt-2">Your smart contract code looks secure and optimized.</p>
            </div>
          )}

          {results.vulnerabilities?.map((vuln: any, idx: number) => (
            <div
              key={`vuln-${idx}`}
              className="group p-4 rounded-xl bg-gradient-to-br from-red-500/5 to-transparent border-l-4 border-l-red-500 border-y border-r border-red-500/10 hover:border-red-500/30 transition-all duration-300"
            >
              <div className="flex items-start gap-4">
                <div className="p-2 bg-red-500/10 rounded-lg border border-red-500/20 group-hover:bg-red-500/20 transition-colors">
                  <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <h4 className="font-bold text-slate-200 break-words text-base">{vuln.type}</h4>
                    {vuln.severity && (
                      <span
                        className={`text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded border ${vuln.severity === "critical"
                            ? "bg-red-500/20 text-red-400 border-red-500/30"
                            : vuln.severity === "high"
                              ? "bg-orange-500/20 text-orange-400 border-orange-500/30"
                              : "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
                          }`}
                      >
                        {vuln.severity}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed mb-3">{vuln.description}</p>

                  {vuln.suggestion && (
                    <div className="p-3 bg-black/20 rounded-lg border border-white/5 text-sm">
                      <span className="text-cyan-400 font-medium block text-xs mb-1 uppercase tracking-wide">Remediation</span>
                      <span className="text-slate-300">{vuln.suggestion}</span>
                    </div>
                  )}
                  {vuln.line && <p className="text-xs text-slate-600 mt-2 font-mono">Line: {vuln.line}</p>}
                </div>
              </div>
            </div>
          ))}

          {/* Optimizations */}
          {results.optimizations?.map((opt: any, idx: number) => (
            <div
              key={`opt-${idx}`}
              className="group p-4 rounded-xl bg-gradient-to-br from-blue-500/5 to-transparent border-l-4 border-l-blue-500 border-y border-r border-blue-500/10 hover:border-blue-500/30 transition-all duration-300"
            >
              <div className="flex items-start gap-4">
                <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20 group-hover:bg-blue-500/20 transition-colors">
                  <TrendingDown className="w-5 h-5 text-blue-400 flex-shrink-0" />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-bold text-slate-200 break-words text-base mb-1">{opt.type}</h4>
                  <p className="text-sm text-slate-400 leading-relaxed">{opt.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
