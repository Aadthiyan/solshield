"use client"

import AnalyzerEditor from "@/components/analyzer/analyzer-editor"

export default function AnalyzerPage() {
  return (
    <div className="relative min-h-[calc(100vh-4rem)] w-full overflow-hidden bg-slate-950">
       <div className="absolute inset-0 -z-10">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-[100px] opacity-30" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-[100px] opacity-30" />
      </div>
      
      <div className="relative z-10 px-4 md:px-8 py-12">
        <div className="max-w-7xl mx-auto mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">
            Smart Contract <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">Analyzer</span>
          </h1>
          <p className="text-slate-400 max-w-2xl">
            Detect vulnerabilities and optimize your Solidity code with AI-powered analysis.
          </p>
        </div>
        <AnalyzerEditor />
      </div>
    </div>
  )
}
