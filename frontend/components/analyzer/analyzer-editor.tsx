"use client"

import { useState } from "react"
import EditorPanel from "./editor-panel"
import ResultsPanel from "./results-panel"
import { useAnalyzerStore } from "@/store/analyzer-store"
import { useHistoryStore } from "@/store/history-store"

export default function AnalyzerEditor() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const { code, setCode, setResults, setError } = useAnalyzerStore()
  const { addItem } = useHistoryStore()

  const handleAnalyze = async () => {
    if (!code.trim()) {
      setError("Please enter Solidity code to analyze")
      return
    }

    setIsAnalyzing(true)
    setError(null)

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      })

      if (!response.ok) throw new Error("Analysis failed")
      const data = await response.json()
      setResults(data)

      addItem(code, data)
    } catch (error) {
      setError(error instanceof Error ? error.message : "Analysis failed")
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-7xl mx-auto">
      <EditorPanel code={code} onChange={setCode} onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
      <ResultsPanel isAnalyzing={isAnalyzing} />
    </div>
  )
}
