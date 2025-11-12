"use client"

import { Button } from "@/components/ui/button"
import { Zap, Copy } from "lucide-react"
import { useState } from "react"
import { Loader2 } from "lucide-react"
import { MonacoEditor } from "./monaco-editor"

export default function EditorPanel({
  code,
  onChange,
  onAnalyze,
  isAnalyzing,
}: {
  code: string
  onChange: (code: string) => void
  onAnalyze: () => void
  isAnalyzing: boolean
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="glass rounded-xl overflow-hidden flex flex-col h-[600px]">
      <div className="flex items-center justify-between px-6 py-4 border-b border-border/50">
        <h2 className="text-lg font-semibold text-foreground">Solidity Code</h2>
        <Button variant="ghost" size="sm" onClick={handleCopy} className="text-muted-foreground hover:text-foreground">
          {copied ? "Copied!" : <Copy className="w-4 h-4" />}
        </Button>
      </div>

      <div className="flex-1 overflow-hidden">
        <MonacoEditor code={code} onChange={onChange} language="solidity" />
      </div>

      <div className="px-6 py-4 border-t border-border/50 flex gap-3">
        <Button onClick={onAnalyze} disabled={isAnalyzing} className="flex-1 bg-primary hover:bg-primary/90">
          {isAnalyzing ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 mr-2" />
              Analyze Code
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
