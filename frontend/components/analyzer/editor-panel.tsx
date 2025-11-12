"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Zap, Copy, Upload, FileText } from "lucide-react"
import { useState, useRef } from "react"
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
  const [inputMode, setInputMode] = useState<"paste" | "file">("paste")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      const content = e.target?.result as string
      onChange(content)
      setInputMode("paste")
    }
    reader.readAsText(file)
  }

  return (
    <div className="glass rounded-xl overflow-hidden flex flex-col h-[600px]">
      {/* Header with mode toggle */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border/50">
        <h2 className="text-lg font-semibold text-foreground">Solidity Code</h2>

        <div className="flex items-center gap-2">
          <div className="flex gap-1 bg-card rounded-lg p-1">
            <button
              onClick={() => setInputMode("paste")}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                inputMode === "paste" ? "bg-primary/20 text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <FileText className="w-4 h-4 inline mr-1" />
              Paste
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                inputMode === "file" ? "bg-primary/20 text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Upload className="w-4 h-4 inline mr-1" />
              File
            </button>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="text-muted-foreground hover:text-foreground"
          >
            {copied ? "Copied!" : <Copy className="w-4 h-4" />}
          </Button>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".sol,.solidity,.txt"
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>

      {/* Editor */}
      <div className="flex-1 overflow-hidden">
        <MonacoEditor code={code} onChange={onChange} language="solidity" />
      </div>

      {/* Footer with analyze button */}
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
