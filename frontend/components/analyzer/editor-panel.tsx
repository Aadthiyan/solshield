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
    <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl overflow-hidden flex flex-col h-[600px] shadow-2xl">
      {/* Header with mode toggle */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/5">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          Solidity Code
        </h2>

        <div className="flex items-center gap-2">
          <div className="flex gap-1 bg-black/20 rounded-lg p-1 border border-white/5">
            <button
              onClick={() => setInputMode("paste")}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${inputMode === "paste"
                  ? "bg-cyan-500/10 text-cyan-400 shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
            >
              <FileText className="w-4 h-4 inline mr-2" />
              Paste
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${inputMode === "file"
                  ? "bg-cyan-500/10 text-cyan-400 shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
            >
              <Upload className="w-4 h-4 inline mr-2" />
              File
            </button>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="text-slate-400 hover:text-white hover:bg-white/5"
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
      <div className="px-6 py-4 border-t border-white/10 bg-white/5 flex gap-3">
        <Button
          onClick={onAnalyze}
          disabled={isAnalyzing}
          className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold py-6 shadow-lg shadow-cyan-900/20 border-0"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Analyzing Contract...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5 mr-2 fill-current" />
              Analyze Smart Contract
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
