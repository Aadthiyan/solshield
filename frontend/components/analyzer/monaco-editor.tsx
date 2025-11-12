"use client"

import { useEffect, useRef } from "react"
import * as monaco from "monaco-editor"

interface MonacoEditorProps {
  code: string
  onChange: (code: string) => void
  language?: string
  height?: string | number
}

export function MonacoEditor({ code, onChange, language = "solidity", height = "600px" }: MonacoEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null)
  const editorInstanceRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)

  useEffect(() => {
    if (!editorRef.current) return

    // Create editor instance
    editorInstanceRef.current = monaco.editor.create(editorRef.current, {
      value: code,
      language,
      theme: "vs-dark",
      automaticLayout: true,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      fontSize: 14,
      fontFamily: "'Geist Mono', monospace",
      wordWrap: "on",
      padding: { top: 16, bottom: 16 },
    })

    // Handle code changes
    const changeSubscription = editorInstanceRef.current.onDidChangeModelContent(() => {
      const value = editorInstanceRef.current?.getValue()
      if (value !== undefined) {
        onChange(value)
      }
    })

    return () => {
      changeSubscription.dispose()
    }
  }, [])

  // Update editor content when code prop changes (from external source)
  useEffect(() => {
    if (editorInstanceRef.current && editorInstanceRef.current.getValue() !== code) {
      editorInstanceRef.current.setValue(code)
    }
  }, [code])

  return <div ref={editorRef} style={{ height, width: "100%" }} className="rounded-lg overflow-hidden" />
}
