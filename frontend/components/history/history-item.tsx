"use client"

import { useState } from "react"
import { format } from "date-fns"
import { ChevronDown, Trash2, Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useHistoryStore } from "@/store/history-store"
import { ScrollArea } from "@/components/ui/scroll-area"

export default function HistoryItem({ item }: { item: any }) {
  const [expanded, setExpanded] = useState(false)
  const { removeItem } = useHistoryStore()
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(item.code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const vulnerabilityCount = item.results?.vulnerabilities?.length || 0
  const optimizationCount = item.results?.optimizations?.length || 0

  return (
    <div className="glass rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-card/50 transition-colors"
      >
        <div className="flex-1 text-left">
          <p className="text-sm text-muted-foreground">{format(new Date(item.timestamp), "MMM dd, yyyy HH:mm")}</p>
          <div className="flex gap-4 mt-2">
            <span className="text-sm font-medium text-destructive">{vulnerabilityCount} Vulnerabilities</span>
            <span className="text-sm font-medium text-accent">{optimizationCount} Optimizations</span>
          </div>
        </div>
        <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>

      {expanded && (
        <div className="border-t border-border/50 px-6 py-4 space-y-4">
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-foreground">Code Preview</h4>
            <ScrollArea className="h-32 rounded-lg bg-secondary/50 p-3">
              <pre className="text-xs text-muted-foreground font-mono">
                {item.code.slice(0, 500)}
                {item.code.length > 500 ? "..." : ""}
              </pre>
            </ScrollArea>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleCopy} className="flex-1 bg-transparent">
              <Copy className="w-4 h-4 mr-2" />
              {copied ? "Copied" : "Copy Code"}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => removeItem(item.id)}
              className="text-destructive hover:text-destructive/80 hover:bg-destructive/10"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
