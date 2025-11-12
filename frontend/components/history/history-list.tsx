"use client"

import { useHistoryStore } from "@/store/history-store"
import { useAuthStore } from "@/store/auth-store"
import { Trash2, Copy, Code2, CheckCircle2 } from "lucide-react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { formatDistanceToNow } from "date-fns"

export default function HistoryList() {
  const { history, removeItem } = useHistoryStore()
  const { user } = useAuthStore()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    console.log("[v0] History list mounted. User:", user?.id, "Total history items:", history.length)
  }, [user, history])

  if (!mounted) {
    return (
      <div className="glass rounded-xl p-8 h-full flex items-center justify-center">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    )
  }

  const userHistory = history.filter((item) => item.userId === user?.id)
  console.log("[v0] Filtered user history:", userHistory.length, "User ID:", user?.id)

  if (userHistory.length === 0) {
    return (
      <div className="glass rounded-xl p-8 h-full flex items-center justify-center">
        <div className="text-center space-y-3">
          <Code2 className="w-12 h-12 text-muted-foreground mx-auto opacity-50" />
          <div>
            <p className="text-foreground font-medium">No analysis history yet</p>
            <p className="text-sm text-muted-foreground">Your analyzed contracts will appear here</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Analysis History</h2>
          <p className="text-sm text-muted-foreground mt-1">{userHistory.length} analyses saved</p>
        </div>
      </div>

      {/* History List */}
      <ScrollArea className="h-[calc(100vh-200px)]">
        <div className="space-y-3 pr-4">
          {userHistory.map((item) => {
            const vulnCount = item.results?.vulnerabilities?.length || 0
            const isSelected = selectedId === item.id

            return (
              <div
                key={item.id}
                onClick={() => setSelectedId(item.id)}
                className={`glass p-4 rounded-lg cursor-pointer transition-all ${
                  isSelected ? "bg-card/60 border border-primary/50" : "hover:bg-card/40"
                }`}
              >
                <div className="space-y-3">
                  {/* Top row: Time and vulnerability count */}
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-muted-foreground">
                      {formatDistanceToNow(new Date(item.timestamp), { addSuffix: true })}
                    </p>
                    <div className="flex items-center gap-2">
                      {vulnCount === 0 ? (
                        <span className="flex items-center gap-1 text-xs text-accent">
                          <CheckCircle2 className="w-3 h-3" />
                          No issues
                        </span>
                      ) : (
                        <span
                          className={`text-xs font-medium ${vulnCount > 3 ? "text-destructive" : "text-orange-400"}`}
                        >
                          {vulnCount} vulnerabilities
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Code preview */}
                  <div className="bg-background/50 rounded p-2 border border-border/30">
                    <p className="text-xs font-mono text-foreground/60 line-clamp-3 whitespace-pre-wrap break-words">
                      {item.code.substring(0, 200)}
                      {item.code.length > 200 ? "..." : ""}
                    </p>
                  </div>

                  {/* Actions */}
                  {isSelected && (
                    <div className="flex gap-2 pt-2 border-t border-border/30">
                      <Button
                        onClick={(e) => {
                          e.stopPropagation()
                          navigator.clipboard.writeText(item.code)
                        }}
                        size="sm"
                        variant="outline"
                        className="flex-1"
                      >
                        <Copy className="w-3 h-3 mr-1" />
                        Copy Code
                      </Button>
                      <Button
                        onClick={(e) => {
                          e.stopPropagation()
                          removeItem(item.id)
                          setSelectedId(null)
                        }}
                        size="sm"
                        variant="destructive"
                        className="flex-1"
                      >
                        <Trash2 className="w-3 h-3 mr-1" />
                        Delete
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
