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
      <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl p-12 flex items-center justify-center min-h-[400px] animate-pulse">
        <div className="flex flex-col items-center gap-3 text-slate-400">
          <div className="w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          <span>Loading history...</span>
        </div>
      </div>
    )
  }

  const userHistory = history.filter((item) => item.userId === user?.id)
  console.log("[v0] Filtered user history:", userHistory.length, "User ID:", user?.id)

  if (userHistory.length === 0) {
    return (
      <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl p-16 flex items-center justify-center min-h-[400px]">
        <div className="text-center space-y-4">
          <div className="bg-white/5 p-4 rounded-full w-fit mx-auto mb-4 ring-1 ring-white/10">
            <Code2 className="w-10 h-10 text-slate-500" />
          </div>
          <div>
            <p className="text-white text-lg font-medium">No analysis history yet</p>
            <p className="text-sm text-slate-400 mt-1">Your analyzed contracts will appear here automatically.</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-slate-900/50 border border-white/10 backdrop-blur-md rounded-xl overflow-hidden shadow-2xl">
      {/* Header Bar */}
      <div className="px-6 py-4 border-b border-white/10 bg-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]" />
          <span className="text-sm font-medium text-slate-300">Saved Analyses</span>
        </div>
        <span className="text-xs text-slate-500 font-mono bg-black/20 px-2 py-1 rounded border border-white/5">
          {userHistory.length} ITEMS
        </span>
      </div>

      {/* History List */}
      <ScrollArea className="h-[calc(100vh-16rem)]">
        <div className="p-4 space-y-3">
          {userHistory.map((item) => {
            const vulnCount = item.results?.vulnerabilities?.length || 0
            const isSelected = selectedId === item.id

            return (
              <div
                key={item.id}
                onClick={() => setSelectedId(isSelected ? null : item.id)}
                className={`group rounded-xl cursor-pointer border transition-all duration-200 ${isSelected
                    ? "bg-cyan-950/20 border-cyan-500/50 shadow-[0_0_20px_rgba(6,182,212,0.1)]"
                    : "bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/10"
                  }`}
              >
                <div className="p-4 space-y-4">
                  {/* Top row: Time and vulnerability count */}
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-slate-400 group-hover:text-slate-300 transition-colors">
                      {formatDistanceToNow(new Date(item.timestamp), { addSuffix: true })}
                    </p>
                    <div className="flex items-center gap-2">
                      {vulnCount === 0 ? (
                        <span className="flex items-center gap-1.5 text-xs font-medium text-green-400 bg-green-500/10 px-2.5 py-1 rounded-full border border-green-500/20">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Secure
                        </span>
                      ) : (
                        <span
                          className={`text-xs font-bold px-2.5 py-1 rounded-full border ${vulnCount > 3
                              ? "bg-red-500/10 text-red-400 border-red-500/20"
                              : "bg-orange-500/10 text-orange-400 border-orange-500/20"
                            }`}
                        >
                          {vulnCount} Vulnerabilities
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Code preview */}
                  <div className="relative group/code">
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5 font-mono text-xs text-slate-300 overflow-hidden relative">
                      <div className="absolute top-0 right-0 p-2 opacity-0 group-hover/code:opacity-100 transition-opacity">
                        {/* Optional overlay icon could go here */}
                      </div>
                      <p className="line-clamp-2 opacity-80 leading-relaxed">
                        {item.code}
                      </p>
                      <div className="absolute inset-x-0 bottom-0 h-8 bg-gradient-to-t from-black/20 to-transparent" />
                    </div>
                  </div>

                  {/* Actions (Expandable) */}
                  <div
                    className={`overflow-hidden transition-all duration-300 ease-in-out ${isSelected ? "max-h-20 opacity-100 mt-4" : "max-h-0 opacity-0"
                      }`}
                  >
                    <div className="flex gap-3 pt-2 border-t border-white/10">
                      <Button
                        onClick={(e) => {
                          e.stopPropagation()
                          navigator.clipboard.writeText(item.code)
                        }}
                        size="sm"
                        className="flex-1 bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 hover:border-white/20"
                      >
                        <Copy className="w-3.5 h-3.5 mr-2 text-cyan-400" />
                        Copy Code
                      </Button>
                      <Button
                        onClick={(e) => {
                          e.stopPropagation()
                          removeItem(item.id)
                          setSelectedId(null)
                        }}
                        size="sm"
                        className="flex-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 hover:border-red-500/30"
                      >
                        <Trash2 className="w-3.5 h-3.5 mr-2" />
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
