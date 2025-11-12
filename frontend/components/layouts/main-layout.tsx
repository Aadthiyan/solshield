"use client"

import type React from "react"
import { TopBar } from "@/components/navigation/topbar"

export function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background">
      {/* TopBar Navigation */}
      <TopBar />

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="min-h-full">{children}</div>
      </main>
    </div>
  )
}
