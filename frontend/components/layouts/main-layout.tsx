"use client"

import type React from "react"
import { useState } from "react"
import { Sidebar } from "@/components/navigation/sidebar"
import { Header } from "@/components/navigation/header"

export function MainLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} sidebarOpen={sidebarOpen} />

        {/* Page Content */}
        <main className="flex-1 overflow-auto">
          <div className="min-h-full">{children}</div>
        </main>
      </div>
    </div>
  )
}
