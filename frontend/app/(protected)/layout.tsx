import type React from "react"
import { ProtectedRoute } from "@/components/auth/protected-route"
import { TopBar } from "@/components/navigation/topbar"

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ProtectedRoute>
      <div className="flex flex-col h-screen overflow-hidden bg-background">
        <TopBar />
        <main className="flex-1 overflow-auto">
          <div className="min-h-full">{children}</div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
