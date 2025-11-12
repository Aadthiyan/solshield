"use client"

import type React from "react"

import Navigation from "@/components/navigation"

export default function PagesLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/5">
      <Navigation />
      <main className="pt-20">{children}</main>
    </div>
  )
}
