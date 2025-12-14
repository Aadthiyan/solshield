"use client"

import type React from "react"
import { ClerkProvider } from "@clerk/nextjs"

export function ClerkWrapper({ children }: { children: React.ReactNode }) {
  return <ClerkProvider>{children}</ClerkProvider>
}
