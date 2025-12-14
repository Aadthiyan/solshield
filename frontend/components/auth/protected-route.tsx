"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth, useUser } from "@clerk/nextjs"
import { useAuthStore } from "@/store/auth-store"

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const router = useRouter()
  const { isSignedIn, isLoaded } = useAuth()
  const { user: clerkUser } = useUser()
  const { initializeFromClerk } = useAuthStore()
  const [isHydrated, setIsHydrated] = useState(false)

  useEffect(() => {
    setIsHydrated(true)
  }, [])

  useEffect(() => {
    if (isHydrated && isLoaded) {
      if (!isSignedIn || !clerkUser) {
        router.push("/sign-in")
      } else {
        // Initialize auth store with Clerk user data
        initializeFromClerk(clerkUser, null)
      }
    }
  }, [isSignedIn, clerkUser, router, isHydrated, isLoaded, initializeFromClerk])

  if (!isHydrated || !isLoaded || !isSignedIn || !clerkUser) {
    return null
  }

  return <>{children}</>
}
