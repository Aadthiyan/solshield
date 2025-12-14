"use client"

import { SignUp } from "@clerk/nextjs"
import { useEffect, useState } from "react"
import { useUser } from "@clerk/nextjs"
import { useRouter } from "next/navigation"

export default function SignupPage() {
  const [mounted, setMounted] = useState(false)
  const router = useRouter()
  const { user, isLoaded } = useUser()

  useEffect(() => {
    setMounted(true)
  }, [])

  // Redirect to analyzer if user is signed up and verified
  useEffect(() => {
    if (isLoaded && user && user.emailAddresses.length > 0) {
      const primaryEmail = user.emailAddresses.find(email => email.id === user.primaryEmailAddressId)
      if (primaryEmail && primaryEmail.verification?.status === "verified") {
        router.push("/analyzer")
      }
    }
  }, [user, isLoaded, router])

  if (!mounted) return null

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20 relative overflow-hidden flex items-center justify-center">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 right-20 w-80 h-80 bg-primary/30 rounded-full mix-blend-screen blur-3xl animate-pulse" />
        <div
          className="absolute bottom-20 left-10 w-96 h-96 bg-primary/20 rounded-full mix-blend-screen blur-3xl animate-pulse"
          style={{ animationDelay: "1s" }}
        />
        <div
          className="absolute top-1/2 right-1/4 w-72 h-72 bg-accent/20 rounded-full mix-blend-screen blur-3xl animate-pulse"
          style={{ animationDelay: "2s" }}
        />

        <svg className="absolute inset-0 w-full h-full opacity-20" preserveAspectRatio="none">
          <defs>
            <pattern id="wave2" x="0" y="0" width="120" height="120" patternUnits="userSpaceOnUse">
              <path d="M0,60 Q30,50 60,60 T120,60" stroke="url(#gradient2)" strokeWidth="1" fill="none" opacity="0.3" />
            </pattern>
            <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#00d4ff" />
              <stop offset="100%" stopColor="#7c3aed" />
            </linearGradient>
          </defs>
          <rect width="100%" height="100%" fill="url(#wave2)" />
        </svg>
      </div>

      {/* Clerk SignUp Component */}
      <div className="relative z-10 w-full max-w-md mx-4">
        <SignUp 
          signInUrl="/sign-in"
          redirectUrl="/analyzer"
          routing="path"
        />
      </div>
    </div>
  )
}
