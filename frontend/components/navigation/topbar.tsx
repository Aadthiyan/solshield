"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useAuthStore } from "@/store/auth-store"
import { Code2, Home, History, LogOut, ChevronDown } from "lucide-react"
import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"

export function TopBar() {
  const pathname = usePathname()
  const { user, logout } = useAuthStore()
  const [profileOpen, setProfileOpen] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setProfileOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const navItems = [
    { label: "Home", href: "/home", icon: Home },
    { label: "Analyzer", href: "/analyzer", icon: Code2 },
    { label: "History", href: "/history", icon: History },
  ]

  return (
    <nav className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between px-6 sticky top-0 z-40">
      {/* Logo & Brand */}
      <Link href="/home" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
        <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
          <Code2 className="w-5 h-5 text-primary-foreground" />
        </div>
        <span className="font-bold text-lg hidden sm:block">SolidityGuard</span>
      </Link>

      {/* Navigation Links */}
      <div className="flex items-center gap-1 flex-1 justify-center">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 text-sm font-medium",
                isActive
                  ? "bg-primary/20 text-primary border border-primary/50"
                  : "text-foreground/60 hover:text-foreground hover:bg-primary/10",
              )}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline">{item.label}</span>
            </Link>
          )
        })}
      </div>

      {/* Profile Dropdown */}
      <div className="relative" ref={profileRef}>
        <button
          onClick={() => setProfileOpen(!profileOpen)}
          className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-card/50 transition-colors"
        >
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-sm font-bold text-primary-foreground">
            {user?.username?.[0]?.toUpperCase() || "U"}
          </div>
          <span className="text-sm font-medium hidden sm:block text-foreground">{user?.username || "User"}</span>
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </button>

        {/* Dropdown Menu */}
        {profileOpen && (
          <div className="absolute right-0 mt-2 w-48 rounded-lg border border-border bg-card/95 backdrop-blur-sm shadow-lg overflow-hidden animate-in fade-in-0 zoom-in-95">
            <div className="px-4 py-3 border-b border-border">
              <p className="text-sm font-medium text-foreground">{user?.username}</p>
              <p className="text-xs text-muted-foreground mt-1">{user?.email}</p>
            </div>

            <div className="py-2">
              <button
                onClick={() => {
                  logout()
                  setProfileOpen(false)
                  window.location.href = "/login"
                }}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm text-foreground hover:bg-destructive/10 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
