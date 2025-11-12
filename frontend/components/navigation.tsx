"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ShieldAlert } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Navigation() {
  const pathname = usePathname()

  const navItems = [
    { href: "/home", label: "Home" },
    { href: "/analyzer", label: "Analyzer" },
    { href: "/history", label: "History" },
  ]

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass border-b">
      <div className="px-4 md:px-8 py-4 flex items-center justify-between max-w-7xl mx-auto">
        <Link href="/home" className="flex items-center gap-2 group">
          <div className="p-2 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
            <ShieldAlert className="w-6 h-6 text-primary" />
          </div>
          <span className="hidden sm:inline text-xl font-bold text-foreground">SolidityGuard</span>
        </Link>

        <div className="flex items-center gap-8">
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <Link key={item.href} href={item.href}>
                <Button
                  variant="ghost"
                  className={`${
                    pathname === item.href
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {item.label}
                </Button>
              </Link>
            ))}
          </div>

          <Link href="/analyzer">
            <Button className="bg-primary hover:bg-primary/90">Start Analysis</Button>
          </Link>
        </div>
      </div>
    </nav>
  )
}
