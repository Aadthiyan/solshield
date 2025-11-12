"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Code2, Home, History } from "lucide-react"

interface SidebarProps {
  open: boolean
  onToggle: () => void
}

export function Sidebar({ open }: SidebarProps) {
  const pathname = usePathname()

  const navItems = [
    {
      label: "Home",
      href: "/home",
      icon: Home,
    },
    {
      label: "Analyzer",
      href: "/analyzer",
      icon: Code2,
    },
    {
      label: "History",
      href: "/history",
      icon: History,
    },
  ]

  return (
    <aside
      className={cn(
        "bg-sidebar border-r border-sidebar-border transition-all duration-300 ease-in-out flex flex-col",
        open ? "w-64" : "w-20",
      )}
    >
      {/* Logo */}
      <div className="h-16 border-b border-sidebar-border flex items-center justify-center px-4">
        <Link href="/home" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
            <Code2 className="w-5 h-5 text-sidebar-primary-foreground" />
          </div>
          {open && <span className="font-bold text-sm whitespace-nowrap">SolidityAI</span>}
        </Link>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 px-3 py-6 space-y-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200",
                isActive
                  ? "bg-sidebar-primary/20 text-sidebar-primary border border-sidebar-primary/50"
                  : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-primary/10",
              )}
              title={!open ? item.label : undefined}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {open && <span className="text-sm font-medium">{item.label}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="h-16 border-t border-sidebar-border flex items-center justify-center px-3">
        <button
          onClick={() => {}}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-primary/10 transition-all"
          title={!open ? "Profile" : undefined}
        >
          <div className="w-5 h-5 rounded-full bg-gradient-to-br from-primary to-accent flex-shrink-0" />
          {open && <span className="text-sm font-medium">Profile</span>}
        </button>
      </div>
    </aside>
  )
}
