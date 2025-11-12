"use client"
import { Menu } from "lucide-react"

interface HeaderProps {
  onMenuClick: () => void
  sidebarOpen: boolean
}

export function Header({ onMenuClick, sidebarOpen }: HeaderProps) {
  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        {/* Menu Button - visible when sidebar is closed on mobile */}
        <button
          onClick={onMenuClick}
          className="p-2 hover:bg-card rounded-lg transition-colors md:hidden"
          aria-label="Toggle sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Page Title */}
        <div>
          <h1 className="text-lg font-semibold text-foreground">Solidity Vulnerability Analyzer</h1>
          <p className="text-xs text-muted-foreground">Advanced smart contract security analysis</p>
        </div>
      </div>

      {/* Header Actions */}
      <div className="flex items-center gap-4">
        <button
          className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
          onClick={() => {}}
        >
          Documentation
        </button>
      </div>
    </header>
  )
}
