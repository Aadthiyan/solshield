"use client"

export default function HeroBackground() {
  return (
    <div className="absolute inset-0 -z-10 bg-gradient-to-br from-[#0a0e27] to-[#1e1b4b] overflow-hidden">
      {/* CSS Animated Background */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '4s' }} />
        <div className="absolute bottom-20 right-10 w-72 h-72 bg-accent/20 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '7s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '10s' }} />
      </div>

      {/* Grid Pattern */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />

      {/* Floating Elements (CSS only) */}
      <div className="absolute top-1/4 left-1/4 w-4 h-4 bg-primary rounded-full opacity-50 animate-bounce" style={{ animationDuration: '3s' }} />
      <div className="absolute top-3/4 right-1/4 w-6 h-6 bg-accent rounded-full opacity-50 animate-bounce" style={{ animationDuration: '5s' }} />
    </div>
  )
}
