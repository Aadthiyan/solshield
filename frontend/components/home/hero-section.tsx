"use client"

import Link from "next/link"
import { ArrowRight, Zap } from "lucide-react"
import dynamic from "next/dynamic"

const DynamicHeroBackground = dynamic(
  () => import("./3d-hero-background"),
  {
    ssr: false,
    loading: () => (
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl opacity-30 animate-pulse" />
        <div className="absolute bottom-20 right-10 w-72 h-72 bg-accent/20 rounded-full blur-3xl opacity-30 animate-pulse" />
      </div>
    ),
  },
)

export function HeroSection() {
  return (
    <section className="relative min-h-[600px] flex items-center justify-center px-4 py-20 overflow-hidden">
      {/* Background gradient orbs */}
      <DynamicHeroBackground />

      <div className="max-w-4xl mx-auto text-center space-y-8 relative z-10">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 text-sm font-medium backdrop-blur-md">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-white/90">AI-Powered Security Analysis</span>
        </div>

        {/* Main Heading */}
        <h1 className="text-5xl md:text-7xl font-bold text-white text-balance leading-tight drop-shadow-sm">
          Secure Your Smart Contracts with <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">AI</span>
        </h1>

        {/* Subheading */}
        <p className="text-lg md:text-xl text-white/70 text-balance max-w-2xl mx-auto leading-relaxed">
          Advanced vulnerability detection powered by artificial intelligence. Analyze Solidity code in seconds and
          identify security risks before deployment.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8">
          <Link
            href="/sign-in"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold hover:shadow-[0_0_20px_rgba(6,182,212,0.5)] hover:scale-105 transition-all duration-200 group relative overflow-hidden"
          >
            <span className="relative z-10 flex items-center gap-2">
              Launch Analyzer
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </span>
          </Link>
          <a
            href="#features"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-white/5 border border-white/10 text-white backdrop-blur-sm hover:bg-white/10 transition-all duration-200 font-medium"
          >
            Learn More
          </a>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 md:gap-8 pt-12">
          {[
            { label: "Analyses", value: "1000+" },
            { label: "Vulnerabilities Caught", value: "5000+" },
            { label: "Users Protected", value: "500+" },
          ].map((stat) => (
            <div key={stat.label} className="space-y-1">
              <p className="text-2xl md:text-3xl font-bold text-accent">{stat.value}</p>
              <p className="text-sm text-foreground/60">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
