"use client"

import Link from "next/link"
import { ArrowRight } from "lucide-react"

export function CTASection() {
  return (
    <section className="py-20 px-4 relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-accent/10 rounded-full blur-3xl opacity-50" />
      </div>

      <div className="max-w-2xl mx-auto text-center space-y-8">
        <h2 className="text-4xl md:text-5xl font-bold text-foreground text-balance">Ready to Secure Your Contracts?</h2>

        <p className="text-lg text-foreground/70">
          Start analyzing your Solidity code with AI-powered vulnerability detection today.
        </p>

        <Link
          href="/analyzer"
          className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-lg bg-primary text-primary-foreground font-semibold hover:opacity-90 transition-all duration-200 group"
        >
          Launch Analyzer
          <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
        </Link>
      </div>
    </section>
  )
}
