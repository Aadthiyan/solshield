"use client"

import Link from "next/link"
import { ArrowRight } from "lucide-react"

export function CTASection() {
  return (
    <section className="py-32 px-4 relative overflow-hidden bg-[#0a0e27]">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-t from-cyan-900/20 to-transparent pointer-events-none" />
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="max-w-3xl mx-auto text-center space-y-8 relative z-10">
        <h2 className="text-4xl md:text-5xl font-bold text-white text-balance drop-shadow-lg">
          Ready to Secure Your Contracts?
        </h2>

        <p className="text-xl text-gray-300 max-w-xl mx-auto">
          Start analyzing your Solidity code with AI-powered vulnerability detection today.
        </p>

        <Link
          href="/analyzer"
          className="inline-flex items-center justify-center gap-2 px-10 py-5 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold hover:shadow-[0_0_30px_rgba(6,182,212,0.4)] hover:scale-105 transition-all duration-300 group"
        >
          Launch Analyzer
          <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
        </Link>
      </div>
    </section>
  )
}
