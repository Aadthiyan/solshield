"use client"

import { CheckCircle2 } from "lucide-react"

export function BenefitsSection() {
  const benefits = [
    "Save hours of manual code review",
    "Catch vulnerabilities before production",
    "Reduce security audit costs",
    "Comply with best practices",
    "Get detailed remediation guidance",
    "Track analysis history",
  ]

  return (
    <section className="py-24 px-4 bg-[#0a0e27] border-t border-white/5">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          {/* Left Column */}
          <div className="space-y-6">
            <h2 className="text-4xl font-bold text-white">Why Choose SolidityAI?</h2>
            <p className="text-gray-400 text-lg leading-relaxed">
              Our AI-powered analyzer combines machine learning with security expertise to provide unmatched
              vulnerability detection for smart contracts.
            </p>
            <ul className="space-y-4">
              {benefits.map((benefit, index) => (
                <li key={index} className="flex items-center gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                  </div>
                  <span className="text-white/80 font-medium">{benefit}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Right Column - Visual */}
          <div className="relative group">
            {/* Glow behind */}
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-xl blur opacity-50 group-hover:opacity-100 transition duration-500"></div>
            <div className="relative glass bg-[#1e293b]/50 p-8 rounded-xl space-y-6 border border-white/10 backdrop-blur-md">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="h-2 w-20 bg-white/20 rounded-full"></div>
                  <div className="h-2 w-8 bg-green-500/50 rounded-full"></div>
                </div>
                <div className="h-40 bg-black/40 rounded-lg border border-white/5 p-4 space-y-2">
                  <div className="h-2 bg-white/10 rounded w-3/4 animate-pulse"></div>
                  <div className="h-2 bg-white/10 rounded w-1/2 animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                  <div className="h-2 bg-white/10 rounded w-2/3 animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>

              <div className="pt-4 border-t border-white/10">
                <p className="text-xs text-gray-400 uppercase tracking-widest font-semibold">Security Score</p>
                <div className="flex items-center gap-3 mt-2">
                  <span className="text-3xl font-bold text-white">98<span className="text-sm text-gray-500">/100</span></span>
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-500/20 text-green-400">Secure</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
