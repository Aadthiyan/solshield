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
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          {/* Left Column */}
          <div className="space-y-6">
            <h2 className="text-4xl font-bold text-foreground">Why Choose SolidityAI?</h2>
            <p className="text-foreground/70 text-lg leading-relaxed">
              Our AI-powered analyzer combines machine learning with security expertise to provide unmatched
              vulnerability detection for smart contracts.
            </p>
            <ul className="space-y-3">
              {benefits.map((benefit, index) => (
                <li key={index} className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-accent flex-shrink-0" />
                  <span className="text-foreground/80">{benefit}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Right Column - Visual */}
          <div className="relative">
            <div className="glass p-8 rounded-xl space-y-4">
              <div className="h-2 bg-gradient-to-r from-primary to-accent rounded-full w-3/4" />
              <div className="space-y-3">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="flex gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary/50 mt-1.5 flex-shrink-0" />
                    <div className="h-3 bg-muted rounded w-full" />
                  </div>
                ))}
              </div>
              <div className="pt-4 border-t border-border">
                <p className="text-xs text-foreground/50">Analysis Report</p>
                <p className="text-sm text-accent font-semibold mt-2">5 Vulnerabilities Found</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
