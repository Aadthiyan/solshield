"use client"

import { Code2, Zap, Shield, BarChart3 } from "lucide-react"

export function FeaturesSection() {
  const features = [
    {
      icon: Code2,
      title: "Smart Code Analysis",
      description:
        "Analyze Solidity contracts with advanced pattern recognition and vulnerability detection algorithms.",
    },
    {
      icon: Zap,
      title: "Instant Results",
      description: "Get comprehensive analysis results in seconds with detailed vulnerability classifications.",
    },
    {
      icon: Shield,
      title: "Security First",
      description: "Identify reentrancy, overflow, underflow, and other critical security issues automatically.",
    },
    {
      icon: BarChart3,
      title: "Detailed Reports",
      description: "Generate detailed reports with remediation suggestions for each identified vulnerability.",
    },
  ]

  return (
    <section id="features" className="py-20 px-4 bg-card/30">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground">Powerful Features</h2>
          <p className="text-foreground/70 max-w-2xl mx-auto text-lg">
            Everything you need to secure your smart contracts
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div key={index} className="glass p-6 rounded-xl hover:bg-card/60 transition-all duration-300 group">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0 group-hover:bg-primary/30 transition-colors">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <div className="flex-1 space-y-2">
                    <h3 className="text-lg font-semibold text-foreground">{feature.title}</h3>
                    <p className="text-foreground/60 text-sm leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
