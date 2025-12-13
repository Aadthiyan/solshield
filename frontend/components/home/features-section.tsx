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
    <section id="features" className="py-24 px-4 bg-[#0a0e27] relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-purple-500/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-cyan-500/10 rounded-full blur-[100px] pointer-events-none" />

      <div className="max-w-6xl mx-auto relative z-10">
        <div className="text-center mb-16 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold text-white">Powerful Features</h2>
          <p className="text-gray-400 max-w-2xl mx-auto text-lg">
            Everything you need to secure your smart contracts against modern threats.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div
                key={index}
                className="group relative p-8 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300 hover:-translate-y-1 backdrop-blur-sm"
              >
                {/* Hover Glow Effect */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-cyan-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />

                <div className="relative flex flex-col gap-6">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-white/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                    <Icon className="w-7 h-7 text-cyan-400" />
                  </div>

                  <div className="space-y-3">
                    <h3 className="text-xl font-bold text-white">{feature.title}</h3>
                    <p className="text-gray-400 leading-relaxed">{feature.description}</p>
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
