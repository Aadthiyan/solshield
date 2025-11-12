"use client"

import HeroSection from "@/components/home/hero-section"
import FeaturesSection from "@/components/home/features-section"
import BenefitsSection from "@/components/home/benefits-section"
import CTASection from "@/components/home/cta-section"

export default function HomePage() {
  return (
    <div className="space-y-0 pb-20">
      <HeroSection />
      <FeaturesSection />
      <BenefitsSection />
      <CTASection />
    </div>
  )
}
