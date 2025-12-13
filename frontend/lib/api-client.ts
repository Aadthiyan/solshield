/**
 * API Client for Solidity Analyzer
 * Provides methods for interacting with the analyzer backend
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface AnalyzeRequest {
  code: string
}

interface AnalyzeResponse {
  vulnerabilities: Array<{
    type: string
    severity: string
    description: string
    line?: number
    suggestion?: string
  }>
  optimizations: Array<{
    type: string
    description: string
  }>
  timestamp: string
}

export async function analyzeCode(code: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ code } as AnalyzeRequest),
  })

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`)
  }

  return response.json()
}

export async function validateCode(code: string): Promise<{ valid: boolean; error?: string }> {
  // Basic validation - check if code looks like Solidity
  if (!code.trim()) {
    return { valid: false, error: "Code is empty" }
  }

  if (!code.includes("pragma solidity") && !code.includes("contract")) {
    return { valid: false, error: "Does not appear to be valid Solidity code" }
  }

  return { valid: true }
}
