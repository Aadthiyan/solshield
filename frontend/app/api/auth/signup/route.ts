import { type NextRequest, NextResponse } from "next/server"

// Backend API URL - use environment variable in production
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // Ensure username is included as the backend expects it inside the user_data schema
    // The backend endpoint /auth/signup expects UserSignup schema which is EmailStr, Username (optional maybe?), Password
    // Let's check UserSignup schema in python.
    // Wait, the python UserSignup schema is not visible here, but api/routers/auth.py handles it.
    // Line 24 in auth.py: async def signup(user_data: UserSignup, ...):

    const response = await fetch(`${BACKEND_URL}/auth/signup`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(
        { message: data.detail || "Signup failed" },
        { status: response.status }
      )
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Signup proxy error:", error)
    return NextResponse.json(
      { message: "Backend connection failed. Is the server running?" },
      { status: 500 }
    )
  }
}
