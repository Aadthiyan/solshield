import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json()

    // Mock authentication - replace with your backend call
    if (email && password) {
      const user = {
        id: "user-" + Date.now(),
        email,
        username: email.split("@")[0],
      }

      // Mock JWT token
      const token = btoa(JSON.stringify({ id: user.id, email: user.email, exp: Date.now() + 86400000 }))

      return NextResponse.json({ user, token })
    }

    return NextResponse.json({ message: "Invalid credentials" }, { status: 401 })
  } catch (error) {
    return NextResponse.json({ message: "Login failed" }, { status: 500 })
  }
}
