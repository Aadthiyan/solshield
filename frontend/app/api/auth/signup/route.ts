import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { email, username, password } = await request.json()

    // Mock user creation - replace with your backend call
    if (email && username && password) {
      const user = {
        id: "user-" + Date.now(),
        email,
        username,
      }

      const token = btoa(JSON.stringify({ id: user.id, email: user.email, exp: Date.now() + 86400000 }))

      return NextResponse.json({ user, token })
    }

    return NextResponse.json({ message: "Signup failed" }, { status: 400 })
  } catch (error) {
    return NextResponse.json({ message: "Signup failed" }, { status: 500 })
  }
}
