import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware"
import { useAuth, useUser } from "@clerk/nextjs"

interface User {
  id: string
  email: string
  username: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  initializeFromClerk: (clerkUser: any, token: string | null) => void
  logout: () => void
  setError: (error: string | null) => void
  setUser: (user: User | null) => void
  setToken: (token: string | null) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      initializeFromClerk: (clerkUser: any, token: string | null) => {
        if (clerkUser) {
          const user: User = {
            id: clerkUser.id,
            email: clerkUser.emailAddresses?.[0]?.emailAddress || "",
            username: clerkUser.username || clerkUser.firstName || "",
          }
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          })
        } else {
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          })
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        })
      },

      setError: (error: string | null) => set({ error }),
      setUser: (user: User | null) => set({ user }),
      setToken: (token: string | null) => set({ token }),
    }),
    {
      name: "auth-storage",
      storage: createJSONStorage(() => {
        if (typeof window === "undefined") {
          return {
            getItem: () => null,
            setItem: () => {},
            removeItem: () => {},
          }
        }
        return localStorage
      }),
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
)

