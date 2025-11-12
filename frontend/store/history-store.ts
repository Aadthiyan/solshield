import { create } from "zustand"
import { persist } from "zustand/middleware"

interface HistoryItem {
  id: string
  code: string
  results: any
  timestamp: number
  userId?: string
}

interface HistoryState {
  history: HistoryItem[]
  addItem: (code: string, results: any, userId?: string) => void
  removeItem: (id: string) => void
  clearHistory: () => void
  loadHistory: () => void
  getHistoryByUser: (userId: string) => HistoryItem[]
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      history: [],
      addItem: (code, results, userId) => {
        const newItem: HistoryItem = {
          id: Date.now().toString(),
          code,
          results,
          timestamp: Date.now(),
          userId,
        }
        set((state) => ({
          history: [newItem, ...state.history].slice(0, 50), // Keep last 50 items
        }))
      },
      removeItem: (id) => {
        set((state) => ({
          history: state.history.filter((item) => item.id !== id),
        }))
      },
      clearHistory: () => set({ history: [] }),
      loadHistory: () => {
        // History automatically loads from localStorage via persist middleware
      },
      getHistoryByUser: (userId: string) => {
        const state = get()
        return state.history.filter((item) => item.userId === userId)
      },
    }),
    {
      name: "analyzer-history-storage",
      storage: {
        getItem: (name) => {
          const value = localStorage.getItem(name)
          return value ? JSON.parse(value) : null
        },
        setItem: (name, value) => {
          localStorage.setItem(name, JSON.stringify(value))
        },
        removeItem: (name) => {
          localStorage.removeItem(name)
        },
      },
    },
  ),
)
