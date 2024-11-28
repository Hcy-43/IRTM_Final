'use client'

import { useState } from 'react'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { processInput } from './actions'

export default function Home() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    try {
      const processedResult = await processInput(input)
      setResult(processedResult)
    } catch (err) {
      setError('An error occurred while processing your input.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-5xl font-bold mb-8">Real Or Fake News</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md space-y-4">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Please enter news title"
          disabled={isLoading}
        />
        <Button type="submit" disabled={isLoading} className="w-full">
          {isLoading ? 'Processing...' : 'Send'}
        </Button>
      </form>
      {error && (
        <p className="mt-4 text-red-500">{error}</p>
      )}
      {result !== null && (
        <div className="mt-8 p-4 bg-gray-100 rounded-md">
          <h2 className="text-xl font-semibold mb-2">Result:</h2>
          <p>{result}</p>
        </div>
      )}
    </main>
  )
}

