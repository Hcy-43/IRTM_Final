'use client'

import { useState } from 'react'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import Image from 'next/image'

type Option = 'all' | 'title' | 'text'

interface ApiResponse {
  content: string;
  lime_figure: string;
  positive_features: string[];
  predicted_class: number;
  probabilities: number[];
  query: string;
}

export default function Home() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState<ApiResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedOption, setSelectedOption] = useState<Option>('all')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: selectedOption,
          query: input
        })
      })

      if (!response.ok) {
        throw new Error('API request failed')
      }

      const data: ApiResponse = await response.json()
      setResult(data)
    } catch (err) {
      setError('An error occurred while processing your input.')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-5xl font-bold mb-8">Real Or Fake News</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md space-y-4">
        <div className="flex justify-center space-x-2">
          {(['all', 'title', 'text'] as const).map((option) => (
            <Button
              key={option}
              type="button"
              variant={selectedOption === option ? "default" : "outline"}
              onClick={() => setSelectedOption(option)}
              disabled={isLoading}
            >
              {option.charAt(0).toUpperCase() + option.slice(1)}
            </Button>
          ))}
        </div>
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Please enter news title or text"
          disabled={isLoading}
        />
        <Button type="submit" disabled={isLoading} className="w-full">
          {isLoading ? 'Processing...' : 'Send'}
        </Button>
      </form>
      {error && (
        <p className="mt-4 text-red-500">{error}</p>
      )}
      {result && (
        <div className="mt-8 p-4 bg-gray-100 rounded-md w-full max-w-2xl">
          <h2 className="text-xl font-semibold mb-2">Result:</h2>
          <p><strong>Query:</strong> {result.query}</p>
          <p><strong>Content Type:</strong> {result.content}</p>
          <p><strong>Predicted Class:</strong> {result.predicted_class === 1 ? 'Real' : 'Fake'}</p>
          <p><strong>Probabilities:</strong> Real: {(result.probabilities[1] * 100).toFixed(2)}%, Fake: {(result.probabilities[0] * 100).toFixed(2)}%</p>
          <p><strong>Positive Features:</strong> {result.positive_features.join(', ')}</p>
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">LIME Explanation:</h3>
            <Image
              src={`data:image/png;base64,${result.lime_figure}`}
              alt="LIME Explanation"
              width={600}
              height={400}
            />
          </div>
        </div>
      )}
    </main>
  )
}

