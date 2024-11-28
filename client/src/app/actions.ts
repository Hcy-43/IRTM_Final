'use server'

export async function processInput(input: string) {
  // Simulate a delay to mimic backend processing
  await new Promise(resolve => setTimeout(resolve, 2000))
  
  // Simple processing: reverse the input string
  return input.split('').reverse().join('')
}
