package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
     * Generates content based on a given prompt.
     * @param prompt The input prompt.
     * @return The generated content as a String, or null on failure.
     * TODO: Needs actual implementation in implementing classes.
     */
    suspend fun generateContent(prompt: String): String?

    // Add other methods like startChat, listModels, etc. as needed
}
