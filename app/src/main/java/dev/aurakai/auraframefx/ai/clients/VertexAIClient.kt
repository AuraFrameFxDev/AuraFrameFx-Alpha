package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from a text prompt using Vertex AI.
 *
 * @param prompt The input text prompt to generate content from.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates code according to the provided specification, programming language, and coding style.
 *
 * @param specification The description or requirements for the desired code.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether a connection to Vertex AI can be established.
 *
 * @return `true` if the connection is successful, or `false` if it fails.
 */
    suspend fun validateConnection(): Boolean

    // Add other methods like startChat, listModels, etc. as needed
}
