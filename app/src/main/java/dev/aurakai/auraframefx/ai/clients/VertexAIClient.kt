package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the provided text prompt using Vertex AI.
 *
 * @param prompt The input text prompt for content generation.
 * @return The generated content as a string, or null if content generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates code based on a given specification, programming language, and coding style.
 *
 * @param specification The requirements or description for the code to be generated.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Attempts to establish a connection to Vertex AI.
 *
 * @return `true` if the connection is established successfully, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    // Add other methods like startChat, listModels, etc. as needed
}
