package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt using Vertex AI.
 *
 * @param prompt The input text prompt for content generation.
 * @return The generated content, or null if generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates code based on a given specification, target programming language, and coding style.
 *
 * @param specification The requirements or description for the code to be generated.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to apply to the generated code.
 * @return The generated code as a String, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies if a connection to Vertex AI can be established.
 *
 * @return `true` if the connection is successful; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    // Add other methods like startChat, listModels, etc. as needed
}
