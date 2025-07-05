package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from the specified prompt using Vertex AI.
 *
 * @param prompt The input text prompt to guide content generation.
 * @return The generated content as a string, or null if generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates code according to the provided specification, programming language, and coding style.
 *
 * @param specification The description or requirements for the code to generate.
 * @param language The target programming language for the generated code.
 * @param style The desired coding style or conventions for the output.
 * @return The generated code as a string, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether a connection to Vertex AI can be successfully established.
 *
 * @return `true` if the connection is established; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    // Add other methods like startChat, listModels, etc. as needed
}
