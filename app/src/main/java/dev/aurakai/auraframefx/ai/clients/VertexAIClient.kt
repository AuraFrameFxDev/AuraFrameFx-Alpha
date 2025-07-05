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

    /**
     * Generates code based on specification.
     * @param specification The specification for the code to generate.
     * @param language The programming language to use.
     * @param style The coding style to apply.
     * @return The generated code as a String, or null on failure.
     */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
     * Validates the connection to Vertex AI.
     * @return true if connected successfully, false otherwise.
     */
    suspend fun validateConnection(): Boolean

    // Add other methods like startChat, listModels, etc. as needed
}
