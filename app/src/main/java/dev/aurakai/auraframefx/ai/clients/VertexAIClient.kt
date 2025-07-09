package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given text prompt.
 *
 * @param prompt The text prompt to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on a prompt, with options to control output length and creativity.
 *
 * @param prompt The text prompt to guide text generation.
 * @param maxTokens Maximum number of tokens allowed in the generated text.
 * @param temperature Controls randomness in the output; higher values produce more varied results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to apply to the generated code.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies whether the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service can be reached; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models in Vertex AI for use in content generation operations.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data based on a guiding text prompt and returns the analysis result.
 *
 * @param imageData Raw bytes of the image to be analyzed.
 * @param prompt Text prompt that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
