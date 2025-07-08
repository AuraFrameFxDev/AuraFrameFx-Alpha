package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text that guides the content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text using the provided prompt, allowing customization of output length and creativity.
     *
     * @param prompt The input prompt that guides the generated text.
     * @param maxTokens The maximum number of tokens to generate in the output.
     * @param temperature Controls the randomness and creativity of the generated text; higher values yield more diverse results.
     * @return The generated text.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int = 1000,
        temperature: Float = 0.7f
    ): String

    /**
 * Generates source code based on a specification, programming language, and coding style.
 *
 * @param specification Description of the functionality or requirements for the code to be generated.
 * @param language The programming language in which the code should be written.
 * @param style The coding style or conventions to follow.
 * @return The generated source code, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies whether the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service is reachable and operational, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares and configures creative AI models within Vertex AI to enable content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes raw image data using a guiding text prompt and returns the analysis result as a string.
 *
 * @param imageData The image data to be analyzed.
 * @param prompt The prompt that guides the analysis process.
 * @return The analysis result.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
