package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using the provided prompt.
 *
 * @param prompt The input text that guides the content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text from a prompt with adjustable length and creativity.
     *
     * @param prompt The input text that guides the generation.
     * @param maxTokens The maximum number of tokens allowed in the generated output.
     * @param temperature The degree of randomness in the output; higher values yield more diverse results.
     * @return The generated text.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int = 1000,
        temperature: Float = 0.7f
    ): String

    /**
 * Generates source code based on a specification, target programming language, and coding style.
 *
 * @param specification Details the desired functionality or requirements for the generated code.
 * @param language Specifies the programming language for the output code.
 * @param style Indicates the coding style or conventions to follow.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service is accessible; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models in Vertex AI for content generation.
 *
 * Prepares the necessary models to enable creative content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data based on a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt that directs the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
