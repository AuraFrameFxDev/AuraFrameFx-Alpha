package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text used to guide the content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text from a prompt with configurable maximum token length and temperature.
     *
     * @param prompt The input prompt to guide text generation.
     * @param maxTokens The maximum number of tokens to generate in the output.
     * @param temperature Controls the randomness of the generated text; higher values yield more diverse results.
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
 * @param specification Description of the functionality or requirements for the code to be generated.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to follow.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity and responsiveness of the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares and configures creative AI models in Vertex AI to enable content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt to guide the interpretation and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The textual instruction or question that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
