package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text from a prompt with configurable length and creativity.
     *
     * @param prompt The input prompt for text generation.
     * @param maxTokens The maximum number of tokens in the generated output.
     * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
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
 * @param specification Description of the functionality or features the generated code should implement.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated source code, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies the availability and responsiveness of the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI to enable content generation features.
 *
 * Prepares and configures the required models for subsequent creative tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt to guide the interpretation and returns the analysis result.
 *
 * @param imageData Raw bytes of the image to be analyzed.
 * @param prompt Text prompt providing context or direction for the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
