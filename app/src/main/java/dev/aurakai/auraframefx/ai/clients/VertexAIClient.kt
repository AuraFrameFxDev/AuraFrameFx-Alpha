package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text used to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text from the provided prompt, allowing customization of output length and randomness.
     *
     * @param prompt The input prompt to guide text generation.
     * @param maxTokens The maximum number of tokens to generate in the output.
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
 * @param specification The requirements or description for the code to be generated.
 * @param language The target programming language.
 * @param style The coding style or conventions to follow.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks the connectivity and responsiveness of the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive; `false` if the connection fails.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI to enable content generation features.
 *
 * Prepares and configures the required models for creative content generation tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt and returns the analysis result as a string.
 *
 * @param imageData The image data to be analyzed.
 * @param prompt The prompt guiding the analysis.
 * @return The analysis result.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
