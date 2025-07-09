package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on a text prompt.
 *
 * @param prompt The text prompt to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from a prompt with configurable maximum length and randomness.
 *
 * @param prompt The text prompt that guides the generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the code.
 * @param language The programming language in which to generate the code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt and returns the analysis result as a string.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt guiding the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
