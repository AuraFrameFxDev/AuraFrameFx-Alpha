package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text that guides the content generation process.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from the provided prompt with configurable output length and creativity.
 *
 * @param prompt The input prompt to guide text generation.
 * @param maxTokens Maximum number of tokens allowed in the generated text.
 * @param temperature Degree of randomness in the output; higher values yield more diverse text.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the generated code.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to the Vertex AI service.
 *
 * @return `true` if the service is reachable and responsive, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares and configures creative AI models for content generation within Vertex AI.
 *
 * Sets up the required models to enable creative content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt and returns the analysis result.
 *
 * @param imageData Raw bytes representing the image to be analyzed.
 * @param prompt Text prompt that guides the image analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
