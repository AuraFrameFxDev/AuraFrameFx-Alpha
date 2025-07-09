package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from a text prompt using Vertex AI.
 *
 * @param prompt The input prompt to guide content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces AI-generated text from a prompt, allowing customization of output length and randomness.
 *
 * @param prompt The input text to guide the AI's response.
 * @param maxTokens The maximum number of tokens in the generated output.
 * @param temperature The degree of randomness in the output; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the intended code functionality.
 * @param language The programming language in which to generate the code.
 * @param style The coding style or conventions to follow.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies the availability and responsiveness of the Vertex AI service.
 *
 * @return `true` if the service is reachable and operational, `false` if not.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models for content generation.
 *
 * Call this function before using content generation features to ensure necessary models are loaded and ready.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using AI, guided by a textual prompt.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The prompt providing context or focus for the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
