package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates AI-driven content from a given text prompt.
 *
 * @param prompt The input text used to guide content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces AI-generated text from a prompt, allowing customization of output length and randomness.
 *
 * @param prompt The input text to guide the generation.
 * @param maxTokens The maximum number of tokens in the generated output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to a given specification, programming language, and coding style.
 *
 * @param specification The description or requirements that define the desired code functionality.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to apply to the generated code.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies the availability and responsiveness of the Vertex AI service.
 *
 * @return `true` if the service is accessible and responsive, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models within Vertex AI for content generation tasks.
 *
 * This function must be called before using content generation features to ensure the necessary models are properly initialized and configured.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using AI based on a provided textual prompt.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt guiding the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
