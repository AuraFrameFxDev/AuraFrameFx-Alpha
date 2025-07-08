package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using AI based on the provided text prompt.
 *
 * @param prompt The input text that guides the content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on a prompt, with options to control output length and creativity.
 *
 * @param prompt The input text that guides the content generation.
 * @param maxTokens The maximum number of tokens in the generated text.
 * @param temperature Controls the randomness of the output; higher values produce more varied results.
 * @return The generated text as a string.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target programming language, and coding style.
 *
 * @param specification The requirements or description guiding the code generation.
 * @param language The programming language in which the code should be generated.
 * @param style The coding style or conventions to follow.
 * @return The generated source code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service responds successfully; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models in Vertex AI to enable content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Performs AI-driven analysis of image data guided by a text prompt.
 *
 * @param imageData The raw bytes representing the image to analyze.
 * @param prompt The textual instruction or question to guide the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
