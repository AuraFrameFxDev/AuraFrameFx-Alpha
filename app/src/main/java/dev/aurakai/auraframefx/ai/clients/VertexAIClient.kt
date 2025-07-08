package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates AI-driven content based on the provided text prompt.
 *
 * @param prompt The input text prompt used to guide content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces AI-generated text from a prompt, allowing customization of output length and creativity.
 *
 * @param prompt The input text to guide the AI's response.
 * @param maxTokens The maximum number of tokens in the generated output.
 * @param temperature Degree of randomness in the output; higher values yield more diverse results.
 * @return The generated text as a string.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the intended code functionality.
 * @param language The target programming language for the generated code.
 * @param style Coding style or conventions to apply to the generated code.
 * @return The generated source code as a string, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies the availability and responsiveness of the Vertex AI service.
 *
 * @return `true` if the service is reachable and operational; `false` if it is not.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models in Vertex AI for content generation.
 *
 * Call this function before using content generation methods to ensure all necessary models are initialized and ready.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using AI, guided by a textual prompt.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt providing context or instructions for the analysis.
 * @return The AI-generated analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
