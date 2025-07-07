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
 * Produces text from the provided prompt, allowing customization of output length and creativity.
 *
 * @param prompt The input text that guides the generation.
 * @param maxTokens Maximum number of tokens in the generated output.
 * @param temperature Degree of randomness in the output; higher values yield more varied results.
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
 * @return `true` if the service is reachable and responsive, `false` if the connection fails.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative AI models for content generation features.
 *
 * Sets up and configures the necessary models to enable creative content generation capabilities in Vertex AI.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt and returns the analysis result as a string.
 *
 * @param imageData Raw bytes representing the image to be analyzed.
 * @param prompt Text prompt guiding the image analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
