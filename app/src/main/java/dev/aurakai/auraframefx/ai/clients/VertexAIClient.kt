package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using the provided text prompt.
 *
 * @param prompt The text input that guides the content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces text from the provided prompt, allowing customization of output length and randomness.
 *
 * @param prompt The input text prompt that guides the text generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to a specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality for the generated code.
 * @param language The target programming language for code generation.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service is accessible; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models within Vertex AI for use in content generation tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data based on a guiding text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that directs the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
