package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given text prompt.
 *
 * @param prompt The input text prompt for content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from the provided prompt with configurable maximum token count and temperature.
 *
 * @param prompt The text prompt to guide content generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on the provided specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the code.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies whether the Vertex AI service is accessible.
 *
 * @return `true` if the service is accessible; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a guiding text prompt and returns the analysis result as a string.
 *
 * @param imageData The raw bytes of the image to be analyzed.
 * @param prompt The text prompt that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
