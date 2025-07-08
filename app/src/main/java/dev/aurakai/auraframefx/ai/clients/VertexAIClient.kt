package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input text that guides the content generation.
 * @return The generated content as a string, or null if generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from a prompt with configurable maximum token length and temperature for output randomness.
 *
 * @param prompt The input prompt to guide text generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature The degree of randomness in the generated text; higher values yield more diverse results.
 * @return The generated text as a string.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target programming language, and coding style.
 *
 * @param specification Description or requirements guiding the code generation.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated source code, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies whether the Vertex AI service is accessible and responsive.
 *
 * @return `true` if the service responds successfully; `false` if it is unreachable or unresponsive.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a text prompt and returns the analysis result as a string.
 *
 * @param imageData Raw image data to be analyzed.
 * @param prompt Text prompt guiding the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
