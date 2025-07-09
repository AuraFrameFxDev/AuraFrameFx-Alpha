package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content using the provided text prompt.
 *
 * @param prompt The text prompt that guides content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text from a prompt with configurable maximum token length and output randomness.
 *
 * @param prompt The text prompt to guide the generation.
 * @param maxTokens The maximum number of tokens allowed in the generated text.
 * @param temperature The degree of randomness in the output; higher values yield more diverse results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, programming language, and coding style.
 *
 * @param specification The requirements or description for the code to be generated.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code as a string, or null if generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service is available; `false` if the connection fails.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models within Vertex AI for content generation tasks.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data using a text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt guiding the image analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
