package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content from the provided prompt.
 *
 * @param prompt The input text to guide content generation.
 * @return The generated content, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with options to control output length and randomness.
 *
 * @param prompt The input prompt for text generation.
 * @param maxTokens The maximum number of tokens in the generated text.
 * @param temperature Controls the randomness of the output; higher values produce more varied results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the given specification, programming language, and coding style.
 *
 * @param specification The requirements or description for the code to generate.
 * @param language The target programming language.
 * @param style The desired coding style or conventions.
 * @return The generated code, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the connection is successful, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes creative AI models in Vertex AI for content generation.
 *
 * Sets up and configures models required to enable creative generation capabilities.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data according to the given prompt and returns the analysis result.
 *
 * @param imageData The binary data of the image to analyze.
 * @param prompt The text prompt that guides the analysis.
 * @return The result of the image analysis.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
