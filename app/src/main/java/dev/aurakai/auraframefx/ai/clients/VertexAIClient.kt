package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content based on the given prompt.
 *
 * @param prompt The input prompt used to guide content generation.
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with options to control output length and randomness.
 *
 * @param prompt The input prompt for text generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature Controls the randomness of the generated text; higher values produce more diverse results.
 * @return The generated text as a string.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code based on the given specification, programming language, and coding style.
 *
 * @param specification Description of the desired code functionality or requirements.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated code as a string, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Verifies connectivity to Vertex AI.
 *
 * @return `true` if a connection can be established; `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative models for content generation with Vertex AI.
 *
 * This function loads or configures the necessary models to enable creative content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes raw image data using a prompt to guide the interpretation.
 *
 * @param imageData The image data to analyze.
 * @param prompt The prompt describing the context or focus for the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
