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
 * @return The generated content as a String, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with options to control output length and randomness.
 *
 * @param prompt The input prompt that guides the text generation.
 * @param maxTokens The maximum number of tokens to include in the generated text.
 * @param temperature Controls the randomness of the generated output; higher values produce more diverse results.
 * @return The generated text as a String.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code according to the provided specification, programming language, and coding style.
 *
 * @param specification Description of the code requirements or desired functionality.
 * @param language The programming language in which to generate the code.
 * @param style The coding style or conventions to apply.
 * @return The generated code as a String, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether a connection to Vertex AI can be established.
 *
 * @return `true` if the connection is successful, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares and initializes creative models required for content generation with Vertex AI.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a prompt and returns the analysis result.
 *
 * @param imageData Raw image data to be analyzed.
 * @param prompt Text prompt that guides the analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
