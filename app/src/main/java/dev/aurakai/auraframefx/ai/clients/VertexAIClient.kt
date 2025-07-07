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
 * Generates text based on the given prompt, with options to control output length and creativity.
 *
 * @param prompt The input prompt that guides the text generation.
 * @param maxTokens The maximum number of tokens to include in the generated text.
 * @param temperature Controls the randomness of the generated output; higher values produce more creative results.
 * @return The generated text as a String.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code based on a given specification, programming language, and coding style.
 *
 * @param specification Description of the desired code functionality or requirements.
 * @param language The programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated code as a String, or null if code generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks whether a connection to Vertex AI can be established.
 *
 * @return `true` if the connection is successful, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Prepares creative models for use in content generation tasks.
 *
 * This function initializes the necessary models to enable creative content generation with Vertex AI.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes image data using a prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The prompt that guides the analysis process.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
