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
 * Generates text from the provided prompt using configurable maximum token count and temperature.
 *
 * @param prompt The input text prompt to guide text generation.
 * @param maxTokens The maximum number of tokens to generate in the response.
 * @param temperature Sampling temperature to control randomness in generation.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates code based on the given specification, programming language, and coding style.
 *
 * @param specification The requirements or description for the code to be generated.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to follow.
 * @return The generated code as a string, or null if generation is unsuccessful.
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
 * This function must be called before generating creative content to ensure required models are initialized.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes raw image data according to the provided prompt and returns the analysis result.
 *
 * @param imageData Raw image data to be analyzed.
 * @param prompt Instruction or question guiding the image analysis.
 * @return The analysis result as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
