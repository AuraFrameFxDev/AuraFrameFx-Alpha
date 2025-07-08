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
 * @return The generated content as a string, or null if content generation is unsuccessful.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Generates text based on the given prompt, with adjustable output length and creativity.
 *
 * @param prompt The prompt that guides the generated text.
 * @param maxTokens The maximum number of tokens in the generated output.
 * @param temperature Controls the randomness of the output; higher values produce more varied results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code according to the provided specification, programming language, and coding style.
 *
 * @param specification The requirements or description for the code to be generated.
 * @param language The target programming language for the generated code.
 * @param style The coding style or conventions to apply.
 * @return The generated source code, or null if code generation fails.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service can be contacted successfully, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models in Vertex AI for content generation.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data according to the given prompt and returns the analysis result.
 *
 * @param imageData The image data to analyze.
 * @param prompt The text prompt guiding the analysis.
 * @return The result of the image analysis.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
