package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
 * Generates content in response to the provided prompt.
 *
 * @param prompt The input text used to guide content generation.
 * @return The generated content as a string, or null if generation fails.
 */
    suspend fun generateContent(prompt: String): String?

    /**
 * Produces text based on the given prompt, allowing customization of output length and creativity.
 *
 * @param prompt The input text that guides the content generation.
 * @param maxTokens The maximum number of tokens to generate in the output.
 * @param temperature Controls the randomness of the generated text; higher values produce more varied results.
 * @return The generated text.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, target programming language, and coding style.
 *
 * @param specification The functional requirements or description for the code to be generated.
 * @param language The programming language in which the code should be written.
 * @param style The coding style or conventions to follow.
 * @return The generated source code as a string, or null if generation is unsuccessful.
 */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
 * Checks if the Vertex AI service is reachable and responsive.
 *
 * @return `true` if the service can be contacted successfully, `false` otherwise.
 */
    suspend fun validateConnection(): Boolean

    /**
 * Initializes and configures creative AI models in Vertex AI for content generation tasks.
 *
 * This method prepares the necessary models to support creative content generation features.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data according to the given text prompt and returns the analysis result.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt guiding the analysis.
 * @return The result of the image analysis as a string.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
