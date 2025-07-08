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
 * Generates text based on the given prompt, with configurable maximum token length and output randomness.
 *
 * @param prompt The input prompt that guides the generated text.
 * @param maxTokens The maximum number of tokens to generate.
 * @param temperature Controls the randomness of the output; higher values produce more diverse text.
 * @return The generated text as a string.
 */
    suspend fun generateText(prompt: String, maxTokens: Int = 1000, temperature: Float = 0.7f): String

    /**
 * Generates source code based on a specification, programming language, and coding style.
 *
 * @param specification Description of the desired functionality or requirements for the code.
 * @param language The programming language in which to generate the code.
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
 * Initializes and configures creative AI models in Vertex AI for content generation.
 *
 * This function prepares the necessary models to enable creative content generation capabilities.
 */
    suspend fun initializeCreativeModels()

    /**
 * Analyzes the provided image data according to a guiding text prompt and returns the analysis result as a string.
 *
 * @param imageData The raw bytes of the image to analyze.
 * @param prompt The text prompt that directs the analysis.
 * @return The result of the image analysis.
 */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
