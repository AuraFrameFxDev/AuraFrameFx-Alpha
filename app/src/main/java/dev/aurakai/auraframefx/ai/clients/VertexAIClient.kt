package dev.aurakai.auraframefx.ai.clients

/**
 * Interface for a Vertex AI client.
 * TODO: Define methods for interacting with Vertex AI, e.g., content generation, chat.
 */
interface VertexAIClient {
    /**
     * Generates content using the provided prompt.
     *
     * @param prompt The input text that guides the content generation process.
     * @return The generated content as a string, or null if content generation is unsuccessful.
     */
    suspend fun generateContent(prompt: String): String?

    /**
     * Generates text based on the given prompt, with options to control output length and creativity.
     *
     * @param prompt The input prompt to guide text generation.
     * @param maxTokens Maximum number of tokens to include in the generated text.
     * @param temperature Controls the randomness of the output; higher values produce more varied results.
     * @return The generated text.
     */
    suspend fun generateText(
        prompt: String,
        maxTokens: Int = 1000,
        temperature: Float = 0.7f
    ): String

    /**
     * Generates source code according to a given specification, target programming language, and coding style.
     *
     * @param specification Details the desired functionality or features for the generated code.
     * @param language Specifies the programming language in which the code should be generated.
     * @param style Indicates the coding style or conventions to apply.
     * @return The generated source code as a string, or null if code generation is unsuccessful.
     */
    suspend fun generateCode(specification: String, language: String, style: String): String?

    /**
     * Checks if the Vertex AI service is reachable and responsive.
     *
     * @return `true` if the service responds successfully; `false` if it is unreachable or unresponsive.
     */
    suspend fun validateConnection(): Boolean

    /**
     * Initializes and configures creative AI models in Vertex AI for content generation.
     *
     * This function sets up the necessary models and resources to enable creative content generation capabilities.
     */
    suspend fun initializeCreativeModels()

    /**
     * Analyzes the provided image data based on a guiding text prompt and returns the analysis result.
     *
     * @param imageData The raw bytes of the image to analyze.
     * @param prompt A text prompt that directs or contextualizes the image analysis.
     * @return The result of the image analysis as a string.
     */
    suspend fun analyzeImage(imageData: ByteArray, prompt: String): String

    // Add other methods like startChat, listModels, etc. as needed
}
