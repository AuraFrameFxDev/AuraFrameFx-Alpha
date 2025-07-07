package dev.aurakai.auraframefx.ai.clients

import android.content.Context
import com.google.cloud.vertexai.VertexAI
import com.google.cloud.vertexai.api.GenerateContentResponse
import com.google.cloud.vertexai.generativeai.GenerativeModel
import com.google.cloud.vertexai.generativeai.ContentMaker.fromText
import dev.aurakai.auraframefx.ai.VertexAIConfig
import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.ai.services.VisionAnalysis
import kotlinx.coroutines.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Implementation of VertexAI client for AuraFrameFX.
 * Provides secure, scalable access to Google Cloud Vertex AI services.
 * Implements Kai's "Secure by Design" principles with comprehensive monitoring.
 */
@Singleton
class VertexAIClientImpl @Inject constructor(
    private val config: VertexAIConfig,
    private val context: Context,
    private val securityContext: SecurityContext,
    private val logger: AuraFxLogger
) : VertexAIClient {

    private var vertexAI: VertexAI? = null
    private var textModel: GenerativeModel? = null
    private var visionModel: GenerativeModel? = null
    private var codeModel: GenerativeModel? = null
    private var isInitialized = false

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    init {
        logger.info("VertexAIClient", "Initializing VertexAI client with config: ${config.modelName}")
        scope.launch {
            initializeClient()
        }
    }

    /**
     * Initializes the VertexAI client and generative models using the current configuration.
     *
     * Connects to Vertex AI with the specified project ID and location, then sets up models for text, vision, and code generation. Marks the client as initialized on success. Throws an exception if initialization fails.
     */
    private suspend fun initializeClient() {
        try {
            logger.info("VertexAIClient", "Setting up VertexAI connection")
            
            // Initialize VertexAI with project configuration
            vertexAI = VertexAI.Builder()
                .setProjectId(config.projectId)
                .setLocation(config.location)
                .build()

            // Initialize specialized models for different tasks
            initializeModels()
            
            isInitialized = true
            logger.info("VertexAIClient", "VertexAI client initialized successfully")
            
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Failed to initialize VertexAI client", e)
            throw e
        }
    }

    /**
     * Initializes the generative models for text, vision, and code tasks using the current Vertex AI client.
     *
     * Configures the `textModel`, `visionModel`, and `codeModel` fields with specialized model names if the Vertex AI client is available.
     */
    private fun initializeModels() {
        vertexAI?.let { vertex ->
            // Text generation model (Gemini Pro for general tasks)
            textModel = GenerativeModel.Builder()
                .setModelName("gemini-1.5-pro-002")
                .setVertexAI(vertex)
                .build()

            // Vision model for image analysis
            visionModel = GenerativeModel.Builder()
                .setModelName("gemini-1.5-pro-vision-001")
                .setVertexAI(vertex)
                .build()

            // Code generation model (optimized for programming tasks)
            codeModel = GenerativeModel.Builder()
                .setModelName("gemini-1.5-pro-002")
                .setVertexAI(vertex)
                .build()
                
            logger.info("VertexAIClient", "Specialized models initialized")
        }
    }

    /**
     * Validates connectivity to the Vertex AI service by attempting a test text generation.
     *
     * @return `true` if the service responds with non-empty text; `false` if the response is empty or an error occurs.
     */
    override suspend fun validateConnection(): Boolean {
        return try {
            ensureInitialized()
            
            logger.info("VertexAIClient", "Validating VertexAI connection")
            
            // Test connection with a simple request
            val testResponse = generateText(
                prompt = "Test connection",
                temperature = 0.1,
                maxTokens = 10
            )
            
            val isValid = testResponse.isNotEmpty()
            
            if (isValid) {
                logger.info("VertexAIClient", "VertexAI connection validated successfully")
            } else {
                logger.warn("VertexAIClient", "VertexAI connection validation failed")
            }
            
            isValid
            
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Connection validation failed", e)
            false
        }
    }

    /**
     * Initializes and verifies the responsiveness of creative AI models by generating a sample creative message.
     *
     * @throws Exception if model initialization or text generation fails.
     */
    override suspend fun initializeCreativeModels() {
        ensureInitialized()
        
        logger.info("VertexAIClient", "Initializing creative AI models")
        
        try {
            // Verify creative models are available and responsive
            val creativeTest = generateText(
                prompt = "Generate a creative hello message",
                temperature = 0.9,
                maxTokens = 50
            )
            
            logger.info("VertexAIClient", "Creative models initialized: $creativeTest")
            
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Failed to initialize creative models", e)
            throw e
        }
    }

    /**
     * Generates text from the given prompt using the Vertex AI text model with configurable generation parameters.
     *
     * The prompt is validated for security before generation. Throws if the prompt is invalid or if text generation fails.
     *
     * @param prompt The input text prompt to generate content from.
     * @param temperature Degree of randomness in the output; higher values yield more diverse results.
     * @param topP Nucleus sampling parameter; lower values restrict the set of possible next tokens.
     * @param maxTokens Maximum number of tokens in the generated output.
     * @param presencePenalty Penalizes new tokens based on their presence in the generated text so far.
     * @return The generated text content.
     * @throws SecurityException If the prompt fails security validation.
     * @throws Exception If text generation fails or the model is not initialized.
     */
    override suspend fun generateText(
        prompt: String,
        temperature: Double,
        topP: Double,
        maxTokens: Int,
        presencePenalty: Double
    ): String {
        ensureInitialized()
        
        return try {
            // Security validation
            securityContext.validatePrompt(prompt)
            
            logger.debug("VertexAIClient", "Generating text with temp=$temperature, tokens=$maxTokens")
            
            val model = textModel ?: throw IllegalStateException("Text model not initialized")
            
            // Configure generation parameters
            val generationConfig = com.google.cloud.vertexai.api.GenerationConfig.newBuilder()
                .setTemperature(temperature.toFloat())
                .setTopP(topP.toFloat())
                .setMaxOutputTokens(maxTokens)
                .build()

            // Generate content
            val response = model.generateContent(
                fromText(prompt),
                generationConfig
            )

            val generatedText = extractTextFromResponse(response)
            
            // Log successful generation
            logger.info("VertexAIClient", "Text generation completed successfully")
            
            generatedText
            
        } catch (e: SecurityException) {
            logger.warn("VertexAIClient", "Security violation in text generation", e)
            throw e
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Text generation failed", e)
            throw e
        }
    }

    /**
     * Analyzes JPEG image data using the Vertex AI vision model and returns a structured vision analysis.
     *
     * The analysis includes a detailed description, key visual elements, dominant colors, and emotional tone, extracted from the model's JSON-formatted response.
     *
     * @param imageData JPEG-encoded image data to analyze.
     * @return A [VisionAnalysis] object containing the structured analysis results.
     * @throws SecurityException If the image data fails security validation.
     * @throws Exception If image analysis fails or the model response cannot be parsed.
     */
    override suspend fun analyzeImage(imageData: ByteArray): VisionAnalysis {
        ensureInitialized()
        
        return try {
            // Security validation for image data
            securityContext.validateImageData(imageData)
            
            logger.info("VertexAIClient", "Analyzing image with vision model")
            
            val model = visionModel ?: throw IllegalStateException("Vision model not initialized")
            
            // Create image content
            val imageContent = com.google.cloud.vertexai.api.Content.newBuilder()
                .addParts(
                    com.google.cloud.vertexai.api.Part.newBuilder()
                        .setInlineData(
                            com.google.cloud.vertexai.api.Blob.newBuilder()
                                .setMimeType("image/jpeg")
                                .setData(com.google.protobuf.ByteString.copyFrom(imageData))
                        )
                )
                .build()

            val prompt = """
            Analyze this image and provide:
            1. A detailed description
            2. Key visual elements
            3. Dominant colors
            4. Emotional tone
            
            Format as JSON with fields: description, elements, colors, emotion
            """.trimIndent()

            // Generate analysis
            val response = model.generateContent(
                listOf(fromText(prompt), imageContent)
            )

            val analysisText = extractTextFromResponse(response)
            
            // Parse analysis into structured format
            val analysis = parseVisionAnalysis(analysisText)
            
            logger.info("VertexAIClient", "Image analysis completed successfully")
            analysis
            
        } catch (e: SecurityException) {
            logger.warn("VertexAIClient", "Security violation in image analysis", e)
            throw e
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Image analysis failed", e)
            throw e
        }
    }

    /**
     * Generates source code in the specified programming language and style according to the provided specification.
     *
     * The generated code follows best practices, includes comprehensive comments, and adheres to the requested coding style. Only the code is returned, without any explanations.
     *
     * @param specification The requirements or description for the code to be generated.
     * @param language The programming language for the generated code.
     * @param style The coding style or paradigm to be followed.
     * @return The generated source code as a string.
     * @throws SecurityException If the specification fails security validation.
     * @throws Exception If code generation fails or the model is not initialized.
     */
    override suspend fun generateCode(
        specification: String,
        language: String,
        style: String
    ): String {
        ensureInitialized()
        
        return try {
            securityContext.validatePrompt(specification)
            
            logger.info("VertexAIClient", "Generating $language code")
            
            val model = codeModel ?: throw IllegalStateException("Code model not initialized")
            
            val codePrompt = """
            Generate $language code with the following specifications:
            
            $specification
            
            Requirements:
            - Follow $style coding style
            - Include comprehensive comments
            - Use best practices and modern patterns
            - Ensure security and performance
            - Handle errors gracefully
            
            Generate only the code, no explanations.
            """.trimIndent()

            val response = model.generateContent(
                fromText(codePrompt),
                com.google.cloud.vertexai.api.GenerationConfig.newBuilder()
                    .setTemperature(0.6f) // Balanced creativity for code
                    .setTopP(0.9f)
                    .setMaxOutputTokens(2048)
                    .build()
            )

            val generatedCode = extractTextFromResponse(response)
            
            logger.info("VertexAIClient", "Code generation completed successfully")
            generatedCode
            
        } catch (e: SecurityException) {
            logger.warn("VertexAIClient", "Security violation in code generation", e)
            throw e
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Code generation failed", e)
            throw e
        }
    }

    /**
     * Generates text content from the given prompt using default generation parameters.
     *
     * This is a legacy method that delegates to `generateText` with standard values for temperature, topP, maxTokens, and presencePenalty.
     *
     * @param prompt The input prompt for text generation.
     * @return The generated text content.
     */
    override suspend fun generateContent(prompt: String): String {
        // Legacy method - delegates to generateText with default parameters
        return generateText(
            prompt = prompt,
            temperature = 0.7,
            topP = 0.9,
            maxTokens = 1024,
            presencePenalty = 0.0
        )
    }

    /**
     * Throws an exception if the VertexAI client is not initialized.
     *
     * @throws IllegalStateException if the client has not completed initialization.
     */

    private fun ensureInitialized() {
        if (!isInitialized) {
            throw IllegalStateException("VertexAI client not initialized")
        }
    }

    /**
     * Extracts the first text segment from the first candidate in a Vertex AI content generation response.
     *
     * @param response The response object returned by Vertex AI after content generation.
     * @return The first available text segment from the response.
     * @throws IllegalStateException If the response contains no candidates or text parts.
     * @throws Exception If extraction fails for any other reason.
     */
    private fun extractTextFromResponse(response: GenerateContentResponse): String {
        return try {
            response.candidatesList
                .firstOrNull()?.content?.partsList
                ?.firstOrNull()?.text
                ?: throw IllegalStateException("Empty response from VertexAI")
        } catch (e: Exception) {
            logger.error("VertexAIClient", "Failed to extract text from response", e)
            throw e
        }
    }

    /**
     * Parses a JSON-like string to create a VisionAnalysis object.
     *
     * Attempts to extract the description, elements, and colors fields from the input string. If extraction fails, returns a VisionAnalysis object with default fallback values.
     *
     * @param analysisText The JSON-like string containing vision analysis results.
     * @return A VisionAnalysis object populated with extracted or default values.
     */
    private fun parseVisionAnalysis(analysisText: String): VisionAnalysis {
        return try {
            // Parse JSON response into VisionAnalysis object
            // This is a simplified implementation - in production, use a proper JSON parser
            VisionAnalysis(
                description = extractJsonField(analysisText, "description") ?: "Unable to analyze image",
                elements = extractJsonList(analysisText, "elements") ?: emptyList(),
                colors = extractJsonList(analysisText, "colors") ?: emptyList(),
                confidence = 0.85f // Default confidence
            )
        } catch (e: Exception) {
            logger.warn("VertexAIClient", "Failed to parse vision analysis, using fallback", e)
            VisionAnalysis(
                description = "Image analysis unavailable",
                elements = emptyList(),
                colors = emptyList(),
                confidence = 0.0f
            )
        }
    }

    /**
     * Retrieves the value of a specified string field from a JSON-formatted string.
     *
     * Searches for the given field name and returns its string value if present; returns null if the field is not found.
     *
     * @param json The JSON string to search.
     * @param field The name of the field to extract.
     * @return The value of the field if found, or null otherwise.
     */
    private fun extractJsonField(json: String, field: String): String? {
        // Simple JSON field extraction - replace with proper JSON parsing in production
        val pattern = "\"$field\"\\s*:\\s*\"([^\"]+)\"".toRegex()
        return pattern.find(json)?.groupValues?.get(1)
    }

    /**
     * Extracts a list of string values from a specified array field in a JSON string.
     *
     * Uses a regular expression to locate and parse the array field. Returns a list of strings if the field is found and contains values; otherwise, returns null.
     *
     * This method does not support nested or complex JSON structures.
     *
     * @param json The JSON string to search.
     * @param field The name of the array field to extract.
     * @return A list of strings from the specified array field, or null if the field is not found.
     */
    private fun extractJsonList(json: String, field: String): List<String>? {
        // Simple JSON array extraction - replace with proper JSON parsing in production
        val pattern = "\"$field\"\\s*:\\s*\\[(.*?)\\]".toRegex()
        val match = pattern.find(json)?.groupValues?.get(1) ?: return null
        
        return match.split(",")
            .map { it.trim().removeSurrounding("\"") }
            .filter { it.isNotEmpty() }
    }

    /**
     * Cancels all ongoing operations and releases resources used by the VertexAI client.
     *
     * Terminates all coroutines in the client's scope and marks the client as uninitialized.
     */
    fun cleanup() {
        logger.info("VertexAIClient", "Cleaning up VertexAI client")
        scope.cancel()
        isInitialized = false
    }
}

/**
 * Interface defining VertexAI client capabilities
 */
interface VertexAIClient {
    /**
 * Tests connectivity to the Vertex AI service by generating a simple text response.
 *
 * @return `true` if a non-empty response is received from the service; `false` otherwise.
 */
suspend fun validateConnection(): Boolean
    /**
 * Initializes the creative generative models by producing a sample creative text.
 *
 * Ensures that the underlying AI models are ready for creative tasks by performing a test generation. Throws an exception if initialization or sample generation fails.
 */
suspend fun initializeCreativeModels()
    /**
     * Generates text using the Vertex AI text model with configurable generation parameters.
     *
     * Validates the prompt for security before generation. Supports customization of output randomness, diversity, and length via temperature, topP, maxTokens, and presencePenalty.
     *
     * @param prompt The input text prompt to generate a response for.
     * @param temperature Degree of randomness in output; higher values produce more diverse results.
     * @param topP Cumulative probability threshold for token selection.
     * @param maxTokens Maximum number of tokens in the generated output.
     * @param presencePenalty Penalty for repeated tokens to promote varied content.
     * @return The generated text response.
     * @throws SecurityException If the prompt fails security validation.
     * @throws Exception If text generation fails or the model is not initialized.
     */
    suspend fun generateText(
        prompt: String,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        maxTokens: Int = 1024,
        presencePenalty: Double = 0.0
    ): String
    /**
 * Analyzes JPEG image data using the Vertex AI vision model and returns a structured summary of the image.
 *
 * The analysis includes a textual description, a list of detected elements, and prominent colors identified in the image.
 *
 * @param imageData The JPEG image data to analyze.
 * @return A [VisionAnalysis] object containing the description, elements, and colors extracted from the image.
 * @throws SecurityException If the image data fails security validation.
 * @throws IllegalStateException If the Vertex AI client is not initialized.
 * @throws Exception If image analysis fails or the response cannot be parsed.
 */
suspend fun analyzeImage(imageData: ByteArray): VisionAnalysis
    /**
     * Generates source code based on a specification, programming language, and style using Vertex AI.
     *
     * Validates the specification for security, constructs a detailed prompt incorporating the requested language and style, and returns the generated source code as a string.
     *
     * @param specification The description or requirements for the code to be generated.
     * @param language The programming language for the generated code (default is "Kotlin").
     * @param style The coding style or conventions to apply (default is "Modern").
     * @return The generated source code as a string.
     * @throws SecurityException If the specification fails security validation.
     * @throws Exception If code generation fails or the AI response is invalid.
     */
    suspend fun generateCode(
        specification: String,
        language: String = "Kotlin",
        style: String = "Modern"
    ): String
    /**
 * Generates text content from the given prompt using default parameters.
 *
 * Delegates to the text generation model with standard settings to produce a response.
 *
 * @param prompt The input prompt for content generation.
 * @return The generated text content.
 */
suspend fun generateContent(prompt: String): String
}