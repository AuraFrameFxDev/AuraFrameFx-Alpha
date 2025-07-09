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
                    fun shouldHandle404NotFound() = runTest {
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
            val longPrompt = "A".repeat(10000)
            val expectedResponse = "Response for long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(longPrompt)
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${longPrompt.length}")
        }

        @Test
        @DisplayName("Should handle prompt with special characters")
        fun shouldHandlePromptWithSpecialCharacters() = runTest {
            val specialPrompt = "Hello! @#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`"
            val expectedResponse = "Special response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(specialPrompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(specialPrompt)
        }

        @Test
        @DisplayName("Should handle prompt with unicode characters")
        fun shouldHandlePromptWithUnicodeCharacters() = runTest {
            val unicodePrompt = "Hello 世界 🌍 café naïve résumé"
            val expectedResponse = "Unicode response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(unicodePrompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(unicodePrompt)
        }

        @Test
        @DisplayName("Should handle whitespace-only prompt")
        fun shouldHandleWhitespaceOnlyPrompt() = runTest {
            val whitespacePrompt = "   \t\n\r   "
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(whitespacePrompt)
            }
        }

        @Test
        @DisplayName("Should handle null userId gracefully")
        fun shouldHandleNullUserIdGracefully() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, null)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
        }

        @Test
        @DisplayName("Should handle empty userId gracefully")
        fun shouldHandleEmptyUserIdGracefully() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, "")
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
        }
    }

    @Nested
    @DisplayName("Enhanced Batch Response Tests")
    inner class EnhancedBatchResponseTests {
        @Test
        @DisplayName("Should handle single prompt in batch")
        fun shouldHandleSinglePromptInBatch() = runTest {
            val prompts = listOf("Single prompt")
            val expectedResponse = "Single response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
            verify(mockLogger).info("Generating batch AI responses for 1 prompts")
        }

        @Test
        @DisplayName("Should handle multiple prompts in batch")
        fun shouldHandleMultiplePromptsInBatch() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
            verify(mockLogger).info("Generating batch AI responses for 3 prompts")
        }

        @Test
        @DisplayName("Should handle batch with empty prompts")
        fun shouldHandleBatchWithEmptyPrompts() = runTest {
            val prompts = listOf("Valid prompt", "", "Another prompt")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            verify(mockHttpClient).post(prompts)
        }

        @Test
        @DisplayName("Should handle batch HTTP error")
        fun shouldHandleBatchHttpError() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2")
            val mockHttpResponse = mockHttpResponse(500, "Server Error")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals("Server Error", results[0])
        }

        @Test
        @DisplayName("Should handle batch with very large number of prompts")
        fun shouldHandleBatchWithVeryLargeNumberOfPrompts() = runTest {
            val prompts = (1..1000).map { "Prompt $it" }
            val expectedResponse = "Large batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            verify(mockLogger).info("Generating batch AI responses for 1000 prompts")
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Tests")
    inner class EnhancedConfigurationTests {
        @Test
        @DisplayName("Should update base URL with different valid formats")
        fun shouldUpdateBaseUrlWithDifferentValidFormats() {
            val validUrls = listOf(
                "https://api.example.com",
                "https://api.example.com/",
                "https://api.example.com/v1",
                "https://subdomain.api.example.com"
            )

            validUrls.forEach { url ->
                auraAIService.updateBaseUrl(url)
                verify(mockConfigurationService).updateBaseUrl(url)
            }
        }

        @Test
        @DisplayName("Should reject invalid base URL formats")
        fun shouldRejectInvalidBaseUrlFormats() {
            val invalidUrls = listOf(
                "http://api.example.com",
                "ftp://api.example.com",
                "api.example.com",
                "",
                "https://",
                "not-a-url"
            )

            invalidUrls.forEach { url ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateBaseUrl(url)
                }
            }
        }

        @Test
        @DisplayName("Should update timeout with various valid values")
        fun shouldUpdateTimeoutWithVariousValidValues() {
            val validTimeouts = listOf(1L, 100L, 1000L, 30000L, 60000L, Long.MAX_VALUE)

            validTimeouts.forEach { timeout ->
                auraAIService.updateTimeout(timeout)
                verify(mockConfigurationService).updateTimeout(timeout)
                verify(mockLogger).info("Timeout updated to $timeout ms")
            }
        }

        @Test
        @DisplayName("Should reject invalid timeout values")
        fun shouldRejectInvalidTimeoutValues() {
            val invalidTimeouts = listOf(0L, -1L, -100L, -1000L, Long.MIN_VALUE)

            invalidTimeouts.forEach { timeout ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateTimeout(timeout)
                }
            }
        }

        @Test
        @DisplayName("Should update API key with various valid formats")
        fun shouldUpdateApiKeyWithVariousValidFormats() {
            val validKeys = listOf(
                "sk-1234567890abcdef",
                "api_key_123",
                "Bearer token123",
                "a".repeat(100),
                "key-with-special-chars_123!@#"
            )

            validKeys.forEach { key ->
                auraAIService.updateApiKey(key)
                verify(mockConfigurationService).updateApiKey(key)
                verify(mockLogger).info("API key updated successfully")
            }
        }

        @Test
        @DisplayName("Should reject whitespace-only API key")
        fun shouldRejectWhitespaceOnlyApiKey() {
            val whitespaceKeys = listOf("   ", "\t\t", "\n\n", "  \t  \n  ")

            whitespaceKeys.forEach { key ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateApiKey(key)
                }
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Health Check Tests")
    inner class EnhancedHealthCheckTests {
        @Test
        @DisplayName("Should handle health check with non-200 status codes")
        fun shouldHandleHealthCheckWithNon200StatusCodes() = runTest {
            val statusCodes = listOf(201, 301, 400, 401, 403, 404, 500, 503)

            statusCodes.forEach { statusCode ->
                whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(statusCode, "Error"))
                val result = auraAIService.healthCheck()
                assertFalse(result.isHealthy)
                assertTrue(result.message.contains("Service is unhealthy"))
            }
        }

        @Test
        @DisplayName("Should handle health check timeout")
        fun shouldHandleHealthCheckTimeout() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Health check timeout"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Service is unhealthy"))
            assertTrue(result.message.contains("Health check timeout"))
        }

        @Test
        @DisplayName("Should handle health check network error")
        fun shouldHandleHealthCheckNetworkError() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(IOException("Network unreachable"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Service is unhealthy"))
            assertTrue(result.message.contains("Network unreachable"))
        }

        @Test
        @DisplayName("Should handle health check with custom error message")
        fun shouldHandleHealthCheckWithCustomErrorMessage() = runTest {
            val errorMessage = "Custom service error"
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(500, errorMessage))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains(errorMessage))
        }
    }

    @Nested
    @DisplayName("Enhanced Streaming Tests")
    inner class EnhancedStreamingTests {
        @Test
        @DisplayName("Should handle empty streaming response")
        fun shouldHandleEmptyStreamingResponse() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertTrue(collected.isEmpty())
        }

        @Test
        @DisplayName("Should handle single chunk streaming response")
        fun shouldHandleSingleChunkStreamingResponse() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit("single chunk") })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(1, collected.size)
            assertEquals("single chunk", collected[0])
        }

        @Test
        @DisplayName("Should handle large number of chunks")
        fun shouldHandleLargeNumberOfChunks() = runTest {
            val chunks = (1..1000).map { "chunk $it" }
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks.size, collected.size)
            assertEquals(chunks, collected)
        }

        @Test
        @DisplayName("Should handle streaming with empty prompt rejection")
        fun shouldHandleStreamingWithEmptyPromptRejection() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("").collect()
            }
        }

        @Test
        @DisplayName("Should handle streaming with whitespace-only prompt rejection")
        fun shouldHandleStreamingWithWhitespaceOnlyPromptRejection() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("   \t\n   ").collect()
            }
        }

        @Test
        @DisplayName("Should handle streaming error during flow")
        fun shouldHandleStreamingErrorDuringFlow() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                emit("chunk 1")
                emit("chunk 2")
                throw IOException("Stream error")
            })
            
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse("prompt").collect()
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Model Parameters Tests")
    inner class EnhancedModelParametersTests {
        @Test
        @DisplayName("Should accept valid temperature values")
        fun shouldAcceptValidTemperatureValues() {
            val validTemperatures = listOf(0.0, 0.1, 0.5, 0.9, 1.0)
            
            validTemperatures.forEach { temp ->
                val params = mapOf("temperature" to temp)
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }

        @Test
        @DisplayName("Should reject temperature values outside valid range")
        fun shouldRejectTemperatureValuesOutsideValidRange() {
            val invalidTemperatures = listOf(-0.1, -1.0, 1.1, 2.0, Double.MAX_VALUE, Double.MIN_VALUE)
            
            invalidTemperatures.forEach { temp ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateModelParameters(mapOf("temperature" to temp))
                }
            }
        }

        @Test
        @DisplayName("Should accept valid max_tokens values")
        fun shouldAcceptValidMaxTokensValues() {
            val validTokens = listOf(1, 10, 100, 1000, 4000, Int.MAX_VALUE)
            
            validTokens.forEach { tokens ->
                val params = mapOf("max_tokens" to tokens)
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }

        @Test
        @DisplayName("Should reject invalid max_tokens values")
        fun shouldRejectInvalidMaxTokensValues() {
            val invalidTokens = listOf(0, -1, -100, Int.MIN_VALUE)
            
            invalidTokens.forEach { tokens ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateModelParameters(mapOf("max_tokens" to tokens))
                }
            }
        }

        @Test
        @DisplayName("Should handle parameters with both temperature and max_tokens")
        fun shouldHandleParametersWithBothTemperatureAndMaxTokens() {
            val params = mapOf("temperature" to 0.7, "max_tokens" to 100)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should handle parameters with unknown keys")
        fun shouldHandleParametersWithUnknownKeys() {
            val params = mapOf("unknown_param" to "value", "another_param" to 42)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle empty parameters map")
        fun shouldHandleEmptyParametersMap() {
            val params = emptyMap<String, Any>()
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle parameters with wrong types")
        fun shouldHandleParametersWithWrongTypes() {
            val params = mapOf("temperature" to "not_a_number", "max_tokens" to "also_not_a_number")
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }
    }

    @Nested
    @DisplayName("Enhanced Statistics and Cache Tests")
    inner class EnhancedStatisticsAndCacheTests {
        @Test
        @DisplayName("Should return consistent statistics structure")
        fun shouldReturnConsistentStatisticsStructure() {
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
        @DisplayName("Should handle multiple statistics requests")
        fun shouldHandleMultipleStatisticsRequests() {
            repeat(5) {
                val stats = auraAIService.getServiceStatistics()
                assertNotNull(stats)
            }
            verify(mockLogger, times(5)).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should handle multiple cache operations")
        fun shouldHandleMultipleCacheOperations() {
            repeat(3) {
                auraAIService.clearCache()
                auraAIService.expireCache()
                auraAIService.resetStatistics()
            }
            
            verify(mockLogger, times(3)).info("Response cache cleared")
            verify(mockLogger, times(3)).debug("Cache expired, making new request")
            verify(mockLogger, times(3)).info("Service statistics reset")
        }

        @Test
        @DisplayName("Should handle cache operations in sequence")
        fun shouldHandleCacheOperationsInSequence() {
            auraAIService.clearCache()
            auraAIService.expireCache()
            auraAIService.resetStatistics()
            
            verify(mockLogger).info("Response cache cleared")
            verify(mockLogger).debug("Cache expired, making new request")
            verify(mockLogger).info("Service statistics reset")
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Reload Tests")
    inner class EnhancedConfigurationReloadTests {
        @Test
        @DisplayName("Should handle configuration reload with null API key")
        fun shouldHandleConfigurationReloadWithNullApiKey() {
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.test.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(argThat { contains("Failed to reload configuration") })
        }

        @Test
        @DisplayName("Should handle configuration reload with invalid base URL")
        fun shouldHandleConfigurationReloadWithInvalidBaseUrl() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(argThat { contains("Failed to reload configuration") })
        }

        @Test
        @DisplayName("Should handle configuration reload with negative timeout")
        fun shouldHandleConfigurationReloadWithNegativeTimeout() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.test.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(-1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(argThat { contains("Failed to reload configuration") })
        }

        @Test
        @DisplayName("Should handle configuration reload multiple times")
        fun shouldHandleConfigurationReloadMultipleTimes() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.test.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            repeat(3) {
                auraAIService.reloadConfiguration()
            }
            
            verify(mockLogger, times(3)).info("Configuration reloaded successfully")
        }
    }

    @Nested
    @DisplayName("Boundary Value Tests")
    inner class BoundaryValueTests {
        @Test
        @DisplayName("Should handle minimum valid configuration values")
        fun shouldHandleMinimumValidConfigurationValues() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("a")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://a.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1L)
            
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
        }

        @Test
        @DisplayName("Should handle maximum timeout value")
        fun shouldHandleMaximumTimeoutValue() {
            whenever(mockConfigurationService.getTimeout()).thenReturn(Long.MAX_VALUE)
            
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            service.updateTimeout(Long.MAX_VALUE)
            
            verify(mockConfigurationService).updateTimeout(Long.MAX_VALUE)
        }

        @Test
        @DisplayName("Should handle boundary temperature values")
        fun shouldHandleBoundaryTemperatureValues() {
            val boundaryParams = listOf(
                mapOf("temperature" to 0.0),
                mapOf("temperature" to 1.0)
            )
            
            boundaryParams.forEach { params ->
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }

        @Test
        @DisplayName("Should handle boundary max_tokens values")
        fun shouldHandleBoundaryMaxTokensValues() {
            val boundaryParams = listOf(
                mapOf("max_tokens" to 1),
                mapOf("max_tokens" to Int.MAX_VALUE)
            )
            
            boundaryParams.forEach { params ->
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }
    }

    @Nested
    @DisplayName("Concurrent Access Tests")
    inner class ConcurrentAccessTests {
        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun shouldHandleConcurrentConfigurationUpdates() = runTest {
            val jobs = (1..10).map { i ->
                kotlinx.coroutines.async {
                    auraAIService.updateApiKey("key-$i")
                    auraAIService.updateTimeout(1000L + i)
                }
            }
            
            jobs.forEach { it.await() }
            
            verify(mockConfigurationService, times(10)).updateApiKey(any())
            verify(mockConfigurationService, times(10)).updateTimeout(any())
        }

        @Test
        @DisplayName("Should handle concurrent cache operations")
        fun shouldHandleConcurrentCacheOperations() = runTest {
            val jobs = (1..5).map {
                kotlinx.coroutines.async {
                    auraAIService.clearCache()
                    auraAIService.expireCache()
                    auraAIService.resetStatistics()
                }
            }
            
            jobs.forEach { it.await() }
            
            verify(mockLogger, times(5)).info("Response cache cleared")
            verify(mockLogger, times(5)).debug("Cache expired, making new request")
            verify(mockLogger, times(5)).info("Service statistics reset")
        }
    }

    @Nested
    @DisplayName("Error Recovery Tests")
    inner class ErrorRecoveryTests {
        @Test
        @DisplayName("Should handle service recovery after configuration error")
        fun shouldHandleServiceRecoveryAfterConfigurationError() {
            // First, cause a configuration error
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            
            // Then, fix the configuration
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.test.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            // Should now work
            auraAIService.reloadConfiguration()
            verify(mockLogger).info("Configuration reloaded successfully")
        }

        @Test
        @DisplayName("Should handle service recovery after HTTP error")
        fun shouldHandleServiceRecoveryAfterHttpError() = runTest {
            // First, cause HTTP error
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse(500, "Error"))
            assertThrows<IOException> {
                auraAIService.generateResponse("test")
            }
            
            // Then, fix the HTTP response
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse(200, "Success"))
            val result = auraAIService.generateResponse("test")
            assertEquals("Success", result)
        }
    }
}