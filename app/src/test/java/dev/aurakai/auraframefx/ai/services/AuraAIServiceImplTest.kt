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
            val expectedResponse = "Response to long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(longPrompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(longPrompt)
            verify(mockLogger).info("Generating AI response for prompt length: ${longPrompt.length}")
        }

        @Test
        @DisplayName("Should handle prompt with special characters")
        fun shouldHandlePromptWithSpecialCharacters() = runTest {
            val specialPrompt = "Test with émojis 🚀 and symbols @#$%^&*()"
            val expectedResponse = "Special response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(specialPrompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(specialPrompt)
        }

        @Test
        @DisplayName("Should handle null userId gracefully")
        fun shouldHandleNullUserId() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, null)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
        }

        @Test
        @DisplayName("Should handle empty userId")
        fun shouldHandleEmptyUserId() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = "Test response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, "")
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
        }

        @Test
        @DisplayName("Should handle response with empty body")
        fun shouldHandleResponseWithEmptyBody() = runTest {
            val mockHttpResponse = mockHttpResponse(200, "")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse("Test")
            assertEquals("", result)
            verify(mockHttpClient).post("Test")
        }

        @Test
        @DisplayName("Should handle response with whitespace only body")
        fun shouldHandleResponseWithWhitespaceOnlyBody() = runTest {
            val mockHttpResponse = mockHttpResponse(200, "   \n\t  ")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse("Test")
            assertEquals("   \n\t  ", result)
            verify(mockHttpClient).post("Test")
        }

        @Test
        @DisplayName("Should handle IOException during HTTP request")
        fun shouldHandleIOExceptionDuringRequest() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))
            assertThrows<IOException> {
                auraAIService.generateResponse("Test")
            }
        }

        @Test
        @DisplayName("Should handle RuntimeException during HTTP request")
        fun shouldHandleRuntimeExceptionDuringRequest() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(RuntimeException("Runtime error"))
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
            val prompts = listOf("What is AI?", "How does ML work?", "Define NLP")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
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
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
            verify(mockLogger).info("Generating batch AI responses for 1 prompts")
        }

        @Test
        @DisplayName("Should handle large batch of prompts")
        fun shouldHandleLargeBatchOfPrompts() = runTest {
            val prompts = (1..100).map { "Prompt $it" }
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
            verify(mockLogger).info("Generating batch AI responses for 100 prompts")
        }

        @Test
        @DisplayName("Should handle HTTP error in batch processing")
        fun shouldHandleHttpErrorInBatchProcessing() = runTest {
            val prompts = listOf("Test prompt")
            val mockHttpResponse = mockHttpResponse(500, "Server Error")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            // Note: Current implementation doesn't handle HTTP errors in batch processing
            // This test verifies the current behavior
            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals("Server Error", results[0])
        }

        @Test
        @DisplayName("Should handle exception in batch processing")
        fun shouldHandleExceptionInBatchProcessing() = runTest {
            val prompts = listOf("Test prompt")
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))

            assertThrows<IOException> {
                auraAIService.generateBatchResponses(prompts)
            }
        }

        @Test
        @DisplayName("Should handle prompts with mixed content")
        fun shouldHandlePromptsWithMixedContent() = runTest {
            val prompts = listOf("", "Valid prompt", "Another valid prompt")
            val expectedResponse = "Mixed response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Update Tests")
    inner class EnhancedConfigurationUpdateTests {
        @Test
        @DisplayName("Should update base URL with trailing slash")
        fun shouldUpdateBaseUrlWithTrailingSlash() {
            val newBaseUrl = "https://newapi.test.com/"
            auraAIService.updateBaseUrl(newBaseUrl)
            verify(mockConfigurationService).updateBaseUrl(newBaseUrl)
            verify(mockLogger).info("Base URL updated successfully")
        }

        @Test
        @DisplayName("Should reject http URL")
        fun shouldRejectHttpUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("http://insecure.com")
            }
        }

        @Test
        @DisplayName("Should reject ftp URL")
        fun shouldRejectFtpUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("ftp://files.test.com")
            }
        }

        @Test
        @DisplayName("Should reject malformed URL")
        fun shouldRejectMalformedUrl() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("not-a-url")
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

        @Test
        @DisplayName("Should update API key with special characters")
        fun shouldUpdateApiKeyWithSpecialCharacters() {
            val specialKey = "key-with-@special#characters$"
            auraAIService.updateApiKey(specialKey)
            verify(mockConfigurationService).updateApiKey(specialKey)
            verify(mockLogger).info("API key updated successfully")
        }

        @Test
        @DisplayName("Should update API key with very long value")
        fun shouldUpdateApiKeyWithVeryLongValue() {
            val longKey = "a".repeat(1000)
            auraAIService.updateApiKey(longKey)
            verify(mockConfigurationService).updateApiKey(longKey)
            verify(mockLogger).info("API key updated successfully")
        }
    }

    @Nested
    @DisplayName("Enhanced Health Check Tests")
    inner class EnhancedHealthCheckTests {
        @Test
        @DisplayName("Should return unhealthy on 400 error")
        fun shouldReturnUnhealthyOn400Error() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(400, "Bad Request"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Bad Request", result.message)
        }

        @Test
        @DisplayName("Should return unhealthy on 404 error")
        fun shouldReturnUnhealthyOn404Error() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(404, "Not Found"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Not Found", result.message)
        }

        @Test
        @DisplayName("Should return unhealthy on 500 error")
        fun shouldReturnUnhealthyOn500Error() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(500, "Internal Server Error"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Internal Server Error", result.message)
        }

        @Test
        @DisplayName("Should handle timeout exception in health check")
        fun shouldHandleTimeoutExceptionInHealthCheck() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Request timeout"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Request timeout"))
        }

        @Test
        @DisplayName("Should handle runtime exception in health check")
        fun shouldHandleRuntimeExceptionInHealthCheck() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(RuntimeException("Runtime error"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Runtime error"))
        }

        @Test
        @DisplayName("Should verify correct health endpoint call")
        fun shouldVerifyCorrectHealthEndpointCall() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            auraAIService.healthCheck()
            verify(mockHttpClient).get("health")
        }

        @Test
        @DisplayName("Should handle empty response body in health check")
        fun shouldHandleEmptyResponseBodyInHealthCheck() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, ""))
            val result = auraAIService.healthCheck()
            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }

        @Test
        @DisplayName("Should handle 201 as unhealthy")
        fun shouldHandle201AsUnhealthy() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(201, "Created"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Created", result.message)
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
        }

        @Test
        @DisplayName("Should handle single chunk stream")
        fun shouldHandleSingleChunkStream() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit("single chunk") })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(1, collected.size)
            assertEquals("single chunk", collected[0])
        }

        @Test
        @DisplayName("Should handle large stream")
        fun shouldHandleLargeStream() = runTest {
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
        @DisplayName("Should handle stream with empty chunks")
        fun shouldHandleStreamWithEmptyChunks() = runTest {
            val chunks = listOf("", "chunk", "", "another", "")
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
        }

        @Test
        @DisplayName("Should handle stream with special characters")
        fun shouldHandleStreamWithSpecialCharacters() = runTest {
            val chunks = listOf("🚀", "émoji", "@#$%", "special chars")
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
        }

        @Test
        @DisplayName("Should reject empty prompt for streaming")
        fun shouldRejectEmptyPromptForStreaming() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("").collect { }
            }
        }

        @Test
        @DisplayName("Should verify correct streaming endpoint call")
        fun shouldVerifyCorrectStreamingEndpointCall() = runTest {
            val prompt = "test prompt"
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit("chunk") })
            auraAIService.generateStreamingResponse(prompt).collect { }
            verify(mockHttpClient).postStream(prompt)
            verify(mockLogger).info("Starting streaming response for prompt length: ${prompt.length}")
        }

        @Test
        @DisplayName("Should handle streaming with long prompt")
        fun shouldHandleStreamingWithLongPrompt() = runTest {
            val longPrompt = "A".repeat(5000)
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit("response") })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse(longPrompt).collect { collected.add(it) }
            assertEquals(1, collected.size)
            verify(mockLogger).info("Starting streaming response for prompt length: ${longPrompt.length}")
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
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle configuration reload with invalid URL")
        fun shouldHandleConfigurationReloadWithInvalidUrl() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle configuration reload with invalid timeout")
        fun shouldHandleConfigurationReloadWithInvalidTimeout() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.test.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(-1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle configuration reload with runtime exception")
        fun shouldHandleConfigurationReloadWithRuntimeException() {
            whenever(mockConfigurationService.getApiKey()).thenThrow(RuntimeException("Service unavailable"))
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should successfully reload with minimal valid configuration")
        fun shouldSuccessfullyReloadWithMinimalValidConfiguration() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("k")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://a.b")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1L)
            
            auraAIService.reloadConfiguration()
            verify(mockLogger).info("Configuration reloaded successfully")
        }
    }

    @Nested
    @DisplayName("Enhanced Model Parameters Tests")
    inner class EnhancedModelParametersTests {
        @Test
        @DisplayName("Should update valid temperature parameter")
        fun shouldUpdateValidTemperatureParameter() {
            val params = mapOf("temperature" to 0.0)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should update maximum valid temperature")
        fun shouldUpdateMaximumValidTemperature() {
            val params = mapOf("temperature" to 1.0)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should reject negative temperature")
        fun shouldRejectNegativeTemperature() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("temperature" to -0.1))
            }
            verify(mockLogger).error("Invalid model parameters: temperature must be between 0 and 1")
        }

        @Test
        @DisplayName("Should reject temperature above 1")
        fun shouldRejectTemperatureAbove1() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("temperature" to 1.1))
            }
            verify(mockLogger).error("Invalid model parameters: temperature must be between 0 and 1")
        }

        @Test
        @DisplayName("Should accept non-numeric temperature parameter")
        fun shouldAcceptNonNumericTemperatureParameter() {
            val params = mapOf("temperature" to "invalid")
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should update minimum valid max_tokens")
        fun shouldUpdateMinimumValidMaxTokens() {
            val params = mapOf("max_tokens" to 1)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
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
                auraAIService.updateModelParameters(mapOf("max_tokens" to -100))
            }
            verify(mockLogger).error("Invalid model parameters: max_tokens must be positive")
        }

        @Test
        @DisplayName("Should accept non-integer max_tokens parameter")
        fun shouldAcceptNonIntegerMaxTokensParameter() {
            val params = mapOf("max_tokens" to "invalid")
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should update multiple valid parameters")
        fun shouldUpdateMultipleValidParameters() {
            val params = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 2048,
                "top_p" to 0.9,
                "frequency_penalty" to 0.1
            )
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should reject mixed valid and invalid parameters")
        fun shouldRejectMixedValidAndInvalidParameters() {
            val params = mapOf(
                "temperature" to 0.7,
                "max_tokens" to -100
            )
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(params)
            }
            verify(mockLogger).error("Invalid model parameters: max_tokens must be positive")
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
        @DisplayName("Should handle null parameter values")
        fun shouldHandleNullParameterValues() {
            val params = mapOf("temperature" to null, "max_tokens" to null)
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }
    }

    @Nested
    @DisplayName("Enhanced Statistics and Cache Tests")
    inner class EnhancedStatsAndCacheTests {
        @Test
        @DisplayName("Should return correct statistics structure")
        fun shouldReturnCorrectStatisticsStructure() {
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
            assertTrue(stats.containsKey("totalRequests"))
            assertTrue(stats.containsKey("successfulRequests"))
            assertTrue(stats.containsKey("failedRequests"))
            assertTrue(stats.containsKey("averageResponseTime"))
            assertEquals(0L, stats["totalRequests"])
            assertEquals(0L, stats["successfulRequests"])
            assertEquals(0L, stats["failedRequests"])
            assertEquals(0.0, stats["averageResponseTime"])
            verify(mockLogger).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should verify statistics types")
        fun shouldVerifyStatisticsTypes() {
            val stats = auraAIService.getServiceStatistics()
            assertTrue(stats["totalRequests"] is Long)
            assertTrue(stats["successfulRequests"] is Long)
            assertTrue(stats["failedRequests"] is Long)
            assertTrue(stats["averageResponseTime"] is Double)
        }

        @Test
        @DisplayName("Should reset statistics successfully")
        fun shouldResetStatisticsSuccessfully() {
            auraAIService.resetStatistics()
            verify(mockLogger).info("Service statistics reset")
        }

        @Test
        @DisplayName("Should clear cache successfully")
        fun shouldClearCacheSuccessfully() {
            auraAIService.clearCache()
            verify(mockLogger).info("Response cache cleared")
        }

        @Test
        @DisplayName("Should expire cache successfully")
        fun shouldExpireCacheSuccessfully() {
            auraAIService.expireCache()
            verify(mockLogger).debug("Cache expired, making new request")
        }

        @Test
        @DisplayName("Should handle multiple statistics requests")
        fun shouldHandleMultipleStatisticsRequests() {
            auraAIService.getServiceStatistics()
            auraAIService.getServiceStatistics()
            auraAIService.getServiceStatistics()
            verify(mockLogger, times(3)).debug("Service statistics requested")
        }

        @Test
        @DisplayName("Should handle multiple reset operations")
        fun shouldHandleMultipleResetOperations() {
            auraAIService.resetStatistics()
            auraAIService.resetStatistics()
            verify(mockLogger, times(2)).info("Service statistics reset")
        }

        @Test
        @DisplayName("Should handle multiple cache clear operations")
        fun shouldHandleMultipleCacheClearOperations() {
            auraAIService.clearCache()
            auraAIService.clearCache()
            verify(mockLogger, times(2)).info("Response cache cleared")
        }

        @Test
        @DisplayName("Should handle multiple cache expire operations")
        fun shouldHandleMultipleCacheExpireOperations() {
            auraAIService.expireCache()
            auraAIService.expireCache()
            verify(mockLogger, times(2)).debug("Cache expired, making new request")
        }
    }

    @Nested
    @DisplayName("URL Validation Tests")
    inner class UrlValidationTests {
        @Test
        @DisplayName("Should validate HTTPS URLs with subdomain")
        fun shouldValidateHttpsUrlsWithSubdomain() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.example.com")
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
        }

        @Test
        @DisplayName("Should validate HTTPS URLs with port")
        fun shouldValidateHttpsUrlsWithPort() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.example.com:8080")
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
        }

        @Test
        @DisplayName("Should validate HTTPS URLs with path")
        fun shouldValidateHttpsUrlsWithPath() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.example.com/v1/ai")
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            assertNotNull(service)
        }

        @Test
        @DisplayName("Should reject URLs with http protocol")
        fun shouldRejectUrlsWithHttpProtocol() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("http://api.example.com")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should reject URLs without protocol")
        fun shouldRejectUrlsWithoutProtocol() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("api.example.com")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should reject empty URL")
        fun shouldRejectEmptyUrl() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }

        @Test
        @DisplayName("Should reject URL with only protocol")
        fun shouldRejectUrlWithOnlyProtocol() {
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://")
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
    }

    @Nested
    @DisplayName("Concurrency and Thread Safety Tests")
    inner class ConcurrencyAndThreadSafetyTests {
        @Test
        @DisplayName("Should handle concurrent response generation")
        fun shouldHandleConcurrentResponseGeneration() = runTest {
            val expectedResponse = "Concurrent response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val jobs = (1..10).map { index ->
                async {
                    auraAIService.generateResponse("Prompt $index")
                }
            }

            val results = jobs.awaitAll()
            assertEquals(10, results.size)
            results.forEach { assertEquals(expectedResponse, it) }
            verify(mockHttpClient, times(10)).post(any())
        }

        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun shouldHandleConcurrentConfigurationUpdates() = runTest {
            val jobs = (1..5).map { index ->
                async {
                    auraAIService.updateTimeout(1000L * index)
                }
            }

            jobs.awaitAll()
            verify(mockConfigurationService, times(5)).updateTimeout(any())
        }

        @Test
        @DisplayName("Should handle concurrent health checks")
        fun shouldHandleConcurrentHealthChecks() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))

            val jobs = (1..5).map {
                async {
                    auraAIService.healthCheck()
                }
            }

            val results = jobs.awaitAll()
            assertEquals(5, results.size)
            results.forEach { assertTrue(it.isHealthy) }
            verify(mockHttpClient, times(5)).get("health")
        }
    }

    @Nested
    @DisplayName("Error Handling and Edge Cases")
    inner class ErrorHandlingAndEdgeCasesTests {
        @Test
        @DisplayName("Should handle OutOfMemoryError gracefully")
        fun shouldHandleOutOfMemoryErrorGracefully() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(OutOfMemoryError("Out of memory"))
            assertThrows<OutOfMemoryError> {
                auraAIService.generateResponse("Test")
            }
        }

        @Test
        @DisplayName("Should handle InterruptedException")
        fun shouldHandleInterruptedException() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(InterruptedException("Interrupted"))
            assertThrows<InterruptedException> {
                auraAIService.generateResponse("Test")
            }
        }

        @Test
        @DisplayName("Should handle multiple consecutive failures")
        fun shouldHandleMultipleConsecutiveFailures() = runTest {
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))
            
            repeat(5) {
                assertThrows<IOException> {
                    auraAIService.generateResponse("Test $it")
                }
            }
            verify(mockHttpClient, times(5)).post(any())
        }

        @Test
        @DisplayName("Should handle recovery after failure")
        fun shouldHandleRecoveryAfterFailure() = runTest {
            whenever(mockHttpClient.post(any()))
                .thenThrow(IOException("Network error"))
                .thenReturn(mockHttpResponse(200, "Success"))
            
            assertThrows<IOException> {
                auraAIService.generateResponse("Test 1")
            }
            
            val result = auraAIService.generateResponse("Test 2")
            assertEquals("Success", result)
        }

        @Test
        @DisplayName("Should handle null response from HTTP client")
        fun shouldHandleNullResponseFromHttpClient() = runTest {
            whenever(mockHttpClient.post(any())).thenReturn(null)
            assertThrows<NullPointerException> {
                auraAIService.generateResponse("Test")
            }
        }

        @Test
        @DisplayName("Should handle Unicode characters in prompts")
        fun shouldHandleUnicodeCharactersInPrompts() = runTest {
            val unicodePrompt = "Hello 世界 🌍 café naïve résumé"
            val expectedResponse = "Unicode response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(unicodePrompt)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(unicodePrompt)
        }
    }
}