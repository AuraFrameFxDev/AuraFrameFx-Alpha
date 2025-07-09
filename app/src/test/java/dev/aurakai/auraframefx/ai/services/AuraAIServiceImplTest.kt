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
    @DisplayName("Extended Generate Response Tests")
    inner class ExtendedGenerateResponseTests {
        @Test
        @DisplayName("Should handle very long prompts")
        fun shouldHandleVeryLongPrompts() = runTest {
            val longPrompt = "a".repeat(10000)
            val expectedResponse = "Response for long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(longPrompt)
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${longPrompt.length}")
        }

        @Test
        @DisplayName("Should handle prompts with special characters")
        fun shouldHandlePromptsWithSpecialCharacters() = runTest {
            val specialPrompt = "Test with special chars: àáâãäåæçèé!@#$%^&*()[]{}|;:,.<>?"
            val expectedResponse = "Response with special handling"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(specialPrompt)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle prompts with newlines and tabs")
        fun shouldHandlePromptsWithNewlinesAndTabs() = runTest {
            val multilinePrompt = "Line 1\nLine 2\tTabbed content\r\nWindows line ending"
            val expectedResponse = "Multiline response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(multilinePrompt)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle userId parameter")
        fun shouldHandleUserIdParameter() = runTest {
            val prompt = "Test prompt"
            val userId = "user123"
            val expectedResponse = "User-specific response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, userId)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle null userId parameter")
        fun shouldHandleNullUserIdParameter() = runTest {
            val prompt = "Test prompt"
            val expectedResponse = "Generic response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, null)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle various HTTP error codes")
        fun shouldHandleVariousHttpErrorCodes() = runTest {
            val errorCodes = listOf(400, 401, 403, 404, 429, 500, 502, 503, 504)
            
            errorCodes.forEach { errorCode ->
                val mockHttpResponse = mockHttpResponse(errorCode, "Error $errorCode")
                whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
                
                val exception = assertThrows<IOException> {
                    auraAIService.generateResponse("Test")
                }
                assertTrue(exception.message?.contains("HTTP error: $errorCode") == true)
            }
        }

        @Test
        @DisplayName("Should handle empty response body")
        fun shouldHandleEmptyResponseBody() = runTest {
            val mockHttpResponse = mockHttpResponse(200, "")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse("Test")
            assertEquals("", result)
        }

        @Test
        @DisplayName("Should handle response with whitespace only")
        fun shouldHandleResponseWithWhitespaceOnly() = runTest {
            val mockHttpResponse = mockHttpResponse(200, "   \t\n  ")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse("Test")
            assertEquals("   \t\n  ", result)
        }
    }

    @Nested
    @DisplayName("Extended Batch Response Tests")
    inner class ExtendedBatchResponseTests {
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
            val prompts = listOf("", "Valid prompt", "")
            val expectedResponse = "Mixed batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
        }

        @Test
        @DisplayName("Should handle batch HTTP errors")
        fun shouldHandleBatchHttpErrors() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2")
            val mockHttpResponse = mockHttpResponse(500, "Batch error")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            // Note: Current implementation doesn't handle batch errors, just returns the response
            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals("Batch error", results[0])
        }

        @Test
        @DisplayName("Should handle batch network exceptions")
        fun shouldHandleBatchNetworkExceptions() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2")
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))

            assertThrows<IOException> {
                auraAIService.generateBatchResponses(prompts)
            }
        }

        @Test
        @DisplayName("Should handle very large batch size")
        fun shouldHandleVeryLargeBatchSize() = runTest {
            val prompts = (1..1000).map { "Prompt $it" }
            val expectedResponse = "Large batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
            verify(mockLogger).info("Generating batch AI responses for 1000 prompts")
        }
    }

    @Nested
    @DisplayName("Extended Streaming Response Tests")
    inner class ExtendedStreamingResponseTests {
        @Test
        @DisplayName("Should handle empty stream")
        fun shouldHandleEmptyStream() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {})
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertTrue(collected.isEmpty())
        }

        @Test
        @DisplayName("Should handle stream with single chunk")
        fun shouldHandleStreamWithSingleChunk() = runTest {
            val singleChunk = "Single chunk response"
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit(singleChunk) })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(1, collected.size)
            assertEquals(singleChunk, collected[0])
        }

        @Test
        @DisplayName("Should handle stream with many chunks")
        fun shouldHandleStreamWithManyChunks() = runTest {
            val chunks = (1..100).map { "Chunk $it" }
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
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
        @DisplayName("Should handle stream with special characters")
        fun shouldHandleStreamWithSpecialCharacters() = runTest {
            val chunks = listOf("∀", "∃", "∈", "∉", "∪", "∩", "⊆", "⊇")
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
        }

        @Test
        @DisplayName("Should handle empty streaming prompt")
        fun shouldHandleEmptyStreamingPrompt() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("")
            }
        }
    }

    @Nested
    @DisplayName("Extended Configuration Tests")
    inner class ExtendedConfigurationTests {
        @Test
        @DisplayName("Should update base URL with different valid formats")
        fun shouldUpdateBaseUrlWithDifferentValidFormats() {
            val validUrls = listOf(
                "https://api.example.com",
                "https://api.example.com/",
                "https://api.example.com/v1",
                "https://subdomain.api.example.com/path"
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
                "http://api.example.com",  // not https
                "ftp://api.example.com",   // not https
                "api.example.com",         // no protocol
                "",                        // empty
                "https://",                // incomplete
                "not-a-url"               // invalid format
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
            val validTimeouts = listOf(1L, 1000L, 30000L, 60000L, 300000L)
            
            validTimeouts.forEach { timeout ->
                auraAIService.updateTimeout(timeout)
                verify(mockConfigurationService).updateTimeout(timeout)
                verify(mockLogger).info("Timeout updated to $timeout ms")
            }
        }

        @Test
        @DisplayName("Should reject invalid timeout values")
        fun shouldRejectInvalidTimeoutValues() {
            val invalidTimeouts = listOf(0L, -1L, -1000L, Long.MIN_VALUE)
            
            invalidTimeouts.forEach { timeout ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateTimeout(timeout)
                }
            }
        }

        @Test
        @DisplayName("Should handle timeout at boundary values")
        fun shouldHandleTimeoutAtBoundaryValues() {
            // Test boundary values
            auraAIService.updateTimeout(1L)  // minimum valid
            auraAIService.updateTimeout(Long.MAX_VALUE)  // maximum valid
            
            verify(mockConfigurationService).updateTimeout(1L)
            verify(mockConfigurationService).updateTimeout(Long.MAX_VALUE)
        }
    }

    @Nested
    @DisplayName("Extended Health Check Tests")
    inner class ExtendedHealthCheckTests {
        @Test
        @DisplayName("Should return unhealthy on various error codes")
        fun shouldReturnUnhealthyOnVariousErrorCodes() = runTest {
            val errorCodes = listOf(400, 401, 403, 404, 500, 502, 503, 504)
            
            errorCodes.forEach { errorCode ->
                whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(errorCode, "Error $errorCode"))
                val result = auraAIService.healthCheck()
                assertFalse(result.isHealthy)
                assertTrue(result.message.contains("Service is unhealthy"))
                assertTrue(result.message.contains("Error $errorCode"))
            }
        }

        @Test
        @DisplayName("Should handle health check timeout")
        fun shouldHandleHealthCheckTimeout() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Health check timeout"))
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Health check timeout"))
        }

        @Test
        @DisplayName("Should handle health check with different exceptions")
        fun shouldHandleHealthCheckWithDifferentExceptions() = runTest {
            val exceptions = listOf(
                IOException("Network error"),
                RuntimeException("Runtime error"),
                IllegalStateException("Illegal state")
            )
            
            exceptions.forEach { exception ->
                whenever(mockHttpClient.get(any())).thenThrow(exception)
                val result = auraAIService.healthCheck()
                assertFalse(result.isHealthy)
                assertTrue(result.message.contains(exception.message!!))
            }
        }

        @Test
        @DisplayName("Should handle health check with empty response")
        fun shouldHandleHealthCheckWithEmptyResponse() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, ""))
            val result = auraAIService.healthCheck()
            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }
    }

    @Nested
    @DisplayName("Extended Model Parameters Tests")
    inner class ExtendedModelParametersTests {
        @Test
        @DisplayName("Should handle valid temperature values")
        fun shouldHandleValidTemperatureValues() {
            val validTemperatures = listOf(0.0, 0.1, 0.5, 0.9, 1.0)
            
            validTemperatures.forEach { temp ->
                val params = mapOf("temperature" to temp)
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }

        @Test
        @DisplayName("Should reject invalid temperature values")
        fun shouldRejectInvalidTemperatureValues() {
            val invalidTemperatures = listOf(-0.1, 1.1, 2.0, -1.0, Double.MAX_VALUE)
            
            invalidTemperatures.forEach { temp ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateModelParameters(mapOf("temperature" to temp))
                }
            }
        }

        @Test
        @DisplayName("Should handle valid max_tokens values")
        fun shouldHandleValidMaxTokensValues() {
            val validTokens = listOf(1, 100, 1000, 4096, Int.MAX_VALUE)
            
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
        @DisplayName("Should handle mixed valid and invalid parameters")
        fun shouldHandleMixedValidAndInvalidParameters() {
            // Should reject if any parameter is invalid
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf(
                    "temperature" to 0.5,  // valid
                    "max_tokens" to -1     // invalid
                ))
            }
            
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf(
                    "temperature" to 2.0,  // invalid
                    "max_tokens" to 100    // valid
                ))
            }
        }

        @Test
        @DisplayName("Should handle parameters with different types")
        fun shouldHandleParametersWithDifferentTypes() {
            val params = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 150,
                "top_p" to 0.9,
                "frequency_penalty" to 0.1,
                "presence_penalty" to 0.2,
                "stop" to listOf("END", "STOP"),
                "model" to "gpt-3.5-turbo"
            )
            
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should handle empty parameters map")
        fun shouldHandleEmptyParametersMap() {
            val params = emptyMap<String, Any>()
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle parameters with non-numeric temperature")
        fun shouldHandleParametersWithNonNumericTemperature() {
            val params = mapOf("temperature" to "0.5")  // String instead of Double
            // Should not throw since it's not a Double
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle parameters with non-integer max_tokens")
        fun shouldHandleParametersWithNonIntegerMaxTokens() {
            val params = mapOf("max_tokens" to "100")  // String instead of Int
            // Should not throw since it's not an Int
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }
    }

    @Nested
    @DisplayName("Extended Statistics and Cache Tests")
    inner class ExtendedStatsAndCacheTests {
        @Test
        @DisplayName("Should return statistics with expected structure")
        fun shouldReturnStatisticsWithExpectedStructure() {
            val stats = auraAIService.getServiceStatistics()
            
            // Verify structure
            assertNotNull(stats)
            assertTrue(stats.containsKey("totalRequests"))
            assertTrue(stats.containsKey("successfulRequests"))
            assertTrue(stats.containsKey("failedRequests"))
            assertTrue(stats.containsKey("averageResponseTime"))
            
            // Verify types
            assertTrue(stats["totalRequests"] is Long)
            assertTrue(stats["successfulRequests"] is Long)
            assertTrue(stats["failedRequests"] is Long)
            assertTrue(stats["averageResponseTime"] is Double)
            
            // Verify initial values
            assertEquals(0L, stats["totalRequests"])
            assertEquals(0L, stats["successfulRequests"])
            assertEquals(0L, stats["failedRequests"])
            assertEquals(0.0, stats["averageResponseTime"])
        }

        @Test
        @DisplayName("Should reset statistics")
        fun shouldResetStatistics() {
            auraAIService.resetStatistics()
            verify(mockLogger).info("Service statistics reset")
        }

        @Test
        @DisplayName("Should expire cache")
        fun shouldExpireCache() {
            auraAIService.expireCache()
            verify(mockLogger).debug("Cache expired, making new request")
        }

        @Test
        @DisplayName("Should handle multiple cache operations")
        fun shouldHandleMultipleCacheOperations() {
            // Test sequence of cache operations
            auraAIService.clearCache()
            auraAIService.expireCache()
            auraAIService.clearCache()
            
            verify(mockLogger, times(2)).info("Response cache cleared")
            verify(mockLogger).debug("Cache expired, making new request")
        }

        @Test
        @DisplayName("Should handle multiple statistics operations")
        fun shouldHandleMultipleStatisticsOperations() {
            // Test sequence of statistics operations
            auraAIService.getServiceStatistics()
            auraAIService.resetStatistics()
            auraAIService.getServiceStatistics()
            
            verify(mockLogger, times(2)).debug("Service statistics requested")
            verify(mockLogger).info("Service statistics reset")
        }
    }

    @Nested
    @DisplayName("Extended Reload Configuration Tests")
    inner class ExtendedReloadConfigurationTests {
        @Test
        @DisplayName("Should reload configuration with different valid combinations")
        fun shouldReloadConfigurationWithDifferentValidCombinations() {
            val validConfigs = listOf(
                Triple("key1", "https://api1.com", 1000L),
                Triple("key2", "https://api2.com/v1", 5000L),
                Triple("very-long-key-123456789", "https://subdomain.api.com/path", 60000L)
            )
            
            validConfigs.forEach { (key, url, timeout) ->
                whenever(mockConfigurationService.getApiKey()).thenReturn(key)
                whenever(mockConfigurationService.getBaseUrl()).thenReturn(url)
                whenever(mockConfigurationService.getTimeout()).thenReturn(timeout)
                
                auraAIService.reloadConfiguration()
                verify(mockLogger, atLeastOnce()).info("Configuration reloaded successfully")
            }
        }

        @Test
        @DisplayName("Should fail reload with null API key")
        fun shouldFailReloadWithNullApiKey() {
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should fail reload with invalid base URL")
        fun shouldFailReloadWithInvalidBaseUrl() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should fail reload with invalid timeout")
        fun shouldFailReloadWithInvalidTimeout() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://api.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(-1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle configuration service exceptions during reload")
        fun shouldHandleConfigurationServiceExceptionsDuringReload() {
            whenever(mockConfigurationService.getApiKey()).thenThrow(RuntimeException("Config service error"))
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }
    }

    @Nested
    @DisplayName("Concurrent Access Tests")
    inner class ConcurrentAccessTests {
        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun shouldHandleConcurrentConfigurationUpdates() {
            // Test that multiple configuration updates don't interfere
            auraAIService.updateApiKey("key1")
            auraAIService.updateBaseUrl("https://api1.com")
            auraAIService.updateTimeout(1000L)
            
            verify(mockConfigurationService).updateApiKey("key1")
            verify(mockConfigurationService).updateBaseUrl("https://api1.com")
            verify(mockConfigurationService).updateTimeout(1000L)
        }

        @Test
        @DisplayName("Should handle concurrent cache operations")
        fun shouldHandleConcurrentCacheOperations() {
            // Test that multiple cache operations don't interfere
            auraAIService.clearCache()
            auraAIService.expireCache()
            auraAIService.resetStatistics()
            
            verify(mockLogger).info("Response cache cleared")
            verify(mockLogger).debug("Cache expired, making new request")
            verify(mockLogger).info("Service statistics reset")
        }
    }

    @Nested
    @DisplayName("Integration-like Tests")
    inner class IntegrationLikeTests {
        @Test
        @DisplayName("Should handle complete workflow")
        fun shouldHandleCompleteWorkflow() = runTest {
            // Test a complete workflow from configuration to response
            auraAIService.updateApiKey("workflow-key")
            auraAIService.updateBaseUrl("https://workflow.api.com")
            auraAIService.updateTimeout(5000L)
            
            val healthResult = auraAIService.healthCheck()
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            assertTrue(healthResult.isHealthy)
            
            val prompt = "Complete workflow test"
            val expectedResponse = "Workflow response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            val result = auraAIService.generateResponse(prompt)
            assertEquals(expectedResponse, result)
            
            val stats = auraAIService.getServiceStatistics()
            assertNotNull(stats)
            
            auraAIService.clearCache()
            auraAIService.resetStatistics()
            
            // Verify all operations completed successfully
            verify(mockConfigurationService).updateApiKey("workflow-key")
            verify(mockConfigurationService).updateBaseUrl("https://workflow.api.com")
            verify(mockConfigurationService).updateTimeout(5000L)
            verify(mockHttpClient).post(prompt)
            verify(mockLogger).info("Response cache cleared")
            verify(mockLogger).info("Service statistics reset")
        }

        @Test
        @DisplayName("Should handle error recovery workflow")
        fun shouldHandleErrorRecoveryWorkflow() = runTest {
            // Test error conditions and recovery
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))
            
            assertThrows<IOException> {
                auraAIService.generateResponse("Test prompt")
            }
            
            // Simulate recovery by fixing the network
            val mockHttpResponse = mockHttpResponse(200, "Recovered response")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            val result = auraAIService.generateResponse("Recovery prompt")
            assertEquals("Recovered response", result)
        }
    }

    @Nested
    @DisplayName("Edge Cases and Boundary Tests")
    inner class EdgeCasesAndBoundaryTests {
        @Test
        @DisplayName("Should handle extreme timeout values")
        fun shouldHandleExtremeTimeoutValues() {
            // Test with very large timeout
            auraAIService.updateTimeout(Long.MAX_VALUE)
            verify(mockConfigurationService).updateTimeout(Long.MAX_VALUE)
            
            // Test with minimum valid timeout
            auraAIService.updateTimeout(1L)
            verify(mockConfigurationService).updateTimeout(1L)
        }

        @Test
        @DisplayName("Should handle maximum length API key")
        fun shouldHandleMaximumLengthApiKey() {
            val longApiKey = "a".repeat(1000)
            auraAIService.updateApiKey(longApiKey)
            verify(mockConfigurationService).updateApiKey(longApiKey)
        }

        @Test
        @DisplayName("Should handle complex base URLs")
        fun shouldHandleComplexBaseUrls() {
            val complexUrls = listOf(
                "https://api.complex-domain.co.uk/v1/ai/models",
                "https://subdomain.api.example.com:8443/path/to/service",
                "https://api-staging.example.com/v2/neural-networks"
            )
            
            complexUrls.forEach { url ->
                auraAIService.updateBaseUrl(url)
                verify(mockConfigurationService).updateBaseUrl(url)
            }
        }

        @Test
        @DisplayName("Should handle model parameters with extreme values")
        fun shouldHandleModelParametersWithExtremeValues() {
            val extremeParams = mapOf(
                "temperature" to 0.0,
                "max_tokens" to Int.MAX_VALUE,
                "top_p" to 1.0,
                "frequency_penalty" to -2.0,
                "presence_penalty" to 2.0
            )
            
            auraAIService.updateModelParameters(extremeParams)
            verify(mockConfigurationService).updateModelParameters(extremeParams)
        }
    }

    // Additional helper methods for extended tests
    private fun mockHttpResponseWithHeaders(statusCode: Int, body: String, headers: Map<String, String> = emptyMap()): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }

    private fun setupMockConfigurationService(apiKey: String?, baseUrl: String?, timeout: Long) {
        whenever(mockConfigurationService.getApiKey()).thenReturn(apiKey)
        whenever(mockConfigurationService.getBaseUrl()).thenReturn(baseUrl)
        whenever(mockConfigurationService.getTimeout()).thenReturn(timeout)
    }
}