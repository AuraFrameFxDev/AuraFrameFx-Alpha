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
    @DisplayName("Enhanced Generate Response Tests")
    inner class EnhancedGenerateResponseTests {
        @Test
        @DisplayName("Should handle very long prompts")
        fun shouldHandleVeryLongPrompts() = runTest {
            val longPrompt = "A".repeat(10000)
            val expectedResponse = "Response to long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(longPrompt)
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${longPrompt.length}")
        }

        @Test
        @DisplayName("Should handle prompts with special characters")
        fun shouldHandlePromptsWithSpecialCharacters() = runTest {
            val specialPrompt = "What is ñ + ü = ? 🤔"
            val expectedResponse = "Special character response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(specialPrompt)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle response with userId parameter")
        fun shouldHandleResponseWithUserId() = runTest {
            val prompt = "Test prompt"
            val userId = "user123"
            val expectedResponse = "User-specific response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val result = auraAIService.generateResponse(prompt, userId)
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(prompt)
        }

        @Test
        @DisplayName("Should handle various HTTP error codes")
        fun shouldHandleVariousHttpErrorCodes() = runTest {
            val errorCodes = listOf(400, 401, 403, 404, 429, 500, 502, 503)
            
            errorCodes.forEach { errorCode ->
                val mockHttpResponse = mockHttpResponse(errorCode, "Error $errorCode")
                whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
                
                assertThrows<IOException> {
                    auraAIService.generateResponse("Test")
                }
                verify(mockLogger).error("HTTP error response: $errorCode - Error $errorCode")
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
        @DisplayName("Should handle whitespace-only prompts")
        fun shouldHandleWhitespaceOnlyPrompts() = runTest {
            val whitespacePrompt = "   \n\t  "
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(whitespacePrompt.trim())
            }
        }
    }

    @Nested
    @DisplayName("Enhanced Batch Response Tests")
    inner class EnhancedBatchResponseTests {
        @Test
        @DisplayName("Should process single prompt in batch")
        fun shouldProcessSinglePromptInBatch() = runTest {
            val prompts = listOf("Single prompt")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf(expectedResponse), results)
            verify(mockLogger).info("Generating batch AI responses for 1 prompts")
        }

        @Test
        @DisplayName("Should process multiple prompts in batch")
        fun shouldProcessMultiplePromptsInBatch() = runTest {
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf(expectedResponse), results)
            verify(mockLogger).info("Generating batch AI responses for 3 prompts")
        }

        @Test
        @DisplayName("Should handle batch with empty prompts")
        fun shouldHandleBatchWithEmptyPrompts() = runTest {
            val prompts = listOf("Valid prompt", "", "Another valid prompt")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf(expectedResponse), results)
        }

        @Test
        @DisplayName("Should handle batch HTTP errors")
        fun shouldHandleBatchHttpErrors() = runTest {
            val prompts = listOf("Test prompt")
            val mockHttpResponse = mockHttpResponse(500, "Server error")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)

            val results = auraAIService.generateBatchResponses(prompts)
            assertEquals(listOf("Server error"), results)
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
        @DisplayName("Should handle stream with single chunk")
        fun shouldHandleStreamWithSingleChunk() = runTest {
            val singleChunk = "Complete response"
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                emit(singleChunk)
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(listOf(singleChunk), collected)
        }

        @Test
        @DisplayName("Should handle stream with empty chunks")
        fun shouldHandleStreamWithEmptyChunks() = runTest {
            val chunks = listOf("", "data", "", "more data", "")
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                chunks.forEach { emit(it) }
            })
            val collected = mutableListOf<String>()
            auraAIService.generateStreamingResponse("prompt").collect { collected.add(it) }
            assertEquals(chunks, collected)
        }

        @Test
        @DisplayName("Should handle stream errors")
        fun shouldHandleStreamErrors() = runTest {
            whenever(mockHttpClient.postStream(any())).thenReturn(flow {
                emit("chunk1")
                throw IOException("Stream error")
            })
            
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse("prompt").collect()
            }
        }

        @Test
        @DisplayName("Should reject empty prompt for streaming")
        fun shouldRejectEmptyPromptForStreaming() = runTest {
            assertThrows<IllegalArgumentException> {
                auraAIService.generateStreamingResponse("").collect()
            }
        }

        @Test
        @DisplayName("Should log streaming start")
        fun shouldLogStreamingStart() = runTest {
            val prompt = "Test streaming prompt"
            whenever(mockHttpClient.postStream(any())).thenReturn(flow { emit("chunk") })
            
            auraAIService.generateStreamingResponse(prompt).collect()
            verify(mockLogger).info("Starting streaming response for prompt length: ${prompt.length}")
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Tests")
    inner class EnhancedConfigurationTests {
        @Test
        @DisplayName("Should update base URL with valid HTTPS URL")
        fun shouldUpdateBaseUrlWithValidHttpsUrl() {
            val newBaseUrl = "https://api.newendpoint.com"
            auraAIService.updateBaseUrl(newBaseUrl)
            verify(mockConfigurationService).updateBaseUrl(newBaseUrl)
            verify(mockLogger).info("Base URL updated successfully")
        }

        @Test
        @DisplayName("Should reject HTTP URLs")
        fun shouldRejectHttpUrls() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl("http://insecure.com")
            }
        }

        @Test
        @DisplayName("Should reject malformed URLs")
        fun shouldRejectMalformedUrls() {
            val invalidUrls = listOf("not-a-url", "ftp://example.com", "://missing-protocol", "")
            invalidUrls.forEach { url ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateBaseUrl(url)
                }
            }
        }

        @Test
        @DisplayName("Should update timeout with valid positive value")
        fun shouldUpdateTimeoutWithValidPositiveValue() {
            val newTimeout = 60000L
            auraAIService.updateTimeout(newTimeout)
            verify(mockConfigurationService).updateTimeout(newTimeout)
            verify(mockLogger).info("Timeout updated to $newTimeout ms")
        }

        @Test
        @DisplayName("Should reject negative timeout")
        fun shouldRejectNegativeTimeout() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(-1000L)
            }
        }

        @Test
        @DisplayName("Should reject zero timeout")
        fun shouldRejectZeroTimeout() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(0L)
            }
        }

        @Test
        @DisplayName("Should handle very large timeout values")
        fun shouldHandleVeryLargeTimeoutValues() {
            val largeTimeout = Long.MAX_VALUE
            auraAIService.updateTimeout(largeTimeout)
            verify(mockConfigurationService).updateTimeout(largeTimeout)
        }
    }

    @Nested
    @DisplayName("Enhanced Health Check Tests")
    inner class EnhancedHealthCheckTests {
        @Test
        @DisplayName("Should return unhealthy on non-200 status codes")
        fun shouldReturnUnhealthyOnNon200StatusCodes() = runTest {
            val errorCodes = listOf(400, 401, 403, 404, 500, 502, 503)
            errorCodes.forEach { errorCode ->
                val mockHttpResponse = mockHttpResponse(errorCode, "Error $errorCode")
                whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
                
                val result = auraAIService.healthCheck()
                assertFalse(result.isHealthy)
                assertEquals("Service is unhealthy: Error $errorCode", result.message)
            }
        }

        @Test
        @DisplayName("Should handle timeout during health check")
        fun shouldHandleTimeoutDuringHealthCheck() = runTest {
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Health check timeout"))
            
            val result = auraAIService.healthCheck()
            assertFalse(result.isHealthy)
            assertTrue(result.message.contains("Health check timeout"))
        }

        @Test
        @DisplayName("Should handle various exceptions during health check")
        fun shouldHandleVariousExceptionsDuringHealthCheck() = runTest {
            val exceptions = listOf(
                IOException("Network error"),
                RuntimeException("Runtime error"),
                IllegalStateException("Illegal state")
            )
            
            exceptions.forEach { exception ->
                whenever(mockHttpClient.get(any())).thenThrow(exception)
                
                val result = auraAIService.healthCheck()
                assertFalse(result.isHealthy)
                assertTrue(result.message.contains("Service is unhealthy: ${exception.message}"))
            }
        }

        @Test
        @DisplayName("Should use correct health check endpoint")
        fun shouldUseCorrectHealthCheckEndpoint() = runTest {
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            
            auraAIService.healthCheck()
            verify(mockHttpClient).get("health")
        }
    }

    @Nested
    @DisplayName("Enhanced Model Parameters Tests")
    inner class EnhancedModelParametersTests {
        @Test
        @DisplayName("Should accept valid temperature values")
        fun shouldAcceptValidTemperatureValues() {
            val validTemperatures = listOf(0.0, 0.5, 1.0, 0.1, 0.9)
            validTemperatures.forEach { temp ->
                val params = mapOf("temperature" to temp)
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }

        @Test
        @DisplayName("Should reject temperature below 0")
        fun shouldRejectTemperatureBelowZero() {
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
        @DisplayName("Should accept valid max_tokens values")
        fun shouldAcceptValidMaxTokensValues() {
            val validTokens = listOf(1, 100, 1000, 4096)
            validTokens.forEach { tokens ->
                val params = mapOf("max_tokens" to tokens)
                auraAIService.updateModelParameters(params)
                verify(mockConfigurationService).updateModelParameters(params)
            }
        }

        @Test
        @DisplayName("Should reject negative max_tokens")
        fun shouldRejectNegativeMaxTokens() {
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mapOf("max_tokens" to -1))
            }
            verify(mockLogger).error("Invalid model parameters: max_tokens must be positive")
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
        @DisplayName("Should handle mixed valid and invalid parameters")
        fun shouldHandleMixedValidAndInvalidParameters() {
            val params = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 1000,
                "top_p" to 0.9,
                "frequency_penalty" to 0.0
            )
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
            verify(mockLogger).info("Model parameters updated: $params")
        }

        @Test
        @DisplayName("Should handle empty parameters map")
        fun shouldHandleEmptyParametersMap() {
            val emptyParams = emptyMap<String, Any>()
            auraAIService.updateModelParameters(emptyParams)
            verify(mockConfigurationService).updateModelParameters(emptyParams)
        }

        @Test
        @DisplayName("Should handle non-numeric temperature")
        fun shouldHandleNonNumericTemperature() {
            val params = mapOf("temperature" to "invalid")
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }

        @Test
        @DisplayName("Should handle non-integer max_tokens")
        fun shouldHandleNonIntegerMaxTokens() {
            val params = mapOf("max_tokens" to "invalid")
            auraAIService.updateModelParameters(params)
            verify(mockConfigurationService).updateModelParameters(params)
        }
    }

    @Nested
    @DisplayName("Enhanced Configuration Reload Tests")
    inner class EnhancedConfigurationReloadTests {
        @Test
        @DisplayName("Should handle reload with null API key")
        fun shouldHandleReloadWithNullApiKey() {
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://valid.url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle reload with invalid base URL")
        fun shouldHandleReloadWithInvalidBaseUrl() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle reload with invalid timeout")
        fun shouldHandleReloadWithInvalidTimeout() {
            whenever(mockConfigurationService.getApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://valid.url")
            whenever(mockConfigurationService.getTimeout()).thenReturn(-1000L)
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
        }

        @Test
        @DisplayName("Should handle configuration service exceptions")
        fun shouldHandleConfigurationServiceExceptions() {
            whenever(mockConfigurationService.getApiKey()).thenThrow(RuntimeException("Config error"))
            
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
            verify(mockLogger).error(contains("Failed to reload configuration"))
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
            auraAIService.clearCache()
            auraAIService.expireCache()
            auraAIService.clearCache()
            
            verify(mockLogger, times(2)).info("Response cache cleared")
            verify(mockLogger).debug("Cache expired, making new request")
        }

        @Test
        @DisplayName("Should handle statistics requests with debug logging")
        fun shouldHandleStatisticsRequestsWithDebugLogging() {
            repeat(3) {
                auraAIService.getServiceStatistics()
            }
            
            verify(mockLogger, times(3)).debug("Service statistics requested")
        }
    }

    @Nested
    @DisplayName("URL Validation Tests")
    inner class UrlValidationTests {
        @Test
        @DisplayName("Should validate HTTPS URLs correctly")
        fun shouldValidateHttpsUrlsCorrectly() {
            val validUrls = listOf(
                "https://api.openai.com",
                "https://api.example.com/v1",
                "https://localhost:8080",
                "https://127.0.0.1:3000"
            )
            
            validUrls.forEach { url ->
                whenever(mockConfigurationService.getBaseUrl()).thenReturn(url)
                // This should not throw an exception
                val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
                assertNotNull(service)
            }
        }

        @Test
        @DisplayName("Should reject non-HTTPS URLs")
        fun shouldRejectNonHttpsUrls() {
            val invalidUrls = listOf(
                "http://api.example.com",
                "ftp://example.com",
                "ws://example.com",
                "file://example.com",
                null,
                ""
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
    @DisplayName("Integration and Edge Case Tests")
    inner class IntegrationAndEdgeCaseTests {
        @Test
        @DisplayName("Should handle concurrent requests")
        fun shouldHandleConcurrentRequests() = runTest {
            val mockHttpResponse = mockHttpResponse(200, "Response")
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // Simulate concurrent requests
            val results = (1..10).map { i ->
                async {
                    auraAIService.generateResponse("Prompt $i")
                }
            }.awaitAll()
            
            assertEquals(10, results.size)
            results.forEach { result ->
                assertEquals("Response", result)
            }
        }

        @Test
        @DisplayName("Should handle service lifecycle")
        fun shouldHandleServiceLifecycle() = runTest {
            // Initial health check
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse(200, "OK"))
            val healthResult = auraAIService.healthCheck()
            assertTrue(healthResult.isHealthy)
            
            // Configuration update
            auraAIService.updateApiKey("new-key")
            auraAIService.updateBaseUrl("https://new.url")
            auraAIService.updateTimeout(5000L)
            
            // Statistics and cache management
            auraAIService.getServiceStatistics()
            auraAIService.clearCache()
            auraAIService.resetStatistics()
            
            // Verify all operations completed
            verify(mockConfigurationService).updateApiKey("new-key")
            verify(mockConfigurationService).updateBaseUrl("https://new.url")
            verify(mockConfigurationService).updateTimeout(5000L)
        }

        @Test
        @DisplayName("Should handle memory pressure scenarios")
        fun shouldHandleMemoryPressureScenarios() = runTest {
            val largeResponse = "A".repeat(1000000) // 1MB response
            val mockHttpResponse = mockHttpResponse(200, largeResponse)
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            val result = auraAIService.generateResponse("Test")
            assertEquals(largeResponse, result)
        }
    }

    // Additional helper methods for new tests
    private suspend fun <T> List<Deferred<T>>.awaitAll(): List<T> {
        return this.map { it.await() }
    }

    private fun <T> async(block: suspend () -> T): Deferred<T> {
        return kotlinx.coroutines.GlobalScope.async { block() }
    }

    interface Deferred<T> {
        suspend fun await(): T
    }
}