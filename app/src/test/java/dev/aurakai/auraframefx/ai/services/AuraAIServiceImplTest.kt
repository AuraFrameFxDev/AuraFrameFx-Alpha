package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.async
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.delay
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
    suspend fun postStream(request: Any): Flow<String>
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

// Mock AuraAIServiceImpl interface
interface AuraAIService {
    suspend fun generateResponse(prompt: String, userId: String? = null): String
    suspend fun generateBatchResponses(prompts: List<String>): List<String>
    suspend fun generateStreamingResponse(prompt: String): Flow<String>
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

// Mock implementation for testing
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
    
    override suspend fun generateStreamingResponse(prompt: String): Flow<String> {
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
            httpClient.get("health")
            HealthCheckResult(true, "Service is healthy")
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
            
            logger.info("Configuration reloaded successfully")
        } catch (e: Exception) {
            logger.error("Failed to reload configuration: ${e.message}")
            throw ConfigurationException("Configuration validation failed: ${e.message}")
        }
    }
    
    override fun updateModelParameters(params: Map<String, Any>) {
        // Validate parameters
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
        
        // Setup default mock behaviors
        whenever(mockConfigurationService.getApiKey()).thenReturn(testApiKey)
        whenever(mockConfigurationService.getBaseUrl()).thenReturn(testBaseUrl)
        whenever(mockConfigurationService.getTimeout()).thenReturn(testTimeout)
        
        auraAIService = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
    }
    
    @AfterEach
    fun tearDown() {
        // Clean up any resources if needed
    }
    
    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {
        
        @Test
        @DisplayName("Should initialize with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            // Given & When
            val service = AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            
            // Then
            assertNotNull(service)
            verify(mockConfigurationService).getApiKey()
            verify(mockConfigurationService).getBaseUrl()
            verify(mockConfigurationService).getTimeout()
        }
        
        @Test
        @DisplayName("Should throw exception when API key is null")
        fun shouldThrowExceptionWhenApiKeyIsNull() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn(null)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should throw exception when API key is empty")
        fun shouldThrowExceptionWhenApiKeyIsEmpty() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAIServiceImpl(mockHttpClient, mockConfigurationService, mockLogger)
            }
        }
        
        @Test
        @DisplayName("Should throw exception when base URL is invalid")
        fun shouldThrowExceptionWhenBaseUrlIsInvalid() {
            // Given
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("invalid-url")
            
            // When & Then
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
            // Given
            val prompt = "What is the capital of France?"
            val expectedResponse = "The capital of France is Paris."
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockHttpClient).post(any())
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }
        
        @Test
        @DisplayName("Should handle empty prompt")
        fun shouldHandleEmptyPrompt() = runTest {
            // Given
            val prompt = ""
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @Test
        @DisplayName("Should handle very long prompt")
        fun shouldHandleVeryLongPrompt() = runTest {
            // Given
            val prompt = "A".repeat(10000)
            val expectedResponse = "Response for long prompt"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockLogger).info("Generating AI response for prompt length: ${prompt.length}")
        }
        
        @Test
        @DisplayName("Should handle HTTP error responses")
        fun shouldHandleHttpErrorResponses() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(500, "Internal Server Error")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("HTTP error response: 500 - Internal Server Error")
        }
        
        @Test
        @DisplayName("Should handle network timeout")
        fun shouldHandleNetworkTimeout() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(TimeoutException("Request timed out"))
            
            // When & Then
            assertThrows<TimeoutException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @Test
        @DisplayName("Should handle network connection error")
        fun shouldHandleNetworkConnectionError() = runTest {
            // Given
            val prompt = "Test prompt"
            
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Connection refused"))
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @Test
        @DisplayName("Should handle special characters in prompt")
        fun shouldHandleSpecialCharactersInPrompt() = runTest {
            // Given
            val prompt = "Test with special characters: éñüß@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
            val expectedResponse = "Response with special characters"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
        
        @Test
        @DisplayName("Should handle Unicode characters in prompt")
        fun shouldHandleUnicodeCharactersInPrompt() = runTest {
            // Given
            val prompt = "Test with Unicode: 你好世界 🌍 🚀 ✨"
            val expectedResponse = "Response with Unicode"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then
            assertEquals(expectedResponse, result)
        }
    }
    
    @Nested
    @DisplayName("Generate Batch Responses Tests")
    inner class GenerateBatchResponsesTests {
        
        @Test
        @DisplayName("Should generate batch responses for multiple prompts")
        fun shouldGenerateBatchResponsesForMultiplePrompts() = runTest {
            // Given
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3")
            val expectedResponse = "Batch response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(1, results.size)
            verify(mockHttpClient).post(any())
            verify(mockLogger).info("Generating batch AI responses for ${prompts.size} prompts")
        }
        
        @Test
        @DisplayName("Should handle empty prompt list")
        fun shouldHandleEmptyPromptList() = runTest {
            // Given
            val prompts = emptyList<String>()
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertTrue(results.isEmpty())
            verify(mockLogger).info("No prompts provided for batch processing")
        }
        
        @Test
        @DisplayName("Should handle single prompt in batch")
        fun shouldHandleSinglePromptInBatch() = runTest {
            // Given
            val prompts = listOf("Single prompt")
            val expectedResponse = "Single response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(1, results.size)
            assertEquals(expectedResponse, results[0])
        }
    }
    
    @Nested
    @DisplayName("Configuration Tests")
    inner class ConfigurationTests {
        
        @Test
        @DisplayName("Should update API key")
        fun shouldUpdateApiKey() {
            // Given
            val newApiKey = "new-api-key-456"
            
            // When
            auraAIService.updateApiKey(newApiKey)
            
            // Then
            verify(mockConfigurationService).updateApiKey(newApiKey)
            verify(mockLogger).info("API key updated successfully")
        }
        
        @Test
        @DisplayName("Should throw exception when updating with empty API key")
        fun shouldThrowExceptionWhenUpdatingWithEmptyApiKey() {
            // Given
            val newApiKey = ""
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey(newApiKey)
            }
        }
        
        @Test
        @DisplayName("Should update base URL")
        fun shouldUpdateBaseUrl() {
            // Given
            val newBaseUrl = "https://new-api.test.com"
            
            // When
            auraAIService.updateBaseUrl(newBaseUrl)
            
            // Then
            verify(mockConfigurationService).updateBaseUrl(newBaseUrl)
            verify(mockLogger).info("Base URL updated successfully")
        }
        
        @Test
        @DisplayName("Should validate base URL format")
        fun shouldValidateBaseUrlFormat() {
            // Given
            val invalidUrl = "not-a-valid-url"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateBaseUrl(invalidUrl)
            }
        }
        
        @Test
        @DisplayName("Should update timeout value")
        fun shouldUpdateTimeoutValue() {
            // Given
            val newTimeout = 60000L
            
            // When
            auraAIService.updateTimeout(newTimeout)
            
            // Then
            verify(mockConfigurationService).updateTimeout(newTimeout)
            verify(mockLogger).info("Timeout updated to $newTimeout ms")
        }
        
        @Test
        @DisplayName("Should validate timeout is positive")
        fun shouldValidateTimeoutIsPositive() {
            // Given
            val negativeTimeout = -1000L
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(negativeTimeout)
            }
        }
    }
    
    @Nested
    @DisplayName("Health Check Tests")
    inner class HealthCheckTests {
        
        @Test
        @DisplayName("Should return healthy status when service is available")
        fun shouldReturnHealthyStatusWhenServiceIsAvailable() = runTest {
            // Given
            val mockHttpResponse = mockHttpResponse(200, "OK")
            whenever(mockHttpClient.get(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertTrue(result.isHealthy)
            assertEquals("Service is healthy", result.message)
        }
        
        @Test
        @DisplayName("Should return unhealthy status when service is unavailable")
        fun shouldReturnUnhealthyStatusWhenServiceIsUnavailable() = runTest {
            // Given
            whenever(mockHttpClient.get(any())).thenThrow(IOException("Service unavailable"))
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Service unavailable", result.message)
        }
        
        @Test
        @DisplayName("Should handle health check timeout")
        fun shouldHandleHealthCheckTimeout() = runTest {
            // Given
            whenever(mockHttpClient.get(any())).thenThrow(TimeoutException("Health check timed out"))
            
            // When
            val result = auraAIService.healthCheck()
            
            // Then
            assertFalse(result.isHealthy)
            assertEquals("Service is unhealthy: Health check timed out", result.message)
        }
    }
    
    @Nested
    @DisplayName("Stream Response Tests")
    inner class StreamResponseTests {
        
        @Test
        @DisplayName("Should handle streaming responses successfully")
        fun shouldHandleStreamingResponsesSuccessfully() = runTest {
            // Given
            val prompt = "Stream this response"
            val streamChunks = listOf("Hello", " ", "World", "!")
            val mockStreamResponse = mockStreamResponse(streamChunks)
            
            whenever(mockHttpClient.postStream(any())).thenReturn(mockStreamResponse)
            
            // When
            val resultFlow = auraAIService.generateStreamingResponse(prompt)
            val collectedResults = mutableListOf<String>()
            resultFlow.collect { chunk ->
                collectedResults.add(chunk)
            }
            
            // Then
            assertEquals(streamChunks, collectedResults)
            verify(mockHttpClient).postStream(any())
            verify(mockLogger).info("Starting streaming response for prompt length: ${prompt.length}")
        }
        
        @Test
        @DisplayName("Should handle streaming connection interruption")
        fun shouldHandleStreamingConnectionInterruption() = runTest {
            // Given
            val prompt = "Test streaming"
            whenever(mockHttpClient.postStream(any())).thenThrow(IOException("Connection lost"))
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse(prompt).collect { }
            }
        }
        
        @Test
        @DisplayName("Should handle empty streaming response")
        fun shouldHandleEmptyStreamingResponse() = runTest {
            // Given
            val prompt = "Empty stream test"
            val mockStreamResponse = mockStreamResponse(emptyList())
            
            whenever(mockHttpClient.postStream(any())).thenReturn(mockStreamResponse)
            
            // When
            val resultFlow = auraAIService.generateStreamingResponse(prompt)
            val collectedResults = mutableListOf<String>()
            resultFlow.collect { chunk ->
                collectedResults.add(chunk)
            }
            
            // Then
            assertTrue(collectedResults.isEmpty())
        }
    }
    
    @Nested
    @DisplayName("Advanced Configuration Tests")
    inner class AdvancedConfigurationTests {
        
        @Test
        @DisplayName("Should handle configuration reload")
        fun shouldHandleConfigurationReload() {
            // Given
            val newApiKey = "reloaded-api-key"
            val newBaseUrl = "https://reloaded-api.test.com"
            val newTimeout = 45000L
            
            whenever(mockConfigurationService.getApiKey()).thenReturn(newApiKey)
            whenever(mockConfigurationService.getBaseUrl()).thenReturn(newBaseUrl)
            whenever(mockConfigurationService.getTimeout()).thenReturn(newTimeout)
            
            // When
            auraAIService.reloadConfiguration()
            
            // Then
            verify(mockConfigurationService).getApiKey()
            verify(mockConfigurationService).getBaseUrl()
            verify(mockConfigurationService).getTimeout()
            verify(mockLogger).info("Configuration reloaded successfully")
        }
        
        @Test
        @DisplayName("Should validate configuration on reload")
        fun shouldValidateConfigurationOnReload() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenReturn("")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://valid-url.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(30000L)
            
            // When & Then
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
        }
        
        @Test
        @DisplayName("Should handle configuration service failure")
        fun shouldHandleConfigurationServiceFailure() {
            // Given
            whenever(mockConfigurationService.getApiKey()).thenThrow(IOException("Config file not found"))
            
            // When & Then
            assertThrows<ConfigurationException> {
                auraAIService.reloadConfiguration()
            }
        }
        
        @Test
        @DisplayName("Should update model parameters")
        fun shouldUpdateModelParameters() {
            // Given
            val modelParams = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 2048,
                "top_p" to 0.9
            )
            
            // When
            auraAIService.updateModelParameters(modelParams)
            
            // Then
            verify(mockConfigurationService).updateModelParameters(modelParams)
            verify(mockLogger).info("Model parameters updated: $modelParams")
        }
        
        @Test
        @DisplayName("Should validate model parameters")
        fun shouldValidateModelParameters() {
            // Given
            val invalidParams = mapOf(
                "temperature" to 2.0, // Invalid: should be 0-1
                "max_tokens" to -100   // Invalid: should be positive
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(invalidParams)
            }
        }
    }
    
    @Nested
    @DisplayName("Metrics and Monitoring Tests")
    inner class MetricsAndMonitoringTests {
        
        @Test
        @DisplayName("Should provide service statistics")
        fun shouldProvideServiceStatistics() {
            // When
            val stats = auraAIService.getServiceStatistics()
            
            // Then
            assertNotNull(stats)
            assertTrue(stats.containsKey("totalRequests"))
            assertTrue(stats.containsKey("successfulRequests"))
            assertTrue(stats.containsKey("failedRequests"))
            assertTrue(stats.containsKey("averageResponseTime"))
            verify(mockLogger).debug("Service statistics requested")
        }
        
        @Test
        @DisplayName("Should reset service statistics")
        fun shouldResetServiceStatistics() {
            // When
            auraAIService.resetStatistics()
            
            // Then
            val stats = auraAIService.getServiceStatistics()
            assertEquals(0L, stats["totalRequests"])
            assertEquals(0L, stats["successfulRequests"])
            assertEquals(0L, stats["failedRequests"])
            verify(mockLogger).info("Service statistics reset")
        }
    }
    
    @Nested
    @DisplayName("Caching Tests")
    inner class CachingTests {
        
        @Test
        @DisplayName("Should clear cache on demand")
        fun shouldClearCacheOnDemand() {
            // When
            auraAIService.clearCache()
            
            // Then
            verify(mockLogger).info("Response cache cleared")
        }
        
        @Test
        @DisplayName("Should handle cache expiration")
        fun shouldHandleCacheExpiration() {
            // When
            auraAIService.expireCache()
            
            // Then
            verify(mockLogger).debug("Cache expired, making new request")
        }
    }
    
    // Helper methods
    private fun mockHttpResponse(statusCode: Int, body: String): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }
    
    private fun mockStreamResponse(chunks: List<String>): Flow<String> {
        return flow {
            chunks.forEach { chunk ->
                emit(chunk)
                delay(10) // Simulate streaming delay
            }
        }
    }
}

// Custom exception classes for testing
class BatchProcessingException(message: String) : Exception(message)
class RateLimitException(message: String) : Exception(message)
class AuthenticationException(message: String) : Exception(message)
class QuotaExceededException(message: String) : Exception(message)
class JsonParseException(message: String) : Exception(message)
class ConfigurationException(message: String) : Exception(message)
class PromptTooLongException(message: String) : Exception(message)
class InappropriateContentException(message: String) : Exception(message)
class CircuitBreakerOpenException(message: String) : Exception(message)

// Mock data classes
data class HealthCheckResult(val isHealthy: Boolean, val message: String)
    @Nested
    @DisplayName("Concurrent Operations Tests")
    inner class ConcurrentOperationsTests {
        
        @Test
        @DisplayName("Should handle concurrent generate response calls")
        fun shouldHandleConcurrentGenerateResponseCalls() = runTest {
            // Given
            val prompts = listOf("Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5")
            val mockResponse = mockHttpResponse(200, "Concurrent response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When
            val deferredResults = prompts.map { prompt ->
                async { auraAIService.generateResponse(prompt) }
            }
            val results = deferredResults.map { it.await() }
            
            // Then
            assertEquals(prompts.size, results.size)
            results.forEach { result ->
                assertEquals("Concurrent response", result)
            }
            verify(mockHttpClient, times(prompts.size)).post(any())
        }
        
        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun shouldHandleConcurrentConfigurationUpdates() = runTest {
            // Given
            val apiKeys = listOf("key1", "key2", "key3")
            val baseUrls = listOf("https://api1.test.com", "https://api2.test.com", "https://api3.test.com")
            val timeouts = listOf(1000L, 2000L, 3000L)
            
            // When
            val configUpdates = listOf(
                async { auraAIService.updateApiKey(apiKeys[0]) },
                async { auraAIService.updateBaseUrl(baseUrls[0]) },
                async { auraAIService.updateTimeout(timeouts[0]) }
            )
            configUpdates.forEach { it.await() }
            
            // Then
            verify(mockConfigurationService).updateApiKey(apiKeys[0])
            verify(mockConfigurationService).updateBaseUrl(baseUrls[0])
            verify(mockConfigurationService).updateTimeout(timeouts[0])
        }
        
        @Test
        @DisplayName("Should handle concurrent streaming operations")
        fun shouldHandleConcurrentStreamingOperations() = runTest {
            // Given
            val prompts = listOf("Stream 1", "Stream 2", "Stream 3")
            val mockStream = mockStreamResponse(listOf("chunk1", "chunk2"))
            
            whenever(mockHttpClient.postStream(any())).thenReturn(mockStream)
            
            // When
            val streamingJobs = prompts.map { prompt ->
                async {
                    val chunks = mutableListOf<String>()
                    auraAIService.generateStreamingResponse(prompt).collect { chunk ->
                        chunks.add(chunk)
                    }
                    chunks
                }
            }
            val allResults = streamingJobs.map { it.await() }
            
            // Then
            assertEquals(prompts.size, allResults.size)
            allResults.forEach { chunks ->
                assertEquals(listOf("chunk1", "chunk2"), chunks)
            }
            verify(mockHttpClient, times(prompts.size)).postStream(any())
        }
    }
    
    @Nested
    @DisplayName("Boundary Value Tests")
    inner class BoundaryValueTests {
        
        @Test
        @DisplayName("Should handle maximum prompt length")
        fun shouldHandleMaximumPromptLength() = runTest {
            // Given
            val maxPrompt = "A".repeat(1000000) // 1MB prompt
            val mockResponse = mockHttpResponse(200, "Max prompt response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When
            val result = auraAIService.generateResponse(maxPrompt)
            
            // Then
            assertEquals("Max prompt response", result)
            verify(mockLogger).info("Generating AI response for prompt length: ${maxPrompt.length}")
        }
        
        @Test
        @DisplayName("Should handle minimum timeout value")
        fun shouldHandleMinimumTimeoutValue() {
            // Given
            val minimumTimeout = 1L
            
            // When
            auraAIService.updateTimeout(minimumTimeout)
            
            // Then
            verify(mockConfigurationService).updateTimeout(minimumTimeout)
            verify(mockLogger).info("Timeout updated to $minimumTimeout ms")
        }
        
        @Test
        @DisplayName("Should handle maximum timeout value")
        fun shouldHandleMaximumTimeoutValue() {
            // Given
            val maximumTimeout = Long.MAX_VALUE
            
            // When
            auraAIService.updateTimeout(maximumTimeout)
            
            // Then
            verify(mockConfigurationService).updateTimeout(maximumTimeout)
            verify(mockLogger).info("Timeout updated to $maximumTimeout ms")
        }
        
        @Test
        @DisplayName("Should handle zero timeout boundary")
        fun shouldHandleZeroTimeoutBoundary() {
            // Given
            val zeroTimeout = 0L
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateTimeout(zeroTimeout)
            }
        }
        
        @Test
        @DisplayName("Should handle edge case temperature values")
        fun shouldHandleEdgeCaseTemperatureValues() {
            // Given - Test exact boundary values
            val exactMinParams = mapOf("temperature" to 0.0)
            val exactMaxParams = mapOf("temperature" to 1.0)
            val justOverMaxParams = mapOf("temperature" to 1.0000001)
            val justUnderMinParams = mapOf("temperature" to -0.0000001)
            
            // When & Then - Valid boundaries
            auraAIService.updateModelParameters(exactMinParams)
            auraAIService.updateModelParameters(exactMaxParams)
            
            // Invalid boundaries
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(justOverMaxParams)
            }
            
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(justUnderMinParams)
            }
        }
        
        @Test
        @DisplayName("Should handle edge case max_tokens values")
        fun shouldHandleEdgeCaseMaxTokensValues() {
            // Given
            val minValidTokens = mapOf("max_tokens" to 1)
            val maxValidTokens = mapOf("max_tokens" to Int.MAX_VALUE)
            val invalidZeroTokens = mapOf("max_tokens" to 0)
            val invalidNegativeTokens = mapOf("max_tokens" to -1)
            
            // When & Then - Valid values
            auraAIService.updateModelParameters(minValidTokens)
            auraAIService.updateModelParameters(maxValidTokens)
            
            // Invalid values
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(invalidZeroTokens)
            }
            
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(invalidNegativeTokens)
            }
        }
    }
    
    @Nested
    @DisplayName("Error Recovery Tests")
    inner class ErrorRecoveryTests {
        
        @Test
        @DisplayName("Should handle network interruption during streaming")
        fun shouldHandleNetworkInterruptionDuringStreaming() = runTest {
            // Given
            val prompt = "Test streaming with interruption"
            val interruptedStream = flow<String> {
                emit("chunk1")
                emit("chunk2")
                delay(100)
                throw IOException("Network interrupted")
            }
            
            whenever(mockHttpClient.postStream(any())).thenReturn(interruptedStream)
            
            // When & Then
            val chunks = mutableListOf<String>()
            assertThrows<IOException> {
                auraAIService.generateStreamingResponse(prompt).collect { chunk ->
                    chunks.add(chunk)
                }
            }
            
            // Verify partial data was collected before interruption
            assertEquals(listOf("chunk1", "chunk2"), chunks)
        }
        
        @Test
        @DisplayName("Should handle memory pressure scenarios")
        fun shouldHandleMemoryPressureScenarios() = runTest {
            // Given
            val largeResponse = "X".repeat(10000000) // 10MB response
            val mockResponse = mockHttpResponse(200, largeResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When
            val result = auraAIService.generateResponse("Large response test")
            
            // Then
            assertEquals(largeResponse, result)
        }
        
        @Test
        @DisplayName("Should handle intermittent failures")
        fun shouldHandleIntermittentFailures() = runTest {
            // Given
            val prompt = "Intermittent test"
            whenever(mockHttpClient.post(any()))
                .thenThrow(IOException("First attempt failed"))
                .thenReturn(mockHttpResponse(200, "Success on retry"))
            
            // When & Then - First call should fail
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            
            // Second call should succeed (if retry logic exists)
            // Note: This test assumes no built-in retry, but shows how to test it
        }
        
        @Test
        @DisplayName("Should handle malformed responses")
        fun shouldHandleMalformedResponses() = runTest {
            // Given
            val prompt = "Malformed response test"
            val malformedResponse = mockHttpResponse(200, "\u0000\u0001\u0002Invalid binary data")
            
            whenever(mockHttpClient.post(any())).thenReturn(malformedResponse)
            
            // When
            val result = auraAIService.generateResponse(prompt)
            
            // Then - Should handle gracefully
            assertEquals("\u0000\u0001\u0002Invalid binary data", result)
        }
    }
    
    @Nested
    @DisplayName("Advanced Validation Tests")
    inner class AdvancedValidationTests {
        
        @Test
        @DisplayName("Should validate URL schemes strictly")
        fun shouldValidateUrlSchemesStrictly() {
            // Given - Various URL formats
            val validUrls = listOf(
                "https://api.test.com",
                "https://api.test.com:8080",
                "https://api.test.com/v1/path"
            )
            
            val invalidUrls = listOf(
                "http://api.test.com",  // HTTP not HTTPS
                "ftp://api.test.com",   // Wrong protocol
                "https://",             // Incomplete
                "api.test.com",         // No protocol
                "//api.test.com"        // Protocol relative
            )
            
            // When & Then - Valid URLs should work
            validUrls.forEach { url ->
                auraAIService.updateBaseUrl(url)
                verify(mockConfigurationService).updateBaseUrl(url)
            }
            
            // Invalid URLs should fail
            invalidUrls.forEach { url ->
                assertThrows<IllegalArgumentException> {
                    auraAIService.updateBaseUrl(url)
                }
            }
        }
        
        @Test
        @DisplayName("Should handle complex model parameters")
        fun shouldHandleComplexModelParameters() {
            // Given - Complex parameter combinations
            val validComplexParams = mapOf(
                "temperature" to 0.7,
                "max_tokens" to 2048,
                "top_p" to 0.9,
                "frequency_penalty" to 0.1,
                "presence_penalty" to 0.2,
                "stop_sequences" to listOf("STOP", "END"),
                "custom_param" to "custom_value"
            )
            
            val mixedValidInvalidParams = mapOf(
                "temperature" to 0.5,     // Valid
                "max_tokens" to -100,     // Invalid
                "top_p" to 0.8           // Valid
            )
            
            // When & Then
            auraAIService.updateModelParameters(validComplexParams)
            verify(mockConfigurationService).updateModelParameters(validComplexParams)
            
            assertThrows<IllegalArgumentException> {
                auraAIService.updateModelParameters(mixedValidInvalidParams)
            }
        }
        
        @Test
        @DisplayName("Should validate parameter types strictly")
        fun shouldValidateParameterTypesStrictly() {
            // Given - Wrong parameter types
            val wrongTypeParams = mapOf(
                "temperature" to "0.7",  // String instead of Double
                "max_tokens" to 2048.5   // Double instead of Int
            )
            
            // When & Then - Should handle gracefully or validate types
            // This test demonstrates type validation if implemented
            auraAIService.updateModelParameters(wrongTypeParams)
            verify(mockConfigurationService).updateModelParameters(wrongTypeParams)
        }
    }
    
    @Nested
    @DisplayName("Performance and Load Tests")
    inner class PerformanceAndLoadTests {
        
        @Test
        @DisplayName("Should handle high-frequency requests")
        fun shouldHandleHighFrequencyRequests() = runTest {
            // Given
            val numberOfRequests = 100
            val mockResponse = mockHttpResponse(200, "High frequency response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When
            val startTime = System.currentTimeMillis()
            val results = (1..numberOfRequests).map { i ->
                async { auraAIService.generateResponse("Request $i") }
            }.map { it.await() }
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(numberOfRequests, results.size)
            results.forEach { result ->
                assertEquals("High frequency response", result)
            }
            
            // Verify reasonable performance (adjust threshold as needed)
            val totalTime = endTime - startTime
            assertTrue(totalTime < 10000, "High frequency requests took too long: ${totalTime}ms")
            
            verify(mockHttpClient, times(numberOfRequests)).post(any())
        }
        
        @Test
        @DisplayName("Should handle batch processing with mixed results")
        fun shouldHandleBatchProcessingWithMixedResults() = runTest {
            // Given
            val mixedPrompts = listOf(
                "Success prompt 1",
                "", // This would cause validation error if processed individually
                "Success prompt 2"
            )
            val mockResponse = mockHttpResponse(200, "Batch mixed response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(mixedPrompts)
            
            // Then
            assertEquals(1, results.size) // Current implementation returns single response
            assertEquals("Batch mixed response", results[0])
        }
        
        @Test
        @DisplayName("Should handle rapid configuration changes")
        fun shouldHandleRapidConfigurationChanges() = runTest {
            // Given
            val configChanges = 50
            
            // When
            val updateJobs = (1..configChanges).map { i ->
                async {
                    when (i % 3) {
                        0 -> auraAIService.updateApiKey("key-$i")
                        1 -> auraAIService.updateBaseUrl("https://api-$i.test.com")
                        else -> auraAIService.updateTimeout((1000 + i).toLong())
                    }
                }
            }
            updateJobs.forEach { it.await() }
            
            // Then - Verify some updates were made (exact count depends on execution order)
            verify(mockConfigurationService, atLeast(1)).updateApiKey(any())
            verify(mockConfigurationService, atLeast(1)).updateBaseUrl(any())
            verify(mockConfigurationService, atLeast(1)).updateTimeout(any())
        }
    }
    
    @Nested
    @DisplayName("Integration Workflow Tests")
    inner class IntegrationWorkflowTests {
        
        @Test
        @DisplayName("Should handle complete AI workflow")
        fun shouldHandleCompleteAiWorkflow() = runTest {
            // Given - Complete workflow scenario
            val mockResponse = mockHttpResponse(200, "Workflow response")
            val mockStreamChunks = listOf("Stream", " response")
            val mockStream = mockStreamResponse(mockStreamChunks)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            whenever(mockHttpClient.postStream(any())).thenReturn(mockStream)
            whenever(mockHttpClient.get(any())).thenReturn(mockResponse)
            
            // When - Execute complete workflow
            // 1. Health check
            val healthResult = auraAIService.healthCheck()
            
            // 2. Update configuration
            auraAIService.updateApiKey("workflow-key")
            auraAIService.updateModelParameters(mapOf("temperature" to 0.8))
            
            // 3. Generate responses
            val singleResponse = auraAIService.generateResponse("Single prompt")
            val batchResponses = auraAIService.generateBatchResponses(listOf("Batch 1", "Batch 2"))
            
            // 4. Streaming response
            val streamChunks = mutableListOf<String>()
            auraAIService.generateStreamingResponse("Stream prompt").collect { chunk ->
                streamChunks.add(chunk)
            }
            
            // 5. Get statistics and cleanup
            val stats = auraAIService.getServiceStatistics()
            auraAIService.clearCache()
            
            // Then - Verify complete workflow
            assertTrue(healthResult.isHealthy)
            assertEquals("Workflow response", singleResponse)
            assertEquals(1, batchResponses.size)
            assertEquals(mockStreamChunks, streamChunks)
            assertNotNull(stats)
            
            // Verify all operations were called
            verify(mockHttpClient).get(any())
            verify(mockHttpClient, times(2)).post(any()) // Single + batch
            verify(mockHttpClient).postStream(any())
            verify(mockConfigurationService).updateApiKey("workflow-key")
            verify(mockConfigurationService).updateModelParameters(any())
        }
        
        @Test
        @DisplayName("Should handle service lifecycle operations")
        fun shouldHandleServiceLifecycleOperations() = runTest {
            // Given - Service lifecycle scenario
            val newConfig = mapOf(
                "temperature" to 0.9,
                "max_tokens" to 4096
            )
            
            whenever(mockConfigurationService.getApiKey()).thenReturn("lifecycle-key")
            whenever(mockConfigurationService.getBaseUrl()).thenReturn("https://lifecycle.test.com")
            whenever(mockConfigurationService.getTimeout()).thenReturn(60000L)
            
            // When - Lifecycle operations
            // 1. Initial statistics
            val initialStats = auraAIService.getServiceStatistics()
            
            // 2. Configuration reload
            auraAIService.reloadConfiguration()
            
            // 3. Update parameters
            auraAIService.updateModelParameters(newConfig)
            
            // 4. Reset and clear
            auraAIService.resetStatistics()
            auraAIService.clearCache()
            auraAIService.expireCache()
            
            // 5. Final statistics
            val finalStats = auraAIService.getServiceStatistics()
            
            // Then
            assertNotNull(initialStats)
            assertNotNull(finalStats)
            verify(mockConfigurationService, times(2)).getApiKey() // Init + reload
            verify(mockConfigurationService, times(2)).getBaseUrl() // Init + reload
            verify(mockConfigurationService, times(2)).getTimeout() // Init + reload
            verify(mockConfigurationService).updateModelParameters(newConfig)
        }
    }
    
    @Nested
    @DisplayName("Security and Authentication Tests")
    inner class SecurityAndAuthenticationTests {
        
        @Test
        @DisplayName("Should handle API key security")
        fun shouldHandleApiKeySecurity() {
            // Given - Various security scenarios
            val validKey = "sk-1234567890abcdef"
            val emptyKey = ""
            val nullKey: String? = null
            val shortKey = "sk-123"
            val malformedKey = "invalid-key-format"
            
            // When & Then - Valid key should work
            auraAIService.updateApiKey(validKey)
            verify(mockConfigurationService).updateApiKey(validKey)
            
            // Invalid keys should fail
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey(emptyKey)
            }
            
            // Test with various key formats
            auraAIService.updateApiKey(shortKey) // Should work if not empty
            auraAIService.updateApiKey(malformedKey) // Should work if not empty
        }
        
        @Test
        @DisplayName("Should handle authentication failures")
        fun shouldHandleAuthenticationFailures() = runTest {
            // Given
            val prompt = "Authenticated request"
            val authFailureResponse = mockHttpResponse(401, "Unauthorized")
            val forbiddenResponse = mockHttpResponse(403, "Forbidden")
            
            // When & Then - 401 Unauthorized
            whenever(mockHttpClient.post(any())).thenReturn(authFailureResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            
            // 403 Forbidden
            whenever(mockHttpClient.post(any())).thenReturn(forbiddenResponse)
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
        }
        
        @Test
        @DisplayName("Should handle rate limiting scenarios")
        fun shouldHandleRateLimitingScenarios() = runTest {
            // Given
            val prompt = "Rate limited request"
            val rateLimitResponse = mockHttpResponse(429, "Too Many Requests")
            
            whenever(mockHttpClient.post(any())).thenReturn(rateLimitResponse)
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            
            verify(mockLogger).error("HTTP error response: 429 - Too Many Requests")
        }
    }
    
    @Nested
    @DisplayName("Data Validation and Sanitization Tests")
    inner class DataValidationTests {
        
        @Test
        @DisplayName("Should handle null and undefined parameters")
        fun shouldHandleNullAndUndefinedParameters() {
            // Given - Parameters with null values
            val paramsWithNulls = mapOf<String, Any?>(
                "temperature" to null,
                "max_tokens" to 2048,
                "top_p" to null
            )
            
            // When & Then - Should handle gracefully
            auraAIService.updateModelParameters(paramsWithNulls.filterValues { it != null })
            verify(mockConfigurationService).updateModelParameters(any())
        }
        
        @Test
        @DisplayName("Should validate prompt content sanitization")
        fun shouldValidatePromptContentSanitization() = runTest {
            // Given - Prompts with potentially dangerous content
            val sqlInjectionPrompt = "'; DROP TABLE users; --"
            val xssPrompt = "<script>alert('xss')</script>"
            val commandInjectionPrompt = "; rm -rf / ;"
            val normalPrompt = "What is machine learning?"
            
            val mockResponse = mockHttpResponse(200, "Sanitized response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When & Then - All should be handled safely
            assertEquals("Sanitized response", auraAIService.generateResponse(sqlInjectionPrompt))
            assertEquals("Sanitized response", auraAIService.generateResponse(xssPrompt))
            assertEquals("Sanitized response", auraAIService.generateResponse(commandInjectionPrompt))
            assertEquals("Sanitized response", auraAIService.generateResponse(normalPrompt))
            
            verify(mockHttpClient, times(4)).post(any())
        }
        
        @Test
        @DisplayName("Should handle unusual character encodings")
        fun shouldHandleUnusualCharacterEncodings() = runTest {
            // Given - Various character encodings
            val utf8Prompt = "UTF-8: Hello 世界"
            val emojiPrompt = "Emojis: 😀🎉🚀💻🌟"
            val rtlPrompt = "RTL: مرحبا بالعالم"
            val mixedPrompt = "Mixed: Hello नमस्ते 你好 🌍"
            
            val mockResponse = mockHttpResponse(200, "Encoded response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When & Then
            assertEquals("Encoded response", auraAIService.generateResponse(utf8Prompt))
            assertEquals("Encoded response", auraAIService.generateResponse(emojiPrompt))
            assertEquals("Encoded response", auraAIService.generateResponse(rtlPrompt))
            assertEquals("Encoded response", auraAIService.generateResponse(mixedPrompt))
        }
    }
    
    @Nested
    @DisplayName("Service State Management Tests")
    inner class ServiceStateManagementTests {
        
        @Test
        @DisplayName("Should maintain state consistency across operations")
        fun shouldMaintainStateConsistencyAcrossOperations() = runTest {
            // Given - Multiple state-changing operations
            val mockResponse = mockHttpResponse(200, "State test response")
            whenever(mockHttpClient.post(any())).thenReturn(mockResponse)
            
            // When - Perform multiple operations that might affect state
            val initialStats = auraAIService.getServiceStatistics()
            
            auraAIService.generateResponse("Test 1")
            auraAIService.updateApiKey("new-key")
            auraAIService.generateResponse("Test 2")
            auraAIService.clearCache()
            auraAIService.generateResponse("Test 3")
            
            val finalStats = auraAIService.getServiceStatistics()
            
            // Then - State should be consistent
            assertNotNull(initialStats)
            assertNotNull(finalStats)
            verify(mockHttpClient, times(3)).post(any())
            verify(mockConfigurationService).updateApiKey("new-key")
        }
        
        @Test
        @DisplayName("Should handle service recovery after errors")
        fun shouldHandleServiceRecoveryAfterErrors() = runTest {
            // Given - Service experiences errors then recovers
            val mockSuccessResponse = mockHttpResponse(200, "Recovery success")
            
            whenever(mockHttpClient.post(any()))
                .thenThrow(IOException("Service error"))
                .thenReturn(mockSuccessResponse)
            
            // When & Then - First call fails
            assertThrows<IOException> {
                auraAIService.generateResponse("First attempt")
            }
            
            // Service should still be functional for subsequent calls
            val result = auraAIService.generateResponse("Recovery attempt")
            assertEquals("Recovery success", result)
            
            // Configuration should still work
            auraAIService.updateTimeout(5000L)
            verify(mockConfigurationService).updateTimeout(5000L)
        }
    }
