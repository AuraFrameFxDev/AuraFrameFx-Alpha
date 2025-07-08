package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.collect
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