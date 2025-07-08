package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
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
            verify(mockLogger).error("Prompt cannot be empty")
        }
        
        @Test
        @DisplayName("Should handle null prompt")
        fun shouldHandleNullPrompt() = runTest {
            // Given
            val prompt: String? = null
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.generateResponse(prompt!!)
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
            verify(mockLogger).error("Request timed out")
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
            verify(mockLogger).error("Network connection error: Connection refused")
        }
        
        @Test
        @DisplayName("Should handle malformed JSON response")
        fun shouldHandleMalformedJsonResponse() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(200, "Invalid JSON{")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<JsonParseException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Failed to parse JSON response")
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
            val expectedResponses = listOf("Response 1", "Response 2", "Response 3")
            val mockHttpResponse = mockHttpResponse(200, expectedResponses.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(expectedResponses.size, results.size)
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
            val expectedResponse = listOf("Single response")
            val mockHttpResponse = mockHttpResponse(200, expectedResponse.toString())
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = auraAIService.generateBatchResponses(prompts)
            
            // Then
            assertEquals(1, results.size)
            assertEquals(expectedResponse[0], results[0])
        }
        
        @Test
        @DisplayName("Should handle batch processing with partial failures")
        fun shouldHandleBatchProcessingWithPartialFailures() = runTest {
            // Given
            val prompts = listOf("Valid prompt", "", "Another valid prompt")
            
            // When & Then
            assertThrows<BatchProcessingException> {
                auraAIService.generateBatchResponses(prompts)
            }
            verify(mockLogger).error("Batch processing failed due to invalid prompts")
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
        @DisplayName("Should throw exception when updating with null API key")
        fun shouldThrowExceptionWhenUpdatingWithNullApiKey() {
            // Given
            val newApiKey: String? = null
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.updateApiKey(newApiKey!!)
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
            verify(mockLogger).error("Invalid base URL format: $invalidUrl")
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
            verify(mockLogger).error("Timeout must be positive: $negativeTimeout")
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
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle concurrent requests gracefully")
        fun shouldHandleConcurrentRequestsGracefully() = runTest {
            // Given
            val prompts = (1..10).map { "Prompt $it" }
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val results = prompts.map { prompt ->
                async { auraAIService.generateResponse(prompt) }
            }.awaitAll()
            
            // Then
            assertEquals(10, results.size)
            verify(mockHttpClient, times(10)).post(any())
        }
        
        @Test
        @DisplayName("Should handle rate limiting gracefully")
        fun shouldHandleRateLimitingGracefully() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(429, "Rate limit exceeded")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<RateLimitException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).warn("Rate limit exceeded, retrying...")
        }
        
        @Test
        @DisplayName("Should handle authentication failures")
        fun shouldHandleAuthenticationFailures() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(401, "Unauthorized")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<AuthenticationException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Authentication failed: Invalid API key")
        }
        
        @Test
        @DisplayName("Should handle quota exceeded")
        fun shouldHandleQuotaExceeded() = runTest {
            // Given
            val prompt = "Test prompt"
            val mockHttpResponse = mockHttpResponse(403, "Quota exceeded")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When & Then
            assertThrows<QuotaExceededException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("API quota exceeded")
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
    
    // Helper methods
    private fun mockHttpResponse(statusCode: Int, body: String): HttpResponse {
        val mockResponse = mock<HttpResponse>()
        whenever(mockResponse.statusCode).thenReturn(statusCode)
        whenever(mockResponse.body).thenReturn(body)
        return mockResponse
    }
    
    private fun async(block: suspend () -> Any): Deferred<Any> {
        // Mock async implementation for testing
        return mock<Deferred<Any>>()
    }
    
    private fun <T> List<Deferred<T>>.awaitAll(): List<T> {
        // Mock awaitAll implementation for testing
        return this.map { mock<T>() }
    }
}

// Custom exception classes for testing
class BatchProcessingException(message: String) : Exception(message)
class RateLimitException(message: String) : Exception(message)
class AuthenticationException(message: String) : Exception(message)
class QuotaExceededException(message: String) : Exception(message)
class JsonParseException(message: String) : Exception(message)

// Mock data classes
data class HealthCheckResult(val isHealthy: Boolean, val message: String)
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
                auraAIService.generateStreamingResponse(prompt).collect()
            }
            verify(mockLogger).error("Streaming connection interrupted: Connection lost")
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
            verify(mockLogger).warn("Received empty streaming response")
        }
        
        @Test
        @DisplayName("Should handle streaming with backpressure")
        fun shouldHandleStreamingWithBackpressure() = runTest {
            // Given
            val prompt = "Backpressure test"
            val largeChunks = (1..1000).map { "Chunk $it" }
            val mockStreamResponse = mockStreamResponse(largeChunks)
            
            whenever(mockHttpClient.postStream(any())).thenReturn(mockStreamResponse)
            
            // When
            val resultFlow = auraAIService.generateStreamingResponse(prompt)
            var chunkCount = 0
            resultFlow.collect { chunk ->
                chunkCount++
                delay(1) // Simulate slow processing
            }
            
            // Then
            assertEquals(1000, chunkCount)
            verify(mockLogger).info("Streaming completed with $chunkCount chunks")
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
            verify(mockLogger).error("Configuration validation failed: API key cannot be empty")
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
            verify(mockLogger).error("Failed to reload configuration: Config file not found")
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
            verify(mockLogger).error("Invalid model parameters: temperature must be between 0 and 1")
        }
    }
    
    @Nested
    @DisplayName("Metrics and Monitoring Tests")
    inner class MetricsAndMonitoringTests {
        
        @Test
        @DisplayName("Should track request metrics")
        fun shouldTrackRequestMetrics() = runTest {
            // Given
            val prompt = "Test metrics"
            val expectedResponse = "Metrics response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockLogger).debug("Request latency: {} ms", any())
            verify(mockLogger).debug("Response size: {} bytes", any())
        }
        
        @Test
        @DisplayName("Should track error metrics")
        fun shouldTrackErrorMetrics() = runTest {
            // Given
            val prompt = "Error metrics test"
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).error("Request failed with error type: IOException")
        }
        
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
        @DisplayName("Should cache successful responses")
        fun shouldCacheSuccessfulResponses() = runTest {
            // Given
            val prompt = "Cacheable prompt"
            val expectedResponse = "Cached response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result1 = auraAIService.generateResponse(prompt)
            val result2 = auraAIService.generateResponse(prompt) // Should use cache
            
            // Then
            assertEquals(expectedResponse, result1)
            assertEquals(expectedResponse, result2)
            verify(mockHttpClient, times(1)).post(any()) // Only one actual HTTP call
            verify(mockLogger).debug("Cache hit for prompt hash: {}", any())
        }
        
        @Test
        @DisplayName("Should not cache error responses")
        fun shouldNotCacheErrorResponses() = runTest {
            // Given
            val prompt = "Error prompt"
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Network error"))
            
            // When & Then
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt)
            }
            assertThrows<IOException> {
                auraAIService.generateResponse(prompt) // Should try again, not use cache
            }
            verify(mockHttpClient, times(2)).post(any()) // Two HTTP calls attempted
        }
        
        @Test
        @DisplayName("Should respect cache TTL")
        fun shouldRespectCacheTTL() = runTest {
            // Given
            val prompt = "TTL test prompt"
            val expectedResponse = "TTL response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            
            // Simulate cache expiration
            auraAIService.expireCache()
            
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient, times(2)).post(any()) // Two HTTP calls due to cache expiration
            verify(mockLogger).debug("Cache expired, making new request")
        }
        
        @Test
        @DisplayName("Should clear cache on demand")
        fun shouldClearCacheOnDemand() = runTest {
            // Given
            val prompt = "Clear cache test"
            val expectedResponse = "Cache cleared response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            auraAIService.generateResponse(prompt)
            auraAIService.clearCache()
            auraAIService.generateResponse(prompt)
            
            // Then
            verify(mockHttpClient, times(2)).post(any()) // Two HTTP calls due to cache clear
            verify(mockLogger).info("Response cache cleared")
        }
    }
    
    @Nested
    @DisplayName("Security and Validation Tests")
    inner class SecurityAndValidationTests {
        
        @Test
        @DisplayName("Should sanitize input prompts")
        fun shouldSanitizeInputPrompts() = runTest {
            // Given
            val maliciousPrompt = "Valid prompt <script>alert('xss')</script>"
            val sanitizedPrompt = "Valid prompt &lt;script&gt;alert('xss')&lt;/script&gt;"
            val expectedResponse = "Sanitized response"
            val mockHttpResponse = mockHttpResponse(200, expectedResponse)
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val result = auraAIService.generateResponse(maliciousPrompt)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockLogger).debug("Input sanitized: {} characters removed", any())
        }
        
        @Test
        @DisplayName("Should validate prompt length limits")
        fun shouldValidatePromptLengthLimits() = runTest {
            // Given
            val oversizedPrompt = "A".repeat(100000) // Assuming 50k is the limit
            
            // When & Then
            assertThrows<PromptTooLongException> {
                auraAIService.generateResponse(oversizedPrompt)
            }
            verify(mockLogger).error("Prompt exceeds maximum length: {} characters", oversizedPrompt.length)
        }
        
        @Test
        @DisplayName("Should detect and block inappropriate content")
        fun shouldDetectAndBlockInappropriateContent() = runTest {
            // Given
            val inappropriatePrompt = "Generate harmful content about violence"
            
            // When & Then
            assertThrows<InappropriateContentException> {
                auraAIService.generateResponse(inappropriatePrompt)
            }
            verify(mockLogger).warn("Inappropriate content detected and blocked")
        }
        
        @Test
        @DisplayName("Should rate limit requests per user")
        fun shouldRateLimitRequestsPerUser() = runTest {
            // Given
            val userId = "test-user-123"
            val prompt = "Rate limit test"
            val mockHttpResponse = mockHttpResponse(200, "Response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When - Make multiple rapid requests
            repeat(10) {
                auraAIService.generateResponse(prompt, userId)
            }
            
            // Then - Should trigger rate limiting after threshold
            assertThrows<RateLimitException> {
                auraAIService.generateResponse(prompt, userId)
            }
            verify(mockLogger).warn("Rate limit exceeded for user: {}", userId)
        }
    }
    
    @Nested
    @DisplayName("Integration and Performance Tests")
    inner class IntegrationAndPerformanceTests {
        
        @Test
        @DisplayName("Should handle high load scenarios")
        fun shouldHandleHighLoadScenarios() = runTest {
            // Given
            val prompts = (1..100).map { "High load prompt $it" }
            val mockHttpResponse = mockHttpResponse(200, "Load test response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            val startTime = System.currentTimeMillis()
            val results = prompts.map { prompt ->
                async { auraAIService.generateResponse(prompt) }
            }.awaitAll()
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(100, results.size)
            val totalTime = endTime - startTime
            assertTrue(totalTime < 10000) // Should complete within 10 seconds
            verify(mockLogger).info("High load test completed: {} requests in {} ms", 100, totalTime)
        }
        
        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() = runTest {
            // Given
            val largePrompt = "Large prompt: " + "X".repeat(50000)
            val mockHttpResponse = mockHttpResponse(200, "Memory test response")
            
            whenever(mockHttpClient.post(any())).thenReturn(mockHttpResponse)
            
            // When
            repeat(10) {
                auraAIService.generateResponse(largePrompt)
            }
            
            // Then
            verify(mockHttpClient, times(10)).post(any())
            verify(mockLogger, atLeastOnce()).debug("Memory usage: {} MB", any())
        }
        
        @Test
        @DisplayName("Should handle circuit breaker activation")
        fun shouldHandleCircuitBreakerActivation() = runTest {
            // Given
            val prompt = "Circuit breaker test"
            whenever(mockHttpClient.post(any())).thenThrow(IOException("Service down"))
            
            // When - Trigger multiple failures to activate circuit breaker
            repeat(5) {
                try {
                    auraAIService.generateResponse(prompt)
                } catch (e: IOException) {
                    // Expected failures
                }
            }
            
            // Then - Circuit breaker should be open
            assertThrows<CircuitBreakerOpenException> {
                auraAIService.generateResponse(prompt)
            }
            verify(mockLogger).warn("Circuit breaker is now OPEN")
        }
    }
    
    // Additional helper methods
    private fun mockStreamResponse(chunks: List<String>): Flow<String> {
        return flow {
            chunks.forEach { chunk ->
                emit(chunk)
                delay(10) // Simulate streaming delay
            }
        }
    }
}

// Additional custom exception classes for comprehensive testing
class ConfigurationException(message: String) : Exception(message)
class PromptTooLongException(message: String) : Exception(message)
class InappropriateContentException(message: String) : Exception(message)
class CircuitBreakerOpenException(message: String) : Exception(message)

// Additional mock extensions
private fun mockHttpClient(response: HttpResponse): HttpClient {
    val client = mock<HttpClient>()
    whenever(client.post(any())).thenReturn(response)
    whenever(client.get(any())).thenReturn(response)
    return client
}

private fun mockConfigurationService(apiKey: String, baseUrl: String, timeout: Long): ConfigurationService {
    val service = mock<ConfigurationService>()
    whenever(service.getApiKey()).thenReturn(apiKey)
    whenever(service.getBaseUrl()).thenReturn(baseUrl)
    whenever(service.getTimeout()).thenReturn(timeout)
    return service
}