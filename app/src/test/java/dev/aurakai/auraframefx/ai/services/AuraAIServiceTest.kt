package dev.aurakai.auraframefx.ai.services

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import org.mockito.junit.jupiter.MockitoExtension
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import kotlin.test.assertFailsWith

@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("AuraAIService Tests")
class AuraAIServiceTest {

    @Mock
    private lateinit var mockHttpClient: HttpClient
    
    @Mock
    private lateinit var mockConfigurationService: ConfigurationService
    
    @Mock
    private lateinit var mockLogger: Logger
    
    private lateinit var auraAIService: AuraAIService
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        auraAIService = AuraAIService(mockHttpClient, mockConfigurationService, mockLogger)
    }
    
    @AfterEach
    fun tearDown() {
        // Clean up any resources if needed
        reset(mockHttpClient, mockConfigurationService, mockLogger)
    }

    @Nested
    @DisplayName("Service Initialization Tests")
    inner class ServiceInitializationTests {
        
        @Test
        @DisplayName("Should initialize service with valid dependencies")
        fun shouldInitializeServiceWithValidDependencies() {
            // Given
            val service = AuraAIService(mockHttpClient, mockConfigurationService, mockLogger)
            
            // When & Then
            assertNotNull(service)
            assertTrue(service.isInitialized())
        }
        
        @Test
        @DisplayName("Should fail initialization with null dependencies")
        fun shouldFailInitializationWithNullDependencies() {
            // Given & When & Then
            assertFailsWith<IllegalArgumentException> {
                AuraAIService(null, mockConfigurationService, mockLogger)
            }
            
            assertFailsWith<IllegalArgumentException> {
                AuraAIService(mockHttpClient, null, mockLogger)
            }
            
            assertFailsWith<IllegalArgumentException> {
                AuraAIService(mockHttpClient, mockConfigurationService, null)
            }
        }
    }

    @Nested
    @DisplayName("AI Query Processing Tests")
    inner class AIQueryProcessingTests {
        
        @Test
        @DisplayName("Should process valid AI query successfully")
        fun shouldProcessValidAIQuerySuccessfully() = runTest {
            // Given
            val query = "What is the weather like today?"
            val expectedResponse = AIResponse(
                id = "test-id",
                content = "The weather is sunny and warm today.",
                model = "gpt-4",
                usage = AIUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25)
            )
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenReturn(expectedResponse)
            
            // When
            val result = auraAIService.processQuery(query)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse.content, result.content)
            assertEquals(expectedResponse.model, result.model)
            verify(mockHttpClient).post(any<AIRequest>())
            verify(mockLogger).info("Processing AI query: $query")
        }
        
        @Test
        @DisplayName("Should handle empty query gracefully")
        fun shouldHandleEmptyQueryGracefully() = runTest {
            // Given
            val emptyQuery = ""
            
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                auraAIService.processQuery(emptyQuery)
            }
            
            verify(mockLogger).warn("Empty query provided")
        }
        
        @Test
        @DisplayName("Should handle null query gracefully")
        fun shouldHandleNullQueryGracefully() = runTest {
            // Given
            val nullQuery: String? = null
            
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                auraAIService.processQuery(nullQuery)
            }
            
            verify(mockLogger).warn("Null query provided")
        }
        
        @Test
        @DisplayName("Should handle very long query")
        fun shouldHandleVeryLongQuery() = runTest {
            // Given
            val longQuery = "A".repeat(10000)
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockConfigurationService.getMaxQueryLength()).thenReturn(5000)
            
            // When & Then
            assertFailsWith<QueryTooLongException> {
                auraAIService.processQuery(longQuery)
            }
            
            verify(mockLogger).warn("Query exceeds maximum length: ${longQuery.length}")
        }
        
        @Test
        @DisplayName("Should handle special characters in query")
        fun shouldHandleSpecialCharactersInQuery() = runTest {
            // Given
            val specialQuery = "What's the meaning of life? 🤔 \n\t\"Special\" characters & symbols!"
            val expectedResponse = AIResponse(
                id = "test-id",
                content = "The meaning of life is 42.",
                model = "gpt-4",
                usage = AIUsage(promptTokens = 20, completionTokens = 10, totalTokens = 30)
            )
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenReturn(expectedResponse)
            
            // When
            val result = auraAIService.processQuery(specialQuery)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse.content, result.content)
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle network timeout gracefully")
        fun shouldHandleNetworkTimeoutGracefully() = runTest {
            // Given
            val query = "Test query"
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenThrow(TimeoutException("Network timeout"))
            
            // When & Then
            assertFailsWith<AIServiceException> {
                auraAIService.processQuery(query)
            }
            
            verify(mockLogger).error("Network timeout occurred", any<TimeoutException>())
        }
        
        @Test
        @DisplayName("Should handle API rate limiting")
        fun shouldHandleAPIRateLimiting() = runTest {
            // Given
            val query = "Test query"
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenThrow(RateLimitException("Too many requests"))
            
            // When & Then
            assertFailsWith<AIServiceException> {
                auraAIService.processQuery(query)
            }
            
            verify(mockLogger).warn("Rate limit exceeded, retrying...")
        }
        
        @Test
        @DisplayName("Should handle invalid API key")
        fun shouldHandleInvalidAPIKey() = runTest {
            // Given
            val query = "Test query"
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("invalid-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenThrow(UnauthorizedException("Invalid API key"))
            
            // When & Then
            assertFailsWith<AIServiceException> {
                auraAIService.processQuery(query)
            }
            
            verify(mockLogger).error("Authentication failed with AI service")
        }
        
        @Test
        @DisplayName("Should handle malformed API response")
        fun shouldHandleMalformedAPIResponse() = runTest {
            // Given
            val query = "Test query"
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenThrow(JsonParseException("Invalid JSON"))
            
            // When & Then
            assertFailsWith<AIServiceException> {
                auraAIService.processQuery(query)
            }
            
            verify(mockLogger).error("Failed to parse AI service response", any<JsonParseException>())
        }
    }

    @Nested
    @DisplayName("Configuration Management Tests")
    inner class ConfigurationManagementTests {
        
        @Test
        @DisplayName("Should validate configuration on startup")
        fun shouldValidateConfigurationOnStartup() {
            // Given
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockConfigurationService.getMaxQueryLength()).thenReturn(4000)
            
            // When
            val isValid = auraAIService.validateConfiguration()
            
            // Then
            assertTrue(isValid)
            verify(mockConfigurationService).getAIApiKey()
            verify(mockConfigurationService).getAIModel()
            verify(mockConfigurationService).getMaxQueryLength()
        }
        
        @Test
        @DisplayName("Should fail validation with missing API key")
        fun shouldFailValidationWithMissingAPIKey() {
            // Given
            whenever(mockConfigurationService.getAIApiKey()).thenReturn(null)
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            
            // When
            val isValid = auraAIService.validateConfiguration()
            
            // Then
            assertFalse(isValid)
            verify(mockLogger).error("AI API key is not configured")
        }
        
        @Test
        @DisplayName("Should fail validation with invalid model")
        fun shouldFailValidationWithInvalidModel() {
            // Given
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("valid-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("invalid-model")
            
            // When
            val isValid = auraAIService.validateConfiguration()
            
            // Then
            assertFalse(isValid)
            verify(mockLogger).error("Unsupported AI model: invalid-model")
        }
        
        @Test
        @DisplayName("Should update configuration dynamically")
        fun shouldUpdateConfigurationDynamically() {
            // Given
            val newApiKey = "new-api-key"
            val newModel = "gpt-4-turbo"
            
            // When
            auraAIService.updateConfiguration(newApiKey, newModel)
            
            // Then
            verify(mockConfigurationService).updateAIApiKey(newApiKey)
            verify(mockConfigurationService).updateAIModel(newModel)
            verify(mockLogger).info("AI service configuration updated")
        }
    }

    @Nested
    @DisplayName("Batch Processing Tests")
    inner class BatchProcessingTests {
        
        @Test
        @DisplayName("Should process multiple queries in batch")
        fun shouldProcessMultipleQueriesInBatch() = runTest {
            // Given
            val queries = listOf("Query 1", "Query 2", "Query 3")
            val expectedResponses = queries.mapIndexed { index, query ->
                AIResponse(
                    id = "id-$index",
                    content = "Response to $query",
                    model = "gpt-4",
                    usage = AIUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25)
                )
            }
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenReturn(*expectedResponses.toTypedArray())
            
            // When
            val results = auraAIService.processBatchQueries(queries)
            
            // Then
            assertEquals(queries.size, results.size)
            results.forEachIndexed { index, result ->
                assertEquals(expectedResponses[index].content, result.content)
            }
            verify(mockHttpClient, times(queries.size)).post(any<AIRequest>())
        }
        
        @Test
        @DisplayName("Should handle batch processing with some failures")
        fun shouldHandleBatchProcessingWithSomeFailures() = runTest {
            // Given
            val queries = listOf("Query 1", "Query 2", "Query 3")
            val successResponse = AIResponse(
                id = "success-id",
                content = "Success response",
                model = "gpt-4",
                usage = AIUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25)
            )
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>()))
                .thenReturn(successResponse)
                .thenThrow(RuntimeException("API Error"))
                .thenReturn(successResponse)
            
            // When
            val results = auraAIService.processBatchQueries(queries)
            
            // Then
            assertEquals(2, results.size) // Only successful responses
            verify(mockLogger).error("Failed to process query: Query 2", any<RuntimeException>())
        }
        
        @Test
        @DisplayName("Should handle empty batch gracefully")
        fun shouldHandleEmptyBatchGracefully() = runTest {
            // Given
            val emptyQueries = emptyList<String>()
            
            // When
            val results = auraAIService.processBatchQueries(emptyQueries)
            
            // Then
            assertTrue(results.isEmpty())
            verify(mockLogger).info("Empty batch provided for processing")
        }
    }

    @Nested
    @DisplayName("Async Operation Tests")
    inner class AsyncOperationTests {
        
        @Test
        @DisplayName("Should handle concurrent requests properly")
        fun shouldHandleConcurrentRequestsProperly() = runTest {
            // Given
            val query = "Concurrent test query"
            val expectedResponse = AIResponse(
                id = "concurrent-id",
                content = "Concurrent response",
                model = "gpt-4",
                usage = AIUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25)
            )
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenReturn(expectedResponse)
            
            // When
            val futures = (1..5).map { 
                CompletableFuture.supplyAsync { 
                    runBlocking { auraAIService.processQuery(query) }
                }
            }
            
            val results = futures.map { it.get(10, TimeUnit.SECONDS) }
            
            // Then
            assertEquals(5, results.size)
            results.forEach { result ->
                assertEquals(expectedResponse.content, result.content)
            }
            verify(mockHttpClient, times(5)).post(any<AIRequest>())
        }
        
        @Test
        @DisplayName("Should handle timeout in async operations")
        fun shouldHandleTimeoutInAsyncOperations() = runTest {
            // Given
            val query = "Timeout test query"
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenAnswer { 
                Thread.sleep(15000) // Simulate long response
                throw TimeoutException("Request timed out")
            }
            
            // When & Then
            assertFailsWith<AIServiceException> {
                auraAIService.processQueryWithTimeout(query, 5000)
            }
            
            verify(mockLogger).warn("Request timed out after 5000ms")
        }
    }

    @Nested
    @DisplayName("Metrics and Monitoring Tests")
    inner class MetricsAndMonitoringTests {
        
        @Test
        @DisplayName("Should track query processing metrics")
        fun shouldTrackQueryProcessingMetrics() = runTest {
            // Given
            val query = "Metrics test query"
            val expectedResponse = AIResponse(
                id = "metrics-id",
                content = "Metrics response",
                model = "gpt-4",
                usage = AIUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25)
            )
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenReturn(expectedResponse)
            
            // When
            val result = auraAIService.processQuery(query)
            val metrics = auraAIService.getMetrics()
            
            // Then
            assertNotNull(result)
            assertEquals(1, metrics.totalQueries)
            assertEquals(1, metrics.successfulQueries)
            assertEquals(0, metrics.failedQueries)
            assertTrue(metrics.averageResponseTime > 0)
        }
        
        @Test
        @DisplayName("Should track error metrics")
        fun shouldTrackErrorMetrics() = runTest {
            // Given
            val query = "Error metrics test"
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenThrow(RuntimeException("API Error"))
            
            // When
            try {
                auraAIService.processQuery(query)
            } catch (e: AIServiceException) {
                // Expected
            }
            
            val metrics = auraAIService.getMetrics()
            
            // Then
            assertEquals(1, metrics.totalQueries)
            assertEquals(0, metrics.successfulQueries)
            assertEquals(1, metrics.failedQueries)
        }
        
        @Test
        @DisplayName("Should reset metrics when requested")
        fun shouldResetMetricsWhenRequested() = runTest {
            // Given
            val query = "Reset metrics test"
            val expectedResponse = AIResponse(
                id = "reset-id",
                content = "Reset response",
                model = "gpt-4",
                usage = AIUsage(promptTokens = 10, completionTokens = 15, totalTokens = 25)
            )
            
            whenever(mockConfigurationService.getAIApiKey()).thenReturn("test-api-key")
            whenever(mockConfigurationService.getAIModel()).thenReturn("gpt-4")
            whenever(mockHttpClient.post(any<AIRequest>())).thenReturn(expectedResponse)
            
            // When
            auraAIService.processQuery(query)
            var metrics = auraAIService.getMetrics()
            assertEquals(1, metrics.totalQueries)
            
            auraAIService.resetMetrics()
            metrics = auraAIService.getMetrics()
            
            // Then
            assertEquals(0, metrics.totalQueries)
            assertEquals(0, metrics.successfulQueries)
            assertEquals(0, metrics.failedQueries)
        }
    }
}