package dev.aurakai.auraframefx.ai.services

import io.mockk.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import java.util.stream.Stream
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeoutException

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAIServiceTest {

    private lateinit var auraAIService: AuraAIService
    private val mockHttpClient = mockk<HttpClient>()
    private val mockApiClient = mockk<ApiClient>()
    private val mockConfigService = mockk<ConfigService>()
    private val mockLogger = mockk<Logger>()

    @BeforeEach
    fun setUp() {
        clearAllMocks()
        auraAIService = AuraAIService(mockHttpClient, mockApiClient, mockConfigService, mockLogger)
    }

    @AfterEach
    fun tearDown() {
        unmockkAll()
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize service with valid configuration")
        fun `should initialize service with valid configuration`() {
            // Given
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig

            // When
            val result = auraAIService.initialize()

            // Then
            assertTrue(result)
            assertTrue(auraAIService.isInitialized())
            verify { mockConfigService.getConfig("ai") }
        }

        @Test
        @DisplayName("Should fail initialization with invalid configuration")
        fun `should fail initialization with invalid configuration`() {
            // Given
            val invalidConfig = mapOf<String, String>()
            every { mockConfigService.getConfig("ai") } returns invalidConfig

            // When
            val result = auraAIService.initialize()

            // Then
            assertFalse(result)
            assertFalse(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle null configuration gracefully")
        fun `should handle null configuration gracefully`() {
            // Given
            every { mockConfigService.getConfig("ai") } returns null

            // When
            val result = auraAIService.initialize()

            // Then
            assertFalse(result)
            assertFalse(auraAIService.isInitialized())
        }
    }

    @Nested
    @DisplayName("AI Query Tests")
    inner class AIQueryTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should successfully process valid query")
        fun `should successfully process valid query`() = runTest {
            // Given
            val query = "What is the meaning of life?"
            val expectedResponse = AIResponse(
                content = "The meaning of life is 42",
                confidence = 0.95,
                tokensUsed = 15
            )
            coEvery { mockApiClient.sendQuery(any()) } returns expectedResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedResponse, result)
            coVerify { mockApiClient.sendQuery(query) }
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\t\n"])
        @DisplayName("Should handle empty or whitespace queries")
        fun `should handle empty or whitespace queries`(query: String) = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle null query gracefully")
        fun `should handle null query gracefully`() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.processQuery(null)
            }
        }

        @ParameterizedTest
        @CsvSource(
            "Simple query, 50",
            "Medium length query with more words, 100",
            "This is a very long query that contains many words and should test the token counting functionality properly, 200"
        )
        @DisplayName("Should handle queries of different lengths")
        fun `should handle queries of different lengths`(query: String, expectedTokens: Int) = runTest {
            // Given
            val response = AIResponse(
                content = "Test response",
                confidence = 0.8,
                tokensUsed = expectedTokens
            )
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedTokens, result.tokensUsed)
        }

        @Test
        @DisplayName("Should handle network timeout gracefully")
        fun `should handle network timeout gracefully`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws SocketTimeoutException("Request timeout")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle API rate limiting")
        fun `should handle API rate limiting`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws ApiRateLimitException("Rate limit exceeded")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should retry on transient failures")
        fun `should retry on transient failures`() = runTest {
            // Given
            val query = "Test query"
            val expectedResponse = AIResponse("Success", 0.9, 10)
            coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error") andThen expectedResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedResponse, result)
            coVerify(exactly = 2) { mockApiClient.sendQuery(query) }
        }
    }

    @Nested
    @DisplayName("Context Management Tests")
    inner class ContextManagementTests {

        @ParameterizedTest
        @MethodSource("dev.aurakai.auraframefx.ai.services.AuraAIServiceTest#provideContextSizes")
        @DisplayName("Should handle contexts of various sizes")
        fun `should handle contexts of various sizes`(contextSize: Int, description: String) {
            // Given
            val sessionId = "test-session-$contextSize"
            val context = "A".repeat(contextSize)

            // When
            auraAIService.storeContext(sessionId, context)
            val retrieved = auraAIService.getContext(sessionId)

            // Then
            assertEquals(context, retrieved.joinToString(""), "Should handle $description")
        }


        @Test
        @DisplayName("Should maintain conversation context")
        fun `should maintain conversation context`() = runTest {
            // Given
            val sessionId = "test-session-123"
            val firstQuery = "Hello"
            val secondQuery = "What did I just say?"
            
            val firstResponse = AIResponse("Hello there!", 0.9, 5)
            val secondResponse = AIResponse("You said 'Hello'", 0.95, 8)
            
            coEvery { mockApiClient.sendQueryWithContext(firstQuery, emptyList()) } returns firstResponse
            coEvery { mockApiClient.sendQueryWithContext(secondQuery, any()) } returns secondResponse

            // When
            val result1 = auraAIService.processQueryWithContext(firstQuery, sessionId)
            val result2 = auraAIService.processQueryWithContext(secondQuery, sessionId)

            // Then
            assertEquals(firstResponse, result1)
            assertEquals(secondResponse, result2)
            coVerify { mockApiClient.sendQueryWithContext(secondQuery, match { it.isNotEmpty() }) }
        }

        @Test
        @DisplayName("Should clear context when requested")
        fun `should clear context when requested`() {
            // Given
            val sessionId = "test-session-123"
            auraAIService.storeContext(sessionId, "Previous context")

            // When
            auraAIService.clearContext(sessionId)

            // Then
            assertTrue(auraAIService.getContext(sessionId).isEmpty())
        }

        @Test
        @DisplayName("Should handle multiple concurrent sessions")
        fun `should handle multiple concurrent sessions`() = runTest {
            // Given
            val session1 = "session-1"
            val session2 = "session-2"
            val query = "Test query"
            
            val response1 = AIResponse("Response 1", 0.8, 10)
            val response2 = AIResponse("Response 2", 0.9, 12)
            
            coEvery { mockApiClient.sendQueryWithContext(query, emptyList()) } returns response1 andThen response2

            // When
            val result1 = auraAIService.processQueryWithContext(query, session1)
            val result2 = auraAIService.processQueryWithContext(query, session2)

            // Then
            assertEquals(response1, result1)
            assertEquals(response2, result2)
            assertNotEquals(auraAIService.getContext(session1), auraAIService.getContext(session2))
        }
    }

    @Nested
    @DisplayName("Configuration Management Tests")
    inner class ConfigurationManagementTests {

        @ParameterizedTest
        @MethodSource("dev.aurakai.auraframefx.ai.services.AuraAIServiceTest#provideInvalidConfigurations")
        @DisplayName("Should reject invalid configurations")
        fun `should reject invalid configurations`(config: Map<String, String>, description: String) {
            // When
            val result = auraAIService.updateConfiguration(config)

            // Then
            assertFalse(result, "Should reject configuration: $description")
        }


        @Test
        @DisplayName("Should update configuration at runtime")
        fun `should update configuration at runtime`() {
            // Given
            val newConfig = mapOf(
                "apiKey" to "new-test-key",
                "baseUrl" to "https://new-api.test.com",
                "timeout" to "45000"
            )
            every { mockConfigService.updateConfig("ai", newConfig) } returns true

            // When
            val result = auraAIService.updateConfiguration(newConfig)

            // Then
            assertTrue(result)
            verify { mockConfigService.updateConfig("ai", newConfig) }
        }

        @Test
        @DisplayName("Should validate configuration before updating")
        fun `should validate configuration before updating`() {
            // Given
            val invalidConfig = mapOf("invalidKey" to "invalidValue")

            // When
            val result = auraAIService.updateConfiguration(invalidConfig)

            // Then
            assertFalse(result)
            verify(exactly = 0) { mockConfigService.updateConfig(any(), any()) }
        }

        @Test
        @DisplayName("Should get current configuration")
        fun `should get current configuration`() {
            // Given
            val expectedConfig = mapOf(
                "apiKey" to "current-key",
                "baseUrl" to "https://current-api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns expectedConfig

            // When
            val result = auraAIService.getCurrentConfiguration()

            // Then
            assertEquals(expectedConfig, result)
        }
    }

    @Nested
    @DisplayName("Service State Management Tests")
    inner class ServiceStateManagementTests {

        @Test
        @DisplayName("Should report correct service status")
        fun `should report correct service status`() {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            
            // When
            auraAIService.initialize()
            val status = auraAIService.getServiceStatus()

            // Then
            assertTrue(status.isHealthy)
            assertTrue(status.isInitialized)
            assertNotNull(status.lastHealthCheck)
        }

        @Test
        @DisplayName("Should perform health checks")
        fun `should perform health checks`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } returns true

            // When
            val result = auraAIService.performHealthCheck()

            // Then
            assertTrue(result)
            coVerify { mockApiClient.healthCheck() }
        }

        @Test
        @DisplayName("Should handle failed health checks")
        fun `should handle failed health checks`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } throws IOException("Service unavailable")

            // When
            val result = auraAIService.performHealthCheck()

            // Then
            assertFalse(result)
        }

        @Test
        @DisplayName("Should shutdown gracefully")
        fun `should shutdown gracefully`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            verify { mockLogger.info("AuraAI service shutting down") }
        }
    }

    @Nested
    @DisplayName("Error Handling and Recovery Tests")
    inner class ErrorHandlingTests {

        @ParameterizedTest
        @MethodSource("dev.aurakai.auraframefx.ai.services.AuraAIServiceTest#provideErrorScenarios")
        @DisplayName("Should handle various error scenarios")
        fun `should handle various error scenarios`(exception: Exception, description: String) = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws exception

            // When & Then
            assertThrows<ServiceException>("Should handle $description") {
                auraAIService.processQuery("Test query")
            }
        }


        @Test
        @DisplayName("Should handle service not initialized error")
        fun `should handle service not initialized error`() = runTest {
            // Given - service not initialized
            
            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should handle authentication errors")
        fun `should handle authentication errors`() = runTest {
            // Given
            val config = mapOf("apiKey" to "invalid-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws AuthenticationException("Invalid API key")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should handle quota exceeded errors")
        fun `should handle quota exceeded errors`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws QuotaExceededException("API quota exceeded")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should recover from temporary network failures")
        fun `should recover from temporary network failures`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            val query = "Test query"
            val expectedResponse = AIResponse("Success after retry", 0.8, 15)
            
            coEvery { mockApiClient.sendQuery(any()) } throws 
                IOException("Network error") andThen 
                IOException("Network error") andThen 
                expectedResponse

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(expectedResponse, result)
            coVerify(exactly = 3) { mockApiClient.sendQuery(query) }
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle thread-safe operations")
        fun `should handle thread-safe operations`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            val sessionIds = (1..10).map { "session-$it" }
            val response = AIResponse("Thread-safe response", 0.8, 10)
            coEvery { mockApiClient.sendQueryWithContext(any(), any()) } returns response

            // When
            val results = sessionIds.map { sessionId ->
                kotlinx.coroutines.async {
                    auraAIService.processQueryWithContext("Query for $sessionId", sessionId)
                }
            }.awaitAll()

            // Then
            assertEquals(10, results.size)
            results.forEach { result ->
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should cleanup resources on shutdown")
        fun `should cleanup resources on shutdown`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            // Create some contexts
            repeat(5) { index ->
                auraAIService.storeContext("session-$index", "Context $index")
            }

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            verify { mockLogger.info(match { it.contains("cleanup") || it.contains("shutdown") }) }
        }

        @Test
        @DisplayName("Should handle resource exhaustion gracefully")
        fun `should handle resource exhaustion gracefully`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws OutOfMemoryError("Memory exhausted")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }


        @Test
        @DisplayName("Should handle concurrent requests efficiently")
        fun `should handle concurrent requests efficiently`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            val queries = (1..10).map { "Query $it" }
            val responses = queries.map { AIResponse("Response for $it", 0.8, 10) }
            
            queries.zip(responses).forEach { (query, response) ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val results = queries.map { query ->
                auraAIService.processQuery(query)
            }

            // Then
            assertEquals(10, results.size)
            results.zip(responses).forEach { (result, expected) ->
                assertEquals(expected, result)
            }
        }

        @Test
        @DisplayName("Should manage memory efficiently with large contexts")
        fun `should manage memory efficiently with large contexts`() = runTest {
            // Given
            val sessionId = "large-context-session"
            val largeContext = (1..1000).map { "Context item $it" }.joinToString(" ")
            
            // When
            auraAIService.storeContext(sessionId, largeContext)
            val retrievedContext = auraAIService.getContext(sessionId)

            // Then
            assertEquals(largeContext, retrievedContext.joinToString(" "))
        }

        @Test
        @DisplayName("Should implement proper timeout handling")
        fun `should implement proper timeout handling`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com", "timeout" to "1000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            coEvery { mockApiClient.sendQuery(any()) } throws TimeoutException("Request timeout")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }
        }
    }


    @Nested
    @DisplayName("Advanced Edge Cases and Validation Tests")
    inner class AdvancedEdgeCaseTests {

        @BeforeEach
        fun setUpAdvancedTests() {
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000",
                "maxRetries" to "3",
                "maxContextSize" to "10000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should handle extremely long queries gracefully")
        fun `should handle extremely long queries gracefully`() = runTest {
            // Given
            val longQuery = "A".repeat(10000)
            val response = AIResponse("Response for long query", 0.8, 1000)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(longQuery)

            // Then
            assertEquals(response, result)
            coVerify { mockApiClient.sendQuery(longQuery) }
        }

        @Test
        @DisplayName("Should handle queries with special Unicode characters")
        fun `should handle queries with special Unicode characters`() = runTest {
            // Given
            val unicodeQuery = "你好世界 🌍 émojis and spëcial chars ñ § ™"
            val response = AIResponse("Unicode response", 0.9, 20)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(unicodeQuery)

            // Then
            assertEquals(response, result)
            coVerify { mockApiClient.sendQuery(unicodeQuery) }
        }

        @Test
        @DisplayName("Should handle malformed JSON responses")
        fun `should handle malformed JSON responses`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws IllegalArgumentException("Malformed JSON")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should validate API response integrity")
        fun `should validate API response integrity`() = runTest {
            // Given
            val query = "Test query"
            val invalidResponse = AIResponse("", -1.0, -5) // Invalid confidence and token count
            coEvery { mockApiClient.sendQuery(any()) } returns invalidResponse

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle concurrent context modifications")
        fun `should handle concurrent context modifications`() = runTest {
            // Given
            val sessionId = "concurrent-test-session"
            val query = "Test query"
            val response = AIResponse("Response", 0.8, 10)
            coEvery { mockApiClient.sendQueryWithContext(any(), any()) } returns response

            // When
            val results = (1..5).map { index ->
                auraAIService.processQueryWithContext("Query $index", sessionId)
            }

            // Then
            assertEquals(5, results.size)
            assertTrue(auraAIService.getContext(sessionId).isNotEmpty())
        }

        @Test
        @DisplayName("Should enforce context size limits")
        fun `should enforce context size limits`() {
            // Given
            val sessionId = "large-context-session"
            val largeContext = "A".repeat(15000) // Exceeds max context size

            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAIService.storeContext(sessionId, largeContext)
            }
        }

        @Test
        @DisplayName("Should handle API key rotation")
        fun `should handle API key rotation`() = runTest {
            // Given
            val query = "Test query"
            val newConfig = mapOf(
                "apiKey" to "new-rotated-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.updateConfig("ai", newConfig) } returns true
            every { mockConfigService.getConfig("ai") } returns newConfig
            val response = AIResponse("Response with new key", 0.9, 15)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.updateConfiguration(newConfig)
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(response, result)
            verify { mockConfigService.updateConfig("ai", newConfig) }
        }

        @Test
        @DisplayName("Should handle service degradation gracefully")
        fun `should handle service degradation gracefully`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.healthCheck() } returns false
            coEvery { mockApiClient.sendQuery(any()) } throws IOException("Service degraded")

            // When
            val healthResult = auraAIService.performHealthCheck()

            // Then
            assertFalse(healthResult)
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle null API responses")
        fun `should handle null API responses`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } returns null

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should log all critical operations")
        fun `should log all critical operations`() = runTest {
            // Given
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            every { mockLogger.debug(any()) } returns Unit
            every { mockLogger.info(any()) } returns Unit

            // When
            auraAIService.processQuery(query)

            // Then
            verify { mockLogger.debug(match { it.contains("Processing query") }) }
            verify { mockLogger.info(match { it.contains("Query processed successfully") }) }
        }

        @Test
        @DisplayName("Should handle configuration validation edge cases")
        fun `should handle configuration validation edge cases`() {
            // Given
            val edgeCaseConfigs = listOf(
                mapOf("apiKey" to "", "baseUrl" to "https://test.com"), // Empty API key
                mapOf("apiKey" to "test", "baseUrl" to ""), // Empty base URL
                mapOf("apiKey" to "test", "baseUrl" to "invalid-url"), // Invalid URL format
                mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "invalid"), // Invalid timeout
                mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "-1") // Negative timeout
            )

            // When & Then
            edgeCaseConfigs.forEach { config ->
                assertFalse(auraAIService.updateConfiguration(config))
            }
        }

        @Test
        @DisplayName("Should handle context cleanup after session timeout")
        fun `should handle context cleanup after session timeout`() {
            // Given
            val sessionId = "timeout-session"
            val context = "Some context"
            auraAIService.storeContext(sessionId, context)

            // When
            auraAIService.cleanupExpiredSessions()

            // Then
            // Context should be cleaned up based on timeout logic
            verify { mockLogger.debug(match { it.contains("cleanup") }) }
        }

        @Test
        @DisplayName("Should handle batch query processing")
        fun `should handle batch query processing`() = runTest {
            // Given
            val queries = listOf("Query 1", "Query 2", "Query 3")
            val responses = queries.mapIndexed { index, query ->
                AIResponse("Response $index", 0.8, 10)
            }
            
            queries.zip(responses).forEach { (query, response) ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val results = auraAIService.processBatchQueries(queries)

            // Then
            assertEquals(3, results.size)
            results.zip(responses).forEach { (result, expected) ->
                assertEquals(expected, result)
            }
        }

        @Test
        @DisplayName("Should handle service metrics collection")
        fun `should handle service metrics collection`() = runTest {
            // Given
            val query = "Test query"
            val response = AIResponse("Response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            repeat(5) { auraAIService.processQuery(query) }
            val metrics = auraAIService.getServiceMetrics()

            // Then
            assertEquals(5, metrics.totalQueries)
            assertTrue(metrics.averageResponseTime > 0)
            assertEquals(50, metrics.totalTokensUsed) // 5 queries × 10 tokens each
        }

        @Test
        @DisplayName("Should handle streaming responses")
        fun `should handle streaming responses`() = runTest {
            // Given
            val query = "Test streaming query"
            val streamChunks = listOf("Chunk 1", "Chunk 2", "Chunk 3")
            coEvery { mockApiClient.sendStreamingQuery(any()) } returns streamChunks.asSequence()

            // When
            val result = auraAIService.processStreamingQuery(query)

            // Then
            val collectedChunks = result.toList()
            assertEquals(streamChunks, collectedChunks)
        }

        @Test
        @DisplayName("Should handle custom model selection")
        fun `should handle custom model selection`() = runTest {
            // Given
            val query = "Test query"
            val modelName = "gpt-4-turbo"
            val response = AIResponse("Response from custom model", 0.9, 20)
            coEvery { mockApiClient.sendQueryWithModel(any(), any()) } returns response

            // When
            val result = auraAIService.processQueryWithModel(query, modelName)

            // Then
            assertEquals(response, result)
            coVerify { mockApiClient.sendQueryWithModel(query, modelName) }
        }

        @Test
        @DisplayName("Should handle response caching")
        fun `should handle response caching`() = runTest {
            // Given
            val query = "Cacheable query"
            val response = AIResponse("Cached response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result1 = auraAIService.processQueryWithCache(query)
            val result2 = auraAIService.processQueryWithCache(query) // Should use cache

            // Then
            assertEquals(result1, result2)
            coVerify(exactly = 1) { mockApiClient.sendQuery(query) } // Only called once due to caching
        }

        @Test
        @DisplayName("Should handle API version compatibility")
        fun `should handle API version compatibility`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws ApiVersionMismatchException("API version not supported")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle graceful shutdown with pending requests")
        fun `should handle graceful shutdown with pending requests`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            val query = "Long running query"
            coEvery { mockApiClient.sendQuery(any()) } coAnswers {
                kotlinx.coroutines.delay(1000)
                AIResponse("Delayed response", 0.8, 10)
            }

            // When
            val queryJob = kotlinx.coroutines.async { auraAIService.processQuery(query) }
            kotlinx.coroutines.delay(100) // Let query start
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            verify { mockLogger.info(match { it.contains("shutting down") }) }
        }
    }

    companion object {
        @JvmStatic
        fun provideTestQueries(): Stream<Arguments> = Stream.of(

        @JvmStatic
        fun provideInvalidConfigurations(): Stream<Arguments> = Stream.of(
            Arguments.of(mapOf<String, String>(), "Empty configuration"),
            Arguments.of(mapOf("apiKey" to ""), "Empty API key"),
            Arguments.of(mapOf("apiKey" to "test", "baseUrl" to ""), "Empty base URL"),
            Arguments.of(mapOf("apiKey" to "test", "baseUrl" to "invalid-url"), "Invalid URL format"),
            Arguments.of(mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "invalid"), "Invalid timeout"),
            Arguments.of(mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "-1"), "Negative timeout")
        )

        @JvmStatic
        fun provideErrorScenarios(): Stream<Arguments> = Stream.of(
            Arguments.of(IOException("Network error"), "Network failure"),
            Arguments.of(SocketTimeoutException("Timeout"), "Request timeout"),
            Arguments.of(IllegalArgumentException("Invalid argument"), "Invalid argument"),
            Arguments.of(RuntimeException("Unknown error"), "Unknown runtime error")
        )

        @JvmStatic
        fun provideContextSizes(): Stream<Arguments> = Stream.of(
            Arguments.of(100, "Small context"),
            Arguments.of(1000, "Medium context"),
            Arguments.of(5000, "Large context"),
            Arguments.of(9999, "Max allowed context")
        )

            Arguments.of("Simple question", "Simple answer"),
            Arguments.of("Complex multi-part question with details", "Detailed response"),
            Arguments.of("Question with special characters !@#$%^&*()", "Response with handling"),
            Arguments.of("Unicode question with émojis 🤖", "Unicode response 🚀")
        )
    }
}