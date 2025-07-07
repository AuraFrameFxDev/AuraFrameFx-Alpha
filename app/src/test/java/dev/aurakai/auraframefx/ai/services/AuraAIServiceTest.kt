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

    companion object {
        @JvmStatic
        fun provideTestQueries(): Stream<Arguments> = Stream.of(
            Arguments.of("Simple question", "Simple answer"),
            Arguments.of("Complex multi-part question with details", "Detailed response"),
            Arguments.of("Question with special characters !@#$%^&*()", "Response with handling"),
            Arguments.of("Unicode question with émojis 🤖", "Unicode response 🚀")
        )
    }
}