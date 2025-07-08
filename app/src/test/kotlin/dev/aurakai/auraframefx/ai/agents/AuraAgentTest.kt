package dev.aurakai.auraframefx.ai.agents

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.test.assertFailsWith

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("AuraAgent Tests")
class AuraAgentTest {

    @Mock
    private lateinit var mockAgentConfig: AgentConfig
    
    @Mock
    private lateinit var mockMessageHandler: MessageHandler
    
    @Mock
    private lateinit var mockResponseGenerator: ResponseGenerator
    
    @Mock
    private lateinit var mockContextManager: ContextManager
    
    private lateinit var auraAgent: AuraAgent
    private lateinit var autoCloseable: AutoCloseable

    @BeforeEach
    fun setUp() {
        autoCloseable = MockitoAnnotations.openMocks(this)
        
        // Setup default mock behavior
        whenever(mockAgentConfig.maxResponseTime).thenReturn(5000L)
        whenever(mockAgentConfig.maxRetries).thenReturn(3)
        whenever(mockAgentConfig.enabled).thenReturn(true)
        whenever(mockAgentConfig.agentName).thenReturn("TestAgent")
        
        auraAgent = AuraAgent(
            config = mockAgentConfig,
            messageHandler = mockMessageHandler,
            responseGenerator = mockResponseGenerator,
            contextManager = mockContextManager
        )
    }

    @AfterEach
    fun tearDown() {
        autoCloseable.close()
    }

    @Nested
    @DisplayName("Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize with valid configuration")
        fun shouldInitializeWithValidConfiguration() {
            // Given
            val config = AgentConfig(
                agentName = "TestAgent",
                maxResponseTime = 10000L,
                maxRetries = 5,
                enabled = true
            )
            
            // When
            val agent = AuraAgent(config, mockMessageHandler, mockResponseGenerator, mockContextManager)
            
            // Then
            assertNotNull(agent)
            assertTrue(agent.isEnabled())
            assertEquals("TestAgent", agent.getAgentName())
        }

        @Test
        @DisplayName("Should throw exception when initialized with null config")
        fun shouldThrowExceptionWhenInitializedWithNullConfig() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAgent(null, mockMessageHandler, mockResponseGenerator, mockContextManager)
            }
        }

        @Test
        @DisplayName("Should throw exception when initialized with invalid max response time")
        fun shouldThrowExceptionWhenInitializedWithInvalidMaxResponseTime() {
            // Given
            whenever(mockAgentConfig.maxResponseTime).thenReturn(-1L)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAgent(mockAgentConfig, mockMessageHandler, mockResponseGenerator, mockContextManager)
            }
        }

        @Test
        @DisplayName("Should throw exception when initialized with invalid max retries")
        fun shouldThrowExceptionWhenInitializedWithInvalidMaxRetries() {
            // Given
            whenever(mockAgentConfig.maxRetries).thenReturn(-1)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                AuraAgent(mockAgentConfig, mockMessageHandler, mockResponseGenerator, mockContextManager)
            }
        }
    }

    @Nested
    @DisplayName("Message Processing Tests")
    inner class MessageProcessingTests {

        @Test
        @DisplayName("Should process simple message successfully")
        fun shouldProcessSimpleMessageSuccessfully() {
            // Given
            val inputMessage = "Hello, how are you?"
            val expectedResponse = "I'm doing well, thank you for asking!"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenReturn(expectedResponse)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val result = auraAgent.processMessage(inputMessage)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result)
            verify(mockMessageHandler).validateMessage(inputMessage)
            verify(mockResponseGenerator).generateResponse(eq(inputMessage), any())
        }

        @Test
        @DisplayName("Should handle invalid message gracefully")
        fun shouldHandleInvalidMessageGracefully() {
            // Given
            val invalidMessage = ""
            
            whenever(mockMessageHandler.validateMessage(invalidMessage)).thenReturn(false)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAgent.processMessage(invalidMessage)
            }
            
            verify(mockMessageHandler).validateMessage(invalidMessage)
            verifyNoInteractions(mockResponseGenerator)
        }

        @Test
        @DisplayName("Should handle null message gracefully")
        fun shouldHandleNullMessageGracefully() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAgent.processMessage(null)
            }
            
            verifyNoInteractions(mockMessageHandler)
            verifyNoInteractions(mockResponseGenerator)
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\n\t", "null"])
        @DisplayName("Should reject empty or whitespace messages")
        fun shouldRejectEmptyOrWhitespaceMessages(message: String) {
            // Given
            whenever(mockMessageHandler.validateMessage(message)).thenReturn(false)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                auraAgent.processMessage(message)
            }
        }

        @Test
        @DisplayName("Should handle very long messages")
        fun shouldHandleVeryLongMessages() {
            // Given
            val longMessage = "a".repeat(10000)
            val expectedResponse = "Response to long message"
            
            whenever(mockMessageHandler.validateMessage(longMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(longMessage, any())).thenReturn(expectedResponse)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val result = auraAgent.processMessage(longMessage)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle special characters in messages")
        fun shouldHandleSpecialCharactersInMessages() {
            // Given
            val specialMessage = "Hello! @#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
            val expectedResponse = "Response with special chars"
            
            whenever(mockMessageHandler.validateMessage(specialMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(specialMessage, any())).thenReturn(expectedResponse)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val result = auraAgent.processMessage(specialMessage)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle unicode characters in messages")
        fun shouldHandleUnicodeCharactersInMessages() {
            // Given
            val unicodeMessage = "Hello 世界 🌍 émojis"
            val expectedResponse = "Unicode response"
            
            whenever(mockMessageHandler.validateMessage(unicodeMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(unicodeMessage, any())).thenReturn(expectedResponse)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val result = auraAgent.processMessage(unicodeMessage)
            
            // Then
            assertNotNull(result)
            assertEquals(expectedResponse, result)
        }
    }

    @Nested
    @DisplayName("Asynchronous Processing Tests")
    inner class AsynchronousProcessingTests {

        @Test
        @DisplayName("Should process message asynchronously")
        fun shouldProcessMessageAsynchronously() {
            // Given
            val inputMessage = "Async test message"
            val expectedResponse = "Async response"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenReturn(expectedResponse)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val future = auraAgent.processMessageAsync(inputMessage)
            
            // Then
            assertNotNull(future)
            val result = future.get(5, TimeUnit.SECONDS)
            assertEquals(expectedResponse, result)
        }

        @Test
        @DisplayName("Should handle timeout in async processing")
        fun shouldHandleTimeoutInAsyncProcessing() {
            // Given
            val inputMessage = "Timeout test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenAnswer {
                Thread.sleep(10000) // Simulate long processing time
                "Should not reach here"
            }
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val future = auraAgent.processMessageAsync(inputMessage)
            
            // Then
            assertThrows<TimeoutException> {
                future.get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should handle concurrent message processing")
        fun shouldHandleConcurrentMessageProcessing() {
            // Given
            val messages = (1..10).map { "Message $it" }
            val expectedResponses = messages.map { "Response to $it" }
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            messages.zip(expectedResponses).forEach { (message, response) ->
                whenever(mockResponseGenerator.generateResponse(message, any())).thenReturn(response)
            }
            
            // When
            val futures = messages.map { auraAgent.processMessageAsync(it) }
            val results = futures.map { it.get(5, TimeUnit.SECONDS) }
            
            // Then
            assertEquals(expectedResponses.size, results.size)
            expectedResponses.zip(results).forEach { (expected, actual) ->
                assertEquals(expected, actual)
            }
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle message handler exceptions")
        fun shouldHandleMessageHandlerExceptions() {
            // Given
            val inputMessage = "Test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenThrow(RuntimeException("Validation error"))
            
            // When & Then
            assertThrows<RuntimeException> {
                auraAgent.processMessage(inputMessage)
            }
        }

        @Test
        @DisplayName("Should handle response generator exceptions")
        fun shouldHandleResponseGeneratorExceptions() {
            // Given
            val inputMessage = "Test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenThrow(RuntimeException("Generation error"))
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When & Then
            assertThrows<RuntimeException> {
                auraAgent.processMessage(inputMessage)
            }
        }

        @Test
        @DisplayName("Should handle context manager exceptions")
        fun shouldHandleContextManagerExceptions() {
            // Given
            val inputMessage = "Test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenThrow(RuntimeException("Context error"))
            
            // When & Then
            assertThrows<RuntimeException> {
                auraAgent.processMessage(inputMessage)
            }
        }

        @Test
        @DisplayName("Should implement retry logic for transient failures")
        fun shouldImplementRetryLogicForTransientFailures() {
            // Given
            val inputMessage = "Test message"
            val expectedResponse = "Success after retry"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            whenever(mockResponseGenerator.generateResponse(inputMessage, any()))
                .thenThrow(RuntimeException("Transient error"))
                .thenThrow(RuntimeException("Transient error"))
                .thenReturn(expectedResponse)
            
            // When
            val result = auraAgent.processMessage(inputMessage)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockResponseGenerator, times(3)).generateResponse(eq(inputMessage), any())
        }

        @Test
        @DisplayName("Should fail after max retries exceeded")
        fun shouldFailAfterMaxRetriesExceeded() {
            // Given
            val inputMessage = "Test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            whenever(mockResponseGenerator.generateResponse(inputMessage, any()))
                .thenThrow(RuntimeException("Persistent error"))
            
            // When & Then
            assertThrows<RuntimeException> {
                auraAgent.processMessage(inputMessage)
            }
            
            verify(mockResponseGenerator, times(3)).generateResponse(eq(inputMessage), any())
        }
    }

    @Nested
    @DisplayName("Context Management Tests")
    inner class ContextManagementTests {

        @Test
        @DisplayName("Should use context in message processing")
        fun shouldUseContextInMessageProcessing() {
            // Given
            val inputMessage = "What's my name?"
            val context = mapOf("username" to "Alice", "sessionId" to "123")
            val expectedResponse = "Hello Alice!"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(context)
            whenever(mockResponseGenerator.generateResponse(inputMessage, context)).thenReturn(expectedResponse)
            
            // When
            val result = auraAgent.processMessage(inputMessage)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockContextManager).getContext(any())
            verify(mockResponseGenerator).generateResponse(inputMessage, context)
        }

        @Test
        @DisplayName("Should handle empty context gracefully")
        fun shouldHandleEmptyContextGracefully() {
            // Given
            val inputMessage = "Test message"
            val emptyContext = emptyMap<String, Any>()
            val expectedResponse = "Response without context"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyContext)
            whenever(mockResponseGenerator.generateResponse(inputMessage, emptyContext)).thenReturn(expectedResponse)
            
            // When
            val result = auraAgent.processMessage(inputMessage)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockContextManager).getContext(any())
        }

        @Test
        @DisplayName("Should update context after processing")
        fun shouldUpdateContextAfterProcessing() {
            // Given
            val inputMessage = "Remember my favorite color is blue"
            val initialContext = mapOf("username" to "Alice")
            val expectedResponse = "I'll remember that blue is your favorite color"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(initialContext)
            whenever(mockResponseGenerator.generateResponse(inputMessage, initialContext)).thenReturn(expectedResponse)
            
            // When
            val result = auraAgent.processMessage(inputMessage)
            
            // Then
            assertEquals(expectedResponse, result)
            verify(mockContextManager).updateContext(any(), any())
        }
    }

    @Nested
    @DisplayName("Agent State Management Tests")
    inner class AgentStateManagementTests {

        @Test
        @DisplayName("Should check if agent is enabled")
        fun shouldCheckIfAgentIsEnabled() {
            // Given
            whenever(mockAgentConfig.enabled).thenReturn(true)
            
            // When
            val isEnabled = auraAgent.isEnabled()
            
            // Then
            assertTrue(isEnabled)
        }

        @Test
        @DisplayName("Should handle disabled agent")
        fun shouldHandleDisabledAgent() {
            // Given
            whenever(mockAgentConfig.enabled).thenReturn(false)
            val disabledAgent = AuraAgent(mockAgentConfig, mockMessageHandler, mockResponseGenerator, mockContextManager)
            
            // When
            val isEnabled = disabledAgent.isEnabled()
            
            // Then
            assertFalse(isEnabled)
        }

        @Test
        @DisplayName("Should reject messages when agent is disabled")
        fun shouldRejectMessagesWhenAgentIsDisabled() {
            // Given
            whenever(mockAgentConfig.enabled).thenReturn(false)
            val disabledAgent = AuraAgent(mockAgentConfig, mockMessageHandler, mockResponseGenerator, mockContextManager)
            
            // When & Then
            assertThrows<IllegalStateException> {
                disabledAgent.processMessage("Test message")
            }
        }

        @Test
        @DisplayName("Should get agent name")
        fun shouldGetAgentName() {
            // Given
            val expectedName = "TestAgent"
            whenever(mockAgentConfig.agentName).thenReturn(expectedName)
            
            // When
            val actualName = auraAgent.getAgentName()
            
            // Then
            assertEquals(expectedName, actualName)
        }

        @Test
        @DisplayName("Should get agent statistics")
        fun shouldGetAgentStatistics() {
            // Given
            val inputMessage = "Test message"
            val expectedResponse = "Test response"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenReturn(expectedResponse)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            auraAgent.processMessage(inputMessage)
            val stats = auraAgent.getStatistics()
            
            // Then
            assertNotNull(stats)
            assertEquals(1, stats.messagesProcessed)
            assertEquals(0, stats.errorsCount)
            assertTrue(stats.averageResponseTime > 0)
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle burst of messages efficiently")
        fun shouldHandleBurstOfMessagesEfficiently() {
            // Given
            val messageCount = 100
            val messages = (1..messageCount).map { "Message $it" }
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(any(), any())).thenReturn("Response")
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val startTime = System.currentTimeMillis()
            messages.forEach { auraAgent.processMessage(it) }
            val endTime = System.currentTimeMillis()
            
            // Then
            val processingTime = endTime - startTime
            assertTrue(processingTime < 5000) // Should process 100 messages in under 5 seconds
            
            val stats = auraAgent.getStatistics()
            assertEquals(messageCount, stats.messagesProcessed)
        }

        @Test
        @DisplayName("Should maintain performance under load")
        fun shouldMaintainPerformanceUnderLoad() {
            // Given
            val messageCount = 1000
            val messages = (1..messageCount).map { "Load test message $it" }
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockResponseGenerator.generateResponse(any(), any())).thenReturn("Load response")
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            // When
            val futures = messages.map { auraAgent.processMessageAsync(it) }
            val results = futures.map { it.get(10, TimeUnit.SECONDS) }
            
            // Then
            assertEquals(messageCount, results.size)
            assertTrue(results.all { it == "Load response" })
            
            val stats = auraAgent.getStatistics()
            assertEquals(messageCount, stats.messagesProcessed)
            assertTrue(stats.averageResponseTime < 100) // Should average under 100ms per message
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle complete conversation flow")
        fun shouldHandleCompleteConversationFlow() {
            // Given
            val conversation = listOf(
                "Hello" to "Hi there! How can I help you?",
                "What's the weather like?" to "I don't have access to weather data, but I can help with other questions.",
                "Thank you" to "You're welcome! Is there anything else I can help with?"
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            
            conversation.forEach { (message, response) ->
                whenever(mockResponseGenerator.generateResponse(message, any())).thenReturn(response)
            }
            
            // When & Then
            conversation.forEach { (message, expectedResponse) ->
                val actualResponse = auraAgent.processMessage(message)
                assertEquals(expectedResponse, actualResponse)
            }
            
            val stats = auraAgent.getStatistics()
            assertEquals(conversation.size, stats.messagesProcessed)
        }

        @Test
        @DisplayName("Should maintain context across conversation")
        fun shouldMaintainContextAcrossConversation() {
            // Given
            val messages = listOf(
                "My name is Alice",
                "What's my name?",
                "I like pizza",
                "What do I like?"
            )
            
            val contexts = listOf(
                emptyMap<String, Any>(),
                mapOf("username" to "Alice"),
                mapOf("username" to "Alice"),
                mapOf("username" to "Alice", "likes" to "pizza")
            )
            
            val responses = listOf(
                "Nice to meet you, Alice!",
                "Your name is Alice",
                "I'll remember that you like pizza",
                "You like pizza"
            )
            
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            
            messages.zip(contexts).zip(responses).forEach { (messageContext, response) ->
                val (message, context) = messageContext
                whenever(mockContextManager.getContext(any())).thenReturn(context)
                whenever(mockResponseGenerator.generateResponse(message, context)).thenReturn(response)
            }
            
            // When & Then
            messages.zip(responses).forEach { (message, expectedResponse) ->
                val actualResponse = auraAgent.processMessage(message)
                assertEquals(expectedResponse, actualResponse)
            }
            
            verify(mockContextManager, times(messages.size)).updateContext(any(), any())
        }
    }

    @Nested
    @DisplayName("Edge Cases Tests")
    inner class EdgeCasesTests {

        @Test
        @DisplayName("Should handle system shutdown gracefully")
        fun shouldHandleSystemShutdownGracefully() {
            // Given
            val inputMessage = "Test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenAnswer {
                Thread.sleep(1000)
                "Response"
            }
            
            // When
            val future = auraAgent.processMessageAsync(inputMessage)
            auraAgent.shutdown()
            
            // Then
            assertTrue(auraAgent.isShutdown())
            assertThrows<IllegalStateException> {
                auraAgent.processMessage("New message")
            }
        }

        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() {
            // Given - Simulate memory pressure by creating large objects
            val largeMessage = "x".repeat(1000000) // 1MB message
            
            whenever(mockMessageHandler.validateMessage(largeMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            whenever(mockResponseGenerator.generateResponse(largeMessage, any())).thenReturn("Response")
            
            // When & Then - Should not throw OutOfMemoryError
            assertDoesNotThrow {
                auraAgent.processMessage(largeMessage)
            }
        }

        @Test
        @DisplayName("Should handle rapid enable/disable cycles")
        fun shouldHandleRapidEnableDisableCycles() {
            // Given
            val inputMessage = "Test message"
            
            whenever(mockMessageHandler.validateMessage(inputMessage)).thenReturn(true)
            whenever(mockContextManager.getContext(any())).thenReturn(emptyMap())
            whenever(mockResponseGenerator.generateResponse(inputMessage, any())).thenReturn("Response")
            
            // When & Then
            repeat(10) { cycle ->
                whenever(mockAgentConfig.enabled).thenReturn(cycle % 2 == 0)
                
                if (cycle % 2 == 0) {
                    // Agent should be enabled
                    val result = auraAgent.processMessage(inputMessage)
                    assertNotNull(result)
                } else {
                    // Agent should be disabled
                    assertThrows<IllegalStateException> {
                        auraAgent.processMessage(inputMessage)
                    }
                }
            }
        }
    }

    companion object {
        @JvmStatic
        fun messageVariationsProvider(): List<Arguments> {
            return listOf(
                Arguments.of("Simple message", "Simple response"),
                Arguments.of("Message with numbers 123", "Numeric response"),
                Arguments.of("Message with symbols !@#$%", "Symbol response"),
                Arguments.of("Very long message that exceeds normal length boundaries and tests the system's ability to handle extended input gracefully", "Long response"),
                Arguments.of("Mixed 中文 content", "Mixed response")
            )
        }
    }
}