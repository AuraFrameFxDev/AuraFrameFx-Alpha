package dev.aurakai.auraframefx.ai.agents

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.any
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.eq
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.assertFailsWith

@ExperimentalCoroutinesApi
@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAgentTest {

    @Mock
    private lateinit var mockAgentContext: AgentContext

    @Mock
    private lateinit var mockMessageHandler: MessageHandler

    @Mock
    private lateinit var mockEventBus: EventBus

    @Mock
    private lateinit var mockConfigurationProvider: ConfigurationProvider

    private lateinit var auraAgent: AuraAgent
    private val testDispatcher = StandardTestDispatcher(TestCoroutineScheduler())

    @BeforeEach
    fun setUp() {
        Dispatchers.setMain(testDispatcher)

        // Set up default mock behaviors
        whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(
            AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT", "ANALYSIS"),
                maxConcurrentTasks = 5
            )
        )

        whenever(mockAgentContext.getMessageHandler()).thenReturn(mockMessageHandler)
        whenever(mockAgentContext.getEventBus()).thenReturn(mockEventBus)
        whenever(mockAgentContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)

        auraAgent = AuraAgent(mockAgentContext)
    }

    @AfterEach
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Nested
    @DisplayName("Agent Initialization Tests")
    inner class InitializationTests {

        @Test
        @DisplayName("Should initialize successfully with valid context")
        fun shouldInitializeSuccessfullyWithValidContext() {
            // Given
            val validContext = mock<AgentContext>()
            whenever(validContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(validContext.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(validContext.getEventBus()).thenReturn(mockEventBus)

            // When
            val agent = AuraAgent(validContext)

            // Then
            assertNotNull(agent)
            assertEquals("TestAgent", agent.getName())
            assertEquals("1.0.0", agent.getVersion())
            assertTrue(agent.isInitialized())
        }

        @Test
        @DisplayName("Should throw exception when context is null")
        fun shouldThrowExceptionWhenContextIsNull() {
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                AuraAgent(null)
            }
        }

        @Test
        @DisplayName("Should throw exception when configuration provider is null")
        fun shouldThrowExceptionWhenConfigurationProviderIsNull() {
            // Given
            val invalidContext = mock<AgentContext>()
            whenever(invalidContext.getConfigurationProvider()).thenReturn(null)

            // When & Then
            assertFailsWith<IllegalStateException> {
                AuraAgent(invalidContext)
            }
        }

        @Test
        @DisplayName("Should initialize with default configuration when config is missing")
        fun shouldInitializeWithDefaultConfigurationWhenConfigIsMissing() {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(null)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertEquals("AuraAgent", agent.getName())
            assertEquals("1.0.0", agent.getVersion())
        }

        @Test
        @DisplayName("Should initialize with fallback values when configuration is partially invalid")
        fun shouldInitializeWithFallbackValuesWhenConfigurationIsPartiallyInvalid() {
            // Given
            val partialConfig = AgentConfiguration(
                name = null,
                version = null,
                capabilities = null,
                maxConcurrentTasks = -1
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(partialConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertNotNull(agent.getName())
            assertNotNull(agent.getVersion())
            assertTrue(agent.getCapabilities().isEmpty())
            assertTrue(agent.getMaxConcurrentTasks() > 0)
        }
    }

    @Nested
    @DisplayName("Message Processing Tests")
    inner class MessageProcessingTests {

        @Test
        @DisplayName("Should process simple text message successfully")
        fun shouldProcessSimpleTextMessageSuccessfully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Hello, AuraAgent!",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Hello! How can I help you today?",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("Hello! How can I help you today?", response.content)
            verify(mockMessageHandler).validateMessage(message)
            verify(mockMessageHandler).processMessage(message)
        }

        @Test
        @DisplayName("Should handle invalid message gracefully")
        fun shouldHandleInvalidMessageGracefully() = runTest {
            // Given
            val invalidMessage = AgentMessage(
                id = "",
                type = MessageType.TEXT,
                content = "",
                timestamp = 0
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(invalidMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("Invalid message"))
            verify(mockMessageHandler).validateMessage(invalidMessage)
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should handle message processing exceptions")
        fun shouldHandleMessageProcessingExceptions() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-002",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                RuntimeException("Processing failed")
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            assertTrue(response.content.contains("Processing failed"))
        }

        @Test
        @DisplayName("Should handle concurrent message processing")
        fun shouldHandleConcurrentMessageProcessing() = runTest {
            // Given
            val messages = (1..10).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(10, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
            verify(mockMessageHandler, times(10)).validateMessage(any())
            verify(mockMessageHandler, times(10)).processMessage(any())
        }

        @Test
        @DisplayName("Should respect maximum concurrent tasks limit")
        fun shouldRespectMaximumConcurrentTasksLimit() = runTest {
            // Given
            val config = AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 2
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(config)

            val agent = AuraAgent(mockAgentContext)
            val latch = CountDownLatch(2)

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                latch.countDown()
                latch.await(1, TimeUnit.SECONDS)
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            // Then
            val responses = messages.map { message ->
                agent.processMessage(message)
            }

            assertEquals(5, responses.size)
            // Verify that not all messages were processed simultaneously
            assertTrue(agent.getActiveTaskCount() <= 2)
        }

        @Test
        @DisplayName("Should handle message processing timeout")
        fun shouldHandleMessageProcessingTimeout() = runTest {
            // Given
            val message = AgentMessage(
                id = "timeout-001",
                type = MessageType.TEXT,
                content = "Timeout test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(10000) // Simulate very long processing
                AgentResponse(
                    messageId = message.id,
                    content = "Should not reach here",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.TIMEOUT, response.status)
        }

        @Test
        @DisplayName("Should handle null message gracefully")
        fun shouldHandleNullMessageGracefully() = runTest {
            // When
            val response = auraAgent.processMessage(null)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("null"))
        }
    }

    @Nested
    @DisplayName("Event Handling Tests")
    inner class EventHandlingTests {

        @Test
        @DisplayName("Should publish event when message is processed")
        fun shouldPublishEventWhenMessageIsProcessed() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            auraAgent.processMessage(message)

            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus).publish(eventCaptor.capture())

            val publishedEvent = eventCaptor.firstValue
            assertEquals(EventType.MESSAGE_PROCESSED, publishedEvent.type)
            assertEquals(message.id, publishedEvent.data["messageId"])
        }

        @Test
        @DisplayName("Should handle event publishing failures gracefully")
        fun shouldHandleEventPublishingFailuresGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )

            doThrow(RuntimeException("Event publishing failed"))
                .whenever(mockEventBus).publish(any())

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            // Processing should succeed even if event publishing fails
        }

        @Test
        @DisplayName("Should register event listeners during initialization")
        fun shouldRegisterEventListenersDuringInitialization() {
            // Given
            val context = mock<AgentContext>()
            whenever(context.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(context.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(context.getEventBus()).thenReturn(mockEventBus)

            // When
            AuraAgent(context)

            // Then
            verify(mockEventBus).subscribe(eq(EventType.SYSTEM_SHUTDOWN), any())
            verify(mockEventBus).subscribe(eq(EventType.CONFIGURATION_CHANGED), any())
        }

        @Test
        @DisplayName("Should handle event listener registration failures")
        fun shouldHandleEventListenerRegistrationFailures() {
            // Given
            doThrow(RuntimeException("Registration failed"))
                .whenever(mockEventBus).subscribe(any(), any())

            // When & Then
            assertDoesNotThrow {
                AuraAgent(mockAgentContext)
            }
        }
    }

    @Nested
    @DisplayName("Configuration Management Tests")
    inner class ConfigurationManagementTests {

        @Test
        @DisplayName("Should reload configuration when configuration changed event is received")
        fun shouldReloadConfigurationWhenConfigurationChangedEventIsReceived() = runTest {
            // Given
            val newConfig = AgentConfiguration(
                name = "UpdatedAgent",
                version = "2.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION"),
                maxConcurrentTasks = 10
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            assertEquals("UpdatedAgent", auraAgent.getName())
            assertEquals("2.0.0", auraAgent.getVersion())
            assertEquals(3, auraAgent.getCapabilities().size)
            assertEquals(10, auraAgent.getMaxConcurrentTasks())
        }

        @Test
        @DisplayName("Should handle configuration reload failures gracefully")
        fun shouldHandleConfigurationReloadFailuresGracefully() = runTest {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration())
                .thenThrow(RuntimeException("Config load failed"))

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            // Agent should continue with previous configuration
            assertEquals("TestAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should validate configuration before applying changes")
        fun shouldValidateConfigurationBeforeApplyingChanges() = runTest {
            // Given
            val invalidConfig = AgentConfiguration(
                name = "",
                version = "",
                capabilities = emptyList(),
                maxConcurrentTasks = -1
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(invalidConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            // Should not apply invalid configuration
            assertEquals("TestAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should handle partial configuration updates")
        fun shouldHandlePartialConfigurationUpdates() = runTest {
            // Given
            val partialConfig = AgentConfiguration(
                name = "PartialAgent",
                version = null,
                capabilities = null,
                maxConcurrentTasks = 8
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(partialConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            assertEquals("PartialAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion()) // Should retain previous version
            assertEquals(8, auraAgent.getMaxConcurrentTasks())
        }
    }

    @Nested
    @DisplayName("Capability Management Tests")
    inner class CapabilityManagementTests {

        @Test
        @DisplayName("Should return true when agent has requested capability")
        fun shouldReturnTrueWhenAgentHasRequestedCapability() {
            // Given
            val capability = "CHAT"

            // When
            val hasCapability = auraAgent.hasCapability(capability)

            // Then
            assertTrue(hasCapability)
        }

        @Test
        @DisplayName("Should return false when agent does not have requested capability")
        fun shouldReturnFalseWhenAgentDoesNotHaveRequestedCapability() {
            // Given
            val capability = "TRANSLATION"

            // When
            val hasCapability = auraAgent.hasCapability(capability)

            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should handle null capability gracefully")
        fun shouldHandleNullCapabilityGracefully() {
            // When
            val hasCapability = auraAgent.hasCapability(null)

            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should handle empty capability string gracefully")
        fun shouldHandleEmptyCapabilityStringGracefully() {
            // Given
            val capability = ""

            // When
            val hasCapability = auraAgent.hasCapability(capability)

            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should return all configured capabilities")
        fun shouldReturnAllConfiguredCapabilities() {
            // When
            val capabilities = auraAgent.getCapabilities()

            // Then
            assertEquals(2, capabilities.size)
            assertTrue(capabilities.contains("CHAT"))
            assertTrue(capabilities.contains("ANALYSIS"))
        }

        @Test
        @DisplayName("Should handle capability check with whitespace")
        fun shouldHandleCapabilityCheckWithWhitespace() {
            // Given
            val capabilityWithSpaces = " CHAT "

            // When
            val hasCapability = auraAgent.hasCapability(capabilityWithSpaces)

            // Then
            assertTrue(hasCapability)
        }

        @Test
        @DisplayName("Should handle case-insensitive capability checks")
        fun shouldHandleCaseInsensitiveCapabilityChecks() {
            // Given
            val lowerCaseCapability = "chat"
            val upperCaseCapability = "CHAT"
            val mixedCaseCapability = "ChAt"

            // When
            val hasLowerCase = auraAgent.hasCapability(lowerCaseCapability)
            val hasUpperCase = auraAgent.hasCapability(upperCaseCapability)
            val hasMixedCase = auraAgent.hasCapability(mixedCaseCapability)

            // Then
            assertTrue(hasLowerCase)
            assertTrue(hasUpperCase)
            assertTrue(hasMixedCase)
        }
    }

    @Nested
    @DisplayName("Lifecycle Management Tests")
    inner class LifecycleManagementTests {

        @Test
        @DisplayName("Should start agent successfully")
        fun shouldStartAgentSuccessfully() = runTest {
            // Given
            assertFalse(auraAgent.isRunning())

            // When
            auraAgent.start()

            // Then
            assertTrue(auraAgent.isRunning())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
        }

        @Test
        @DisplayName("Should stop agent successfully")
        fun shouldStopAgentSuccessfully() = runTest {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())

            // When
            auraAgent.stop()

            // Then
            assertFalse(auraAgent.isRunning())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
        }

        @Test
        @DisplayName("Should handle multiple start calls gracefully")
        fun shouldHandleMultipleStartCallsGracefully() = runTest {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())

            // When
            auraAgent.start()

            // Then
            assertTrue(auraAgent.isRunning())
            // Should only publish one start event
            verify(mockEventBus, times(1)).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
        }

        @Test
        @DisplayName("Should handle multiple stop calls gracefully")
        fun shouldHandleMultipleStopCallsGracefully() = runTest {
            // Given
            auraAgent.start()
            auraAgent.stop()
            assertFalse(auraAgent.isRunning())

            // When
            auraAgent.stop()

            // Then
            assertFalse(auraAgent.isRunning())
            // Should only publish one stop event
            verify(mockEventBus, times(1)).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
        }

        @Test
        @DisplayName("Should handle shutdown gracefully")
        fun shouldHandleShutdownGracefully() = runTest {
            // Given
            auraAgent.start()
            assertTrue(auraAgent.isRunning())

            // When
            auraAgent.shutdown()

            // Then
            assertFalse(auraAgent.isRunning())
            assertTrue(auraAgent.isShutdown())
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_SHUTDOWN
            })
        }

        @Test
        @DisplayName("Should reject new messages after shutdown")
        fun shouldRejectNewMessagesAfterShutdown() = runTest {
            // Given
            auraAgent.start()
            auraAgent.shutdown()

            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.AGENT_SHUTDOWN, response.status)
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should handle graceful shutdown with pending messages")
        fun shouldHandleGracefulShutdownWithPendingMessages() = runTest {
            // Given
            auraAgent.start()
            val pendingMessages = (1..3).map { index ->
                AgentMessage(
                    id = "pending-$index",
                    type = MessageType.TEXT,
                    content = "Pending message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(100) // Simulate processing time
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            }

            // Process messages
            pendingMessages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // When
            auraAgent.shutdown()

            // Then
            assertTrue(auraAgent.isShutdown())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }
    }

    @Nested
    @DisplayName("Error Handling and Recovery Tests")
    inner class ErrorHandlingAndRecoveryTests {

        @Test
        @DisplayName("Should recover from temporary message handler failures")
        fun shouldRecoverFromTemporaryMessageHandlerFailures() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Temporary failure"))
                .thenReturn(
                    AgentResponse(
                        messageId = message.id,
                        content = "Recovered response",
                        status = ResponseStatus.SUCCESS
                    )
                )

            // When
            val firstResponse = auraAgent.processMessage(message)
            val secondResponse = auraAgent.processMessage(message)

            // Then
            assertEquals(ResponseStatus.ERROR, firstResponse.status)
            assertEquals(ResponseStatus.SUCCESS, secondResponse.status)
            assertEquals("Recovered response", secondResponse.content)
        }

        @Test
        @DisplayName("Should handle out of memory errors gracefully")
        fun shouldHandleOutOfMemoryErrorsGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(OutOfMemoryError("Out of memory"))

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SYSTEM_ERROR, response.status)
            assertTrue(response.content.contains("System error"))
        }

        @Test
        @DisplayName("Should maintain state consistency during concurrent failures")
        fun shouldMaintainStateConsistencyDuringConcurrentFailures() = runTest {
            // Given
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Concurrent failure"))

            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(5, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.ERROR, response.status)
            }

            // Agent should still be in a consistent state
            assertTrue(auraAgent.isInitialized())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }

        @Test
        @DisplayName("Should handle stack overflow errors gracefully")
        fun shouldHandleStackOverflowErrorsGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "stackoverflow-001",
                type = MessageType.TEXT,
                content = "Stack overflow test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(StackOverflowError("Stack overflow"))

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SYSTEM_ERROR, response.status)
            assertTrue(response.content.contains("System error"))
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceAndResourceManagementTests {

        @Test
        @DisplayName("Should not exceed memory limits during message processing")
        fun shouldNotExceedMemoryLimitsDuringMessageProcessing() = runTest {
            // Given
            val initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val largeMessage = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "x".repeat(1000000), // 1MB message
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = largeMessage.id,
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            repeat(10) {
                auraAgent.processMessage(largeMessage)
            }

            // Then
            System.gc()
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val memoryIncrease = finalMemory - initialMemory

            // Memory increase should be reasonable (less than 10MB)
            assertTrue(memoryIncrease < 10 * 1024 * 1024, 
                "Memory increase of ${memoryIncrease / 1024 / 1024}MB exceeded acceptable limit")
        }

        @Test
        @DisplayName("Should handle message processing timeout gracefully")
        fun shouldHandleMessageProcessingTimeoutGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(10000) // Simulate long processing time
                AgentResponse(
                    messageId = message.id,
                    content = "Delayed response",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val startTime = System.currentTimeMillis()
            val response = auraAgent.processMessage(message)
            val endTime = System.currentTimeMillis()

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.TIMEOUT, response.status)
            assertTrue(endTime - startTime < 5000, "Processing should timeout within 5 seconds")
        }

        @Test
        @DisplayName("Should clean up resources after processing")
        fun shouldCleanUpResourcesAfterProcessing() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            auraAgent.processMessage(message)

            // Then
            assertEquals(0, auraAgent.getActiveTaskCount())
            assertTrue(auraAgent.getResourceUsage().memoryUsage < 1024 * 1024) // Less than 1MB
        }

        @Test
        @DisplayName("Should handle high-frequency message processing")
        fun shouldHandleHighFrequencyMessageProcessing() = runTest {
            // Given
            val messages = (1..1000).map { index ->
                AgentMessage(
                    id = "rapid-$index",
                    type = MessageType.TEXT,
                    content = "Rapid message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val startTime = System.currentTimeMillis()
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(1000, responses.size)
            assertTrue(responses.all { it.status == ResponseStatus.SUCCESS })
            assertTrue(endTime - startTime < 5000, "Should process 1000 messages within 5 seconds")
        }
    }

    @Nested
    @DisplayName("Message Type Handling Tests")
    inner class MessageTypeHandlingTests {

        @Test
        @DisplayName("Should process COMMAND message type correctly")
        fun shouldProcessCommandMessageTypeCorrectly() = runTest {
            // Given
            val commandMessage = AgentMessage(
                id = "cmd-001",
                type = MessageType.COMMAND,
                content = "/help",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = commandMessage.id,
                    content = "Available commands: /help, /status, /config",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(commandMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertTrue(response.content.contains("Available commands"))
        }

        @Test
        @DisplayName("Should process QUERY message type correctly")
        fun shouldProcessQueryMessageTypeCorrectly() = runTest {
            // Given
            val queryMessage = AgentMessage(
                id = "query-001",
                type = MessageType.QUERY,
                content = "What is the weather today?",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = queryMessage.id,
                    content = "I don't have access to weather information",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(queryMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should handle unsupported message types gracefully")
        fun shouldHandleUnsupportedMessageTypesGracefully() = runTest {
            // Given
            val unsupportedMessage = AgentMessage(
                id = "unsupported-001",
                type = MessageType.BINARY,
                content = "Binary data",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                UnsupportedOperationException("Binary messages not supported")
            )

            // When
            val response = auraAgent.processMessage(unsupportedMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            assertTrue(response.content.contains("not supported"))
        }

        @Test
        @DisplayName("Should handle null message type gracefully")
        fun shouldHandleNullMessageTypeGracefully() = runTest {
            // Given
            val nullTypeMessage = AgentMessage(
                id = "null-001",
                type = null,
                content = "Message with null type",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(nullTypeMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle FILE message type correctly")
        fun shouldHandleFileMessageTypeCorrectly() = runTest {
            // Given
            val fileMessage = AgentMessage(
                id = "file-001",
                type = MessageType.FILE,
                content = "file://path/to/document.pdf",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = fileMessage.id,
                    content = "File processed successfully",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(fileMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertTrue(response.content.contains("File processed"))
        }
    }

    @Nested
    @DisplayName("Complex Message Validation Tests")
    inner class ComplexMessageValidationTests {

        @Test
        @DisplayName("Should validate message with special characters")
        fun shouldValidateMessageWithSpecialCharacters() = runTest {
            // Given
            val specialMessage = AgentMessage(
                id = "special-001",
                type = MessageType.TEXT,
                content = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = specialMessage.id,
                    content = "Processed special characters",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(specialMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should validate message with unicode characters")
        fun shouldValidateMessageWithUnicodeCharacters() = runTest {
            // Given
            val unicodeMessage = AgentMessage(
                id = "unicode-001",
                type = MessageType.TEXT,
                content = "Hello 世界! 🌍 Café naïve résumé",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = unicodeMessage.id,
                    content = "Processed unicode text",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(unicodeMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should handle extremely long message IDs")
        fun shouldHandleExtremelyLongMessageIds() = runTest {
            // Given
            val longId = "x".repeat(10000)
            val longIdMessage = AgentMessage(
                id = longId,
                type = MessageType.TEXT,
                content = "Message with very long ID",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(longIdMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should validate message timestamps correctly")
        fun shouldValidateMessageTimestampsCorrectly() = runTest {
            // Given
            val futureMessage = AgentMessage(
                id = "future-001",
                type = MessageType.TEXT,
                content = "Message from the future",
                timestamp = System.currentTimeMillis() + 1000000
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(futureMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle message with negative timestamp")
        fun shouldHandleMessageWithNegativeTimestamp() = runTest {
            // Given
            val negativeTimestampMessage = AgentMessage(
                id = "negative-001",
                type = MessageType.TEXT,
                content = "Negative timestamp test",
                timestamp = -1L
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(negativeTimestampMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle message with extremely large content")
        fun shouldHandleMessageWithExtremelyLargeContent() = runTest {
            // Given
            val largeContent = "x".repeat(10 * 1024 * 1024) // 10MB content
            val largeMessage = AgentMessage(
                id = "large-001",
                type = MessageType.TEXT,
                content = largeContent,
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(largeMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("size"))
        }
    }

    @Nested
    @DisplayName("Advanced Event Handling Tests")
    inner class AdvancedEventHandlingTests {

        @Test
        @DisplayName("Should handle rapid event publishing correctly")
        fun shouldHandleRapidEventPublishingCorrectly() = runTest {
            // Given
            val messages = (1..100).map { index ->
                AgentMessage(
                    id = "rapid-$index",
                    type = MessageType.TEXT,
                    content = "Rapid message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            messages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // Then
            verify(mockEventBus, times(100)).publish(any())
        }

        @Test
        @DisplayName("Should handle event subscription failures gracefully")
        fun shouldHandleEventSubscriptionFailuresGracefully() {
            // Given
            doThrow(RuntimeException("Subscription failed"))
                .whenever(mockEventBus).subscribe(any(), any())

            // When & Then
            assertDoesNotThrow {
                AuraAgent(mockAgentContext)
            }
        }

        @Test
        @DisplayName("Should publish different event types based on message processing results")
        fun shouldPublishDifferentEventTypesBasedOnMessageProcessingResults() = runTest {
            // Given
            val successMessage = AgentMessage(
                id = "success-001",
                type = MessageType.TEXT,
                content = "Success message",
                timestamp = System.currentTimeMillis()
            )

            val errorMessage = AgentMessage(
                id = "error-001",
                type = MessageType.TEXT,
                content = "Error message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(successMessage)).thenReturn(
                AgentResponse(
                    messageId = successMessage.id,
                    content = "Success",
                    status = ResponseStatus.SUCCESS
                )
            )
            whenever(mockMessageHandler.processMessage(errorMessage)).thenThrow(
                RuntimeException("Processing failed")
            )

            // When
            auraAgent.processMessage(successMessage)
            auraAgent.processMessage(errorMessage)

            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus, times(2)).publish(eventCaptor.capture())

            val events = eventCaptor.allValues
            assertTrue(events.any { it.type == EventType.MESSAGE_PROCESSED })
            assertTrue(events.any { it.type == EventType.MESSAGE_PROCESSING_FAILED })
        }

        @Test
        @DisplayName("Should handle event bus disconnection gracefully")
        fun shouldHandleEventBusDisconnectionGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "disconnect-001",
                type = MessageType.TEXT,
                content = "Disconnection test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // Simulate event bus disconnection
            doThrow(RuntimeException("Event bus disconnected"))
                .whenever(mockEventBus).publish(any())

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }
    }

    @Nested
    @DisplayName("Configuration Edge Cases Tests")
    inner class ConfigurationEdgeCasesTests {

        @Test
        @DisplayName("Should handle configuration with extremely high max concurrent tasks")
        fun shouldHandleConfigurationWithExtremelyHighMaxConcurrentTasks() = runTest {
            // Given
            val extremeConfig = AgentConfiguration(
                name = "ExtremeAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = Int.MAX_VALUE
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(extremeConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertTrue(agent.getMaxConcurrentTasks() <= 1000) // Should be capped at reasonable limit
        }

        @Test
        @DisplayName("Should handle configuration with zero max concurrent tasks")
        fun shouldHandleConfigurationWithZeroMaxConcurrentTasks() = runTest {
            // Given
            val zeroConfig = AgentConfiguration(
                name = "ZeroAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 0
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(zeroConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertTrue(agent.getMaxConcurrentTasks() >= 1) // Should default to at least 1
        }

        @Test
        @DisplayName("Should handle configuration with duplicate capabilities")
        fun shouldHandleConfigurationWithDuplicateCapabilities() = runTest {
            // Given
            val duplicateConfig = AgentConfiguration(
                name = "DuplicateAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT", "CHAT", "ANALYSIS", "CHAT"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(duplicateConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            val capabilities = agent.getCapabilities()
            assertEquals(2, capabilities.size) // Should deduplicate
            assertTrue(capabilities.contains("CHAT"))
            assertTrue(capabilities.contains("ANALYSIS"))
        }

        @Test
        @DisplayName("Should handle configuration with empty version string")
        fun shouldHandleConfigurationWithEmptyVersionString() = runTest {
            // Given
            val emptyVersionConfig = AgentConfiguration(
                name = "EmptyVersionAgent",
                version = "",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(emptyVersionConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertFalse(agent.getVersion().isEmpty()) // Should have default version
        }

        @Test
        @DisplayName("Should handle configuration with null capabilities list")
        fun shouldHandleConfigurationWithNullCapabilitiesList() = runTest {
            // Given
            val nullCapabilitiesConfig = AgentConfiguration(
                name = "NullCapabilitiesAgent",
                version = "1.0.0",
                capabilities = null,
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(nullCapabilitiesConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertTrue(agent.getCapabilities().isEmpty())
        }
    }

    @Nested
    @DisplayName("Thread Safety Tests")
    inner class ThreadSafetyTests {

        @Test
        @DisplayName("Should handle concurrent start and stop operations safely")
        fun shouldHandleConcurrentStartAndStopOperationsSafely() = runTest {
            // Given
            val latch = CountDownLatch(10)
            val threads = mutableListOf<Thread>()

            // When
            repeat(10) { index ->
                val thread = Thread {
                    try {
                        if (index % 2 == 0) {
                            auraAgent.start()
                        } else {
                            auraAgent.stop()
                        }
                    } finally {
                        latch.countDown()
                    }
                }
                threads.add(thread)
                thread.start()
            }

            latch.await(5, TimeUnit.SECONDS)

            // Then
            // Agent should be in a consistent state
            assertTrue(auraAgent.isInitialized())
            // No exceptions should be thrown
        }

        @Test
        @DisplayName("Should handle concurrent configuration updates safely")
        fun shouldHandleConcurrentConfigurationUpdatesSafely() = runTest {
            // Given
            val configs = listOf(
                AgentConfiguration("Agent1", "1.0.0", listOf("CHAT"), 5),
                AgentConfiguration("Agent2", "2.0.0", listOf("ANALYSIS"), 10),
                AgentConfiguration("Agent3", "3.0.0", listOf("TRANSLATION"), 15)
            )

            val latch = CountDownLatch(3)

            // When
            configs.forEach { config ->
                Thread {
                    try {
                        whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(config)
                        auraAgent.handleConfigurationChanged()
                    } finally {
                        latch.countDown()
                    }
                }.start()
            }

            latch.await(5, TimeUnit.SECONDS)

            // Then
            // Agent should be in a consistent state with one of the configurations
            assertTrue(auraAgent.isInitialized())
            assertNotNull(auraAgent.getName())
            assertNotNull(auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should handle concurrent message processing safely")
        fun shouldHandleConcurrentMessageProcessingSafely() = runTest {
            // Given
            val messages = (1..50).map { index ->
                AgentMessage(
                    id = "concurrent-$index",
                    type = MessageType.TEXT,
                    content = "Concurrent message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            val latch = CountDownLatch(50)

            // When
            messages.forEach { message ->
                Thread {
                    try {
                        auraAgent.processMessage(message)
                    } finally {
                        latch.countDown()
                    }
                }.start()
            }

            latch.await(10, TimeUnit.SECONDS)

            // Then
            // All messages should be processed without race conditions
            verify(mockMessageHandler, times(50)).validateMessage(any())
            verify(mockMessageHandler, times(50)).processMessage(any())
        }
    }

    @Nested
    @DisplayName("Resource Management Edge Cases Tests")
    inner class ResourceManagementEdgeCasesTests {

        @Test
        @DisplayName("Should handle resource cleanup during shutdown")
        fun shouldHandleResourceCleanupDuringShutdown() = runTest {
            // Given
            auraAgent.start()

            // Process some messages to create resources
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "cleanup-$index",
                    type = MessageType.TEXT,
                    content = "Cleanup message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            messages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // When
            auraAgent.shutdown()

            // Then
            assertTrue(auraAgent.isShutdown())
            assertEquals(0, auraAgent.getActiveTaskCount())
            assertTrue(auraAgent.getResourceUsage().memoryUsage < 1024 * 1024)
        }

        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() = runTest {
            // Given
            val largeMessages = (1..100).map { index ->
                AgentMessage(
                    id = "large-$index",
                    type = MessageType.TEXT,
                    content = "x".repeat(100000), // 100KB each
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            largeMessages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // Then
            // Agent should still be functional
            assertTrue(auraAgent.isInitialized())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }

        @Test
        @DisplayName("Should handle resource exhaustion gracefully")
        fun shouldHandleResourceExhaustionGracefully() = runTest {
            // Given
            val config = AgentConfiguration(
                name = "ResourceTestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 1000
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(config)

            val agent = AuraAgent(mockAgentContext)
            agent.start()

            // Simulate resource exhaustion
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                RuntimeException("Resource exhausted")
            )

            val messages = (1..10).map { index ->
                AgentMessage(
                    id = "exhaustion-$index",
                    type = MessageType.TEXT,
                    content = "Exhaustion test $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            // When
            val responses = messages.map { message ->
                agent.processMessage(message)
            }

            // Then
            assertTrue(responses.all { it.status == ResponseStatus.ERROR })
            assertTrue(agent.isInitialized())
        }
    }

    @Nested
    @DisplayName("Integration-Style Tests")
    inner class IntegrationStyleTests {

        @Test
        @DisplayName("Should handle complete message lifecycle")
        fun shouldHandleCompleteMessageLifecycle() = runTest {
            // Given
            val message = AgentMessage(
                id = "lifecycle-001",
                type = MessageType.TEXT,
                content = "Complete lifecycle test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Lifecycle complete",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            auraAgent.start()
            val response = auraAgent.processMessage(message)
            auraAgent.stop()

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)

            // Verify complete interaction chain
            verify(mockMessageHandler).validateMessage(message)
            verify(mockMessageHandler).processMessage(message)
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.MESSAGE_PROCESSED
            })
        }

        @Test
        @DisplayName("Should handle agent restart scenario")
        fun shouldHandleAgentRestartScenario() = runTest {
            // Given
            val message = AgentMessage(
                id = "restart-001",
                type = MessageType.TEXT,
                content = "Restart test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Restart response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            auraAgent.start()
            auraAgent.processMessage(message)
            auraAgent.stop()

            // Restart
            auraAgent.start()
            val response = auraAgent.processMessage(message)
            auraAgent.stop()

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)

            // Verify agent can be restarted successfully
            verify(mockEventBus, times(2)).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
            verify(mockEventBus, times(2)).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
        }

        @Test
        @DisplayName("Should handle full agent lifecycle with configuration changes")
        fun shouldHandleFullAgentLifecycleWithConfigurationChanges() = runTest {
            // Given
            val initialMessage = AgentMessage(
                id = "initial-001",
                type = MessageType.TEXT,
                content = "Initial message",
                timestamp = System.currentTimeMillis()
            )

            val updatedConfig = AgentConfiguration(
                name = "UpdatedAgent",
                version = "2.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION"),
                maxConcurrentTasks = 10
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            auraAgent.start()
            val initialResponse = auraAgent.processMessage(initialMessage)

            // Update configuration
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(updatedConfig)
            auraAgent.handleConfigurationChanged()

            val updatedResponse = auraAgent.processMessage(initialMessage)
            auraAgent.stop()

            // Then
            assertNotNull(initialResponse)
            assertNotNull(updatedResponse)
            assertEquals(ResponseStatus.SUCCESS, initialResponse.status)
            assertEquals(ResponseStatus.SUCCESS, updatedResponse.status)
            assertEquals("UpdatedAgent", auraAgent.getName())
            assertEquals("2.0.0", auraAgent.getVersion())
        }
    }

    @Nested
    @DisplayName("Boundary Value Tests")
    inner class BoundaryValueTests {

        @Test
        @DisplayName("Should handle message at exact character limit")
        fun shouldHandleMessageAtExactCharacterLimit() = runTest {
            // Given
            val maxLength = 65536 // Assuming 64KB limit
            val exactLimitMessage = AgentMessage(
                id = "limit-001",
                type = MessageType.TEXT,
                content = "x".repeat(maxLength),
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = exactLimitMessage.id,
                    content = "Processed at limit",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(exactLimitMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should handle message exceeding character limit")
        fun shouldHandleMessageExceedingCharacterLimit() = runTest {
            // Given
            val oversizedMessage = AgentMessage(
                id = "oversized-001",
                type = MessageType.TEXT,
                content = "x".repeat(1000000), // 1MB
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(oversizedMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle minimum valid timestamp")
        fun shouldHandleMinimumValidTimestamp() = runTest {
            // Given
            val minTimestampMessage = AgentMessage(
                id = "min-timestamp-001",
                type = MessageType.TEXT,
                content = "Minimum timestamp test",
                timestamp = 1L
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = minTimestampMessage.id,
                    content = "Processed min timestamp",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(minTimestampMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should handle maximum valid timestamp")
        fun shouldHandleMaximumValidTimestamp() = runTest {
            // Given
            val maxTimestampMessage = AgentMessage(
                id = "max-timestamp-001",
                type = MessageType.TEXT,
                content = "Maximum timestamp test",
                timestamp = Long.MAX_VALUE
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(maxTimestampMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle zero timestamp")
        fun shouldHandleZeroTimestamp() = runTest {
            // Given
            val zeroTimestampMessage = AgentMessage(
                id = "zero-timestamp-001",
                type = MessageType.TEXT,
                content = "Zero timestamp test",
                timestamp = 0L
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(zeroTimestampMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle empty message ID")
        fun shouldHandleEmptyMessageId() = runTest {
            // Given
            val emptyIdMessage = AgentMessage(
                id = "",
                type = MessageType.TEXT,
                content = "Empty ID test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(emptyIdMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle empty message content")
        fun shouldHandleEmptyMessageContent() = runTest {
            // Given
            val emptyContentMessage = AgentMessage(
                id = "empty-content-001",
                type = MessageType.TEXT,
                content = "",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(emptyContentMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }
    }

    @Nested
    @DisplayName("State Management Tests")
    inner class StateManagementTests {

        @Test
        @DisplayName("Should maintain consistent state across operations")
        fun shouldMaintainConsistentStateAcrossOperations() = runTest {
            // Given
            val message = AgentMessage(
                id = "state-001",
                type = MessageType.TEXT,
                content = "State test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When & Then
            assertTrue(auraAgent.isInitialized())
            assertFalse(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())

            auraAgent.start()
            assertTrue(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())

            auraAgent.processMessage(message)
            assertTrue(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())

            auraAgent.stop()
            assertFalse(auraAgent.isRunning())
            assertFalse(auraAgent.isShutdown())

            auraAgent.shutdown()
            assertFalse(auraAgent.isRunning())
            assertTrue(auraAgent.isShutdown())
        }

        @Test
        @DisplayName("Should handle state transitions correctly")
        fun shouldHandleStateTransitionsCorrectly() = runTest {
            // Given
            val states = mutableListOf<String>()

            // When
            states.add("initial: ${auraAgent.getState()}")
            
            auraAgent.start()
            states.add("after start: ${auraAgent.getState()}")
            
            auraAgent.stop()
            states.add("after stop: ${auraAgent.getState()}")
            
            auraAgent.shutdown()
            states.add("after shutdown: ${auraAgent.getState()}")

            // Then
            assertEquals(4, states.size)
            assertTrue(states[0].contains("INITIALIZED"))
            assertTrue(states[1].contains("RUNNING"))
            assertTrue(states[2].contains("STOPPED"))
            assertTrue(states[3].contains("SHUTDOWN"))
        }

        @Test
        @DisplayName("Should prevent invalid state transitions")
        fun shouldPreventInvalidStateTransitions() = runTest {
            // Given
            auraAgent.shutdown()

            // When & Then
            assertFailsWith<IllegalStateException> {
                auraAgent.start()
            }

            assertFailsWith<IllegalStateException> {
                auraAgent.stop()
            }
        }
    }

    @Nested
    @DisplayName("Security Tests")
    inner class SecurityTests {

        @Test
        @DisplayName("Should sanitize message content")
        fun shouldSanitizeMessageContent() = runTest {
            // Given
            val maliciousMessage = AgentMessage(
                id = "malicious-001",
                type = MessageType.TEXT,
                content = "<script>alert('XSS')</script>",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = maliciousMessage.id,
                    content = "Sanitized content",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(maliciousMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertFalse(response.content.contains("<script>"))
        }

        @Test
        @DisplayName("Should handle suspicious message patterns")
        fun shouldHandleSuspiciousMessagePatterns() = runTest {
            // Given
            val suspiciousMessage = AgentMessage(
                id = "suspicious-001",
                type = MessageType.TEXT,
                content = "DROP TABLE users; --",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(suspiciousMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should rate limit message processing")
        fun shouldRateLimitMessageProcessing() = runTest {
            // Given
            val rapidMessages = (1..1000).map { index ->
                AgentMessage(
                    id = "rate-limit-$index",
                    type = MessageType.TEXT,
                    content = "Rate limit test $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val responses = rapidMessages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            val rateLimitedResponses = responses.filter { it.status == ResponseStatus.RATE_LIMITED }
            assertTrue(rateLimitedResponses.isNotEmpty(), "Should have rate limited some messages")
        }
    }
}