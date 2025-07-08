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
        @DisplayName("Should initialize with minimal configuration")
        fun shouldInitializeWithMinimalConfiguration() {
            // Given
            val minimalConfig = AgentConfiguration(
                name = "MinimalAgent",
                version = "0.1.0",
                capabilities = emptyList(),
                maxConcurrentTasks = 1
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(minimalConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertEquals("MinimalAgent", agent.getName())
            assertEquals("0.1.0", agent.getVersion())
            assertTrue(agent.getCapabilities().isEmpty())
            assertEquals(1, agent.getMaxConcurrentTasks())
        }

        @Test
        @DisplayName("Should handle configuration with null name gracefully")
        fun shouldHandleConfigurationWithNullNameGracefully() {
            // Given
            val nullNameConfig = AgentConfiguration(
                name = null,
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 5
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(nullNameConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertNotNull(agent.getName())
            assertFalse(agent.getName().isEmpty())
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
        @DisplayName("Should handle null message gracefully")
        fun shouldHandleNullMessageGracefully() = runTest {
            // When
            val response = auraAgent.processMessage(null)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("null"))
            verify(mockMessageHandler, never()).validateMessage(any())
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should handle message with null content")
        fun shouldHandleMessageWithNullContent() = runTest {
            // Given
            val messageWithNullContent = AgentMessage(
                id = "msg-null-content",
                type = MessageType.TEXT,
                content = null,
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(messageWithNullContent)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            verify(mockMessageHandler).validateMessage(messageWithNullContent)
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should handle message with future timestamp")
        fun shouldHandleMessageWithFutureTimestamp() = runTest {
            // Given
            val futureMessage = AgentMessage(
                id = "msg-future",
                type = MessageType.TEXT,
                content = "Future message",
                timestamp = System.currentTimeMillis() + 86400000 // 24 hours in future
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(futureMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            verify(mockMessageHandler).validateMessage(futureMessage)
        }

        @Test
        @DisplayName("Should handle message with negative timestamp")
        fun shouldHandleMessageWithNegativeTimestamp() = runTest {
            // Given
            val negativeTimestampMessage = AgentMessage(
                id = "msg-negative",
                type = MessageType.TEXT,
                content = "Negative timestamp message",
                timestamp = -1L
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(negativeTimestampMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            verify(mockMessageHandler).validateMessage(negativeTimestampMessage)
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

        @Test
        @DisplayName("Should publish different event types for different outcomes")
        fun shouldPublishDifferentEventTypesForDifferentOutcomes() = runTest {
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
        @DisplayName("Should handle configuration with null values")
        fun shouldHandleConfigurationWithNullValues() = runTest {
            // Given
            val nullConfig = AgentConfiguration(
                name = null,
                version = null,
                capabilities = null,
                maxConcurrentTasks = null
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(nullConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            // Should maintain previous valid configuration
            assertEquals("TestAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should handle partial configuration updates")
        fun shouldHandlePartialConfigurationUpdates() = runTest {
            // Given
            val partialConfig = AgentConfiguration(
                name = "PartialAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(partialConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            assertEquals("PartialAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
            assertEquals(1, auraAgent.getCapabilities().size)
            assertEquals(5, auraAgent.getMaxConcurrentTasks())
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

        @Test
        @DisplayName("Should handle capability check with special characters")
        fun shouldHandleCapabilityCheckWithSpecialCharacters() {
            // Given
            val specialCapability = "CHAT@#$%"

            // When
            val hasCapability = auraAgent.hasCapability(specialCapability)

            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should return immutable capabilities list")
        fun shouldReturnImmutableCapabilitiesList() {
            // When
            val capabilities = auraAgent.getCapabilities()

            // Then
            assertFailsWith<UnsupportedOperationException> {
                capabilities.add("NEW_CAPABILITY")
            }
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
        @DisplayName("Should handle restart after shutdown")
        fun shouldHandleRestartAfterShutdown() = runTest {
            // Given
            auraAgent.start()
            auraAgent.shutdown()
            assertTrue(auraAgent.isShutdown())

            // When & Then
            assertFailsWith<IllegalStateException> {
                auraAgent.start()
            }
        }

        @Test
        @DisplayName("Should handle graceful shutdown with pending messages")
        fun shouldHandleGracefulShutdownWithPendingMessages() = runTest {
            // Given
            auraAgent.start()
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(100) // Simulate processing time
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            }

            // Start processing a message
            auraAgent.processMessage(message)

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
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
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

        @Test
        @DisplayName("Should handle security exceptions gracefully")
        fun shouldHandleSecurityExceptionsGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(SecurityException("Security violation"))

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SECURITY_ERROR, response.status)
            assertTrue(response.content.contains("Security"))
        }

        @Test
        @DisplayName("Should handle interrupted exceptions gracefully")
        fun shouldHandleInterruptedExceptionsGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(InterruptedException("Thread interrupted"))

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.INTERRUPTED, response.status)
            assertTrue(response.content.contains("interrupted"))
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
        @DisplayName("Should handle rapid message processing without performance degradation")
        fun shouldHandleRapidMessageProcessingWithoutPerformanceDegradation() = runTest {
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
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }

            val processingTime = endTime - startTime
            val averageTimePerMessage = processingTime / 1000.0
            assertTrue(averageTimePerMessage < 10.0, // Less than 10ms per message
                "Average processing time of ${averageTimePerMessage}ms per message is too high")
        }

        @Test
        @DisplayName("Should handle CPU intensive processing gracefully")
        fun shouldHandleCpuIntensiveProcessingGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "cpu-intensive-001",
                type = MessageType.TEXT,
                content = "CPU intensive message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                // Simulate CPU intensive work
                var sum = 0L
                for (i in 1..1000000) {
                    sum += i
                }
                AgentResponse(
                    messageId = message.id,
                    content = "CPU intensive result: $sum",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertTrue(response.content.contains("CPU intensive result"))
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
                content = "file://path/to/document.txt",
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

        @Test
        @DisplayName("Should handle AUDIO message type correctly")
        fun shouldHandleAudioMessageTypeCorrectly() = runTest {
            // Given
            val audioMessage = AgentMessage(
                id = "audio-001",
                type = MessageType.AUDIO,
                content = "audio_data_base64",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = audioMessage.id,
                    content = "Audio transcription: Hello world",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(audioMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertTrue(response.content.contains("Audio transcription"))
        }

        @Test
        @DisplayName("Should handle IMAGE message type correctly")
        fun shouldHandleImageMessageTypeCorrectly() = runTest {
            // Given
            val imageMessage = AgentMessage(
                id = "image-001",
                type = MessageType.IMAGE,
                content = "image_data_base64",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = imageMessage.id,
                    content = "Image analysis: Contains a cat",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(imageMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertTrue(response.content.contains("Image analysis"))
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
        @DisplayName("Should validate message with extremely long content")
        fun shouldValidateMessageWithExtremelyLongContent() = runTest {
            // Given
            val longContentMessage = AgentMessage(
                id = "long-content-001",
                type = MessageType.TEXT,
                content = "x".repeat(1000000), // 1MB content
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(longContentMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle message with malformed JSON content")
        fun shouldHandleMessageWithMalformedJsonContent() = runTest {
            // Given
            val malformedJsonMessage = AgentMessage(
                id = "malformed-001",
                type = MessageType.JSON,
                content = "{ invalid json content }",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(malformedJsonMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle message with SQL injection attempts")
        fun shouldHandleMessageWithSqlInjectionAttempts() = runTest {
            // Given
            val sqlInjectionMessage = AgentMessage(
                id = "sql-injection-001",
                type = MessageType.TEXT,
                content = "'; DROP TABLE users; --",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(sqlInjectionMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle message with script injection attempts")
        fun shouldHandleMessageWithScriptInjectionAttempts() = runTest {
            // Given
            val scriptInjectionMessage = AgentMessage(
                id = "script-injection-001",
                type = MessageType.TEXT,
                content = "<script>alert('XSS')</script>",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(scriptInjectionMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }
    }

    @Nested
    @DisplayName("Advanced Configuration Tests")
    inner class AdvancedConfigurationTests {

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
        @DisplayName("Should handle configuration with version containing special characters")
        fun shouldHandleConfigurationWithVersionContainingSpecialCharacters() = runTest {
            // Given
            val specialVersionConfig = AgentConfiguration(
                name = "SpecialVersionAgent",
                version = "1.0.0-beta.1+build.123",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(specialVersionConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertEquals("1.0.0-beta.1+build.123", agent.getVersion())
        }

        @Test
        @DisplayName("Should handle configuration with capabilities containing special characters")
        fun shouldHandleConfigurationWithCapabilitiesContainingSpecialCharacters() = runTest {
            // Given
            val specialCapabilitiesConfig = AgentConfiguration(
                name = "SpecialCapabilitiesAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT@v1", "ANALYSIS.NEW", "TRANSLATION_BETA"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(specialCapabilitiesConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            val capabilities = agent.getCapabilities()
            assertEquals(3, capabilities.size)
            assertTrue(capabilities.contains("CHAT@v1"))
            assertTrue(capabilities.contains("ANALYSIS.NEW"))
            assertTrue(capabilities.contains("TRANSLATION_BETA"))
        }

        @Test
        @DisplayName("Should handle configuration with very long capability names")
        fun shouldHandleConfigurationWithVeryLongCapabilityNames() = runTest {
            // Given
            val longCapabilityName = "VERY_LONG_CAPABILITY_NAME_" + "X".repeat(1000)
            val longCapabilityConfig = AgentConfiguration(
                name = "LongCapabilityAgent",
                version = "1.0.0",
                capabilities = listOf(longCapabilityName),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(longCapabilityConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            val capabilities = agent.getCapabilities()
            assertTrue(capabilities.size <= 1) // Should be filtered or truncated
        }

        @Test
        @DisplayName("Should handle configuration with numeric agent name")
        fun shouldHandleConfigurationWithNumericAgentName() = runTest {
            // Given
            val numericNameConfig = AgentConfiguration(
                name = "12345",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(numericNameConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertEquals("12345", agent.getName())
        }
    }

    @Nested
    @DisplayName("Integration and End-to-End Tests")
    inner class IntegrationAndEndToEndTests {

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
        @DisplayName("Should handle configuration changes during runtime")
        fun shouldHandleConfigurationChangesDuringRuntime() = runTest {
            // Given
            auraAgent.start()

            val message = AgentMessage(
                id = "config-change-001",
                type = MessageType.TEXT,
                content = "Test before config change",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response before config change",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val responseBefore = auraAgent.processMessage(message)

            // Change configuration
            val newConfig = AgentConfiguration(
                name = "UpdatedRuntimeAgent",
                version = "2.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION"),
                maxConcurrentTasks = 10
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)
            auraAgent.handleConfigurationChanged()

            val responseAfter = auraAgent.processMessage(message)

            auraAgent.stop()

            // Then
            assertNotNull(responseBefore)
            assertNotNull(responseAfter)
            assertEquals(ResponseStatus.SUCCESS, responseBefore.status)
            assertEquals(ResponseStatus.SUCCESS, responseAfter.status)
            assertEquals("UpdatedRuntimeAgent", auraAgent.getName())
            assertEquals("2.0.0", auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should handle mixed message types in sequence")
        fun shouldHandleMixedMessageTypesInSequence() = runTest {
            // Given
            val messages = listOf(
                AgentMessage("msg-1", MessageType.TEXT, "Hello", System.currentTimeMillis()),
                AgentMessage("msg-2", MessageType.COMMAND, "/status", System.currentTimeMillis()),
                AgentMessage("msg-3", MessageType.QUERY, "What time is it?", System.currentTimeMillis()),
                AgentMessage("msg-4", MessageType.FILE, "file://test.txt", System.currentTimeMillis())
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Mixed type response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(4, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
            verify(mockMessageHandler, times(4)).validateMessage(any())
            verify(mockMessageHandler, times(4)).processMessage(any())
        }

        @Test
        @DisplayName("Should handle stress test with high message volume")
        fun shouldHandleStressTestWithHighMessageVolume() = runTest {
            // Given
            val messageCount = 10000
            val messages = (1..messageCount).map { index ->
                AgentMessage(
                    id = "stress-$index",
                    type = MessageType.TEXT,
                    content = "Stress test message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Stress response",
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
            assertEquals(messageCount, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }

            val processingTime = endTime - startTime
            val messagesPerSecond = messageCount * 1000.0 / processingTime
            assertTrue(messagesPerSecond > 100.0, 
                "Processing rate of $messagesPerSecond messages/second is too low")
        }

        @Test
        @DisplayName("Should maintain consistency across multiple agent instances")
        fun shouldMaintainConsistencyAcrossMultipleAgentInstances() = runTest {
            // Given
            val agent1 = AuraAgent(mockAgentContext)
            val agent2 = AuraAgent(mockAgentContext)
            val agent3 = AuraAgent(mockAgentContext)

            val message = AgentMessage(
                id = "consistency-001",
                type = MessageType.TEXT,
                content = "Consistency test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Consistent response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response1 = agent1.processMessage(message)
            val response2 = agent2.processMessage(message)
            val response3 = agent3.processMessage(message)

            // Then
            assertNotNull(response1)
            assertNotNull(response2)
            assertNotNull(response3)
            assertEquals(ResponseStatus.SUCCESS, response1.status)
            assertEquals(ResponseStatus.SUCCESS, response2.status)
            assertEquals(ResponseStatus.SUCCESS, response3.status)
            assertEquals("Consistent response", response1.content)
            assertEquals("Consistent response", response2.content)
            assertEquals("Consistent response", response3.content)
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
        @DisplayName("Should handle message ID with only whitespace")
        fun shouldHandleMessageIdWithOnlyWhitespace() = runTest {
            // Given
            val whitespaceIdMessage = AgentMessage(
                id = "   ",
                type = MessageType.TEXT,
                content = "Whitespace ID test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(whitespaceIdMessage)

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
    @DisplayName("Thread Safety and Concurrency Tests")
    inner class ThreadSafetyAndConcurrencyTests {

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
            val messageCount = 100
            val messages = (1..messageCount).map { index ->
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
                    content = "Concurrent response",
                    status = ResponseStatus.SUCCESS
                )
            )

            val latch = CountDownLatch(messageCount)
            val responses = mutableListOf<AgentResponse>()

            // When
            messages.forEach { message ->
                Thread {
                    try {
                        val response = auraAgent.processMessage(message)
                        synchronized(responses) {
                            responses.add(response)
                        }
                    } finally {
                        latch.countDown()
                    }
                }.start()
            }

            latch.await(10, TimeUnit.SECONDS)

            // Then
            assertEquals(messageCount, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
        }

        @Test
        @DisplayName("Should handle concurrent capability checks safely")
        fun shouldHandleConcurrentCapabilityChecksSafely() = runTest {
            // Given
            val capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION", "UNKNOWN")
            val checkCount = 1000
            val results = mutableListOf<Boolean>()
            val latch = CountDownLatch(checkCount)

            // When
            repeat(checkCount) { index ->
                Thread {
                    try {
                        val capability = capabilities[index % capabilities.size]
                        val hasCapability = auraAgent.hasCapability(capability)
                        synchronized(results) {
                            results.add(hasCapability)
                        }
                    } finally {
                        latch.countDown()
                    }
                }.start()
            }

            latch.await(5, TimeUnit.SECONDS)

            // Then
            assertEquals(checkCount, results.size)
            // Results should be consistent
            val chatResults = results.filterIndexed { index, _ -> 
                capabilities[index % capabilities.size] == "CHAT" 
            }
            assertTrue(chatResults.all { it })
        }

        @Test
        @DisplayName("Should handle deadlock prevention in concurrent operations")
        fun shouldHandleDeadlockPreventionInConcurrentOperations() = runTest {
            // Given
            val operationCount = 50
            val latch = CountDownLatch(operationCount * 3)
            val results = mutableListOf<String>()

            // When
            repeat(operationCount) { index ->
                // Start agent
                Thread {
                    try {
                        auraAgent.start()
                        synchronized(results) {
                            results.add("started-$index")
                        }
                    } finally {
                        latch.countDown()
                    }
                }.start()

                // Process message
                Thread {
                    try {
                        val message = AgentMessage(
                            id = "deadlock-$index",
                            type = MessageType.TEXT,
                            content = "Deadlock test $index",
                            timestamp = System.currentTimeMillis()
                        )
                        whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
                        whenever(mockMessageHandler.processMessage(any())).thenReturn(
                            AgentResponse(
                                messageId = message.id,
                                content = "Deadlock response",
                                status = ResponseStatus.SUCCESS
                            )
                        )
                        auraAgent.processMessage(message)
                        synchronized(results) {
                            results.add("processed-$index")
                        }
                    } finally {
                        latch.countDown()
                    }
                }.start()

                // Stop agent
                Thread {
                    try {
                        auraAgent.stop()
                        synchronized(results) {
                            results.add("stopped-$index")
                        }
                    } finally {
                        latch.countDown()
                    }
                }.start()
            }

            // Then
            val completed = latch.await(10, TimeUnit.SECONDS)
            assertTrue(completed, "Operations should complete without deadlock")
            assertTrue(results.size > 0, "Some operations should have completed")
        }
    }
}