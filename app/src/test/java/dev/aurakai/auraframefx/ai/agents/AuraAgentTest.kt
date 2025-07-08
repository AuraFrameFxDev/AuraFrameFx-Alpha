package dev.aurakai.auraframefx.ai.agents

<<<<<<< HEAD
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
=======
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
>>>>>>> origin/coderabbitai/docstrings/78f34ad
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.jupiter.MockitoExtension
<<<<<<< HEAD
import org.mockito.kotlin.any
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.eq
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
=======
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.any
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.eq
import org.mockito.kotlin.argumentCaptor
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.resetMain
>>>>>>> origin/coderabbitai/docstrings/78f34ad
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.assertFailsWith

@ExperimentalCoroutinesApi
@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AuraAgentTest {

    @Mock
    private lateinit var mockAgentContext: AgentContext
<<<<<<< HEAD

    @Mock
    private lateinit var mockMessageHandler: MessageHandler

    @Mock
    private lateinit var mockEventBus: EventBus

    @Mock
    private lateinit var mockConfigurationProvider: ConfigurationProvider

=======

    @Mock
    private lateinit var mockMessageHandler: MessageHandler

    @Mock
    private lateinit var mockEventBus: EventBus

    @Mock
    private lateinit var mockConfigurationProvider: ConfigurationProvider

>>>>>>> origin/coderabbitai/docstrings/78f34ad
    private lateinit var auraAgent: AuraAgent
    private val testDispatcher = StandardTestDispatcher(TestCoroutineScheduler())

    @BeforeEach
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
        // Set up default mock behaviors
        whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(
            AgentConfiguration(
                name = "TestAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT", "ANALYSIS"),
                maxConcurrentTasks = 5
            )
        )
<<<<<<< HEAD

        whenever(mockAgentContext.getMessageHandler()).thenReturn(mockMessageHandler)
        whenever(mockAgentContext.getEventBus()).thenReturn(mockEventBus)
        whenever(mockAgentContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)

=======

        whenever(mockAgentContext.getMessageHandler()).thenReturn(mockMessageHandler)
        whenever(mockAgentContext.getEventBus()).thenReturn(mockEventBus)
        whenever(mockAgentContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            val agent = AuraAgent(validContext)

=======

            // When
            val agent = AuraAgent(validContext)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            val agent = AuraAgent(mockAgentContext)

=======

            // When
            val agent = AuraAgent(mockAgentContext)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertNotNull(agent)
            assertEquals("AuraAgent", agent.getName())
            assertEquals("1.0.0", agent.getVersion())
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Hello! How can I help you today?",
                    status = ResponseStatus.SUCCESS
                )
            )
<<<<<<< HEAD

            // When
            val response = auraAgent.processMessage(message)

=======

            // When
            val response = auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(invalidMessage)

=======

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(invalidMessage)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                RuntimeException("Processing failed")
            )
<<<<<<< HEAD

            // When
            val response = auraAgent.processMessage(message)

=======

            // When
            val response = auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = "test",
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            val agent = AuraAgent(mockAgentContext)
            val latch = CountDownLatch(2)

=======

            val agent = AuraAgent(mockAgentContext)
            val latch = CountDownLatch(2)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // When
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "msg-$index",
                    type = MessageType.TEXT,
                    content = "Message $index",
                    timestamp = System.currentTimeMillis()
                )
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            val responses = messages.map { message ->
                agent.processMessage(message)
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            assertEquals(5, responses.size)
            // Verify that not all messages were processed simultaneously
            assertTrue(agent.getActiveTaskCount() <= 2)
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )
<<<<<<< HEAD

            // When
            auraAgent.processMessage(message)

            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus).publish(eventCaptor.capture())

=======

            // When
            auraAgent.processMessage(message)

            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus).publish(eventCaptor.capture())

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )
<<<<<<< HEAD

            doThrow(RuntimeException("Event publishing failed"))
                .whenever(mockEventBus).publish(any())

            // When
            val response = auraAgent.processMessage(message)

=======

            doThrow(RuntimeException("Event publishing failed"))
                .whenever(mockEventBus).publish(any())

            // When
            val response = auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            AuraAgent(context)

=======

            // When
            AuraAgent(context)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            verify(mockEventBus).subscribe(eq(EventType.SYSTEM_SHUTDOWN), any())
            verify(mockEventBus).subscribe(eq(EventType.CONFIGURATION_CHANGED), any())
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
<<<<<<< HEAD

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)

            // When
            auraAgent.handleConfigurationChanged()

=======

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)

            // When
            auraAgent.handleConfigurationChanged()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            auraAgent.handleConfigurationChanged()

=======

            // When
            auraAgent.handleConfigurationChanged()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(invalidConfig)

            // When
            auraAgent.handleConfigurationChanged()

=======

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(invalidConfig)

            // When
            auraAgent.handleConfigurationChanged()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            // Should not apply invalid configuration
            assertEquals("TestAgent", auraAgent.getName())
            assertEquals("1.0.0", auraAgent.getVersion())
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
<<<<<<< HEAD

            // When
            val hasCapability = auraAgent.hasCapability(capability)

=======

            // When
            val hasCapability = auraAgent.hasCapability(capability)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertTrue(hasCapability)
        }

        @Test
        @DisplayName("Should return false when agent does not have requested capability")
        fun shouldReturnFalseWhenAgentDoesNotHaveRequestedCapability() {
            // Given
            val capability = "TRANSLATION"
<<<<<<< HEAD

            // When
            val hasCapability = auraAgent.hasCapability(capability)

=======

            // When
            val hasCapability = auraAgent.hasCapability(capability)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should handle null capability gracefully")
        fun shouldHandleNullCapabilityGracefully() {
            // When
            val hasCapability = auraAgent.hasCapability(null)
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should handle empty capability string gracefully")
        fun shouldHandleEmptyCapabilityStringGracefully() {
            // Given
            val capability = ""
<<<<<<< HEAD

            // When
            val hasCapability = auraAgent.hasCapability(capability)

=======

            // When
            val hasCapability = auraAgent.hasCapability(capability)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertFalse(hasCapability)
        }

        @Test
        @DisplayName("Should return all configured capabilities")
        fun shouldReturnAllConfiguredCapabilities() {
            // When
            val capabilities = auraAgent.getCapabilities()
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertEquals(2, capabilities.size)
            assertTrue(capabilities.contains("CHAT"))
            assertTrue(capabilities.contains("ANALYSIS"))
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
<<<<<<< HEAD

            // When
            auraAgent.start()

=======

            // When
            auraAgent.start()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            auraAgent.stop()

=======

            // When
            auraAgent.stop()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            auraAgent.start()

=======

            // When
            auraAgent.start()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            auraAgent.stop()

=======

            // When
            auraAgent.stop()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            // When
            auraAgent.shutdown()

=======

            // When
            auraAgent.shutdown()

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            val message = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "Test message",
                timestamp = System.currentTimeMillis()
            )
<<<<<<< HEAD

            // When
            val response = auraAgent.processMessage(message)

=======

            // When
            val response = auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.AGENT_SHUTDOWN, response.status)
            verify(mockMessageHandler, never()).processMessage(any())
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
<<<<<<< HEAD

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

=======

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Temporary failure"))
                .thenReturn(AgentResponse(
                    messageId = message.id,
                    content = "Recovered response",
                    status = ResponseStatus.SUCCESS
                ))

            // When
            val firstResponse = auraAgent.processMessage(message)
            val secondResponse = auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(OutOfMemoryError("Out of memory"))

            // When
            val response = auraAgent.processMessage(message)

=======

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(OutOfMemoryError("Out of memory"))

            // When
            val response = auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Concurrent failure"))

=======

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("Concurrent failure"))

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertEquals(5, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.ERROR, response.status)
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Agent should still be in a consistent state
            assertTrue(auraAgent.isInitialized())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }
    }

    @Nested
    @DisplayName("Performance and Resource Management Tests")
    inner class PerformanceAndResourceManagementTests {

        @Test
        @DisplayName("Should not exceed memory limits during message processing")
        fun shouldNotExceedMemoryLimitsDuringMessageProcessing() = runTest {
            // Given
<<<<<<< HEAD
            val initialMemory =
                Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
=======
            val initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
>>>>>>> origin/coderabbitai/docstrings/78f34ad
            val largeMessage = AgentMessage(
                id = "msg-001",
                type = MessageType.TEXT,
                content = "x".repeat(1000000), // 1MB message
                timestamp = System.currentTimeMillis()
            )
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = largeMessage.id,
                    content = "Processed",
                    status = ResponseStatus.SUCCESS
                )
            )
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // When
            repeat(10) {
                auraAgent.processMessage(largeMessage)
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            System.gc()
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val memoryIncrease = finalMemory - initialMemory
<<<<<<< HEAD

            // Memory increase should be reasonable (less than 10MB)
            assertTrue(
                memoryIncrease < 10 * 1024 * 1024,
                "Memory increase of ${memoryIncrease / 1024 / 1024}MB exceeded acceptable limit"
            )
=======

            // Memory increase should be reasonable (less than 10MB)
            assertTrue(memoryIncrease < 10 * 1024 * 1024,
                "Memory increase of ${memoryIncrease / 1024 / 1024}MB exceeded acceptable limit")
>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(10000) // Simulate long processing time
                AgentResponse(
                    messageId = message.id,
                    content = "Delayed response",
                    status = ResponseStatus.SUCCESS
                )
            }
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // When
            val startTime = System.currentTimeMillis()
            val response = auraAgent.processMessage(message)
            val endTime = System.currentTimeMillis()
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
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
<<<<<<< HEAD

=======

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Response",
                    status = ResponseStatus.SUCCESS
                )
            )
<<<<<<< HEAD

            // When
            auraAgent.processMessage(message)

=======

            // When
            auraAgent.processMessage(message)

>>>>>>> origin/coderabbitai/docstrings/78f34ad
            // Then
            assertEquals(0, auraAgent.getActiveTaskCount())
            assertTrue(auraAgent.getResourceUsage().memoryUsage < 1024 * 1024) // Less than 1MB
        }
    }
}
<<<<<<< HEAD

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
}

@Nested
@DisplayName("Capability Edge Cases Tests")
inner class CapabilityEdgeCasesTests {

    @Test
    @DisplayName("Should handle capability check with whitespace")
    fun shouldHandleCapabilityCheckWithWhitespace() {
        // Given
        val capabilityWithSpaces = " CHAT "

        // When
        val hasCapability = auraAgent.hasCapability(capabilityWithSpaces)

        // Then
        assertTrue(hasCapability) // Should trim whitespace
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
}
=======
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
    }

    @Nested
    @DisplayName("Capability Edge Cases Tests")
    inner class CapabilityEdgeCasesTests {

        @Test
        @DisplayName("Should handle capability check with whitespace")
        fun shouldHandleCapabilityCheckWithWhitespace() {
            // Given
            val capabilityWithSpaces = " CHAT "

            // When
            val hasCapability = auraAgent.hasCapability(capabilityWithSpaces)

            // Then
            assertTrue(hasCapability) // Should trim whitespace
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
    }
>>>>>>> origin/coderabbitai/docstrings/78f34ad
}
    @Nested
    @DisplayName("Security and Input Validation Tests")
    inner class SecurityAndInputValidationTests {

        @Test
        @DisplayName("Should sanitize malicious script content in messages")
        fun shouldSanitizeMaliciousScriptContentInMessages() = runTest {
            // Given
            val maliciousMessage = AgentMessage(
                id = "malicious-001",
                type = MessageType.TEXT,
                content = "<script>alert('XSS')</script>Hello World",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = maliciousMessage.id,
                    content = "Sanitized response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(maliciousMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            verify(mockMessageHandler).validateMessage(maliciousMessage)
        }

        @Test
        @DisplayName("Should handle SQL injection attempts in message content")
        fun shouldHandleSqlInjectionAttemptsInMessageContent() = runTest {
            // Given
            val sqlInjectionMessage = AgentMessage(
                id = "sql-injection-001",
                type = MessageType.TEXT,
                content = "'; DROP TABLE users; --",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = sqlInjectionMessage.id,
                    content = "Processed safely",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(sqlInjectionMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should reject messages with extremely long nested structures")
        fun shouldRejectMessagesWithExtremelyLongNestedStructures() = runTest {
            // Given
            val nestedContent = "{{{{" + "nested".repeat(1000) + "}}}}"
            val nestedMessage = AgentMessage(
                id = "nested-001",
                type = MessageType.TEXT,
                content = nestedContent,
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(nestedMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle messages with null bytes")
        fun shouldHandleMessagesWithNullBytes() = runTest {
            // Given
            val nullByteMessage = AgentMessage(
                id = "null-byte-001",
                type = MessageType.TEXT,
                content = "Hello\u0000World",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(nullByteMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should validate message ID format against injection attacks")
        fun shouldValidateMessageIdFormatAgainstInjectionAttacks() = runTest {
            // Given
            val maliciousIdMessage = AgentMessage(
                id = "../../../etc/passwd",
                type = MessageType.TEXT,
                content = "Directory traversal attempt",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(maliciousIdMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }
    }

    @Nested
    @DisplayName("State Management and Consistency Tests")
    inner class StateManagementAndConsistencyTests {

        @Test
        @DisplayName("Should maintain consistent state during rapid start/stop cycles")
        fun shouldMaintainConsistentStateDuringRapidStartStopCycles() = runTest {
            // Given
            val initialState = auraAgent.isRunning()

            // When
            repeat(100) { cycle ->
                if (cycle % 2 == 0) {
                    auraAgent.start()
                } else {
                    auraAgent.stop()
                }
            }

            // Then
            // Agent should be in a consistent state
            assertTrue(auraAgent.isInitialized())
            // Final state should be stopped (since we did 100 cycles, ending with stop)
            assertFalse(auraAgent.isRunning())
        }

        @Test
        @DisplayName("Should handle state corruption recovery")
        fun shouldHandleStateCorruptionRecovery() = runTest {
            // Given
            auraAgent.start()
            val message = AgentMessage(
                id = "corruption-001",
                type = MessageType.TEXT,
                content = "State corruption test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                IllegalStateException("State corruption detected")
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            // Agent should still be initialized despite the error
            assertTrue(auraAgent.isInitialized())
        }

        @Test
        @DisplayName("Should preserve configuration state across message processing failures")
        fun shouldPreserveConfigurationStateAcrossMessageProcessingFailures() = runTest {
            // Given
            val originalName = auraAgent.getName()
            val originalVersion = auraAgent.getVersion()
            val originalCapabilities = auraAgent.getCapabilities()

            val message = AgentMessage(
                id = "config-preserve-001",
                type = MessageType.TEXT,
                content = "Configuration preservation test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                RuntimeException("Processing failure")
            )

            // When
            repeat(10) {
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(originalName, auraAgent.getName())
            assertEquals(originalVersion, auraAgent.getVersion())
            assertEquals(originalCapabilities, auraAgent.getCapabilities())
        }

        @Test
        @DisplayName("Should handle concurrent state modifications safely")
        fun shouldHandleConcurrentStateModificationsSafely() = runTest {
            // Given
            val latch = CountDownLatch(20)
            val exceptions = mutableListOf<Exception>()

            // When
            repeat(20) { index ->
                Thread {
                    try {
                        when (index % 4) {
                            0 -> auraAgent.start()
                            1 -> auraAgent.stop()
                            2 -> auraAgent.handleConfigurationChanged()
                            3 -> auraAgent.processMessage(
                                AgentMessage(
                                    id = "concurrent-$index",
                                    type = MessageType.TEXT,
                                    content = "Concurrent test",
                                    timestamp = System.currentTimeMillis()
                                )
                            )
                        }
                    } catch (e: Exception) {
                        synchronized(exceptions) {
                            exceptions.add(e)
                        }
                    } finally {
                        latch.countDown()
                    }
                }.start()
            }

            latch.await(10, TimeUnit.SECONDS)

            // Then
            // No exceptions should occur during concurrent operations
            assertTrue(exceptions.isEmpty(), "Concurrent operations should not cause exceptions: ${exceptions.joinToString()}")
            assertTrue(auraAgent.isInitialized())
        }
    }

    @Nested
    @DisplayName("Performance and Stress Testing")
    inner class PerformanceAndStressTests {

        @Test
        @DisplayName("Should handle burst message processing efficiently")
        fun shouldHandleBurstMessageProcessingEfficiently() = runTest {
            // Given
            val burstSize = 1000
            val messages = (1..burstSize).map { index ->
                AgentMessage(
                    id = "burst-$index",
                    type = MessageType.TEXT,
                    content = "Burst message $index",
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
            assertEquals(burstSize, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
            // Should process 1000 messages in under 5 seconds
            assertTrue(endTime - startTime < 5000, "Burst processing took too long: ${endTime - startTime}ms")
        }

        @Test
        @DisplayName("Should maintain performance under sustained load")
        fun shouldMaintainPerformanceUnderSustainedLoad() = runTest {
            // Given
            val loadDuration = 1000 // 1 second
            val message = AgentMessage(
                id = "sustained-load",
                type = MessageType.TEXT,
                content = "Sustained load test",
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

            // When
            val startTime = System.currentTimeMillis()
            var messageCount = 0
            var totalResponseTime = 0L

            while (System.currentTimeMillis() - startTime < loadDuration) {
                val responseStart = System.currentTimeMillis()
                val response = auraAgent.processMessage(message)
                val responseTime = System.currentTimeMillis() - responseStart
                
                totalResponseTime += responseTime
                messageCount++
                
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }

            // Then
            assertTrue(messageCount > 0, "Should process at least one message")
            val averageResponseTime = totalResponseTime / messageCount
            assertTrue(averageResponseTime < 100, "Average response time should be under 100ms, was ${averageResponseTime}ms")
        }

        @Test
        @DisplayName("Should handle memory-intensive message processing")
        fun shouldHandleMemoryIntensiveMessageProcessing() = runTest {
            // Given
            val initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val memoryIntensiveMessages = (1..50).map { index ->
                AgentMessage(
                    id = "memory-$index",
                    type = MessageType.TEXT,
                    content = "x".repeat(500000), // 500KB per message
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
            memoryIntensiveMessages.forEach { message ->
                auraAgent.processMessage(message)
            }

            // Then
            System.gc()
            Thread.sleep(100) // Give GC time to run
            val finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val memoryIncrease = finalMemory - initialMemory
            
            // Memory increase should be reasonable (less than 50MB)
            assertTrue(memoryIncrease < 50 * 1024 * 1024, 
                "Memory increase of ${memoryIncrease / 1024 / 1024}MB exceeded 50MB limit")
        }

        @Test
        @DisplayName("Should handle processing timeout gracefully under load")
        fun shouldHandleProcessingTimeoutGracefullyUnderLoad() = runTest {
            // Given
            val timeoutMessages = (1..10).map { index ->
                AgentMessage(
                    id = "timeout-$index",
                    type = MessageType.TEXT,
                    content = "Timeout test $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(6000) // Simulate long processing
                AgentResponse(
                    messageId = "test",
                    content = "Should not reach here",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val startTime = System.currentTimeMillis()
            val responses = timeoutMessages.map { message ->
                auraAgent.processMessage(message)
            }
            val endTime = System.currentTimeMillis()

            // Then
            assertEquals(10, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.TIMEOUT, response.status)
            }
            // Should timeout quickly, not wait for full processing
            assertTrue(endTime - startTime < 30000, "Timeout handling took too long")
        }
    }

    @Nested
    @DisplayName("Logging and Monitoring Tests")
    inner class LoggingAndMonitoringTests {

        @Test
        @DisplayName("Should log message processing events correctly")
        fun shouldLogMessageProcessingEventsCorrectly() = runTest {
            // Given
            val message = AgentMessage(
                id = "log-test-001",
                type = MessageType.TEXT,
                content = "Logging test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Logged response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            // Verify logging through event publication
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.MESSAGE_PROCESSED && 
                event.data["messageId"] == message.id
            })
        }

        @Test
        @DisplayName("Should log configuration changes")
        fun shouldLogConfigurationChanges() = runTest {
            // Given
            val newConfig = AgentConfiguration(
                name = "LogTestAgent",
                version = "2.0.0",
                capabilities = listOf("LOGGING_TEST"),
                maxConcurrentTasks = 8
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)

            // When
            auraAgent.handleConfigurationChanged()

            // Then
            assertEquals("LogTestAgent", auraAgent.getName())
            assertEquals("2.0.0", auraAgent.getVersion())
            // Configuration change should be logged through events
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.CONFIGURATION_CHANGED
            })
        }

        @Test
        @DisplayName("Should monitor agent lifecycle events")
        fun shouldMonitorAgentLifecycleEvents() = runTest {
            // Given
            assertFalse(auraAgent.isRunning())

            // When
            auraAgent.start()
            auraAgent.stop()
            auraAgent.shutdown()

            // Then
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_SHUTDOWN
            })
        }

        @Test
        @DisplayName("Should track performance metrics")
        fun shouldTrackPerformanceMetrics() = runTest {
            // Given
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "metrics-$index",
                    type = MessageType.TEXT,
                    content = "Metrics test $index",
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
            // Verify metrics are tracked through resource usage
            val resourceUsage = auraAgent.getResourceUsage()
            assertNotNull(resourceUsage)
            assertTrue(resourceUsage.memoryUsage >= 0)
            assertEquals(0, auraAgent.getActiveTaskCount()) // Should be 0 after processing
        }
    }

    @Nested
    @DisplayName("Data Integrity and Corruption Tests")
    inner class DataIntegrityAndCorruptionTests {

        @Test
        @DisplayName("Should handle corrupted message data gracefully")
        fun shouldHandleCorruptedMessageDataGracefully() = runTest {
            // Given
            val corruptedMessage = AgentMessage(
                id = "corrupt-001",
                type = MessageType.TEXT,
                content = null, // Corrupted content
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(corruptedMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("Invalid message"))
        }

        @Test
        @DisplayName("Should validate message data integrity")
        fun shouldValidateMessageDataIntegrity() = runTest {
            // Given
            val message = AgentMessage(
                id = "integrity-001",
                type = MessageType.TEXT,
                content = "Integrity test message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = message.id,
                    content = "Validated response",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals(message.id, response.messageId)
            verify(mockMessageHandler).validateMessage(message)
        }

        @Test
        @DisplayName("Should handle configuration corruption gracefully")
        fun shouldHandleConfigurationCorruptionGracefully() = runTest {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(null)

            // When
            val originalName = auraAgent.getName()
            val originalVersion = auraAgent.getVersion()
            
            auraAgent.handleConfigurationChanged()

            // Then
            // Should fall back to default configuration
            assertEquals(originalName, auraAgent.getName())
            assertEquals(originalVersion, auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should maintain data consistency during concurrent operations")
        fun shouldMaintainDataConsistencyDuringConcurrentOperations() = runTest {
            // Given
            val messages = (1..20).map { index ->
                AgentMessage(
                    id = "consistency-$index",
                    type = MessageType.TEXT,
                    content = "Consistency test $index",
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

            val latch = CountDownLatch(20)
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

            latch.await(5, TimeUnit.SECONDS)

            // Then
            assertEquals(20, responses.size)
            responses.forEach { response ->
                assertNotNull(response)
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
        }
    }

    @Nested
    @DisplayName("Advanced Error Recovery Scenarios")
    inner class AdvancedErrorRecoveryScenarios {

        @Test
        @DisplayName("Should recover from cascading failures")
        fun shouldRecoverFromCascadingFailures() = runTest {
            // Given
            val messages = (1..5).map { index ->
                AgentMessage(
                    id = "cascade-$index",
                    type = MessageType.TEXT,
                    content = "Cascade test $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any()))
                .thenThrow(RuntimeException("First failure"))
                .thenThrow(RuntimeException("Second failure"))
                .thenThrow(RuntimeException("Third failure"))
                .thenReturn(
                    AgentResponse(
                        messageId = "test",
                        content = "Recovered",
                        status = ResponseStatus.SUCCESS
                    )
                )
                .thenReturn(
                    AgentResponse(
                        messageId = "test",
                        content = "Fully recovered",
                        status = ResponseStatus.SUCCESS
                    )
                )

            // When
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(5, responses.size)
            // First three should be errors, last two should be successful
            assertEquals(ResponseStatus.ERROR, responses[0].status)
            assertEquals(ResponseStatus.ERROR, responses[1].status)
            assertEquals(ResponseStatus.ERROR, responses[2].status)
            assertEquals(ResponseStatus.SUCCESS, responses[3].status)
            assertEquals(ResponseStatus.SUCCESS, responses[4].status)
        }

        @Test
        @DisplayName("Should handle partial system failures")
        fun shouldHandlePartialSystemFailures() = runTest {
            // Given
            val message = AgentMessage(
                id = "partial-failure-001",
                type = MessageType.TEXT,
                content = "Partial failure test",
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

            // Simulate event bus failure
            doThrow(RuntimeException("Event bus failure"))
                .whenever(mockEventBus).publish(any())

            // When
            val response = auraAgent.processMessage(message)

            // Then
            // Message processing should still succeed despite event bus failure
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should handle resource exhaustion gracefully")
        fun shouldHandleResourceExhaustionGracefully() = runTest {
            // Given
            val message = AgentMessage(
                id = "resource-exhaustion-001",
                type = MessageType.TEXT,
                content = "Resource exhaustion test",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                OutOfMemoryError("Heap space exhausted")
            )

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SYSTEM_ERROR, response.status)
            assertTrue(response.content.contains("System error"))
        }

        @Test
        @DisplayName("Should recover from configuration provider failures")
        fun shouldRecoverFromConfigurationProviderFailures() = runTest {
            // Given
            val originalConfig = AgentConfiguration(
                name = "OriginalAgent",
                version = "1.0.0",
                capabilities = listOf("CHAT"),
                maxConcurrentTasks = 5
            )

            whenever(mockConfigurationProvider.getAgentConfiguration())
                .thenReturn(originalConfig)
                .thenThrow(RuntimeException("Config provider failure"))
                .thenReturn(
                    AgentConfiguration(
                        name = "RecoveredAgent",
                        version = "1.1.0",
                        capabilities = listOf("CHAT", "RECOVERY"),
                        maxConcurrentTasks = 10
                    )
                )

            // When
            val agent = AuraAgent(mockAgentContext)
            val initialName = agent.getName()
            
            // Trigger configuration change that fails
            agent.handleConfigurationChanged()
            val nameAfterFailure = agent.getName()
            
            // Trigger configuration change that succeeds
            agent.handleConfigurationChanged()
            val nameAfterRecovery = agent.getName()

            // Then
            assertEquals("OriginalAgent", initialName)
            assertEquals("OriginalAgent", nameAfterFailure) // Should keep original config
            assertEquals("RecoveredAgent", nameAfterRecovery) // Should update to new config
        }
    }

    @Nested
    @DisplayName("Compatibility and Interoperability Tests")
    inner class CompatibilityAndInteroperabilityTests {

        @Test
        @DisplayName("Should handle legacy message formats")
        fun shouldHandleLegacyMessageFormats() = runTest {
            // Given
            val legacyMessage = AgentMessage(
                id = "legacy-001",
                type = MessageType.TEXT,
                content = "Legacy format message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = legacyMessage.id,
                    content = "Legacy processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(legacyMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("Legacy processed", response.content)
        }

        @Test
        @DisplayName("Should maintain backward compatibility with older configurations")
        fun shouldMaintainBackwardCompatibilityWithOlderConfigurations() = runTest {
            // Given
            val oldConfig = AgentConfiguration(
                name = "OldAgent",
                version = "0.9.0",
                capabilities = listOf("LEGACY_CHAT"),
                maxConcurrentTasks = 1
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(oldConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertEquals("OldAgent", agent.getName())
            assertEquals("0.9.0", agent.getVersion())
            assertTrue(agent.hasCapability("LEGACY_CHAT"))
            assertTrue(agent.getMaxConcurrentTasks() >= 1)
        }

        @Test
        @DisplayName("Should handle version mismatch gracefully")
        fun shouldHandleVersionMismatchGracefully() = runTest {
            // Given
            val futureVersionConfig = AgentConfiguration(
                name = "FutureAgent",
                version = "999.0.0",
                capabilities = listOf("FUTURE_FEATURE"),
                maxConcurrentTasks = 1000
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(futureVersionConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertEquals("FutureAgent", agent.getName())
            // Should handle future version gracefully
            assertEquals("999.0.0", agent.getVersion())
        }

        @Test
        @DisplayName("Should handle cross-platform message encoding")
        fun shouldHandleCrossPlatformMessageEncoding() = runTest {
            // Given
            val encodedMessage = AgentMessage(
                id = "encoded-001",
                type = MessageType.TEXT,
                content = "Encoded: 你好世界 🌍 Здравствуй мир",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = encodedMessage.id,
                    content = "Cross-platform processed",
                    status = ResponseStatus.SUCCESS
                )
            )

            // When
            val response = auraAgent.processMessage(encodedMessage)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("Cross-platform processed", response.content)
        }
    }

    @Nested
    @DisplayName("Integration and System Tests")
    inner class IntegrationAndSystemTests {

        @Test
        @DisplayName("Should handle complete system integration workflow")
        fun shouldHandleCompleteSystemIntegrationWorkflow() = runTest {
            // Given
            val messages = listOf(
                AgentMessage("init-001", MessageType.COMMAND, "/initialize", System.currentTimeMillis()),
                AgentMessage("query-001", MessageType.QUERY, "What is your status?", System.currentTimeMillis()),
                AgentMessage("text-001", MessageType.TEXT, "Hello agent", System.currentTimeMillis()),
                AgentMessage("shutdown-001", MessageType.COMMAND, "/shutdown", System.currentTimeMillis())
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer { invocation ->
                val msg = invocation.arguments[0] as AgentMessage
                AgentResponse(
                    messageId = msg.id,
                    content = "Processed ${msg.type}: ${msg.content}",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            auraAgent.start()
            val responses = messages.map { message ->
                auraAgent.processMessage(message)
            }
            auraAgent.stop()

            // Then
            assertEquals(4, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
            
            // Verify complete workflow events
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STARTED
            })
            verify(mockEventBus, times(4)).publish(argThat { event ->
                event.type == EventType.MESSAGE_PROCESSED
            })
            verify(mockEventBus).publish(argThat { event ->
                event.type == EventType.AGENT_STOPPED
            })
        }

        @Test
        @DisplayName("Should handle system-wide stress conditions")
        fun shouldHandleSystemWideStressConditions() = runTest {
            // Given
            val stressMessages = (1..500).map { index ->
                AgentMessage(
                    id = "stress-$index",
                    type = if (index % 3 == 0) MessageType.COMMAND else MessageType.TEXT,
                    content = "Stress test message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer { invocation ->
                // Simulate variable processing time
                Thread.sleep((1..10).random().toLong())
                val msg = invocation.arguments[0] as AgentMessage
                AgentResponse(
                    messageId = msg.id,
                    content = "Stress processed",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            auraAgent.start()
            val startTime = System.currentTimeMillis()
            val responses = stressMessages.map { message ->
                auraAgent.processMessage(message)
            }
            val endTime = System.currentTimeMillis()
            auraAgent.stop()

            // Then
            assertEquals(500, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
            }
            
            // Should handle stress test reasonably quickly
            val totalTime = endTime - startTime
            assertTrue(totalTime < 30000, "Stress test took too long: ${totalTime}ms")
            
            // Agent should still be in good state
            assertTrue(auraAgent.isInitialized())
            assertEquals(0, auraAgent.getActiveTaskCount())
        }
    }
}