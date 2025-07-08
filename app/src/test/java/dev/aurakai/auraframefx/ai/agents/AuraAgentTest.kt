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
            val validContext = mock<AgentContext>()
            whenever(validContext.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(validContext.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(validContext.getEventBus()).thenReturn(mockEventBus)

            val agent = AuraAgent(validContext)

            assertNotNull(agent)
            assertEquals("TestAgent", agent.getName())
            assertEquals("1.0.0", agent.getVersion())
            assertTrue(agent.isInitialized())
        }

        @Test
        @DisplayName("Should throw exception when context is null")
        fun shouldThrowExceptionWhenContextIsNull() {
            assertFailsWith<IllegalArgumentException> {
                AuraAgent(null)
            }
        }

        @Test
        @DisplayName("Should throw exception when configuration provider is null")
        fun shouldThrowExceptionWhenConfigurationProviderIsNull() {
            val invalidContext = mock<AgentContext>()
            whenever(invalidContext.getConfigurationProvider()).thenReturn(null)
            assertFailsWith<IllegalStateException> {
                AuraAgent(invalidContext)
            }
        }

        @Test
        @DisplayName("Should initialize with default configuration when config is missing")
        fun shouldInitializeWithDefaultConfigurationWhenConfigIsMissing() {
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(null)
            val agent = AuraAgent(mockAgentContext)
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
            val message = AgentMessage("msg-001", MessageType.TEXT, "Hello, AuraAgent!", System.currentTimeMillis())
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(message.id, "Hello! How can I help you today?", ResponseStatus.SUCCESS)
            )

            val response = auraAgent.processMessage(message)

            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
            assertEquals("Hello! How can I help you today?", response.content)
            verify(mockMessageHandler).validateMessage(message)
            verify(mockMessageHandler).processMessage(message)
        }

        @Test
        @DisplayName("Should handle invalid message gracefully")
        fun shouldHandleInvalidMessageGracefully() = runTest {
            val invalidMessage = AgentMessage("", MessageType.TEXT, "", 0)
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            val response = auraAgent.processMessage(invalidMessage)

            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
            assertTrue(response.content.contains("Invalid message"))
            verify(mockMessageHandler).validateMessage(invalidMessage)
            verify(mockMessageHandler, never()).processMessage(any())
        }

        @Test
        @DisplayName("Should handle message processing exceptions")
        fun shouldHandleMessageProcessingExceptions() = runTest {
            val message = AgentMessage("msg-002", MessageType.TEXT, "Test message", System.currentTimeMillis())
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenThrow(RuntimeException("Processing failed"))

            val response = auraAgent.processMessage(message)

            assertNotNull(response)
            assertEquals(ResponseStatus.ERROR, response.status)
            assertTrue(response.content.contains("Processing failed"))
        }

        @Test
        @DisplayName("Should handle concurrent message processing")
        fun shouldHandleConcurrentMessageProcessing() = runTest {
            val messages = (1..10).map {
                AgentMessage("msg-$it", MessageType.TEXT, "Message $it", System.currentTimeMillis())
            }
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse("test", "Processed", ResponseStatus.SUCCESS)
            )

            val responses = messages.map { auraAgent.processMessage(it) }

            assertEquals(10, responses.size)
            responses.forEach { assertEquals(ResponseStatus.SUCCESS, it.status) }
            verify(mockMessageHandler, times(10)).validateMessage(any())
            verify(mockMessageHandler, times(10)).processMessage(any())
        }

        @Test
        @DisplayName("Should respect maximum concurrent tasks limit")
        fun shouldRespectMaximumConcurrentTasksLimit() = runTest {
            val config = AgentConfiguration("TestAgent", "1.0.0", listOf("CHAT"), maxConcurrentTasks = 2)
            whenever(mockConfigurationProvider.getAgentConfiguration()).thenReturn(config)
            val agent = AuraAgent(mockAgentContext)
            val latch = CountDownLatch(2)
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenAnswer {
                latch.countDown()
                latch.await(1, TimeUnit.SECONDS)
                AgentResponse("test", "Processed", ResponseStatus.SUCCESS)
            }

            val messages = (1..5).map {
                AgentMessage("msg-$it", MessageType.TEXT, "Message $it", System.currentTimeMillis())
            }
            val responses = messages.map { agent.processMessage(it) }

            assertEquals(5, responses.size)
            assertTrue(agent.getActiveTaskCount() <= 2)
        }
    }

    @Nested
    @DisplayName("Event Handling Tests")
    inner class EventHandlingTests {
        @Test
        @DisplayName("Should publish event when message is processed")
        fun shouldPublishEventWhenMessageIsProcessed() = runTest {
            val message = AgentMessage("msg-001", MessageType.TEXT, "Test message", System.currentTimeMillis())
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(message.id, "Response", ResponseStatus.SUCCESS)
            )

            auraAgent.processMessage(message)

            val eventCaptor = argumentCaptor<AgentEvent>()
            verify(mockEventBus).publish(eventCaptor.capture())
            val publishedEvent = eventCaptor.firstValue
            assertEquals(EventType.MESSAGE_PROCESSED, publishedEvent.type)
            assertEquals(message.id, publishedEvent.data["messageId"])
        }

        @Test
        @DisplayName("Should handle event publishing failures gracefully")
        fun shouldHandleEventPublishingFailuresGracefully() = runTest {
            val message = AgentMessage("msg-001", MessageType.TEXT, "Test message", System.currentTimeMillis())
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(message.id, "Response", ResponseStatus.SUCCESS)
            )
            doThrow(RuntimeException("Event publishing failed")).whenever(mockEventBus).publish(any())

            val response = auraAgent.processMessage(message)
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should register event listeners during initialization")
        fun shouldRegisterEventListenersDuringInitialization() {
            val context = mock<AgentContext>()
            whenever(context.getConfigurationProvider()).thenReturn(mockConfigurationProvider)
            whenever(context.getMessageHandler()).thenReturn(mockMessageHandler)
            whenever(context.getEventBus()).thenReturn(mockEventBus)

            AuraAgent(context)

            verify(mockEventBus).subscribe(eq(EventType.SYSTEM_SHUTDOWN), any())
            verify(mockEventBus).subscribe(eq(EventType.CONFIGURATION_CHANGED), any())
        }
    }

    @Nested
    @DisplayName("Boundary Value Tests")
    inner class BoundaryValueTests {
        @Test
        @DisplayName("Should handle message at exact character limit")
        fun shouldHandleMessageAtExactCharacterLimit() = runTest {
            val maxLength = 65536
            val exactLimitMessage = AgentMessage("limit-001", MessageType.TEXT, "x".repeat(maxLength), System.currentTimeMillis())
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(true)
            whenever(mockMessageHandler.processMessage(any())).thenReturn(
                AgentResponse(exactLimitMessage.id, "Processed at limit", ResponseStatus.SUCCESS)
            )

            val response = auraAgent.processMessage(exactLimitMessage)
            assertNotNull(response)
            assertEquals(ResponseStatus.SUCCESS, response.status)
        }

        @Test
        @DisplayName("Should handle message exceeding character limit")
        fun shouldHandleMessageExceedingCharacterLimit() = runTest {
            val oversizedMessage = AgentMessage("oversized-001", MessageType.TEXT, "x".repeat(1000000), System.currentTimeMillis())
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            val response = auraAgent.processMessage(oversizedMessage)
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }

        @Test
        @DisplayName("Should handle maximum valid timestamp")
        fun shouldHandleMaximumValidTimestamp() = runTest {
            val maxTimestampMessage = AgentMessage("max-timestamp-001", MessageType.TEXT, "Maximum timestamp test", Long.MAX_VALUE)
            whenever(mockMessageHandler.validateMessage(any())).thenReturn(false)

            val response = auraAgent.processMessage(maxTimestampMessage)
            assertNotNull(response)
            assertEquals(ResponseStatus.VALIDATION_ERROR, response.status)
        }
    }
}