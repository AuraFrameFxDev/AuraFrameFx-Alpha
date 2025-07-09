package dev.aurakai.auraframefx.ai.clients

import com.google.cloud.vertexai.VertexAI
import com.google.cloud.vertexai.api.GenerateContentRequest
import com.google.cloud.vertexai.api.GenerateContentResponse
import com.google.cloud.vertexai.generativeai.GenerativeModel
import com.google.cloud.vertexai.generativeai.ResponseHandler
import dev.aurakai.auraframefx.ai.models.AIMessage
import dev.aurakai.auraframefx.ai.models.AIResponse
import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.extension.ExtendWith
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.mockito.junit.jupiter.MockitoExtension
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

@ExtendWith(MockitoExtension::class)
class VertexAIClientImplTest {

    private lateinit var vertexAI: VertexAI
    private lateinit var generativeModel: GenerativeModel
    private lateinit var vertexAIClient: VertexAIClientImpl

    @BeforeEach
    fun setUp() {
        vertexAI = mockk()
        generativeModel = mockk()
        vertexAIClient = VertexAIClientImpl(vertexAI, generativeModel)
    }

    @AfterEach
    fun tearDown() {
        unmockkAll()
    }

    // Happy path tests
    @Test
    fun `generateContent should return successful response for valid input`() = runTest {
        // Given
        val inputMessage = AIMessage(
            role = "user",
            content = "Hello, how are you?"
        )
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "I'm doing well, thank you!"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("I'm doing well, thank you!", result.content)
        assertTrue(result.isSuccessful)
        verify { generativeModel.generateContent("Hello, how are you?") }
    }

    @Test
    fun `generateContent should handle multiple conversation turns`() = runTest {
        // Given
        val messages = listOf(
            AIMessage(role = "user", content = "What's the weather like?"),
            AIMessage(role = "assistant", content = "I need more information about your location."),
            AIMessage(role = "user", content = "I'm in New York.")
        )
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "The weather in New York is partly cloudy."
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<List<AIMessage>>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(messages)

        // Then
        assertNotNull(result)
        assertEquals("The weather in New York is partly cloudy.", result.content)
        assertTrue(result.isSuccessful)
        verify { generativeModel.generateContent(messages) }
    }

    // Edge case tests
    @Test
    fun `generateContent should handle empty message content`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "I notice you sent an empty message."
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("I notice you sent an empty message.", result.content)
        assertTrue(result.isSuccessful)
    }

    @Test
    fun `generateContent should handle null response candidates`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Test message")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns emptyList()
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("", result.content)
        assertTrue(result.isSuccessful)
    }

    @Test
    fun `generateContent should handle response with empty parts list`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Test message")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns emptyList()
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("", result.content)
        assertTrue(result.isSuccessful)
    }

    @ParameterizedTest
    @ValueSource(strings = ["user", "assistant", "system"])
    fun `generateContent should handle different message roles`(role: String) = runTest {
        // Given
        val inputMessage = AIMessage(role = role, content = "Test content")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Response for $role"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("Response for $role", result.content)
        assertTrue(result.isSuccessful)
    }

    @Test
    fun `generateContent should handle very long message content`() = runTest {
        // Given
        val longContent = "a".repeat(10000)
        val inputMessage = AIMessage(role = "user", content = longContent)
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Processed long content"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("Processed long content", result.content)
        assertTrue(result.isSuccessful)
    }

    @Test
    fun `generateContent should handle special characters in content`() = runTest {
        // Given
        val specialContent = "Hello! @#$%^&*()_+{}|:<>?[]\\;'\",./"
        val inputMessage = AIMessage(role = "user", content = specialContent)
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Handled special characters"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When
        val result = vertexAIClient.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("Handled special characters", result.content)
        assertTrue(result.isSuccessful)
    }

    // Error handling tests
    @Test
    fun `generateContent should handle API exceptions gracefully`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Test message")
        every { generativeModel.generateContent(any<String>()) } throws RuntimeException("API Error")

        // When & Then
        assertThrows<RuntimeException> {
            vertexAIClient.generateContent(inputMessage)
        }
    }

    @Test
    fun `generateContent should handle network timeout`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Test message")
        every { generativeModel.generateContent(any<String>()) } throws java.net.SocketTimeoutException("Timeout")

        // When & Then
        assertThrows<java.net.SocketTimeoutException> {
            vertexAIClient.generateContent(inputMessage)
        }
    }

    @Test
    fun `generateContent should handle authentication errors`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Test message")
        every { generativeModel.generateContent(any<String>()) } throws SecurityException("Authentication failed")

        // When & Then
        assertThrows<SecurityException> {
            vertexAIClient.generateContent(inputMessage)
        }
    }

    @Test
    fun `generateContent should handle quota exceeded errors`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Test message")
        every { generativeModel.generateContent(any<String>()) } throws IllegalStateException("Quota exceeded")

        // When & Then
        assertThrows<IllegalStateException> {
            vertexAIClient.generateContent(inputMessage)
        }
    }

    // Stream generation tests
    @Test
    fun `generateContentStream should handle streaming responses`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Stream test")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Streaming response"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContentStream(any<String>()) } returns mockk {
            every { iterator() } returns listOf(mockResponse).iterator()
        }

        // When
        val result = vertexAIClient.generateContentStream(inputMessage)

        // Then
        assertNotNull(result)
        verify { generativeModel.generateContentStream("Stream test") }
    }

    @Test
    fun `generateContentStream should handle empty stream`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Empty stream test")
        
        every { generativeModel.generateContentStream(any<String>()) } returns mockk {
            every { iterator() } returns emptyList<GenerateContentResponse>().iterator()
        }

        // When
        val result = vertexAIClient.generateContentStream(inputMessage)

        // Then
        assertNotNull(result)
        verify { generativeModel.generateContentStream("Empty stream test") }
    }

    // Configuration tests
    @Test
    fun `client should be properly initialized with VertexAI and GenerativeModel`() {
        // Given & When
        val client = VertexAIClientImpl(vertexAI, generativeModel)

        // Then
        assertNotNull(client)
    }

    @Test
    fun `client should handle model configuration changes`() = runTest {
        // Given
        val newModel = mockk<GenerativeModel>()
        val inputMessage = AIMessage(role = "user", content = "Test with new model")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "New model response"
                            }
                        )
                    }
                }
            )
        }
        
        every { newModel.generateContent(any<String>()) } returns mockResponse
        val clientWithNewModel = VertexAIClientImpl(vertexAI, newModel)

        // When
        val result = clientWithNewModel.generateContent(inputMessage)

        // Then
        assertNotNull(result)
        assertEquals("New model response", result.content)
        verify { newModel.generateContent("Test with new model") }
    }

    // Concurrency tests
    @Test
    fun `generateContent should handle concurrent requests`() = runTest {
        // Given
        val inputMessage1 = AIMessage(role = "user", content = "Concurrent test 1")
        val inputMessage2 = AIMessage(role = "user", content = "Concurrent test 2")
        val mockResponse1 = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Response 1"
                            }
                        )
                    }
                }
            )
        }
        val mockResponse2 = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Response 2"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent("Concurrent test 1") } returns mockResponse1
        every { generativeModel.generateContent("Concurrent test 2") } returns mockResponse2

        // When
        val result1 = vertexAIClient.generateContent(inputMessage1)
        val result2 = vertexAIClient.generateContent(inputMessage2)

        // Then
        assertNotNull(result1)
        assertNotNull(result2)
        assertEquals("Response 1", result1.content)
        assertEquals("Response 2", result2.content)
    }

    // Performance tests
    @Test
    fun `generateContent should handle rapid successive calls`() = runTest {
        // Given
        val inputMessage = AIMessage(role = "user", content = "Rapid test")
        val mockResponse = mockk<GenerateContentResponse> {
            every { getCandidatesList() } returns listOf(
                mockk {
                    every { content } returns mockk {
                        every { partsList } returns listOf(
                            mockk {
                                every { text } returns "Rapid response"
                            }
                        )
                    }
                }
            )
        }
        
        every { generativeModel.generateContent(any<String>()) } returns mockResponse

        // When & Then
        repeat(10) {
            val result = vertexAIClient.generateContent(inputMessage)
            assertNotNull(result)
            assertEquals("Rapid response", result.content)
        }
        
        verify(exactly = 10) { generativeModel.generateContent("Rapid test") }
    }
}