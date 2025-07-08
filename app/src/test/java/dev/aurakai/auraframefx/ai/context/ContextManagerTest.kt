package dev.aurakai.auraframefx.ai.context

import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import kotlin.test.assertFailsWith

@DisplayName("ContextManager Tests")
class ContextManagerTest {

    private lateinit var contextManager: ContextManager
    
    @Mock
    private lateinit var mockContextProvider: ContextProvider
    
    @Mock
    private lateinit var mockContextStorage: ContextStorage
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        contextManager = ContextManager(mockContextProvider, mockContextStorage)
    }

    @Nested
    @DisplayName("Context Creation Tests")
    inner class ContextCreationTests {

        @Test
        @DisplayName("Should create context successfully with valid input")
        fun `should create context successfully with valid input`() {
            // Given
            val contextId = "test-context-123"
            val contextData = mapOf("key" to "value")
            val expectedContext = Context(contextId, contextData)
            
            whenever(mockContextProvider.createContext(contextId, contextData))
                .thenReturn(expectedContext)
            
            // When
            val result = contextManager.createContext(contextId, contextData)
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result.id)
            assertEquals(contextData, result.data)
            verify(mockContextProvider).createContext(contextId, contextData)
            verify(mockContextStorage).store(expectedContext)
        }

        @Test
        @DisplayName("Should throw exception when creating context with null id")
        fun `should throw exception when creating context with null id`() {
            // Given
            val contextData = mapOf("key" to "value")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.createContext(null, contextData)
            }
            
            verify(mockContextProvider, never()).createContext(any(), any())
            verify(mockContextStorage, never()).store(any())
        }

        @Test
        @DisplayName("Should throw exception when creating context with empty id")
        fun `should throw exception when creating context with empty id`() {
            // Given
            val contextData = mapOf("key" to "value")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.createContext("", contextData)
            }
            
            verify(mockContextProvider, never()).createContext(any(), any())
            verify(mockContextStorage, never()).store(any())
        }

        @Test
        @DisplayName("Should handle context creation with empty data")
        fun `should handle context creation with empty data`() {
            // Given
            val contextId = "test-context-empty"
            val contextData = emptyMap<String, Any>()
            val expectedContext = Context(contextId, contextData)
            
            whenever(mockContextProvider.createContext(contextId, contextData))
                .thenReturn(expectedContext)
            
            // When
            val result = contextManager.createContext(contextId, contextData)
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result.id)
            assertTrue(result.data.isEmpty())
            verify(mockContextProvider).createContext(contextId, contextData)
            verify(mockContextStorage).store(expectedContext)
        }

        @Test
        @DisplayName("Should handle context creation failure from provider")
        fun `should handle context creation failure from provider`() {
            // Given
            val contextId = "failing-context"
            val contextData = mapOf("key" to "value")
            val exception = RuntimeException("Context creation failed")
            
            whenever(mockContextProvider.createContext(contextId, contextData))
                .thenThrow(exception)
            
            // When & Then
            assertThrows<RuntimeException> {
                contextManager.createContext(contextId, contextData)
            }
            
            verify(mockContextProvider).createContext(contextId, contextData)
            verify(mockContextStorage, never()).store(any())
        }
    }

    @Nested
    @DisplayName("Context Retrieval Tests")
    inner class ContextRetrievalTests {

        @Test
        @DisplayName("Should retrieve existing context successfully")
        fun `should retrieve existing context successfully`() {
            // Given
            val contextId = "existing-context"
            val contextData = mapOf("key" to "value")
            val expectedContext = Context(contextId, contextData)
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(expectedContext)
            
            // When
            val result = contextManager.getContext(contextId)
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result?.id)
            assertEquals(contextData, result?.data)
            verify(mockContextStorage).retrieve(contextId)
        }

        @Test
        @DisplayName("Should return null for non-existing context")
        fun `should return null for non-existing context`() {
            // Given
            val contextId = "non-existing-context"
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(null)
            
            // When
            val result = contextManager.getContext(contextId)
            
            // Then
            assertNull(result)
            verify(mockContextStorage).retrieve(contextId)
        }

        @Test
        @DisplayName("Should throw exception when retrieving context with null id")
        fun `should throw exception when retrieving context with null id`() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.getContext(null)
            }
            
            verify(mockContextStorage, never()).retrieve(any())
        }

        @Test
        @DisplayName("Should throw exception when retrieving context with empty id")
        fun `should throw exception when retrieving context with empty id`() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.getContext("")
            }
            
            verify(mockContextStorage, never()).retrieve(any())
        }

        @Test
        @DisplayName("Should handle storage exception during retrieval")
        fun `should handle storage exception during retrieval`() {
            // Given
            val contextId = "error-context"
            val exception = RuntimeException("Storage error")
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenThrow(exception)
            
            // When & Then
            assertThrows<RuntimeException> {
                contextManager.getContext(contextId)
            }
            
            verify(mockContextStorage).retrieve(contextId)
        }
    }

    @Nested
    @DisplayName("Context Update Tests")
    inner class ContextUpdateTests {

        @Test
        @DisplayName("Should update existing context successfully")
        fun `should update existing context successfully`() {
            // Given
            val contextId = "update-context"
            val originalData = mapOf("key" to "original")
            val updatedData = mapOf("key" to "updated", "new" to "value")
            val originalContext = Context(contextId, originalData)
            val updatedContext = Context(contextId, updatedData)
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(originalContext)
            whenever(mockContextProvider.updateContext(originalContext, updatedData))
                .thenReturn(updatedContext)
            
            // When
            val result = contextManager.updateContext(contextId, updatedData)
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result?.id)
            assertEquals(updatedData, result?.data)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextProvider).updateContext(originalContext, updatedData)
            verify(mockContextStorage).store(updatedContext)
        }

        @Test
        @DisplayName("Should return null when updating non-existing context")
        fun `should return null when updating non-existing context`() {
            // Given
            val contextId = "non-existing-context"
            val updatedData = mapOf("key" to "updated")
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(null)
            
            // When
            val result = contextManager.updateContext(contextId, updatedData)
            
            // Then
            assertNull(result)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextProvider, never()).updateContext(any(), any())
            verify(mockContextStorage, never()).store(any())
        }

        @Test
        @DisplayName("Should throw exception when updating context with null id")
        fun `should throw exception when updating context with null id`() {
            // Given
            val updatedData = mapOf("key" to "updated")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.updateContext(null, updatedData)
            }
            
            verify(mockContextStorage, never()).retrieve(any())
            verify(mockContextProvider, never()).updateContext(any(), any())
        }

        @Test
        @DisplayName("Should handle update with empty data")
        fun `should handle update with empty data`() {
            // Given
            val contextId = "empty-update-context"
            val originalData = mapOf("key" to "original")
            val updatedData = emptyMap<String, Any>()
            val originalContext = Context(contextId, originalData)
            val updatedContext = Context(contextId, updatedData)
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(originalContext)
            whenever(mockContextProvider.updateContext(originalContext, updatedData))
                .thenReturn(updatedContext)
            
            // When
            val result = contextManager.updateContext(contextId, updatedData)
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result?.id)
            assertTrue(result?.data?.isEmpty() == true)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextProvider).updateContext(originalContext, updatedData)
            verify(mockContextStorage).store(updatedContext)
        }
    }

    @Nested
    @DisplayName("Context Deletion Tests")
    inner class ContextDeletionTests {

        @Test
        @DisplayName("Should delete existing context successfully")
        fun `should delete existing context successfully`() {
            // Given
            val contextId = "delete-context"
            val contextData = mapOf("key" to "value")
            val existingContext = Context(contextId, contextData)
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(existingContext)
            
            // When
            val result = contextManager.deleteContext(contextId)
            
            // Then
            assertTrue(result)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextStorage).delete(contextId)
        }

        @Test
        @DisplayName("Should return false when deleting non-existing context")
        fun `should return false when deleting non-existing context`() {
            // Given
            val contextId = "non-existing-context"
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(null)
            
            // When
            val result = contextManager.deleteContext(contextId)
            
            // Then
            assertFalse(result)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextStorage, never()).delete(any())
        }

        @Test
        @DisplayName("Should throw exception when deleting context with null id")
        fun `should throw exception when deleting context with null id`() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                contextManager.deleteContext(null)
            }
            
            verify(mockContextStorage, never()).retrieve(any())
            verify(mockContextStorage, never()).delete(any())
        }

        @Test
        @DisplayName("Should handle storage exception during deletion")
        fun `should handle storage exception during deletion`() {
            // Given
            val contextId = "error-delete-context"
            val contextData = mapOf("key" to "value")
            val existingContext = Context(contextId, contextData)
            val exception = RuntimeException("Storage deletion error")
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(existingContext)
            whenever(mockContextStorage.delete(contextId))
                .thenThrow(exception)
            
            // When & Then
            assertThrows<RuntimeException> {
                contextManager.deleteContext(contextId)
            }
            
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextStorage).delete(contextId)
        }
    }

    @Nested
    @DisplayName("Context Listing Tests")
    inner class ContextListingTests {

        @Test
        @DisplayName("Should list all contexts successfully")
        fun `should list all contexts successfully`() {
            // Given
            val contexts = listOf(
                Context("context-1", mapOf("key1" to "value1")),
                Context("context-2", mapOf("key2" to "value2")),
                Context("context-3", mapOf("key3" to "value3"))
            )
            
            whenever(mockContextStorage.listAll())
                .thenReturn(contexts)
            
            // When
            val result = contextManager.listAllContexts()
            
            // Then
            assertNotNull(result)
            assertEquals(3, result.size)
            assertEquals(contexts, result)
            verify(mockContextStorage).listAll()
        }

        @Test
        @DisplayName("Should return empty list when no contexts exist")
        fun `should return empty list when no contexts exist`() {
            // Given
            whenever(mockContextStorage.listAll())
                .thenReturn(emptyList())
            
            // When
            val result = contextManager.listAllContexts()
            
            // Then
            assertNotNull(result)
            assertTrue(result.isEmpty())
            verify(mockContextStorage).listAll()
        }

        @Test
        @DisplayName("Should handle storage exception during listing")
        fun `should handle storage exception during listing`() {
            // Given
            val exception = RuntimeException("Storage listing error")
            
            whenever(mockContextStorage.listAll())
                .thenThrow(exception)
            
            // When & Then
            assertThrows<RuntimeException> {
                contextManager.listAllContexts()
            }
            
            verify(mockContextStorage).listAll()
        }
    }

    @Nested
    @DisplayName("Context Search Tests")
    inner class ContextSearchTests {

        @Test
        @DisplayName("Should search contexts by criteria successfully")
        fun `should search contexts by criteria successfully`() {
            // Given
            val searchCriteria = mapOf("type" to "test", "status" to "active")
            val matchingContexts = listOf(
                Context("context-1", mapOf("type" to "test", "status" to "active")),
                Context("context-2", mapOf("type" to "test", "status" to "active"))
            )
            
            whenever(mockContextStorage.search(searchCriteria))
                .thenReturn(matchingContexts)
            
            // When
            val result = contextManager.searchContexts(searchCriteria)
            
            // Then
            assertNotNull(result)
            assertEquals(2, result.size)
            assertEquals(matchingContexts, result)
            verify(mockContextStorage).search(searchCriteria)
        }

        @Test
        @DisplayName("Should return empty list when no contexts match criteria")
        fun `should return empty list when no contexts match criteria`() {
            // Given
            val searchCriteria = mapOf("type" to "nonexistent")
            
            whenever(mockContextStorage.search(searchCriteria))
                .thenReturn(emptyList())
            
            // When
            val result = contextManager.searchContexts(searchCriteria)
            
            // Then
            assertNotNull(result)
            assertTrue(result.isEmpty())
            verify(mockContextStorage).search(searchCriteria)
        }

        @Test
        @DisplayName("Should handle search with empty criteria")
        fun `should handle search with empty criteria`() {
            // Given
            val searchCriteria = emptyMap<String, Any>()
            val allContexts = listOf(
                Context("context-1", mapOf("key1" to "value1")),
                Context("context-2", mapOf("key2" to "value2"))
            )
            
            whenever(mockContextStorage.search(searchCriteria))
                .thenReturn(allContexts)
            
            // When
            val result = contextManager.searchContexts(searchCriteria)
            
            // Then
            assertNotNull(result)
            assertEquals(2, result.size)
            assertEquals(allContexts, result)
            verify(mockContextStorage).search(searchCriteria)
        }
    }

    @Nested
    @DisplayName("Context Validation Tests")
    inner class ContextValidationTests {

        @Test
        @DisplayName("Should validate context successfully")
        fun `should validate context successfully`() {
            // Given
            val contextId = "valid-context"
            val validContext = Context(contextId, mapOf("required" to "value"))
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(validContext)
            whenever(mockContextProvider.validateContext(validContext))
                .thenReturn(true)
            
            // When
            val result = contextManager.validateContext(contextId)
            
            // Then
            assertTrue(result)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextProvider).validateContext(validContext)
        }

        @Test
        @DisplayName("Should return false for invalid context")
        fun `should return false for invalid context`() {
            // Given
            val contextId = "invalid-context"
            val invalidContext = Context(contextId, mapOf("incomplete" to "data"))
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(invalidContext)
            whenever(mockContextProvider.validateContext(invalidContext))
                .thenReturn(false)
            
            // When
            val result = contextManager.validateContext(contextId)
            
            // Then
            assertFalse(result)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextProvider).validateContext(invalidContext)
        }

        @Test
        @DisplayName("Should return false for non-existing context validation")
        fun `should return false for non-existing context validation`() {
            // Given
            val contextId = "non-existing-context"
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(null)
            
            // When
            val result = contextManager.validateContext(contextId)
            
            // Then
            assertFalse(result)
            verify(mockContextStorage).retrieve(contextId)
            verify(mockContextProvider, never()).validateContext(any())
        }
    }

    @Nested
    @DisplayName("Concurrent Access Tests")
    inner class ConcurrentAccessTests {

        @Test
        @DisplayName("Should handle concurrent context creation")
        fun `should handle concurrent context creation`() {
            // Given
            val contextId = "concurrent-context"
            val contextData = mapOf("key" to "value")
            val expectedContext = Context(contextId, contextData)
            
            whenever(mockContextProvider.createContext(contextId, contextData))
                .thenReturn(expectedContext)
            
            // When
            val futures = (1..10).map {
                CompletableFuture.supplyAsync {
                    contextManager.createContext("$contextId-$it", contextData)
                }
            }
            
            val results = futures.map { it.get(5, TimeUnit.SECONDS) }
            
            // Then
            assertEquals(10, results.size)
            results.forEachIndexed { index, context ->
                assertEquals("$contextId-${index + 1}", context.id)
                assertEquals(contextData, context.data)
            }
            verify(mockContextProvider, times(10)).createContext(any(), eq(contextData))
            verify(mockContextStorage, times(10)).store(any())
        }

        @Test
        @DisplayName("Should handle concurrent context retrieval")
        fun `should handle concurrent context retrieval`() {
            // Given
            val contextId = "concurrent-retrieval-context"
            val contextData = mapOf("key" to "value")
            val expectedContext = Context(contextId, contextData)
            
            whenever(mockContextStorage.retrieve(contextId))
                .thenReturn(expectedContext)
            
            // When
            val futures = (1..5).map {
                CompletableFuture.supplyAsync {
                    contextManager.getContext(contextId)
                }
            }
            
            val results = futures.map { it.get(5, TimeUnit.SECONDS) }
            
            // Then
            assertEquals(5, results.size)
            results.forEach { context ->
                assertEquals(contextId, context?.id)
                assertEquals(contextData, context?.data)
            }
            verify(mockContextStorage, times(5)).retrieve(contextId)
        }
    }

    @Nested
    @DisplayName("Memory Management Tests")
    inner class MemoryManagementTests {

        @Test
        @DisplayName("Should clean up contexts periodically")
        fun `should clean up contexts periodically`() {
            // Given
            val expiredContexts = listOf(
                Context("expired-1", mapOf("timestamp" to System.currentTimeMillis() - 86400000)),
                Context("expired-2", mapOf("timestamp" to System.currentTimeMillis() - 86400000))
            )
            
            whenever(mockContextStorage.findExpiredContexts())
                .thenReturn(expiredContexts)
            
            // When
            val result = contextManager.cleanupExpiredContexts()
            
            // Then
            assertEquals(2, result)
            verify(mockContextStorage).findExpiredContexts()
            verify(mockContextStorage).delete("expired-1")
            verify(mockContextStorage).delete("expired-2")
        }

        @Test
        @DisplayName("Should handle cleanup when no expired contexts exist")
        fun `should handle cleanup when no expired contexts exist`() {
            // Given
            whenever(mockContextStorage.findExpiredContexts())
                .thenReturn(emptyList())
            
            // When
            val result = contextManager.cleanupExpiredContexts()
            
            // Then
            assertEquals(0, result)
            verify(mockContextStorage).findExpiredContexts()
            verify(mockContextStorage, never()).delete(any())
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle null context provider gracefully")
        fun `should handle null context provider gracefully`() {
            // Given
            val contextManagerWithNulls = ContextManager(null, mockContextStorage)
            
            // When & Then
            assertThrows<IllegalStateException> {
                contextManagerWithNulls.createContext("test", mapOf("key" to "value"))
            }
        }

        @Test
        @DisplayName("Should handle null context storage gracefully")
        fun `should handle null context storage gracefully`() {
            // Given
            val contextManagerWithNulls = ContextManager(mockContextProvider, null)
            
            // When & Then
            assertThrows<IllegalStateException> {
                contextManagerWithNulls.getContext("test")
            }
        }

        @Test
        @DisplayName("Should handle provider timeout gracefully")
        fun `should handle provider timeout gracefully`() {
            // Given
            val contextId = "timeout-context"
            val contextData = mapOf("key" to "value")
            
            whenever(mockContextProvider.createContext(contextId, contextData))
                .thenAnswer { 
                    Thread.sleep(10000) // Simulate long operation
                    Context(contextId, contextData)
                }
            
            // When & Then
            assertThrows<RuntimeException> {
                contextManager.createContextWithTimeout(contextId, contextData, 1000)
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle large context data efficiently")
        fun `should handle large context data efficiently`() {
            // Given
            val contextId = "large-context"
            val largeContextData = (1..1000).associate { "key$it" to "value$it" }
            val expectedContext = Context(contextId, largeContextData)
            
            whenever(mockContextProvider.createContext(contextId, largeContextData))
                .thenReturn(expectedContext)
            
            // When
            val startTime = System.currentTimeMillis()
            val result = contextManager.createContext(contextId, largeContextData)
            val endTime = System.currentTimeMillis()
            
            // Then
            assertNotNull(result)
            assertEquals(contextId, result.id)
            assertEquals(largeContextData, result.data)
            assertTrue(endTime - startTime < 1000, "Operation took too long: ${endTime - startTime}ms")
            verify(mockContextProvider).createContext(contextId, largeContextData)
            verify(mockContextStorage).store(expectedContext)
        }

        @Test
        @DisplayName("Should handle batch operations efficiently")
        fun `should handle batch operations efficiently`() {
            // Given
            val contextIds = (1..100).map { "batch-context-$it" }
            val contextData = mapOf("batch" to "operation")
            
            contextIds.forEach { contextId ->
                whenever(mockContextProvider.createContext(contextId, contextData))
                    .thenReturn(Context(contextId, contextData))
            }
            
            // When
            val startTime = System.currentTimeMillis()
            val results = contextManager.createContextsBatch(contextIds, contextData)
            val endTime = System.currentTimeMillis()
            
            // Then
            assertEquals(100, results.size)
            assertTrue(endTime - startTime < 5000, "Batch operation took too long: ${endTime - startTime}ms")
            verify(mockContextProvider, times(100)).createContext(any(), eq(contextData))
            verify(mockContextStorage, times(100)).store(any())
        }
    }
}