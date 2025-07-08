package dev.aurakai.auraframefx.ai.memory

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.io.TempDir
import org.mockito.Mockito.*
import org.mockito.kotlin.whenever
import java.io.File
import java.nio.file.Path
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Comprehensive unit tests for MemoryManager
 * Testing Framework: JUnit 5 with Mockito for mocking
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MemoryManagerTest {

    private lateinit var memoryManager: MemoryManager
    private lateinit var mockStorage: MemoryStorage
    private lateinit var mockRetriever: MemoryRetriever
    
    @TempDir
    lateinit var tempDir: Path

    @BeforeEach
    fun setUp() {
        mockStorage = mock(MemoryStorage::class.java)
        mockRetriever = mock(MemoryRetriever::class.java)
        memoryManager = MemoryManager(mockStorage, mockRetriever)
    }

    @AfterEach
    fun tearDown() {
        // Clean up any resources
        memoryManager.clear()
    }

    @Nested
    @DisplayName("Memory Storage Tests")
    inner class MemoryStorageTests {

        @Test
        @DisplayName("Should store memory successfully")
        fun shouldStoreMemorySuccessfully() {
            // Arrange
            val testMemory = Memory("test-id", "test content", System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should handle storage failure gracefully")
        fun shouldHandleStorageFailureGracefully() {
            // Arrange
            val testMemory = Memory("test-id", "test content", System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(false)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertFalse(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should throw exception for null memory")
        fun shouldThrowExceptionForNullMemory() {
            // Act & Assert
            assertThrows<IllegalArgumentException> {
                memoryManager.store(null)
            }
        }

        @Test
        @DisplayName("Should handle memory with empty content")
        fun shouldHandleMemoryWithEmptyContent() {
            // Arrange
            val testMemory = Memory("test-id", "", System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should handle memory with very long content")
        fun shouldHandleMemoryWithVeryLongContent() {
            // Arrange
            val longContent = "x".repeat(10000)
            val testMemory = Memory("test-id", longContent, System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should handle concurrent storage operations")
        fun shouldHandleConcurrentStorageOperations() {
            // Arrange
            val memories = (1..10).map { 
                Memory("test-id-$it", "content-$it", System.currentTimeMillis()) 
            }
            memories.forEach { whenever(mockStorage.store(it)).thenReturn(true) }

            // Act
            val futures = memories.map { memory ->
                CompletableFuture.supplyAsync { memoryManager.store(memory) }
            }
            val results = futures.map { it.get(5, TimeUnit.SECONDS) }

            // Assert
            assertTrue(results.all { it })
            memories.forEach { verify(mockStorage).store(it) }
        }
    }

    @Nested
    @DisplayName("Memory Retrieval Tests")
    inner class MemoryRetrievalTests {

        @Test
        @DisplayName("Should retrieve memory by ID successfully")
        fun shouldRetrieveMemoryByIdSuccessfully() {
            // Arrange
            val testMemory = Memory("test-id", "test content", System.currentTimeMillis())
            whenever(mockRetriever.retrieve("test-id")).thenReturn(testMemory)

            // Act
            val result = memoryManager.retrieve("test-id")

            // Assert
            assertNotNull(result)
            assertEquals(testMemory, result)
            verify(mockRetriever).retrieve("test-id")
        }

        @Test
        @DisplayName("Should return null for non-existent memory")
        fun shouldReturnNullForNonExistentMemory() {
            // Arrange
            whenever(mockRetriever.retrieve("non-existent")).thenReturn(null)

            // Act
            val result = memoryManager.retrieve("non-existent")

            // Assert
            assertNull(result)
            verify(mockRetriever).retrieve("non-existent")
        }

        @Test
        @DisplayName("Should handle empty string ID")
        fun shouldHandleEmptyStringId() {
            // Arrange
            whenever(mockRetriever.retrieve("")).thenReturn(null)

            // Act
            val result = memoryManager.retrieve("")

            // Assert
            assertNull(result)
            verify(mockRetriever).retrieve("")
        }

        @Test
        @DisplayName("Should retrieve multiple memories")
        fun shouldRetrieveMultipleMemories() {
            // Arrange
            val memories = listOf(
                Memory("id1", "content1", System.currentTimeMillis()),
                Memory("id2", "content2", System.currentTimeMillis())
            )
            whenever(mockRetriever.retrieveMultiple(listOf("id1", "id2"))).thenReturn(memories)

            // Act
            val result = memoryManager.retrieveMultiple(listOf("id1", "id2"))

            // Assert
            assertEquals(2, result.size)
            assertEquals(memories, result)
            verify(mockRetriever).retrieveMultiple(listOf("id1", "id2"))
        }

        @Test
        @DisplayName("Should handle partial retrieval results")
        fun shouldHandlePartialRetrievalResults() {
            // Arrange
            val memories = listOf(
                Memory("id1", "content1", System.currentTimeMillis())
            )
            whenever(mockRetriever.retrieveMultiple(listOf("id1", "non-existent"))).thenReturn(memories)

            // Act
            val result = memoryManager.retrieveMultiple(listOf("id1", "non-existent"))

            // Assert
            assertEquals(1, result.size)
            assertEquals(memories, result)
            verify(mockRetriever).retrieveMultiple(listOf("id1", "non-existent"))
        }
    }

    @Nested
    @DisplayName("Memory Search Tests")
    inner class MemorySearchTests {

        @Test
        @DisplayName("Should search memories by content")
        fun shouldSearchMemoriesByContent() {
            // Arrange
            val searchResults = listOf(
                Memory("id1", "matching content", System.currentTimeMillis()),
                Memory("id2", "another matching content", System.currentTimeMillis())
            )
            whenever(mockRetriever.search("matching")).thenReturn(searchResults)

            // Act
            val result = memoryManager.search("matching")

            // Assert
            assertEquals(2, result.size)
            assertEquals(searchResults, result)
            verify(mockRetriever).search("matching")
        }

        @Test
        @DisplayName("Should return empty results for no matches")
        fun shouldReturnEmptyResultsForNoMatches() {
            // Arrange
            whenever(mockRetriever.search("nonexistent")).thenReturn(emptyList())

            // Act
            val result = memoryManager.search("nonexistent")

            // Assert
            assertTrue(result.isEmpty())
            verify(mockRetriever).search("nonexistent")
        }

        @Test
        @DisplayName("Should handle empty search query")
        fun shouldHandleEmptySearchQuery() {
            // Arrange
            whenever(mockRetriever.search("")).thenReturn(emptyList())

            // Act
            val result = memoryManager.search("")

            // Assert
            assertTrue(result.isEmpty())
            verify(mockRetriever).search("")
        }

        @Test
        @DisplayName("Should handle search with special characters")
        fun shouldHandleSearchWithSpecialCharacters() {
            // Arrange
            val searchQuery = "!@#$%^&*()"
            whenever(mockRetriever.search(searchQuery)).thenReturn(emptyList())

            // Act
            val result = memoryManager.search(searchQuery)

            // Assert
            assertTrue(result.isEmpty())
            verify(mockRetriever).search(searchQuery)
        }
    }

    @Nested
    @DisplayName("Memory Management Tests")
    inner class MemoryManagementTests {

        @Test
        @DisplayName("Should delete memory successfully")
        fun shouldDeleteMemorySuccessfully() {
            // Arrange
            whenever(mockStorage.delete("test-id")).thenReturn(true)

            // Act
            val result = memoryManager.delete("test-id")

            // Assert
            assertTrue(result)
            verify(mockStorage).delete("test-id")
        }

        @Test
        @DisplayName("Should handle deletion failure")
        fun shouldHandleDeletionFailure() {
            // Arrange
            whenever(mockStorage.delete("test-id")).thenReturn(false)

            // Act
            val result = memoryManager.delete("test-id")

            // Assert
            assertFalse(result)
            verify(mockStorage).delete("test-id")
        }

        @Test
        @DisplayName("Should clear all memories")
        fun shouldClearAllMemories() {
            // Arrange
            whenever(mockStorage.clear()).thenReturn(true)

            // Act
            val result = memoryManager.clear()

            // Assert
            assertTrue(result)
            verify(mockStorage).clear()
        }

        @Test
        @DisplayName("Should get memory count")
        fun shouldGetMemoryCount() {
            // Arrange
            whenever(mockStorage.count()).thenReturn(5)

            // Act
            val result = memoryManager.count()

            // Assert
            assertEquals(5, result)
            verify(mockStorage).count()
        }

        @Test
        @DisplayName("Should handle zero memory count")
        fun shouldHandleZeroMemoryCount() {
            // Arrange
            whenever(mockStorage.count()).thenReturn(0)

            // Act
            val result = memoryManager.count()

            // Assert
            assertEquals(0, result)
            verify(mockStorage).count()
        }

        @Test
        @DisplayName("Should update existing memory")
        fun shouldUpdateExistingMemory() {
            // Arrange
            val updatedMemory = Memory("test-id", "updated content", System.currentTimeMillis())
            whenever(mockStorage.update(updatedMemory)).thenReturn(true)

            // Act
            val result = memoryManager.update(updatedMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).update(updatedMemory)
        }

        @Test
        @DisplayName("Should handle update failure")
        fun shouldHandleUpdateFailure() {
            // Arrange
            val updatedMemory = Memory("test-id", "updated content", System.currentTimeMillis())
            whenever(mockStorage.update(updatedMemory)).thenReturn(false)

            // Act
            val result = memoryManager.update(updatedMemory)

            // Assert
            assertFalse(result)
            verify(mockStorage).update(updatedMemory)
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle storage exception gracefully")
        fun shouldHandleStorageExceptionGracefully() {
            // Arrange
            val testMemory = Memory("test-id", "test content", System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenThrow(RuntimeException("Storage error"))

            // Act & Assert
            assertThrows<RuntimeException> {
                memoryManager.store(testMemory)
            }
        }

        @Test
        @DisplayName("Should handle retrieval exception gracefully")
        fun shouldHandleRetrievalExceptionGracefully() {
            // Arrange
            whenever(mockRetriever.retrieve("test-id")).thenThrow(RuntimeException("Retrieval error"))

            // Act & Assert
            assertThrows<RuntimeException> {
                memoryManager.retrieve("test-id")
            }
        }

        @Test
        @DisplayName("Should handle null pointer exceptions")
        fun shouldHandleNullPointerExceptions() {
            // Arrange
            whenever(mockStorage.store(any())).thenThrow(NullPointerException("Null pointer"))

            // Act & Assert
            assertThrows<NullPointerException> {
                memoryManager.store(Memory("test", "content", System.currentTimeMillis()))
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle large batch operations")
        fun shouldHandleLargeBatchOperations() {
            // Arrange
            val batchSize = 1000
            val memories = (1..batchSize).map { 
                Memory("id-$it", "content-$it", System.currentTimeMillis()) 
            }
            memories.forEach { whenever(mockStorage.store(it)).thenReturn(true) }

            // Act
            val startTime = System.currentTimeMillis()
            val results = memories.map { memoryManager.store(it) }
            val endTime = System.currentTimeMillis()

            // Assert
            assertTrue(results.all { it })
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
            assertEquals(batchSize, results.size)
        }

        @Test
        @DisplayName("Should handle memory cleanup efficiently")
        fun shouldHandleMemoryCleanupEfficiently() {
            // Arrange
            whenever(mockStorage.clear()).thenReturn(true)

            // Act
            val startTime = System.currentTimeMillis()
            val result = memoryManager.clear()
            val endTime = System.currentTimeMillis()

            // Assert
            assertTrue(result)
            assertTrue(endTime - startTime < 1000) // Should complete within 1 second
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle complete memory lifecycle")
        fun shouldHandleCompleteMemoryLifecycle() {
            // Arrange
            val testMemory = Memory("lifecycle-test", "test content", System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(true)
            whenever(mockRetriever.retrieve("lifecycle-test")).thenReturn(testMemory)
            whenever(mockStorage.delete("lifecycle-test")).thenReturn(true)

            // Act & Assert
            // Store
            assertTrue(memoryManager.store(testMemory))
            verify(mockStorage).store(testMemory)

            // Retrieve
            val retrieved = memoryManager.retrieve("lifecycle-test")
            assertEquals(testMemory, retrieved)
            verify(mockRetriever).retrieve("lifecycle-test")

            // Delete
            assertTrue(memoryManager.delete("lifecycle-test"))
            verify(mockStorage).delete("lifecycle-test")
        }

        @Test
        @DisplayName("Should handle store and search workflow")
        fun shouldHandleStoreAndSearchWorkflow() {
            // Arrange
            val testMemory = Memory("search-test", "searchable content", System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(true)
            whenever(mockRetriever.search("searchable")).thenReturn(listOf(testMemory))

            // Act
            val storeResult = memoryManager.store(testMemory)
            val searchResults = memoryManager.search("searchable")

            // Assert
            assertTrue(storeResult)
            assertEquals(1, searchResults.size)
            assertEquals(testMemory, searchResults[0])
            verify(mockStorage).store(testMemory)
            verify(mockRetriever).search("searchable")
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    inner class EdgeCaseTests {

        @Test
        @DisplayName("Should handle memory with future timestamp")
        fun shouldHandleMemoryWithFutureTimestamp() {
            // Arrange
            val futureTimestamp = System.currentTimeMillis() + 86400000 // 24 hours in future
            val testMemory = Memory("future-test", "future content", futureTimestamp)
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should handle memory with zero timestamp")
        fun shouldHandleMemoryWithZeroTimestamp() {
            // Arrange
            val testMemory = Memory("zero-timestamp", "content", 0)
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should handle memory with negative timestamp")
        fun shouldHandleMemoryWithNegativeTimestamp() {
            // Arrange
            val testMemory = Memory("negative-timestamp", "content", -1)
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }

        @Test
        @DisplayName("Should handle memory with Unicode content")
        fun shouldHandleMemoryWithUnicodeContent() {
            // Arrange
            val unicodeContent = "Hello 世界 🌍 💫 ñáéíóú"
            val testMemory = Memory("unicode-test", unicodeContent, System.currentTimeMillis())
            whenever(mockStorage.store(testMemory)).thenReturn(true)

            // Act
            val result = memoryManager.store(testMemory)

            // Assert
            assertTrue(result)
            verify(mockStorage).store(testMemory)
        }
    }
}