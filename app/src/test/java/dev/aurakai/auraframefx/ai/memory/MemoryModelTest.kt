package dev.aurakai.auraframefx.ai.memory

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.verifyNoInteractions
import org.mockito.kotlin.any
import org.mockito.kotlin.times
import java.time.LocalDateTime
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

/**
 * Comprehensive unit tests for MemoryModel class
 * Testing framework: JUnit 5 with Mockito for mocking
 */
@DisplayName("MemoryModel Tests")
class MemoryModelTest {

    private lateinit var memoryModel: MemoryModel
    private lateinit var autoCloseable: AutoCloseable

    @Mock
    private lateinit var mockMemoryStorage: MemoryStorage

    @Mock
    private lateinit var mockMemoryRetriever: MemoryRetriever

    @Mock
    private lateinit var mockMemoryProcessor: MemoryProcessor

    @BeforeEach
    fun setup() {
        autoCloseable = MockitoAnnotations.openMocks(this)
        memoryModel = MemoryModel(mockMemoryStorage, mockMemoryRetriever, mockMemoryProcessor)
    }

    @AfterEach
    fun tearDown() {
        autoCloseable.close()
    }

    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {

        @Test
        @DisplayName("Should initialize with valid dependencies")
        fun shouldInitializeWithValidDependencies() {
            // Given
            val storage = mockMemoryStorage
            val retriever = mockMemoryRetriever
            val processor = mockMemoryProcessor

            // When
            val model = MemoryModel(storage, retriever, processor)

            // Then
            assertNotNull(model)
        }

        @Test
        @DisplayName("Should throw exception with null storage")
        fun shouldThrowExceptionWithNullStorage() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                MemoryModel(null, mockMemoryRetriever, mockMemoryProcessor)
            }
        }

        @Test
        @DisplayName("Should throw exception with null retriever")
        fun shouldThrowExceptionWithNullRetriever() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                MemoryModel(mockMemoryStorage, null, mockMemoryProcessor)
            }
        }

        @Test
        @DisplayName("Should throw exception with null processor")
        fun shouldThrowExceptionWithNullProcessor() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                MemoryModel(mockMemoryStorage, mockMemoryRetriever, null)
            }
        }
    }

    @Nested
    @DisplayName("Store Memory Tests")
    inner class StoreMemoryTests {

        @Test
        @DisplayName("Should store memory successfully")
        fun shouldStoreMemorySuccessfully() {
            // Given
            val memoryData = MemoryData("test-id", "test content", LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }

        @Test
        @DisplayName("Should handle storage failure")
        fun shouldHandleStorageFailure() {
            // Given
            val memoryData = MemoryData("test-id", "test content", LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(false))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertFalse(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }

        @Test
        @DisplayName("Should handle storage exception")
        fun shouldHandleStorageException() {
            // Given
            val memoryData = MemoryData("test-id", "test content", LocalDateTime.now())
            val exception = RuntimeException("Storage error")
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.storeMemory(memoryData).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should throw exception for null memory data")
        fun shouldThrowExceptionForNullMemoryData() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.storeMemory(null)
            }
        }

        @Test
        @DisplayName("Should process memory before storing")
        fun shouldProcessMemoryBeforeStoring() {
            // Given
            val originalMemory = MemoryData("test-id", "test content", LocalDateTime.now())
            val processedMemory = MemoryData("test-id", "processed content", LocalDateTime.now())
            whenever(mockMemoryProcessor.process(any())).thenReturn(processedMemory)
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(originalMemory)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryProcessor).process(originalMemory)
            verify(mockMemoryStorage).store(processedMemory)
        }
    }

    @Nested
    @DisplayName("Retrieve Memory Tests")
    inner class RetrieveMemoryTests {

        @Test
        @DisplayName("Should retrieve memory by ID successfully")
        fun shouldRetrieveMemoryByIdSuccessfully() {
            // Given
            val memoryId = "test-id"
            val expectedMemory = MemoryData(memoryId, "test content", LocalDateTime.now())
            whenever(mockMemoryRetriever.retrieveById(memoryId)).thenReturn(CompletableFuture.completedFuture(expectedMemory))

            // When
            val result = memoryModel.retrieveMemory(memoryId)

            // Then
            val actualMemory = result.get(1, TimeUnit.SECONDS)
            assertEquals(expectedMemory, actualMemory)
            verify(mockMemoryRetriever).retrieveById(memoryId)
        }

        @Test
        @DisplayName("Should handle memory not found")
        fun shouldHandleMemoryNotFound() {
            // Given
            val memoryId = "non-existent-id"
            whenever(mockMemoryRetriever.retrieveById(memoryId)).thenReturn(CompletableFuture.completedFuture(null))

            // When
            val result = memoryModel.retrieveMemory(memoryId)

            // Then
            val actualMemory = result.get(1, TimeUnit.SECONDS)
            assertNull(actualMemory)
            verify(mockMemoryRetriever).retrieveById(memoryId)
        }

        @Test
        @DisplayName("Should handle retrieval exception")
        fun shouldHandleRetrievalException() {
            // Given
            val memoryId = "test-id"
            val exception = RuntimeException("Retrieval error")
            whenever(mockMemoryRetriever.retrieveById(memoryId)).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.retrieveMemory(memoryId).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should throw exception for null memory ID")
        fun shouldThrowExceptionForNullMemoryId() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.retrieveMemory(null)
            }
        }

        @Test
        @DisplayName("Should throw exception for empty memory ID")
        fun shouldThrowExceptionForEmptyMemoryId() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.retrieveMemory("")
            }
        }

        @Test
        @DisplayName("Should throw exception for blank memory ID")
        fun shouldThrowExceptionForBlankMemoryId() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.retrieveMemory("   ")
            }
        }
    }

    @Nested
    @DisplayName("Search Memory Tests")
    inner class SearchMemoryTests {

        @Test
        @DisplayName("Should search memories successfully")
        fun shouldSearchMemoriesSuccessfully() {
            // Given
            val query = "test query"
            val expectedMemories = listOf(
                MemoryData("id1", "content1", LocalDateTime.now()),
                MemoryData("id2", "content2", LocalDateTime.now())
            )
            whenever(mockMemoryRetriever.search(query)).thenReturn(CompletableFuture.completedFuture(expectedMemories))

            // When
            val result = memoryModel.searchMemories(query)

            // Then
            val actualMemories = result.get(1, TimeUnit.SECONDS)
            assertEquals(expectedMemories, actualMemories)
            verify(mockMemoryRetriever).search(query)
        }

        @Test
        @DisplayName("Should return empty list for no matches")
        fun shouldReturnEmptyListForNoMatches() {
            // Given
            val query = "non-matching query"
            whenever(mockMemoryRetriever.search(query)).thenReturn(CompletableFuture.completedFuture(emptyList()))

            // When
            val result = memoryModel.searchMemories(query)

            // Then
            val actualMemories = result.get(1, TimeUnit.SECONDS)
            assertTrue(actualMemories.isEmpty())
            verify(mockMemoryRetriever).search(query)
        }

        @Test
        @DisplayName("Should handle search exception")
        fun shouldHandleSearchException() {
            // Given
            val query = "test query"
            val exception = RuntimeException("Search error")
            whenever(mockMemoryRetriever.search(query)).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.searchMemories(query).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should throw exception for null search query")
        fun shouldThrowExceptionForNullSearchQuery() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.searchMemories(null)
            }
        }

        @Test
        @DisplayName("Should handle empty search query")
        fun shouldHandleEmptySearchQuery() {
            // Given
            val query = ""
            whenever(mockMemoryRetriever.search(query)).thenReturn(CompletableFuture.completedFuture(emptyList()))

            // When
            val result = memoryModel.searchMemories(query)

            // Then
            val actualMemories = result.get(1, TimeUnit.SECONDS)
            assertTrue(actualMemories.isEmpty())
            verify(mockMemoryRetriever).search(query)
        }
    }

    @Nested
    @DisplayName("Update Memory Tests")
    inner class UpdateMemoryTests {

        @Test
        @DisplayName("Should update memory successfully")
        fun shouldUpdateMemorySuccessfully() {
            // Given
            val memoryData = MemoryData("test-id", "updated content", LocalDateTime.now())
            whenever(mockMemoryStorage.update(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.updateMemory(memoryData)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).update(memoryData)
        }

        @Test
        @DisplayName("Should handle update failure")
        fun shouldHandleUpdateFailure() {
            // Given
            val memoryData = MemoryData("test-id", "updated content", LocalDateTime.now())
            whenever(mockMemoryStorage.update(any())).thenReturn(CompletableFuture.completedFuture(false))

            // When
            val result = memoryModel.updateMemory(memoryData)

            // Then
            assertFalse(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).update(memoryData)
        }

        @Test
        @DisplayName("Should handle update exception")
        fun shouldHandleUpdateException() {
            // Given
            val memoryData = MemoryData("test-id", "updated content", LocalDateTime.now())
            val exception = RuntimeException("Update error")
            whenever(mockMemoryStorage.update(any())).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.updateMemory(memoryData).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should throw exception for null memory data in update")
        fun shouldThrowExceptionForNullMemoryDataInUpdate() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.updateMemory(null)
            }
        }

        @Test
        @DisplayName("Should process memory before updating")
        fun shouldProcessMemoryBeforeUpdating() {
            // Given
            val originalMemory = MemoryData("test-id", "original content", LocalDateTime.now())
            val processedMemory = MemoryData("test-id", "processed content", LocalDateTime.now())
            whenever(mockMemoryProcessor.process(any())).thenReturn(processedMemory)
            whenever(mockMemoryStorage.update(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.updateMemory(originalMemory)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryProcessor).process(originalMemory)
            verify(mockMemoryStorage).update(processedMemory)
        }
    }

    @Nested
    @DisplayName("Delete Memory Tests")
    inner class DeleteMemoryTests {

        @Test
        @DisplayName("Should delete memory successfully")
        fun shouldDeleteMemorySuccessfully() {
            // Given
            val memoryId = "test-id"
            whenever(mockMemoryStorage.delete(memoryId)).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.deleteMemory(memoryId)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).delete(memoryId)
        }

        @Test
        @DisplayName("Should handle delete failure")
        fun shouldHandleDeleteFailure() {
            // Given
            val memoryId = "test-id"
            whenever(mockMemoryStorage.delete(memoryId)).thenReturn(CompletableFuture.completedFuture(false))

            // When
            val result = memoryModel.deleteMemory(memoryId)

            // Then
            assertFalse(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).delete(memoryId)
        }

        @Test
        @DisplayName("Should handle delete exception")
        fun shouldHandleDeleteException() {
            // Given
            val memoryId = "test-id"
            val exception = RuntimeException("Delete error")
            whenever(mockMemoryStorage.delete(memoryId)).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.deleteMemory(memoryId).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should throw exception for null memory ID in delete")
        fun shouldThrowExceptionForNullMemoryIdInDelete() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.deleteMemory(null)
            }
        }

        @Test
        @DisplayName("Should throw exception for empty memory ID in delete")
        fun shouldThrowExceptionForEmptyMemoryIdInDelete() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.deleteMemory("")
            }
        }
    }

    @Nested
    @DisplayName("Batch Operations Tests")
    inner class BatchOperationsTests {

        @Test
        @DisplayName("Should store multiple memories successfully")
        fun shouldStoreMultipleMemoriesSuccessfully() {
            // Given
            val memories = listOf(
                MemoryData("id1", "content1", LocalDateTime.now()),
                MemoryData("id2", "content2", LocalDateTime.now())
            )
            whenever(mockMemoryStorage.storeBatch(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemories(memories)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).storeBatch(memories)
        }

        @Test
        @DisplayName("Should handle batch store failure")
        fun shouldHandleBatchStoreFailure() {
            // Given
            val memories = listOf(
                MemoryData("id1", "content1", LocalDateTime.now()),
                MemoryData("id2", "content2", LocalDateTime.now())
            )
            whenever(mockMemoryStorage.storeBatch(any())).thenReturn(CompletableFuture.completedFuture(false))

            // When
            val result = memoryModel.storeMemories(memories)

            // Then
            assertFalse(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).storeBatch(memories)
        }

        @Test
        @DisplayName("Should throw exception for null memories list")
        fun shouldThrowExceptionForNullMemoriesList() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.storeMemories(null)
            }
        }

        @Test
        @DisplayName("Should handle empty memories list")
        fun shouldHandleEmptyMemoriesList() {
            // Given
            val memories = emptyList<MemoryData>()
            whenever(mockMemoryStorage.storeBatch(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemories(memories)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).storeBatch(memories)
        }
    }

    @Nested
    @DisplayName("Memory Statistics Tests")
    inner class MemoryStatisticsTests {

        @Test
        @DisplayName("Should get memory count successfully")
        fun shouldGetMemoryCountSuccessfully() {
            // Given
            val expectedCount = 42L
            whenever(mockMemoryStorage.getMemoryCount()).thenReturn(CompletableFuture.completedFuture(expectedCount))

            // When
            val result = memoryModel.getMemoryCount()

            // Then
            val actualCount = result.get(1, TimeUnit.SECONDS)
            assertEquals(expectedCount, actualCount)
            verify(mockMemoryStorage).getMemoryCount()
        }

        @Test
        @DisplayName("Should handle memory count exception")
        fun shouldHandleMemoryCountException() {
            // Given
            val exception = RuntimeException("Count error")
            whenever(mockMemoryStorage.getMemoryCount()).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.getMemoryCount().get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should get memory size successfully")
        fun shouldGetMemorySizeSuccessfully() {
            // Given
            val expectedSize = 1024L
            whenever(mockMemoryStorage.getTotalMemorySize()).thenReturn(CompletableFuture.completedFuture(expectedSize))

            // When
            val result = memoryModel.getTotalMemorySize()

            // Then
            val actualSize = result.get(1, TimeUnit.SECONDS)
            assertEquals(expectedSize, actualSize)
            verify(mockMemoryStorage).getTotalMemorySize()
        }
    }

    @Nested
    @DisplayName("Memory Cleanup Tests")
    inner class MemoryCleanupTests {

        @Test
        @DisplayName("Should cleanup old memories successfully")
        fun shouldCleanupOldMemoriesSuccessfully() {
            // Given
            val cutoffDate = LocalDateTime.now().minusDays(30)
            whenever(mockMemoryStorage.cleanupOldMemories(cutoffDate)).thenReturn(CompletableFuture.completedFuture(5))

            // When
            val result = memoryModel.cleanupOldMemories(cutoffDate)

            // Then
            val deletedCount = result.get(1, TimeUnit.SECONDS)
            assertEquals(5, deletedCount)
            verify(mockMemoryStorage).cleanupOldMemories(cutoffDate)
        }

        @Test
        @DisplayName("Should handle cleanup exception")
        fun shouldHandleCleanupException() {
            // Given
            val cutoffDate = LocalDateTime.now().minusDays(30)
            val exception = RuntimeException("Cleanup error")
            whenever(mockMemoryStorage.cleanupOldMemories(cutoffDate)).thenReturn(CompletableFuture.failedFuture(exception))

            // When & Then
            assertThrows<RuntimeException> {
                memoryModel.cleanupOldMemories(cutoffDate).get(1, TimeUnit.SECONDS)
            }
        }

        @Test
        @DisplayName("Should throw exception for null cutoff date")
        fun shouldThrowExceptionForNullCutoffDate() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.cleanupOldMemories(null)
            }
        }
    }

    @Nested
    @DisplayName("Concurrent Operations Tests")
    inner class ConcurrentOperationsTests {

        @Test
        @DisplayName("Should handle concurrent store operations")
        fun shouldHandleConcurrentStoreOperations() {
            // Given
            val memory1 = MemoryData("id1", "content1", LocalDateTime.now())
            val memory2 = MemoryData("id2", "content2", LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val future1 = memoryModel.storeMemory(memory1)
            val future2 = memoryModel.storeMemory(memory2)

            // Then
            assertTrue(future1.get(1, TimeUnit.SECONDS))
            assertTrue(future2.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage, times(2)).store(any())
        }

        @Test
        @DisplayName("Should handle concurrent retrieve operations")
        fun shouldHandleConcurrentRetrieveOperations() {
            // Given
            val memory1 = MemoryData("id1", "content1", LocalDateTime.now())
            val memory2 = MemoryData("id2", "content2", LocalDateTime.now())
            whenever(mockMemoryRetriever.retrieveById("id1")).thenReturn(CompletableFuture.completedFuture(memory1))
            whenever(mockMemoryRetriever.retrieveById("id2")).thenReturn(CompletableFuture.completedFuture(memory2))

            // When
            val future1 = memoryModel.retrieveMemory("id1")
            val future2 = memoryModel.retrieveMemory("id2")

            // Then
            assertEquals(memory1, future1.get(1, TimeUnit.SECONDS))
            assertEquals(memory2, future2.get(1, TimeUnit.SECONDS))
            verify(mockMemoryRetriever).retrieveById("id1")
            verify(mockMemoryRetriever).retrieveById("id2")
        }
    }

    @Nested
    @DisplayName("Edge Cases Tests")
    inner class EdgeCasesTests {

        @Test
        @DisplayName("Should handle very large memory content")
        fun shouldHandleVeryLargeMemoryContent() {
            // Given
            val largeContent = "x".repeat(1_000_000)
            val memoryData = MemoryData("large-id", largeContent, LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertTrue(result.get(5, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }

        @Test
        @DisplayName("Should handle special characters in memory content")
        fun shouldHandleSpecialCharactersInMemoryContent() {
            // Given
            val specialContent = "Special chars: !@#$%^&*()_+{}|:<>?[]\\;'\",./"
            val memoryData = MemoryData("special-id", specialContent, LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }

        @Test
        @DisplayName("Should handle Unicode characters in memory content")
        fun shouldHandleUnicodeCharactersInMemoryContent() {
            // Given
            val unicodeContent = "Unicode: 你好世界 🌍 émojis 💡"
            val memoryData = MemoryData("unicode-id", unicodeContent, LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }

        @Test
        @DisplayName("Should handle very long memory IDs")
        fun shouldHandleVeryLongMemoryIds() {
            // Given
            val longId = "id-" + "x".repeat(1000)
            val memoryData = MemoryData(longId, "content", LocalDateTime.now())
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }

        @Test
        @DisplayName("Should handle future timestamps")
        fun shouldHandleFutureTimestamps() {
            // Given
            val futureTime = LocalDateTime.now().plusYears(10)
            val memoryData = MemoryData("future-id", "future content", futureTime)
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))

            // When
            val result = memoryModel.storeMemory(memoryData)

            // Then
            assertTrue(result.get(1, TimeUnit.SECONDS))
            verify(mockMemoryStorage).store(memoryData)
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    inner class IntegrationTests {

        @Test
        @DisplayName("Should handle complete memory lifecycle")
        fun shouldHandleCompleteMemoryLifecycle() {
            // Given
            val originalMemory = MemoryData("lifecycle-id", "original content", LocalDateTime.now())
            val updatedMemory = MemoryData("lifecycle-id", "updated content", LocalDateTime.now())
            
            whenever(mockMemoryStorage.store(any())).thenReturn(CompletableFuture.completedFuture(true))
            whenever(mockMemoryRetriever.retrieveById("lifecycle-id")).thenReturn(CompletableFuture.completedFuture(originalMemory))
            whenever(mockMemoryStorage.update(any())).thenReturn(CompletableFuture.completedFuture(true))
            whenever(mockMemoryStorage.delete("lifecycle-id")).thenReturn(CompletableFuture.completedFuture(true))

            // When & Then
            // Store
            assertTrue(memoryModel.storeMemory(originalMemory).get(1, TimeUnit.SECONDS))
            
            // Retrieve
            assertEquals(originalMemory, memoryModel.retrieveMemory("lifecycle-id").get(1, TimeUnit.SECONDS))
            
            // Update
            assertTrue(memoryModel.updateMemory(updatedMemory).get(1, TimeUnit.SECONDS))
            
            // Delete
            assertTrue(memoryModel.deleteMemory("lifecycle-id").get(1, TimeUnit.SECONDS))
            
            // Verify all operations
            verify(mockMemoryStorage).store(originalMemory)
            verify(mockMemoryRetriever).retrieveById("lifecycle-id")
            verify(mockMemoryStorage).update(updatedMemory)
            verify(mockMemoryStorage).delete("lifecycle-id")
        }
    }
}