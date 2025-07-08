package dev.aurakai.auraframefx.ai.memory

import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.mockito.Mockito.*
import org.mockito.kotlin.*
import java.time.LocalDateTime
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import io.mockk.*
import io.mockk.impl.annotations.MockK
import io.mockk.junit5.MockKExtension
import org.junit.jupiter.api.extension.ExtendWith

@ExtendWith(MockKExtension::class)
class MemoryManagerTest {

    @MockK
    private lateinit var memoryStorage: MemoryStorage
    
    @MockK
    private lateinit var memoryProcessor: MemoryProcessor
    
    private lateinit var memoryManager: MemoryManager
    
    @BeforeEach
    fun setUp() {
        MockKAnnotations.init(this)
        memoryManager = MemoryManager(memoryStorage, memoryProcessor)
    }
    
    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Nested
    @DisplayName("Memory Storage Operations")
    inner class MemoryStorageOperations {
        
        @Test
        @DisplayName("Should store memory successfully")
        fun shouldStoreMemorySuccessfully() = runTest {
            // Given
            val memory = Memory(
                id = "test-id",
                content = "Test memory content",
                timestamp = LocalDateTime.now(),
                importance = MemoryImportance.MEDIUM
            )
            every { memoryStorage.store(memory) } returns Unit
            
            // When
            memoryManager.storeMemory(memory)
            
            // Then
            verify { memoryStorage.store(memory) }
        }
        
        @Test
        @DisplayName("Should handle storage failure gracefully")
        fun shouldHandleStorageFailureGracefully() = runTest {
            // Given
            val memory = Memory(
                id = "test-id",
                content = "Test memory content",
                timestamp = LocalDateTime.now(),
                importance = MemoryImportance.HIGH
            )
            every { memoryStorage.store(memory) } throws StorageException("Storage failed")
            
            // When & Then
            assertThrows<MemoryManagerException> {
                memoryManager.storeMemory(memory)
            }
        }
        
        @Test
        @DisplayName("Should retrieve memory by ID successfully")
        fun shouldRetrieveMemoryByIdSuccessfully() = runTest {
            // Given
            val expectedMemory = Memory(
                id = "test-id",
                content = "Retrieved memory",
                timestamp = LocalDateTime.now(),
                importance = MemoryImportance.LOW
            )
            every { memoryStorage.retrieveById("test-id") } returns expectedMemory
            
            // When
            val result = memoryManager.retrieveMemory("test-id")
            
            // Then
            assertEquals(expectedMemory, result)
            verify { memoryStorage.retrieveById("test-id") }
        }
        
        @Test
        @DisplayName("Should return null for non-existent memory ID")
        fun shouldReturnNullForNonExistentMemoryId() = runTest {
            // Given
            every { memoryStorage.retrieveById("non-existent") } returns null
            
            // When
            val result = memoryManager.retrieveMemory("non-existent")
            
            // Then
            assertNull(result)
            verify { memoryStorage.retrieveById("non-existent") }
        }
        
        @Test
        @DisplayName("Should delete memory successfully")
        fun shouldDeleteMemorySuccessfully() = runTest {
            // Given
            every { memoryStorage.delete("test-id") } returns true
            
            // When
            val result = memoryManager.deleteMemory("test-id")
            
            // Then
            assertTrue(result)
            verify { memoryStorage.delete("test-id") }
        }
        
        @Test
        @DisplayName("Should return false when deleting non-existent memory")
        fun shouldReturnFalseWhenDeletingNonExistentMemory() = runTest {
            // Given
            every { memoryStorage.delete("non-existent") } returns false
            
            // When
            val result = memoryManager.deleteMemory("non-existent")
            
            // Then
            assertFalse(result)
            verify { memoryStorage.delete("non-existent") }
        }
    }
    
    @Nested
    @DisplayName("Memory Search Operations")
    inner class MemorySearchOperations {
        
        @Test
        @DisplayName("Should search memories by content successfully")
        fun shouldSearchMemoriesByContentSuccessfully() = runTest {
            // Given
            val searchQuery = "test query"
            val expectedMemories = listOf(
                Memory("1", "Test content 1", LocalDateTime.now(), MemoryImportance.HIGH),
                Memory("2", "Test content 2", LocalDateTime.now(), MemoryImportance.MEDIUM)
            )
            every { memoryStorage.searchByContent(searchQuery) } returns expectedMemories
            
            // When
            val result = memoryManager.searchMemories(searchQuery)
            
            // Then
            assertEquals(expectedMemories, result)
            verify { memoryStorage.searchByContent(searchQuery) }
        }
        
        @Test
        @DisplayName("Should return empty list for search with no results")
        fun shouldReturnEmptyListForSearchWithNoResults() = runTest {
            // Given
            val searchQuery = "no results"
            every { memoryStorage.searchByContent(searchQuery) } returns emptyList()
            
            // When
            val result = memoryManager.searchMemories(searchQuery)
            
            // Then
            assertTrue(result.isEmpty())
            verify { memoryStorage.searchByContent(searchQuery) }
        }
        
        @Test
        @DisplayName("Should handle search with empty query")
        fun shouldHandleSearchWithEmptyQuery() = runTest {
            // Given
            val emptyQuery = ""
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryManager.searchMemories(emptyQuery)
            }
        }
        
        @Test
        @DisplayName("Should handle search with null query")
        fun shouldHandleSearchWithNullQuery() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryManager.searchMemories(null)
            }
        }
        
        @Test
        @DisplayName("Should search memories by importance level")
        fun shouldSearchMemoriesByImportanceLevel() = runTest {
            // Given
            val importance = MemoryImportance.HIGH
            val expectedMemories = listOf(
                Memory("1", "Important memory 1", LocalDateTime.now(), MemoryImportance.HIGH),
                Memory("2", "Important memory 2", LocalDateTime.now(), MemoryImportance.HIGH)
            )
            every { memoryStorage.searchByImportance(importance) } returns expectedMemories
            
            // When
            val result = memoryManager.searchMemoriesByImportance(importance)
            
            // Then
            assertEquals(expectedMemories, result)
            verify { memoryStorage.searchByImportance(importance) }
        }
        
        @Test
        @DisplayName("Should search memories by date range")
        fun shouldSearchMemoriesByDateRange() = runTest {
            // Given
            val startDate = LocalDateTime.now().minusDays(7)
            val endDate = LocalDateTime.now()
            val expectedMemories = listOf(
                Memory("1", "Recent memory 1", LocalDateTime.now().minusDays(3), MemoryImportance.MEDIUM),
                Memory("2", "Recent memory 2", LocalDateTime.now().minusDays(1), MemoryImportance.LOW)
            )
            every { memoryStorage.searchByDateRange(startDate, endDate) } returns expectedMemories
            
            // When
            val result = memoryManager.searchMemoriesByDateRange(startDate, endDate)
            
            // Then
            assertEquals(expectedMemories, result)
            verify { memoryStorage.searchByDateRange(startDate, endDate) }
        }
    }
    
    @Nested
    @DisplayName("Memory Processing Operations")
    inner class MemoryProcessingOperations {
        
        @Test
        @DisplayName("Should process memory for insights successfully")
        fun shouldProcessMemoryForInsightsSuccessfully() = runTest {
            // Given
            val memory = Memory("test-id", "Test content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            val expectedInsights = listOf("Insight 1", "Insight 2")
            every { memoryProcessor.generateInsights(memory) } returns expectedInsights
            
            // When
            val result = memoryManager.processMemoryForInsights(memory)
            
            // Then
            assertEquals(expectedInsights, result)
            verify { memoryProcessor.generateInsights(memory) }
        }
        
        @Test
        @DisplayName("Should handle processing failure gracefully")
        fun shouldHandleProcessingFailureGracefully() = runTest {
            // Given
            val memory = Memory("test-id", "Test content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            every { memoryProcessor.generateInsights(memory) } throws ProcessingException("Processing failed")
            
            // When & Then
            assertThrows<MemoryManagerException> {
                memoryManager.processMemoryForInsights(memory)
            }
        }
        
        @Test
        @DisplayName("Should compress old memories successfully")
        fun shouldCompressOldMemoriesSuccessfully() = runTest {
            // Given
            val cutoffDate = LocalDateTime.now().minusDays(30)
            val oldMemories = listOf(
                Memory("1", "Old memory 1", LocalDateTime.now().minusDays(40), MemoryImportance.LOW),
                Memory("2", "Old memory 2", LocalDateTime.now().minusDays(35), MemoryImportance.LOW)
            )
            every { memoryStorage.searchByDateRange(any(), cutoffDate) } returns oldMemories
            every { memoryProcessor.compressMemories(oldMemories) } returns "Compressed memories"
            every { memoryStorage.storeCompressed(any()) } returns Unit
            every { memoryStorage.delete(any()) } returns true
            
            // When
            memoryManager.compressOldMemories(cutoffDate)
            
            // Then
            verify { memoryStorage.searchByDateRange(any(), cutoffDate) }
            verify { memoryProcessor.compressMemories(oldMemories) }
            verify { memoryStorage.storeCompressed(any()) }
            verify(exactly = oldMemories.size) { memoryStorage.delete(any()) }
        }
        
        @Test
        @DisplayName("Should calculate memory similarity successfully")
        fun shouldCalculateMemorySimilaritySuccessfully() = runTest {
            // Given
            val memory1 = Memory("1", "Similar content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            val memory2 = Memory("2", "Similar content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            val expectedSimilarity = 0.85
            every { memoryProcessor.calculateSimilarity(memory1, memory2) } returns expectedSimilarity
            
            // When
            val result = memoryManager.calculateSimilarity(memory1, memory2)
            
            // Then
            assertEquals(expectedSimilarity, result, 0.01)
            verify { memoryProcessor.calculateSimilarity(memory1, memory2) }
        }
    }
    
    @Nested
    @DisplayName("Memory Retrieval Strategies")
    inner class MemoryRetrievalStrategies {
        
        @Test
        @DisplayName("Should retrieve recent memories successfully")
        fun shouldRetrieveRecentMemoriesSuccessfully() = runTest {
            // Given
            val limit = 10
            val expectedMemories = (1..limit).map { i ->
                Memory("$i", "Recent memory $i", LocalDateTime.now().minusDays(i.toLong()), MemoryImportance.MEDIUM)
            }
            every { memoryStorage.retrieveRecent(limit) } returns expectedMemories
            
            // When
            val result = memoryManager.retrieveRecentMemories(limit)
            
            // Then
            assertEquals(expectedMemories, result)
            verify { memoryStorage.retrieveRecent(limit) }
        }
        
        @Test
        @DisplayName("Should retrieve most important memories successfully")
        fun shouldRetrieveMostImportantMemoriesSuccessfully() = runTest {
            // Given
            val limit = 5
            val expectedMemories = (1..limit).map { i ->
                Memory("$i", "Important memory $i", LocalDateTime.now(), MemoryImportance.HIGH)
            }
            every { memoryStorage.retrieveByImportance(MemoryImportance.HIGH, limit) } returns expectedMemories
            
            // When
            val result = memoryManager.retrieveMostImportantMemories(limit)
            
            // Then
            assertEquals(expectedMemories, result)
            verify { memoryStorage.retrieveByImportance(MemoryImportance.HIGH, limit) }
        }
        
        @Test
        @DisplayName("Should retrieve related memories successfully")
        fun shouldRetrieveRelatedMemoriesSuccessfully() = runTest {
            // Given
            val baseMemory = Memory("base", "Base memory content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            val relatedMemories = listOf(
                Memory("related1", "Related memory 1", LocalDateTime.now(), MemoryImportance.MEDIUM),
                Memory("related2", "Related memory 2", LocalDateTime.now(), MemoryImportance.LOW)
            )
            every { memoryProcessor.findRelatedMemories(baseMemory, any()) } returns relatedMemories
            
            // When
            val result = memoryManager.retrieveRelatedMemories(baseMemory, 0.7)
            
            // Then
            assertEquals(relatedMemories, result)
            verify { memoryProcessor.findRelatedMemories(baseMemory, 0.7) }
        }
    }
    
    @Nested
    @DisplayName("Memory Maintenance Operations")
    inner class MemoryMaintenanceOperations {
        
        @Test
        @DisplayName("Should clean up expired memories successfully")
        fun shouldCleanupExpiredMemoriesSuccessfully() = runTest {
            // Given
            val expiredMemories = listOf(
                Memory("expired1", "Expired memory 1", LocalDateTime.now().minusMonths(6), MemoryImportance.LOW),
                Memory("expired2", "Expired memory 2", LocalDateTime.now().minusMonths(8), MemoryImportance.LOW)
            )
            every { memoryStorage.findExpiredMemories(any()) } returns expiredMemories
            every { memoryStorage.delete(any()) } returns true
            
            // When
            val result = memoryManager.cleanupExpiredMemories()
            
            // Then
            assertEquals(expiredMemories.size, result)
            verify { memoryStorage.findExpiredMemories(any()) }
            verify(exactly = expiredMemories.size) { memoryStorage.delete(any()) }
        }
        
        @Test
        @DisplayName("Should update memory importance successfully")
        fun shouldUpdateMemoryImportanceSuccessfully() = runTest {
            // Given
            val memoryId = "test-id"
            val newImportance = MemoryImportance.HIGH
            val existingMemory = Memory(memoryId, "Test content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            every { memoryStorage.retrieveById(memoryId) } returns existingMemory
            every { memoryStorage.updateImportance(memoryId, newImportance) } returns Unit
            
            // When
            memoryManager.updateMemoryImportance(memoryId, newImportance)
            
            // Then
            verify { memoryStorage.retrieveById(memoryId) }
            verify { memoryStorage.updateImportance(memoryId, newImportance) }
        }
        
        @Test
        @DisplayName("Should handle updating non-existent memory importance")
        fun shouldHandleUpdatingNonExistentMemoryImportance() = runTest {
            // Given
            val nonExistentId = "non-existent"
            val newImportance = MemoryImportance.HIGH
            every { memoryStorage.retrieveById(nonExistentId) } returns null
            
            // When & Then
            assertThrows<MemoryNotFoundException> {
                memoryManager.updateMemoryImportance(nonExistentId, newImportance)
            }
        }
        
        @Test
        @DisplayName("Should get memory statistics successfully")
        fun shouldGetMemoryStatisticsSuccessfully() = runTest {
            // Given
            val expectedStats = MemoryStatistics(
                totalMemories = 100,
                memoriesByImportance = mapOf(
                    MemoryImportance.HIGH to 20,
                    MemoryImportance.MEDIUM to 50,
                    MemoryImportance.LOW to 30
                ),
                oldestMemory = LocalDateTime.now().minusDays(365),
                newestMemory = LocalDateTime.now()
            )
            every { memoryStorage.getStatistics() } returns expectedStats
            
            // When
            val result = memoryManager.getMemoryStatistics()
            
            // Then
            assertEquals(expectedStats, result)
            verify { memoryStorage.getStatistics() }
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandling {
        
        @Test
        @DisplayName("Should handle null memory gracefully")
        fun shouldHandleNullMemoryGracefully() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryManager.storeMemory(null)
            }
        }
        
        @Test
        @DisplayName("Should handle memory with null content")
        fun shouldHandleMemoryWithNullContent() = runTest {
            // Given
            val invalidMemory = Memory("test-id", null, LocalDateTime.now(), MemoryImportance.MEDIUM)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryManager.storeMemory(invalidMemory)
            }
        }
        
        @Test
        @DisplayName("Should handle memory with empty content")
        fun shouldHandleMemoryWithEmptyContent() = runTest {
            // Given
            val invalidMemory = Memory("test-id", "", LocalDateTime.now(), MemoryImportance.MEDIUM)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryManager.storeMemory(invalidMemory)
            }
        }
        
        @Test
        @DisplayName("Should handle memory with null timestamp")
        fun shouldHandleMemoryWithNullTimestamp() = runTest {
            // Given
            val invalidMemory = Memory("test-id", "Valid content", null, MemoryImportance.MEDIUM)
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryManager.storeMemory(invalidMemory)
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent access gracefully")
        fun shouldHandleConcurrentAccessGracefully() = runTest {
            // Given
            val memory = Memory("test-id", "Test content", LocalDateTime.now(), MemoryImportance.MEDIUM)
            every { memoryStorage.store(memory) } returns Unit
            
            // When - Simulate concurrent access
            val jobs = (1..10).map { 
                async { memoryManager.storeMemory(memory) }
            }
            jobs.forEach { it.await() }
            
            // Then
            verify(exactly = 10) { memoryStorage.store(memory) }
        }
        
        @Test
        @DisplayName("Should handle large memory content efficiently")
        fun shouldHandleLargeMemoryContentEfficiently() = runTest {
            // Given
            val largeContent = "A".repeat(1000000) // 1MB string
            val largeMemory = Memory("large-id", largeContent, LocalDateTime.now(), MemoryImportance.MEDIUM)
            every { memoryStorage.store(largeMemory) } returns Unit
            
            // When
            val startTime = System.currentTimeMillis()
            memoryManager.storeMemory(largeMemory)
            val endTime = System.currentTimeMillis()
            
            // Then
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
            verify { memoryStorage.store(largeMemory) }
        }
        
        @Test
        @DisplayName("Should handle storage corruption gracefully")
        fun shouldHandleStorageCorruptionGracefully() = runTest {
            // Given
            every { memoryStorage.retrieveById(any()) } throws CorruptedDataException("Data corrupted")
            
            // When & Then
            assertThrows<MemoryManagerException> {
                memoryManager.retrieveMemory("test-id")
            }
        }
    }
}