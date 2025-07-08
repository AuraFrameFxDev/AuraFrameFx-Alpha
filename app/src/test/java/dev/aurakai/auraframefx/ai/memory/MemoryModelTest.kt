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
import org.mockito.kotlin.any
import org.mockito.kotlin.never
import java.time.LocalDateTime
import java.util.concurrent.CompletableFuture

@DisplayName("MemoryModel Unit Tests")
class MemoryModelTest {
    
    private lateinit var memoryModel: MemoryModel
    private lateinit var mockClosable: AutoCloseable
    
    @BeforeEach
    fun setUp() {
        mockClosable = MockitoAnnotations.openMocks(this)
        memoryModel = MemoryModel()
    }
    
    @AfterEach
    fun tearDown() {
        mockClosable.close()
    }
    
    @Nested
    @DisplayName("Memory Storage Tests")
    inner class MemoryStorageTests {
        
        @Test
        @DisplayName("Should store memory successfully")
        fun shouldStoreMemorySuccessfully() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            
            // When
            val result = memoryModel.storeMemory(testMemory, testContext)
            
            // Then
            assertTrue(result)
            assertTrue(memoryModel.hasMemory(testMemory))
        }
        
        @Test
        @DisplayName("Should store multiple memories")
        fun shouldStoreMultipleMemories() {
            // Given
            val memories = listOf(
                "Memory 1" to "Context 1",
                "Memory 2" to "Context 2",
                "Memory 3" to "Context 3"
            )
            
            // When
            memories.forEach { (memory, context) ->
                memoryModel.storeMemory(memory, context)
            }
            
            // Then
            memories.forEach { (memory, _) ->
                assertTrue(memoryModel.hasMemory(memory))
            }
            assertEquals(3, memoryModel.getMemoryCount())
        }
        
        @Test
        @DisplayName("Should handle empty memory content")
        fun shouldHandleEmptyMemoryContent() {
            // Given
            val emptyMemory = ""
            val context = "Test context"
            
            // When
            val result = memoryModel.storeMemory(emptyMemory, context)
            
            // Then
            assertFalse(result)
            assertFalse(memoryModel.hasMemory(emptyMemory))
        }
        
        @Test
        @DisplayName("Should handle null memory content")
        fun shouldHandleNullMemoryContent() {
            // Given
            val context = "Test context"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                memoryModel.storeMemory(null, context)
            }
        }
        
        @Test
        @DisplayName("Should handle very large memory content")
        fun shouldHandleVeryLargeMemoryContent() {
            // Given
            val largeMemory = "x".repeat(1000000) // 1MB string
            val context = "Test context"
            
            // When
            val result = memoryModel.storeMemory(largeMemory, context)
            
            // Then
            assertTrue(result)
            assertTrue(memoryModel.hasMemory(largeMemory))
        }
        
        @Test
        @DisplayName("Should handle special characters in memory")
        fun shouldHandleSpecialCharactersInMemory() {
            // Given
            val specialMemory = "Memory with special chars: 特殊字符 !@#$%^&*()_+-=[]{}|;':\",./<>?"
            val context = "Test context"
            
            // When
            val result = memoryModel.storeMemory(specialMemory, context)
            
            // Then
            assertTrue(result)
            assertTrue(memoryModel.hasMemory(specialMemory))
        }
    }
    
    @Nested
    @DisplayName("Memory Retrieval Tests")
    inner class MemoryRetrievalTests {
        
        @Test
        @DisplayName("Should retrieve stored memory")
        fun shouldRetrieveStoredMemory() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            memoryModel.storeMemory(testMemory, testContext)
            
            // When
            val retrievedMemory = memoryModel.getMemory(testMemory)
            
            // Then
            assertNotNull(retrievedMemory)
            assertEquals(testMemory, retrievedMemory?.content)
            assertEquals(testContext, retrievedMemory?.context)
        }
        
        @Test
        @DisplayName("Should return null for non-existent memory")
        fun shouldReturnNullForNonExistentMemory() {
            // Given
            val nonExistentMemory = "Non-existent memory"
            
            // When
            val result = memoryModel.getMemory(nonExistentMemory)
            
            // Then
            assertNull(result)
        }
        
        @Test
        @DisplayName("Should retrieve all memories")
        fun shouldRetrieveAllMemories() {
            // Given
            val memories = listOf(
                "Memory 1" to "Context 1",
                "Memory 2" to "Context 2",
                "Memory 3" to "Context 3"
            )
            memories.forEach { (memory, context) ->
                memoryModel.storeMemory(memory, context)
            }
            
            // When
            val allMemories = memoryModel.getAllMemories()
            
            // Then
            assertEquals(3, allMemories.size)
            memories.forEach { (memory, context) ->
                assertTrue(allMemories.any { it.content == memory && it.context == context })
            }
        }
        
        @Test
        @DisplayName("Should return empty list when no memories exist")
        fun shouldReturnEmptyListWhenNoMemoriesExist() {
            // When
            val allMemories = memoryModel.getAllMemories()
            
            // Then
            assertTrue(allMemories.isEmpty())
        }
        
        @Test
        @DisplayName("Should search memories by keyword")
        fun shouldSearchMemoriesByKeyword() {
            // Given
            memoryModel.storeMemory("Important meeting notes", "Work context")
            memoryModel.storeMemory("Shopping list", "Personal context")
            memoryModel.storeMemory("Important deadline", "Work context")
            
            // When
            val searchResults = memoryModel.searchMemories("important")
            
            // Then
            assertEquals(2, searchResults.size)
            assertTrue(searchResults.all { it.content.contains("Important", ignoreCase = true) })
        }
        
        @Test
        @DisplayName("Should return empty list for no search matches")
        fun shouldReturnEmptyListForNoSearchMatches() {
            // Given
            memoryModel.storeMemory("Test memory", "Test context")
            
            // When
            val searchResults = memoryModel.searchMemories("nonexistent")
            
            // Then
            assertTrue(searchResults.isEmpty())
        }
    }
    
    @Nested
    @DisplayName("Memory Management Tests")
    inner class MemoryManagementTests {
        
        @Test
        @DisplayName("Should delete memory successfully")
        fun shouldDeleteMemorySuccessfully() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            memoryModel.storeMemory(testMemory, testContext)
            
            // When
            val result = memoryModel.deleteMemory(testMemory)
            
            // Then
            assertTrue(result)
            assertFalse(memoryModel.hasMemory(testMemory))
        }
        
        @Test
        @DisplayName("Should return false when deleting non-existent memory")
        fun shouldReturnFalseWhenDeletingNonExistentMemory() {
            // Given
            val nonExistentMemory = "Non-existent memory"
            
            // When
            val result = memoryModel.deleteMemory(nonExistentMemory)
            
            // Then
            assertFalse(result)
        }
        
        @Test
        @DisplayName("Should clear all memories")
        fun shouldClearAllMemories() {
            // Given
            memoryModel.storeMemory("Memory 1", "Context 1")
            memoryModel.storeMemory("Memory 2", "Context 2")
            memoryModel.storeMemory("Memory 3", "Context 3")
            
            // When
            memoryModel.clearAllMemories()
            
            // Then
            assertEquals(0, memoryModel.getMemoryCount())
            assertTrue(memoryModel.getAllMemories().isEmpty())
        }
        
        @Test
        @DisplayName("Should update existing memory")
        fun shouldUpdateExistingMemory() {
            // Given
            val originalMemory = "Original memory"
            val originalContext = "Original context"
            val updatedMemory = "Updated memory"
            val updatedContext = "Updated context"
            memoryModel.storeMemory(originalMemory, originalContext)
            
            // When
            val result = memoryModel.updateMemory(originalMemory, updatedMemory, updatedContext)
            
            // Then
            assertTrue(result)
            assertFalse(memoryModel.hasMemory(originalMemory))
            assertTrue(memoryModel.hasMemory(updatedMemory))
            assertEquals(updatedContext, memoryModel.getMemory(updatedMemory)?.context)
        }
        
        @Test
        @DisplayName("Should return false when updating non-existent memory")
        fun shouldReturnFalseWhenUpdatingNonExistentMemory() {
            // Given
            val nonExistentMemory = "Non-existent memory"
            val updatedMemory = "Updated memory"
            val updatedContext = "Updated context"
            
            // When
            val result = memoryModel.updateMemory(nonExistentMemory, updatedMemory, updatedContext)
            
            // Then
            assertFalse(result)
        }
    }
    
    @Nested
    @DisplayName("Memory Metadata Tests")
    inner class MemoryMetadataTests {
        
        @Test
        @DisplayName("Should track memory creation timestamp")
        fun shouldTrackMemoryCreationTimestamp() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            val beforeTime = LocalDateTime.now()
            
            // When
            memoryModel.storeMemory(testMemory, testContext)
            val afterTime = LocalDateTime.now()
            
            // Then
            val retrievedMemory = memoryModel.getMemory(testMemory)
            assertNotNull(retrievedMemory?.createdAt)
            assertTrue(retrievedMemory?.createdAt?.isAfter(beforeTime) ?: false)
            assertTrue(retrievedMemory?.createdAt?.isBefore(afterTime) ?: false)
        }
        
        @Test
        @DisplayName("Should track memory access count")
        fun shouldTrackMemoryAccessCount() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            memoryModel.storeMemory(testMemory, testContext)
            
            // When
            repeat(3) {
                memoryModel.getMemory(testMemory)
            }
            
            // Then
            val retrievedMemory = memoryModel.getMemory(testMemory)
            assertEquals(4, retrievedMemory?.accessCount) // 3 + 1 for this retrieval
        }
        
        @Test
        @DisplayName("Should track last accessed timestamp")
        fun shouldTrackLastAccessedTimestamp() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            memoryModel.storeMemory(testMemory, testContext)
            val initialMemory = memoryModel.getMemory(testMemory)
            
            // When
            Thread.sleep(100) // Ensure different timestamp
            val secondAccess = memoryModel.getMemory(testMemory)
            
            // Then
            assertTrue(secondAccess?.lastAccessedAt?.isAfter(initialMemory?.lastAccessedAt) ?: false)
        }
    }
    
    @Nested
    @DisplayName("Memory Capacity Tests")
    inner class MemoryCapacityTests {
        
        @Test
        @DisplayName("Should respect memory capacity limit")
        fun shouldRespectMemoryCapacityLimit() {
            // Given
            val maxCapacity = 100
            memoryModel.setMaxCapacity(maxCapacity)
            
            // When
            for (i in 1..150) {
                memoryModel.storeMemory("Memory $i", "Context $i")
            }
            
            // Then
            assertTrue(memoryModel.getMemoryCount() <= maxCapacity)
        }
        
        @Test
        @DisplayName("Should evict oldest memories when capacity exceeded")
        fun shouldEvictOldestMemoriesWhenCapacityExceeded() {
            // Given
            val maxCapacity = 3
            memoryModel.setMaxCapacity(maxCapacity)
            
            // When
            memoryModel.storeMemory("Memory 1", "Context 1")
            memoryModel.storeMemory("Memory 2", "Context 2")
            memoryModel.storeMemory("Memory 3", "Context 3")
            memoryModel.storeMemory("Memory 4", "Context 4") // Should evict Memory 1
            
            // Then
            assertEquals(3, memoryModel.getMemoryCount())
            assertFalse(memoryModel.hasMemory("Memory 1"))
            assertTrue(memoryModel.hasMemory("Memory 2"))
            assertTrue(memoryModel.hasMemory("Memory 3"))
            assertTrue(memoryModel.hasMemory("Memory 4"))
        }
    }
    
    @Nested
    @DisplayName("Concurrency Tests")
    inner class ConcurrencyTests {
        
        @Test
        @DisplayName("Should handle concurrent memory storage")
        fun shouldHandleConcurrentMemoryStorage() {
            // Given
            val futures = mutableListOf<CompletableFuture<Boolean>>()
            
            // When
            for (i in 1..10) {
                val future = CompletableFuture.supplyAsync {
                    memoryModel.storeMemory("Memory $i", "Context $i")
                }
                futures.add(future)
            }
            
            // Then
            CompletableFuture.allOf(*futures.toTypedArray()).join()
            assertEquals(10, memoryModel.getMemoryCount())
        }
        
        @Test
        @DisplayName("Should handle concurrent memory access")
        fun shouldHandleConcurrentMemoryAccess() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            memoryModel.storeMemory(testMemory, testContext)
            
            // When
            val futures = mutableListOf<CompletableFuture<Any?>>()
            for (i in 1..10) {
                val future = CompletableFuture.supplyAsync {
                    memoryModel.getMemory(testMemory)
                }
                futures.add(future)
            }
            
            // Then
            CompletableFuture.allOf(*futures.toTypedArray()).join()
            val retrievedMemory = memoryModel.getMemory(testMemory)
            assertTrue(retrievedMemory?.accessCount!! >= 10)
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle duplicate memory storage")
        fun shouldHandleDuplicateMemoryStorage() {
            // Given
            val testMemory = "Test memory content"
            val testContext = "Test context"
            
            // When
            val firstResult = memoryModel.storeMemory(testMemory, testContext)
            val secondResult = memoryModel.storeMemory(testMemory, testContext)
            
            // Then
            assertTrue(firstResult)
            assertFalse(secondResult) // Should not store duplicate
            assertEquals(1, memoryModel.getMemoryCount())
        }
        
        @Test
        @DisplayName("Should handle memory operations after clear")
        fun shouldHandleMemoryOperationsAfterClear() {
            // Given
            memoryModel.storeMemory("Memory 1", "Context 1")
            memoryModel.clearAllMemories()
            
            // When
            val result = memoryModel.getMemory("Memory 1")
            val hasMemory = memoryModel.hasMemory("Memory 1")
            val deleteResult = memoryModel.deleteMemory("Memory 1")
            
            // Then
            assertNull(result)
            assertFalse(hasMemory)
            assertFalse(deleteResult)
        }
        
        @Test
        @DisplayName("Should handle malformed search queries")
        fun shouldHandleMalformedSearchQueries() {
            // Given
            memoryModel.storeMemory("Test memory", "Test context")
            
            // When & Then
            assertDoesNotThrow {
                memoryModel.searchMemories("")
                memoryModel.searchMemories("   ")
                memoryModel.searchMemories("*")
                memoryModel.searchMemories("[]")
                memoryModel.searchMemories("()")
            }
        }
        
        @Test
        @DisplayName("Should handle memory model reset")
        fun shouldHandleMemoryModelReset() {
            // Given
            memoryModel.storeMemory("Memory 1", "Context 1")
            memoryModel.storeMemory("Memory 2", "Context 2")
            
            // When
            memoryModel.reset()
            
            // Then
            assertEquals(0, memoryModel.getMemoryCount())
            assertTrue(memoryModel.getAllMemories().isEmpty())
        }
    }
    
    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {
        
        @Test
        @DisplayName("Should handle large number of memories efficiently")
        fun shouldHandleLargeNumberOfMemoriesEfficiently() {
            // Given
            val memoryCount = 1000
            val startTime = System.currentTimeMillis()
            
            // When
            for (i in 1..memoryCount) {
                memoryModel.storeMemory("Memory $i", "Context $i")
            }
            val endTime = System.currentTimeMillis()
            
            // Then
            assertTrue(endTime - startTime < 5000) // Should complete within 5 seconds
            assertEquals(memoryCount, memoryModel.getMemoryCount())
        }
        
        @Test
        @DisplayName("Should search large memory collection efficiently")
        fun shouldSearchLargeMemoryCollectionEfficiently() {
            // Given
            for (i in 1..1000) {
                memoryModel.storeMemory("Memory $i with keyword", "Context $i")
            }
            val startTime = System.currentTimeMillis()
            
            // When
            val results = memoryModel.searchMemories("keyword")
            val endTime = System.currentTimeMillis()
            
            // Then
            assertTrue(endTime - startTime < 1000) // Should complete within 1 second
            assertEquals(1000, results.size)
        }
    }
}