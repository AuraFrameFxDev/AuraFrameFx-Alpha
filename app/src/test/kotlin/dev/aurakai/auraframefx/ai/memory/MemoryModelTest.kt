package dev.aurakai.auraframefx.ai.memory

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import io.mockk.*
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.runBlocking
import java.util.stream.Stream
import kotlin.test.assertFailsWith

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("MemoryModel Tests")
class MemoryModelTest {

    private lateinit var memoryModel: MemoryModel
    private lateinit var mockMemoryStorage: MemoryStorage

    @BeforeEach
    fun setup() {
        mockMemoryStorage = mockk<MemoryStorage>()
        memoryModel = MemoryModel(mockMemoryStorage)
    }

    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Nested
    @DisplayName("Memory Creation Tests")
    inner class MemoryCreationTests {

        @Test
        @DisplayName("Should create memory with valid input")
        fun `should create memory with valid input`() = runTest {
            // Arrange
            val content = "Test memory content"
            val expectedMemory = Memory(id = "1", content = content, timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.save(any()) } returns expectedMemory

            // Act
            val result = memoryModel.createMemory(content)

            // Assert
            assertNotNull(result)
            assertEquals(content, result.content)
            verify { mockMemoryStorage.save(any()) }
        }

        @Test
        @DisplayName("Should handle empty content gracefully")
        fun `should handle empty content gracefully`() = runTest {
            // Arrange
            val content = ""
            
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.createMemory(content)
            }
        }

        @Test
        @DisplayName("Should handle null content gracefully")
        fun `should handle null content gracefully`() = runTest {
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.createMemory(null)
            }
        }

        @ParameterizedTest
        @ValueSource(strings = ["", "   ", "\t", "\n"])
        @DisplayName("Should reject whitespace-only content")
        fun `should reject whitespace-only content`(content: String) = runTest {
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.createMemory(content)
            }
        }

        @Test
        @DisplayName("Should handle very long content")
        fun `should handle very long content`() = runTest {
            // Arrange
            val longContent = "x".repeat(10000)
            val expectedMemory = Memory(id = "1", content = longContent, timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.save(any()) } returns expectedMemory

            // Act
            val result = memoryModel.createMemory(longContent)

            // Assert
            assertNotNull(result)
            assertEquals(longContent, result.content)
        }

        @Test
        @DisplayName("Should handle special characters in content")
        fun `should handle special characters in content`() = runTest {
            // Arrange
            val specialContent = "Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?"
            val expectedMemory = Memory(id = "1", content = specialContent, timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.save(any()) } returns expectedMemory

            // Act
            val result = memoryModel.createMemory(specialContent)

            // Assert
            assertNotNull(result)
            assertEquals(specialContent, result.content)
        }

        @Test
        @DisplayName("Should handle storage failure during creation")
        fun `should handle storage failure during creation`() = runTest {
            // Arrange
            val content = "Test content"
            every { mockMemoryStorage.save(any()) } throws RuntimeException("Storage failed")

            // Act & Assert
            assertFailsWith<RuntimeException> {
                memoryModel.createMemory(content)
            }
        }
    }

    @Nested
    @DisplayName("Memory Retrieval Tests")
    inner class MemoryRetrievalTests {

        @Test
        @DisplayName("Should retrieve memory by valid ID")
        fun `should retrieve memory by valid ID`() = runTest {
            // Arrange
            val memoryId = "test-id"
            val expectedMemory = Memory(id = memoryId, content = "Test content", timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.findById(memoryId) } returns expectedMemory

            // Act
            val result = memoryModel.getMemoryById(memoryId)

            // Assert
            assertNotNull(result)
            assertEquals(memoryId, result?.id)
            assertEquals("Test content", result?.content)
            verify { mockMemoryStorage.findById(memoryId) }
        }

        @Test
        @DisplayName("Should return null for non-existent memory ID")
        fun `should return null for non-existent memory ID`() = runTest {
            // Arrange
            val memoryId = "non-existent-id"
            every { mockMemoryStorage.findById(memoryId) } returns null

            // Act
            val result = memoryModel.getMemoryById(memoryId)

            // Assert
            assertNull(result)
            verify { mockMemoryStorage.findById(memoryId) }
        }

        @Test
        @DisplayName("Should handle empty ID gracefully")
        fun `should handle empty ID gracefully`() = runTest {
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.getMemoryById("")
            }
        }

        @Test
        @DisplayName("Should handle null ID gracefully")
        fun `should handle null ID gracefully`() = runTest {
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.getMemoryById(null)
            }
        }

        @Test
        @DisplayName("Should handle storage exceptions during retrieval")
        fun `should handle storage exceptions during retrieval`() = runTest {
            // Arrange
            val memoryId = "test-id"
            every { mockMemoryStorage.findById(memoryId) } throws RuntimeException("Storage error")

            // Act & Assert
            assertFailsWith<RuntimeException> {
                memoryModel.getMemoryById(memoryId)
            }
        }
    }

    @Nested
    @DisplayName("Memory Search Tests")
    inner class MemorySearchTests {

        @Test
        @DisplayName("Should search memories by content")
        fun `should search memories by content`() = runTest {
            // Arrange
            val searchTerm = "test"
            val expectedMemories = listOf(
                Memory(id = "1", content = "This is a test", timestamp = System.currentTimeMillis()),
                Memory(id = "2", content = "Another test memory", timestamp = System.currentTimeMillis())
            )
            every { mockMemoryStorage.searchByContent(searchTerm) } returns expectedMemories

            // Act
            val result = memoryModel.searchMemories(searchTerm)

            // Assert
            assertNotNull(result)
            assertEquals(2, result.size)
            assertTrue(result.all { it.content.contains(searchTerm, ignoreCase = true) })
            verify { mockMemoryStorage.searchByContent(searchTerm) }
        }

        @Test
        @DisplayName("Should return empty list for no matches")
        fun `should return empty list for no matches`() = runTest {
            // Arrange
            val searchTerm = "nonexistent"
            every { mockMemoryStorage.searchByContent(searchTerm) } returns emptyList()

            // Act
            val result = memoryModel.searchMemories(searchTerm)

            // Assert
            assertNotNull(result)
            assertTrue(result.isEmpty())
            verify { mockMemoryStorage.searchByContent(searchTerm) }
        }

        @Test
        @DisplayName("Should handle empty search term")
        fun `should handle empty search term`() = runTest {
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.searchMemories("")
            }
        }

        @Test
        @DisplayName("Should handle null search term")
        fun `should handle null search term`() = runTest {
            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.searchMemories(null)
            }
        }

        @ParameterizedTest
        @MethodSource("searchTermProvider")
        @DisplayName("Should handle various search terms")
        fun `should handle various search terms`(searchTerm: String, expectedCount: Int) = runTest {
            // Arrange
            val memories = listOf(
                Memory(id = "1", content = "Hello world", timestamp = System.currentTimeMillis()),
                Memory(id = "2", content = "Test memory", timestamp = System.currentTimeMillis()),
                Memory(id = "3", content = "Another test", timestamp = System.currentTimeMillis())
            )
            every { mockMemoryStorage.searchByContent(searchTerm) } returns 
                memories.filter { it.content.contains(searchTerm, ignoreCase = true) }

            // Act
            val result = memoryModel.searchMemories(searchTerm)

            // Assert
            assertEquals(expectedCount, result.size)
        }

        fun searchTermProvider(): Stream<Arguments> {
            return Stream.of(
                Arguments.of("test", 2),
                Arguments.of("world", 1),
                Arguments.of("hello", 1),
                Arguments.of("nonexistent", 0)
            )
        }
    }

    @Nested
    @DisplayName("Memory Update Tests")
    inner class MemoryUpdateTests {

        @Test
        @DisplayName("Should update memory content successfully")
        fun `should update memory content successfully`() = runTest {
            // Arrange
            val memoryId = "test-id"
            val newContent = "Updated content"
            val existingMemory = Memory(id = memoryId, content = "Old content", timestamp = System.currentTimeMillis())
            val updatedMemory = existingMemory.copy(content = newContent)
            
            every { mockMemoryStorage.findById(memoryId) } returns existingMemory
            every { mockMemoryStorage.update(any()) } returns updatedMemory

            // Act
            val result = memoryModel.updateMemory(memoryId, newContent)

            // Assert
            assertNotNull(result)
            assertEquals(newContent, result?.content)
            verify { mockMemoryStorage.findById(memoryId) }
            verify { mockMemoryStorage.update(any()) }
        }

        @Test
        @DisplayName("Should return null when updating non-existent memory")
        fun `should return null when updating non-existent memory`() = runTest {
            // Arrange
            val memoryId = "non-existent-id"
            val newContent = "Updated content"
            every { mockMemoryStorage.findById(memoryId) } returns null

            // Act
            val result = memoryModel.updateMemory(memoryId, newContent)

            // Assert
            assertNull(result)
            verify { mockMemoryStorage.findById(memoryId) }
            verify(exactly = 0) { mockMemoryStorage.update(any()) }
        }

        @Test
        @DisplayName("Should handle empty content during update")
        fun `should handle empty content during update`() = runTest {
            // Arrange
            val memoryId = "test-id"
            val newContent = ""

            // Act & Assert
            assertFailsWith<IllegalArgumentException> {
                memoryModel.updateMemory(memoryId, newContent)
            }
        }
    }

    @Nested
    @DisplayName("Memory Deletion Tests")
    inner class MemoryDeletionTests {

        @Test
        @DisplayName("Should delete memory successfully")
        fun `should delete memory successfully`() = runTest {
            // Arrange
            val memoryId = "test-id"
            val existingMemory = Memory(id = memoryId, content = "Test content", timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.findById(memoryId) } returns existingMemory
            every { mockMemoryStorage.delete(memoryId) } returns true

            // Act
            val result = memoryModel.deleteMemory(memoryId)

            // Assert
            assertTrue(result)
            verify { mockMemoryStorage.findById(memoryId) }
            verify { mockMemoryStorage.delete(memoryId) }
        }

        @Test
        @DisplayName("Should return false when deleting non-existent memory")
        fun `should return false when deleting non-existent memory`() = runTest {
            // Arrange
            val memoryId = "non-existent-id"
            every { mockMemoryStorage.findById(memoryId) } returns null

            // Act
            val result = memoryModel.deleteMemory(memoryId)

            // Assert
            assertFalse(result)
            verify { mockMemoryStorage.findById(memoryId) }
            verify(exactly = 0) { mockMemoryStorage.delete(any()) }
        }

        @Test
        @DisplayName("Should handle storage failure during deletion")
        fun `should handle storage failure during deletion`() = runTest {
            // Arrange
            val memoryId = "test-id"
            val existingMemory = Memory(id = memoryId, content = "Test content", timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.findById(memoryId) } returns existingMemory
            every { mockMemoryStorage.delete(memoryId) } throws RuntimeException("Delete failed")

            // Act & Assert
            assertFailsWith<RuntimeException> {
                memoryModel.deleteMemory(memoryId)
            }
        }
    }

    @Nested
    @DisplayName("Memory Listing Tests")
    inner class MemoryListingTests {

        @Test
        @DisplayName("Should retrieve all memories")
        fun `should retrieve all memories`() = runTest {
            // Arrange
            val expectedMemories = listOf(
                Memory(id = "1", content = "First memory", timestamp = System.currentTimeMillis()),
                Memory(id = "2", content = "Second memory", timestamp = System.currentTimeMillis()),
                Memory(id = "3", content = "Third memory", timestamp = System.currentTimeMillis())
            )
            every { mockMemoryStorage.findAll() } returns expectedMemories

            // Act
            val result = memoryModel.getAllMemories()

            // Assert
            assertNotNull(result)
            assertEquals(3, result.size)
            assertEquals(expectedMemories, result)
            verify { mockMemoryStorage.findAll() }
        }

        @Test
        @DisplayName("Should return empty list when no memories exist")
        fun `should return empty list when no memories exist`() = runTest {
            // Arrange
            every { mockMemoryStorage.findAll() } returns emptyList()

            // Act
            val result = memoryModel.getAllMemories()

            // Assert
            assertNotNull(result)
            assertTrue(result.isEmpty())
            verify { mockMemoryStorage.findAll() }
        }

        @Test
        @DisplayName("Should handle storage exceptions during listing")
        fun `should handle storage exceptions during listing`() = runTest {
            // Arrange
            every { mockMemoryStorage.findAll() } throws RuntimeException("Storage error")

            // Act & Assert
            assertFailsWith<RuntimeException> {
                memoryModel.getAllMemories()
            }
        }
    }

    @Nested
    @DisplayName("Memory Validation Tests")
    inner class MemoryValidationTests {

        @Test
        @DisplayName("Should validate memory content length")
        fun `should validate memory content length`() = runTest {
            // Arrange
            val maxLength = 1000
            val validContent = "x".repeat(maxLength)
            val invalidContent = "x".repeat(maxLength + 1)

            // Act & Assert - Valid content should pass
            assertDoesNotThrow {
                memoryModel.validateMemoryContent(validContent)
            }

            // Act & Assert - Invalid content should fail
            assertFailsWith<IllegalArgumentException> {
                memoryModel.validateMemoryContent(invalidContent)
            }
        }

        @Test
        @DisplayName("Should validate memory ID format")
        fun `should validate memory ID format`() = runTest {
            // Arrange
            val validId = "valid-id-123"
            val invalidId = "invalid@id#123"

            // Act & Assert - Valid ID should pass
            assertDoesNotThrow {
                memoryModel.validateMemoryId(validId)
            }

            // Act & Assert - Invalid ID should fail
            assertFailsWith<IllegalArgumentException> {
                memoryModel.validateMemoryId(invalidId)
            }
        }
    }

    @Nested
    @DisplayName("Concurrency Tests")
    inner class ConcurrencyTests {

        @Test
        @DisplayName("Should handle concurrent memory operations")
        fun `should handle concurrent memory operations`() = runTest {
            // Arrange
            val content = "Concurrent test content"
            val memory = Memory(id = "1", content = content, timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.save(any()) } returns memory
            every { mockMemoryStorage.findById("1") } returns memory

            // Act - Simulate concurrent operations
            val results = (1..10).map { index ->
                kotlinx.coroutines.async {
                    memoryModel.createMemory("$content $index")
                }
            }.map { it.await() }

            // Assert
            assertEquals(10, results.size)
            assertTrue(results.all { it.content.startsWith(content) })
            verify(exactly = 10) { mockMemoryStorage.save(any()) }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle large number of memories efficiently")
        fun `should handle large number of memories efficiently`() = runTest {
            // Arrange
            val largeMemoryList = (1..1000).map { index ->
                Memory(id = index.toString(), content = "Memory $index", timestamp = System.currentTimeMillis())
            }
            every { mockMemoryStorage.findAll() } returns largeMemoryList

            // Act
            val startTime = System.currentTimeMillis()
            val result = memoryModel.getAllMemories()
            val endTime = System.currentTimeMillis()

            // Assert
            assertEquals(1000, result.size)
            assertTrue(endTime - startTime < 1000) // Should complete within 1 second
            verify { mockMemoryStorage.findAll() }
        }
    }

    @Nested
    @DisplayName("Edge Case Tests")
    inner class EdgeCaseTests {

        @Test
        @DisplayName("Should handle Unicode content")
        fun `should handle Unicode content`() = runTest {
            // Arrange
            val unicodeContent = "测试内容 🚀 emoji 한글 العربية"
            val expectedMemory = Memory(id = "1", content = unicodeContent, timestamp = System.currentTimeMillis())
            every { mockMemoryStorage.save(any()) } returns expectedMemory

            // Act
            val result = memoryModel.createMemory(unicodeContent)

            // Assert
            assertNotNull(result)
            assertEquals(unicodeContent, result.content)
        }

        @Test
        @DisplayName("Should handle memory with future timestamp")
        fun `should handle memory with future timestamp`() = runTest {
            // Arrange
            val futureTimestamp = System.currentTimeMillis() + 86400000 // 24 hours in future
            val memory = Memory(id = "1", content = "Future memory", timestamp = futureTimestamp)
            every { mockMemoryStorage.save(any()) } returns memory

            // Act
            val result = memoryModel.createMemory("Future memory")

            // Assert
            assertNotNull(result)
            assertEquals("Future memory", result.content)
        }

        @Test
        @DisplayName("Should handle memory with zero timestamp")
        fun `should handle memory with zero timestamp`() = runTest {
            // Arrange
            val memory = Memory(id = "1", content = "Zero timestamp memory", timestamp = 0)
            every { mockMemoryStorage.save(any()) } returns memory

            // Act
            val result = memoryModel.createMemory("Zero timestamp memory")

            // Assert
            assertNotNull(result)
            assertEquals("Zero timestamp memory", result.content)
        }
    }
}