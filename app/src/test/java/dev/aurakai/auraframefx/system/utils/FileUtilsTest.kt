package dev.aurakai.auraframefx.system.utils

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.io.TempDir
import org.mockito.kotlin.*
import java.io.File
import java.io.IOException
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.Files
import kotlin.test.assertFailsWith

/**
 * Comprehensive unit tests for FileUtils class.
 * Testing framework: JUnit 5 with Mockito for mocking.
 */
@DisplayName("FileUtils Tests")
class FileUtilsTest {

    @TempDir
    lateinit var tempDir: Path

    private lateinit var testFile: File
    private lateinit var testDirectory: File

    @BeforeEach
    fun setUp() {
        testFile = tempDir.resolve("test.txt").toFile()
        testDirectory = tempDir.resolve("testDir").toFile()
        testDirectory.mkdirs()
    }

    @AfterEach
    fun tearDown() {
        // Cleanup is handled by @TempDir
    }

    @Nested
    @DisplayName("File Existence Tests")
    inner class FileExistenceTests {

        @Test
        @DisplayName("Should return true when file exists")
        fun shouldReturnTrueWhenFileExists() {
            testFile.createNewFile()
            assertTrue(FileUtils.exists(testFile.path))
        }

        @Test
        @DisplayName("Should return false when file does not exist")
        fun shouldReturnFalseWhenFileDoesNotExist() {
            assertFalse(FileUtils.exists("nonexistent.txt"))
        }

        @Test
        @DisplayName("Should return true when directory exists")
        fun shouldReturnTrueWhenDirectoryExists() {
            assertTrue(FileUtils.exists(testDirectory.path))
        }

        @Test
        @DisplayName("Should handle null path gracefully")
        fun shouldHandleNullPathGracefully() {
            assertFalse(FileUtils.exists(null))
        }

        @Test
        @DisplayName("Should handle empty path gracefully")
        fun shouldHandleEmptyPathGracefully() {
            assertFalse(FileUtils.exists(""))
        }

        @Test
        @DisplayName("Should handle whitespace-only path gracefully")
        fun shouldHandleWhitespaceOnlyPathGracefully() {
            assertFalse(FileUtils.exists("   "))
        }
    }

    @Nested
    @DisplayName("File Reading Tests")
    inner class FileReadingTests {

        @Test
        @DisplayName("Should read file content successfully")
        fun shouldReadFileContentSuccessfully() {
            val content = "Hello, World!"
            testFile.writeText(content)
            
            val result = FileUtils.readFile(testFile.path)
            assertEquals(content, result)
        }

        @Test
        @DisplayName("Should read empty file successfully")
        fun shouldReadEmptyFileSuccessfully() {
            testFile.createNewFile()
            
            val result = FileUtils.readFile(testFile.path)
            assertEquals("", result)
        }

        @Test
        @DisplayName("Should read file with special characters")
        fun shouldReadFileWithSpecialCharacters() {
            val content = "Hello, 世界! 🌍 @#$%^&*()"
            testFile.writeText(content)
            
            val result = FileUtils.readFile(testFile.path)
            assertEquals(content, result)
        }

        @Test
        @DisplayName("Should read multiline file correctly")
        fun shouldReadMultilineFileCorrectly() {
            val content = "Line 1\nLine 2\nLine 3"
            testFile.writeText(content)
            
            val result = FileUtils.readFile(testFile.path)
            assertEquals(content, result)
        }

        @Test
        @DisplayName("Should throw exception when reading non-existent file")
        fun shouldThrowExceptionWhenReadingNonExistentFile() {
            assertFailsWith<IOException> {
                FileUtils.readFile("nonexistent.txt")
            }
        }

        @Test
        @DisplayName("Should throw exception when reading directory")
        fun shouldThrowExceptionWhenReadingDirectory() {
            assertFailsWith<IOException> {
                FileUtils.readFile(testDirectory.path)
            }
        }

        @Test
        @DisplayName("Should handle null path when reading")
        fun shouldHandleNullPathWhenReading() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.readFile(null)
            }
        }

        @Test
        @DisplayName("Should handle empty path when reading")
        fun shouldHandleEmptyPathWhenReading() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.readFile("")
            }
        }
    }

    @Nested
    @DisplayName("File Writing Tests")
    inner class FileWritingTests {

        @Test
        @DisplayName("Should write content to file successfully")
        fun shouldWriteContentToFileSuccessfully() {
            val content = "Hello, World!"
            
            FileUtils.writeFile(testFile.path, content)
            
            assertEquals(content, testFile.readText())
        }

        @Test
        @DisplayName("Should write empty content to file")
        fun shouldWriteEmptyContentToFile() {
            FileUtils.writeFile(testFile.path, "")
            
            assertEquals("", testFile.readText())
        }

        @Test
        @DisplayName("Should overwrite existing file content")
        fun shouldOverwriteExistingFileContent() {
            testFile.writeText("Original content")
            val newContent = "New content"
            
            FileUtils.writeFile(testFile.path, newContent)
            
            assertEquals(newContent, testFile.readText())
        }

        @Test
        @DisplayName("Should create parent directories when writing")
        fun shouldCreateParentDirectoriesWhenWriting() {
            val nestedFile = tempDir.resolve("nested/deep/file.txt")
            val content = "Nested content"
            
            FileUtils.writeFile(nestedFile.toString(), content)
            
            assertTrue(Files.exists(nestedFile))
            assertEquals(content, Files.readString(nestedFile))
        }

        @Test
        @DisplayName("Should write file with special characters")
        fun shouldWriteFileWithSpecialCharacters() {
            val content = "Hello, 世界! 🌍 @#$%^&*()"
            
            FileUtils.writeFile(testFile.path, content)
            
            assertEquals(content, testFile.readText())
        }

        @Test
        @DisplayName("Should write multiline content correctly")
        fun shouldWriteMultilineContentCorrectly() {
            val content = "Line 1\nLine 2\nLine 3"
            
            FileUtils.writeFile(testFile.path, content)
            
            assertEquals(content, testFile.readText())
        }

        @Test
        @DisplayName("Should handle null path when writing")
        fun shouldHandleNullPathWhenWriting() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.writeFile(null, "content")
            }
        }

        @Test
        @DisplayName("Should handle empty path when writing")
        fun shouldHandleEmptyPathWhenWriting() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.writeFile("", "content")
            }
        }

        @Test
        @DisplayName("Should handle null content when writing")
        fun shouldHandleNullContentWhenWriting() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.writeFile(testFile.path, null)
            }
        }
    }

    @Nested
    @DisplayName("File Copying Tests")
    inner class FileCopyingTests {

        @Test
        @DisplayName("Should copy file successfully")
        fun shouldCopyFileSuccessfully() {
            val content = "Content to copy"
            testFile.writeText(content)
            val targetFile = tempDir.resolve("copied.txt").toFile()
            
            FileUtils.copyFile(testFile.path, targetFile.path)
            
            assertTrue(targetFile.exists())
            assertEquals(content, targetFile.readText())
        }

        @Test
        @DisplayName("Should copy empty file successfully")
        fun shouldCopyEmptyFileSuccessfully() {
            testFile.createNewFile()
            val targetFile = tempDir.resolve("copied.txt").toFile()
            
            FileUtils.copyFile(testFile.path, targetFile.path)
            
            assertTrue(targetFile.exists())
            assertEquals("", targetFile.readText())
        }

        @Test
        @DisplayName("Should overwrite existing target file")
        fun shouldOverwriteExistingTargetFile() {
            val originalContent = "Original content"
            val newContent = "New content"
            testFile.writeText(newContent)
            val targetFile = tempDir.resolve("copied.txt").toFile()
            targetFile.writeText(originalContent)
            
            FileUtils.copyFile(testFile.path, targetFile.path)
            
            assertEquals(newContent, targetFile.readText())
        }

        @Test
        @DisplayName("Should create parent directories when copying")
        fun shouldCreateParentDirectoriesWhenCopying() {
            val content = "Content to copy"
            testFile.writeText(content)
            val targetFile = tempDir.resolve("nested/deep/copied.txt")
            
            FileUtils.copyFile(testFile.path, targetFile.toString())
            
            assertTrue(Files.exists(targetFile))
            assertEquals(content, Files.readString(targetFile))
        }

        @Test
        @DisplayName("Should throw exception when copying non-existent file")
        fun shouldThrowExceptionWhenCopyingNonExistentFile() {
            val targetFile = tempDir.resolve("copied.txt").toFile()
            
            assertFailsWith<IOException> {
                FileUtils.copyFile("nonexistent.txt", targetFile.path)
            }
        }

        @Test
        @DisplayName("Should throw exception when copying directory")
        fun shouldThrowExceptionWhenCopyingDirectory() {
            val targetFile = tempDir.resolve("copied.txt").toFile()
            
            assertFailsWith<IOException> {
                FileUtils.copyFile(testDirectory.path, targetFile.path)
            }
        }

        @Test
        @DisplayName("Should handle null source path when copying")
        fun shouldHandleNullSourcePathWhenCopying() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.copyFile(null, "target.txt")
            }
        }

        @Test
        @DisplayName("Should handle null target path when copying")
        fun shouldHandleNullTargetPathWhenCopying() {
            assertFailsWith<IllegalArgumentException> {
                FileUtils.copyFile(testFile.path, null)
            }
        }
    }

    @Nested
    @DisplayName("File Deletion Tests")
    inner class FileDeletionTests {

        @Test
        @DisplayName("Should delete file successfully")
        fun shouldDeleteFileSuccessfully() {
            testFile.createNewFile()
            assertTrue(testFile.exists())
            
            val result = FileUtils.deleteFile(testFile.path)
            
            assertTrue(result)
            assertFalse(testFile.exists())
        }

        @Test
        @DisplayName("Should delete empty directory successfully")
        fun shouldDeleteEmptyDirectorySuccessfully() {
            assertTrue(testDirectory.exists())
            
            val result = FileUtils.deleteFile(testDirectory.path)
            
            assertTrue(result)
            assertFalse(testDirectory.exists())
        }

        @Test
        @DisplayName("Should return false when deleting non-existent file")
        fun shouldReturnFalseWhenDeletingNonExistentFile() {
            val result = FileUtils.deleteFile("nonexistent.txt")
            
            assertFalse(result)
        }

        @Test
        @DisplayName("Should handle null path when deleting")
        fun shouldHandleNullPathWhenDeleting() {
            assertFalse(FileUtils.deleteFile(null))
        }

        @Test
        @DisplayName("Should handle empty path when deleting")
        fun shouldHandleEmptyPathWhenDeleting() {
            assertFalse(FileUtils.deleteFile(""))
        }
    }

    @Nested
    @DisplayName("Directory Operations Tests")
    inner class DirectoryOperationsTests {

        @Test
        @DisplayName("Should create directory successfully")
        fun shouldCreateDirectorySuccessfully() {
            val newDir = tempDir.resolve("newDirectory").toFile()
            
            val result = FileUtils.createDirectory(newDir.path)
            
            assertTrue(result)
            assertTrue(newDir.exists())
            assertTrue(newDir.isDirectory())
        }

        @Test
        @DisplayName("Should create nested directories successfully")
        fun shouldCreateNestedDirectoriesSuccessfully() {
            val nestedDir = tempDir.resolve("nested/deep/directory").toFile()
            
            val result = FileUtils.createDirectory(nestedDir.path)
            
            assertTrue(result)
            assertTrue(nestedDir.exists())
            assertTrue(nestedDir.isDirectory())
        }

        @Test
        @DisplayName("Should return true when directory already exists")
        fun shouldReturnTrueWhenDirectoryAlreadyExists() {
            val result = FileUtils.createDirectory(testDirectory.path)
            
            assertTrue(result)
        }

        @Test
        @DisplayName("Should handle null path when creating directory")
        fun shouldHandleNullPathWhenCreatingDirectory() {
            assertFalse(FileUtils.createDirectory(null))
        }

        @Test
        @DisplayName("Should handle empty path when creating directory")
        fun shouldHandleEmptyPathWhenCreatingDirectory() {
            assertFalse(FileUtils.createDirectory(""))
        }

        @Test
        @DisplayName("Should list directory contents successfully")
        fun shouldListDirectoryContentsSuccessfully() {
            val file1 = tempDir.resolve("file1.txt").toFile()
            val file2 = tempDir.resolve("file2.txt").toFile()
            file1.createNewFile()
            file2.createNewFile()
            
            val contents = FileUtils.listDirectory(tempDir.toString())
            
            assertEquals(3, contents.size) // includes testDir
            assertTrue(contents.contains("file1.txt"))
            assertTrue(contents.contains("file2.txt"))
        }

        @Test
        @DisplayName("Should return empty list for empty directory")
        fun shouldReturnEmptyListForEmptyDirectory() {
            val emptyDir = tempDir.resolve("emptyDir").toFile()
            emptyDir.mkdirs()
            
            val contents = FileUtils.listDirectory(emptyDir.path)
            
            assertTrue(contents.isEmpty())
        }

        @Test
        @DisplayName("Should throw exception when listing non-existent directory")
        fun shouldThrowExceptionWhenListingNonExistentDirectory() {
            assertFailsWith<IOException> {
                FileUtils.listDirectory("nonexistent")
            }
        }

        @Test
        @DisplayName("Should throw exception when listing file instead of directory")
        fun shouldThrowExceptionWhenListingFileInsteadOfDirectory() {
            testFile.createNewFile()
            
            assertFailsWith<IOException> {
                FileUtils.listDirectory(testFile.path)
            }
        }
    }

    @Nested
    @DisplayName("File Information Tests")
    inner class FileInformationTests {

        @Test
        @DisplayName("Should get file size correctly")
        fun shouldGetFileSizeCorrectly() {
            val content = "Hello, World!"
            testFile.writeText(content)
            
            val size = FileUtils.getFileSize(testFile.path)
            
            assertEquals(content.length.toLong(), size)
        }

        @Test
        @DisplayName("Should return zero for empty file")
        fun shouldReturnZeroForEmptyFile() {
            testFile.createNewFile()
            
            val size = FileUtils.getFileSize(testFile.path)
            
            assertEquals(0L, size)
        }

        @Test
        @DisplayName("Should throw exception when getting size of non-existent file")
        fun shouldThrowExceptionWhenGettingSizeOfNonExistentFile() {
            assertFailsWith<IOException> {
                FileUtils.getFileSize("nonexistent.txt")
            }
        }

        @Test
        @DisplayName("Should get file extension correctly")
        fun shouldGetFileExtensionCorrectly() {
            val extension = FileUtils.getFileExtension("document.pdf")
            assertEquals("pdf", extension)
        }

        @Test
        @DisplayName("Should return empty string for file without extension")
        fun shouldReturnEmptyStringForFileWithoutExtension() {
            val extension = FileUtils.getFileExtension("README")
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should handle file with multiple dots")
        fun shouldHandleFileWithMultipleDots() {
            val extension = FileUtils.getFileExtension("archive.tar.gz")
            assertEquals("gz", extension)
        }

        @Test
        @DisplayName("Should handle hidden files")
        fun shouldHandleHiddenFiles() {
            val extension = FileUtils.getFileExtension(".gitignore")
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should handle null filename")
        fun shouldHandleNullFilename() {
            val extension = FileUtils.getFileExtension(null)
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should handle empty filename")
        fun shouldHandleEmptyFilename() {
            val extension = FileUtils.getFileExtension("")
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should check if file is readable")
        fun shouldCheckIfFileIsReadable() {
            testFile.createNewFile()
            
            assertTrue(FileUtils.isReadable(testFile.path))
        }

        @Test
        @DisplayName("Should check if file is writable")
        fun shouldCheckIfFileIsWritable() {
            testFile.createNewFile()
            
            assertTrue(FileUtils.isWritable(testFile.path))
        }

        @Test
        @DisplayName("Should return false for non-existent file permissions")
        fun shouldReturnFalseForNonExistentFilePermissions() {
            assertFalse(FileUtils.isReadable("nonexistent.txt"))
            assertFalse(FileUtils.isWritable("nonexistent.txt"))
        }
    }

    @Nested
    @DisplayName("Path Manipulation Tests")
    inner class PathManipulationTests {

        @Test
        @DisplayName("Should normalize path correctly")
        fun shouldNormalizePathCorrectly() {
            val path = "./folder/../file.txt"
            val normalized = FileUtils.normalizePath(path)
            assertEquals("file.txt", normalized)
        }

        @Test
        @DisplayName("Should join paths correctly")
        fun shouldJoinPathsCorrectly() {
            val joined = FileUtils.joinPaths("folder", "subfolder", "file.txt")
            val expected = Paths.get("folder", "subfolder", "file.txt").toString()
            assertEquals(expected, joined)
        }

        @Test
        @DisplayName("Should get parent directory correctly")
        fun shouldGetParentDirectoryCorrectly() {
            val parent = FileUtils.getParentDirectory("/path/to/file.txt")
            assertEquals("/path/to", parent)
        }

        @Test
        @DisplayName("Should get filename from path")
        fun shouldGetFilenameFromPath() {
            val filename = FileUtils.getFileName("/path/to/file.txt")
            assertEquals("file.txt", filename)
        }

        @Test
        @DisplayName("Should handle root path")
        fun shouldHandleRootPath() {
            val parent = FileUtils.getParentDirectory("/")
            assertNull(parent)
        }

        @Test
        @DisplayName("Should handle relative paths")
        fun shouldHandleRelativePaths() {
            val parent = FileUtils.getParentDirectory("file.txt")
            assertEquals(".", parent)
        }
    }

    @Nested
    @DisplayName("Error Handling and Edge Cases")
    inner class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle very long file paths")
        fun shouldHandleVeryLongFilePaths() {
            val longPath = "a".repeat(1000) + ".txt"
            // This should not throw an exception, but behavior depends on OS
            assertDoesNotThrow {
                FileUtils.exists(longPath)
            }
        }

        @Test
        @DisplayName("Should handle special characters in file names")
        fun shouldHandleSpecialCharactersInFileNames() {
            val specialFile = tempDir.resolve("file with spaces & symbols!@#.txt").toFile()
            val content = "Special content"
            
            FileUtils.writeFile(specialFile.path, content)
            
            assertTrue(specialFile.exists())
            assertEquals(content, FileUtils.readFile(specialFile.path))
        }

        @Test
        @DisplayName("Should handle concurrent file operations")
        fun shouldHandleConcurrentFileOperations() {
            val file1 = tempDir.resolve("concurrent1.txt").toFile()
            val file2 = tempDir.resolve("concurrent2.txt").toFile()
            
            // Simulate concurrent operations
            val thread1 = Thread {
                FileUtils.writeFile(file1.path, "Content 1")
            }
            val thread2 = Thread {
                FileUtils.writeFile(file2.path, "Content 2")
            }
            
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            
            assertTrue(file1.exists())
            assertTrue(file2.exists())
            assertEquals("Content 1", FileUtils.readFile(file1.path))
            assertEquals("Content 2", FileUtils.readFile(file2.path))
        }

        @Test
        @DisplayName("Should handle large file operations")
        fun shouldHandleLargeFileOperations() {
            val largeContent = "Large content line\n".repeat(10000)
            val largeFile = tempDir.resolve("large.txt").toFile()
            
            FileUtils.writeFile(largeFile.path, largeContent)
            val readContent = FileUtils.readFile(largeFile.path)
            
            assertEquals(largeContent, readContent)
        }

        @Test
        @DisplayName("Should handle file system limitations gracefully")
        fun shouldHandleFileSystemLimitationsGracefully() {
            // Test with invalid characters (varies by OS)
            val invalidChars = listOf("<", ">", ":", "\"", "|", "?", "*")
            
            for (char in invalidChars) {
                val filename = "invalid${char}file.txt"
                // Should not crash, but may fail depending on OS
                assertDoesNotThrow {
                    try {
                        FileUtils.writeFile(filename, "content")
                    } catch (e: IOException) {
                        // Expected on some systems
                    }
                }
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should perform file operations within reasonable time")
        fun shouldPerformFileOperationsWithinReasonableTime() {
            val startTime = System.currentTimeMillis()
            
            // Perform multiple operations
            repeat(100) { i ->
                val file = tempDir.resolve("perf_test_$i.txt").toFile()
                FileUtils.writeFile(file.path, "Performance test content $i")
                FileUtils.readFile(file.path)
                FileUtils.deleteFile(file.path)
            }
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            // Should complete within 5 seconds (adjust as needed)
            assertTrue(duration < 5000, "File operations took too long: ${duration}ms")
        }
    }
}