package dev.aurakai.auraframefx.system.utils

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("FileUtils Tests")
class FileUtilsTest {

    @TempDir
    lateinit var tempDir: Path

    private lateinit var testFile: File
    private lateinit var testDir: File

    @BeforeEach
    fun setUp() {
        testFile = tempDir.resolve("test.txt").toFile()
        testDir = tempDir.resolve("testDir").toFile()
        testDir.mkdirs()
    }

    @AfterEach
    fun tearDown() {
        // Clean up is handled by @TempDir
    }

    @Nested
    @DisplayName("File Creation Tests")
    inner class FileCreationTests {

        @Test
        @DisplayName("Should create file successfully")
        fun testCreateFile() {
            val newFile = tempDir.resolve("newFile.txt").toFile()
            assertFalse(newFile.exists())

            val result = FileUtils.createFile(newFile.path)

            assertTrue(result)
            assertTrue(newFile.exists())
            assertTrue(newFile.isFile)
        }

        @Test
        @DisplayName("Should not create file if already exists")
        fun testCreateFileAlreadyExists() {
            testFile.createNewFile()
            assertTrue(testFile.exists())

            val result = FileUtils.createFile(testFile.path)

            assertFalse(result)
        }

        @Test
        @DisplayName("Should handle invalid file path gracefully")
        fun testCreateFileInvalidPath() {
            val invalidPath = "/invalid/path/\u0000/file.txt"

            val result = FileUtils.createFile(invalidPath)

            assertFalse(result)
        }

        @Test
        @DisplayName("Should create parent directories if needed")
        fun testCreateFileWithParentDirectories() {
            val nestedFile = tempDir.resolve("level1/level2/nested.txt").toFile()
            assertFalse(nestedFile.parentFile.exists())

            val result = FileUtils.createFile(nestedFile.path)

            assertTrue(result)
            assertTrue(nestedFile.exists())
            assertTrue(nestedFile.parentFile.exists())
        }
    }

    @Nested
    @DisplayName("File Deletion Tests")
    inner class FileDeletionTests {

        @Test
        @DisplayName("Should delete file successfully")
        fun testDeleteFile() {
            testFile.createNewFile()
            assertTrue(testFile.exists())

            val result = FileUtils.deleteFile(testFile.path)

            assertTrue(result)
            assertFalse(testFile.exists())
        }

        @Test
        @DisplayName("Should return false when deleting non-existent file")
        fun testDeleteNonExistentFile() {
            val nonExistentFile = tempDir.resolve("nonExistent.txt").toFile()
            assertFalse(nonExistentFile.exists())

            val result = FileUtils.deleteFile(nonExistentFile.path)

            assertFalse(result)
        }

        @Test
        @DisplayName("Should delete directory recursively")
        fun testDeleteDirectoryRecursively() {
            val subDir = File(testDir, "subDir")
            subDir.mkdirs()
            val fileInSubDir = File(subDir, "file.txt")
            fileInSubDir.createNewFile()

            val result = FileUtils.deleteFile(testDir.path)

            assertTrue(result)
            assertFalse(testDir.exists())
        }

        @Test
        @DisplayName("Should handle permission issues gracefully")
        fun testDeleteFilePermissionDenied() {
            if (System.getProperty("os.name").lowercase().contains("windows")) {
                return
            }

            testFile.createNewFile()
            testFile.setReadOnly()

            val result = FileUtils.deleteFile(testFile.path)

            if (result) {
                assertFalse(testFile.exists())
            } else {
                assertTrue(testFile.exists())
            }
        }
    }

    @Nested
    @DisplayName("File Reading Tests")
    inner class FileReadingTests {

        @Test
        @DisplayName("Should read file content successfully")
        fun testReadFileContent() {
            val content = "Hello, World!\nThis is a test file."
            testFile.writeText(content)

            val result = FileUtils.readFile(testFile.path)

            assertEquals(content, result)
        }

        @Test
        @DisplayName("Should read empty file successfully")
        fun testReadEmptyFile() {
            testFile.createNewFile()

            val result = FileUtils.readFile(testFile.path)

            assertEquals("", result)
        }

        @Test
        @DisplayName("Should handle non-existent file gracefully")
        fun testReadNonExistentFile() {
            val nonExistentFile = tempDir.resolve("nonExistent.txt").toFile()

            val result = FileUtils.readFile(nonExistentFile.path)

            assertNull(result)
        }

        @Test
        @DisplayName("Should handle large file reading")
        fun testReadLargeFile() {
            val largeContent = "A".repeat(1000000)
            testFile.writeText(largeContent)

            val result = FileUtils.readFile(testFile.path)

            assertEquals(largeContent, result)
        }

        @Test
        @DisplayName("Should handle binary file reading")
        fun testReadBinaryFile() {
            val binaryData = byteArrayOf(0x00, 0x01, 0x02, 0x03, 0xFF.toByte())
            testFile.writeBytes(binaryData)

            val result = FileUtils.readFileBytes(testFile.path)

            assertArrayEquals(binaryData, result)
        }

        @Test
        @DisplayName("Should handle different text encodings")
        fun testReadFileWithEncoding() {
            val unicodeContent = "Hello 世界! 🌍"
            testFile.writeText(unicodeContent, Charsets.UTF_8)

            val result = FileUtils.readFile(testFile.path, Charsets.UTF_8)

            assertEquals(unicodeContent, result)
        }
    }

    @Nested
    @DisplayName("File Writing Tests")
    inner class FileWritingTests {

        @Test
        @DisplayName("Should write to file successfully")
        fun testWriteToFile() {
            val content = "Test content to write"

            val result = FileUtils.writeFile(testFile.path, content)

            assertTrue(result)
            assertEquals(content, testFile.readText())
        }

        @Test
        @DisplayName("Should overwrite existing file")
        fun testOverwriteExistingFile() {
            val originalContent = "Original content"
            val newContent = "New content"
            testFile.writeText(originalContent)

            val result = FileUtils.writeFile(testFile.path, newContent)

            assertTrue(result)
            assertEquals(newContent, testFile.readText())
        }

        @Test
        @DisplayName("Should append to file")
        fun testAppendToFile() {
            val originalContent = "Original content"
            val appendContent = "\nAppended content"
            testFile.writeText(originalContent)

            val result = FileUtils.appendToFile(testFile.path, appendContent)

            assertTrue(result)
            assertEquals(originalContent + appendContent, testFile.readText())
        }

        @Test
        @DisplayName("Should handle empty content")
        fun testWriteEmptyContent() {
            val result = FileUtils.writeFile(testFile.path, "")

            assertTrue(result)
            assertEquals("", testFile.readText())
        }

        @Test
        @DisplayName("Should handle null content gracefully")
        fun testWriteNullContent() {
            val result = FileUtils.writeFile(testFile.path, null)

            assertFalse(result)
        }

        @Test
        @DisplayName("Should create parent directories when writing")
        fun testWriteFileWithParentDirectories() {
            val nestedFile = tempDir.resolve("level1/level2/nested.txt").toFile()
            val content = "Nested file content"

            val result = FileUtils.writeFile(nestedFile.path, content)

            assertTrue(result)
            assertTrue(nestedFile.exists())
            assertEquals(content, nestedFile.readText())
        }

        @Test
        @DisplayName("Should handle different text encodings for writing")
        fun testWriteFileWithEncoding() {
            val unicodeContent = "Hello 世界! 🌍"

            val result = FileUtils.writeFile(testFile.path, unicodeContent, Charsets.UTF_8)

            assertTrue(result)
            assertEquals(unicodeContent, testFile.readText(Charsets.UTF_8))
        }
    }

    @Nested
    @DisplayName("File Information Tests")
    inner class FileInformationTests {

        @Test
        @DisplayName("Should get file size correctly")
        fun testGetFileSize() {
            val content = "Test content"
            testFile.writeText(content)

            val size = FileUtils.getFileSize(testFile.path)

            assertEquals(content.toByteArray().size.toLong(), size)
        }

        @Test
        @DisplayName("Should return -1 for non-existent file size")
        fun testGetFileSizeNonExistent() {
            val nonExistentFile = tempDir.resolve("nonExistent.txt").toFile()

            val size = FileUtils.getFileSize(nonExistentFile.path)

            assertEquals(-1L, size)
        }

        @Test
        @DisplayName("Should check if file exists")
        fun testFileExists() {
            assertFalse(FileUtils.fileExists(testFile.path))

            testFile.createNewFile()

            assertTrue(FileUtils.fileExists(testFile.path))
        }

        @Test
        @DisplayName("Should check if file is directory")
        fun testIsDirectory() {
            assertFalse(FileUtils.isDirectory(testFile.path))
            assertTrue(FileUtils.isDirectory(testDir.path))
        }

        @Test
        @DisplayName("Should check if file is readable")
        fun testIsReadable() {
            testFile.createNewFile()

            assertTrue(FileUtils.isReadable(testFile.path))
        }

        @Test
        @DisplayName("Should check if file is writable")
        fun testIsWritable() {
            testFile.createNewFile()

            assertTrue(FileUtils.isWritable(testFile.path))
        }

        @Test
        @DisplayName("Should get file extension")
        fun testGetFileExtension() {
            val txtFile = tempDir.resolve("test.txt").toFile()
            val noExtFile = tempDir.resolve("test").toFile()
            val multiExtFile = tempDir.resolve("test.tar.gz").toFile()

            assertEquals("txt", FileUtils.getFileExtension(txtFile.path))
            assertEquals("", FileUtils.getFileExtension(noExtFile.path))
            assertEquals("gz", FileUtils.getFileExtension(multiExtFile.path))
        }

        @Test
        @DisplayName("Should get file name without extension")
        fun testGetFileNameWithoutExtension() {
            val txtFile = tempDir.resolve("test.txt").toFile()
            val noExtFile = tempDir.resolve("test").toFile()

            assertEquals("test", FileUtils.getFileNameWithoutExtension(txtFile.path))
            assertEquals("test", FileUtils.getFileNameWithoutExtension(noExtFile.path))
        }
    }

    @Nested
    @DisplayName("File Utility Tests")
    inner class FileUtilityTests {

        @Test
        @DisplayName("Should copy file successfully")
        fun testCopyFile() {
            val content = "Content to copy"
            testFile.writeText(content)
            val destinationFile = tempDir.resolve("copied.txt").toFile()

            val result = FileUtils.copyFile(testFile.path, destinationFile.path)

            assertTrue(result)
            assertTrue(destinationFile.exists())
            assertEquals(content, destinationFile.readText())
        }

        @Test
        @DisplayName("Should move file successfully")
        fun testMoveFile() {
            val content = "Content to move"
            testFile.writeText(content)
            val destinationFile = tempDir.resolve("moved.txt").toFile()

            val result = FileUtils.moveFile(testFile.path, destinationFile.path)

            assertTrue(result)
            assertFalse(testFile.exists())
            assertTrue(destinationFile.exists())
            assertEquals(content, destinationFile.readText())
        }

        @Test
        @DisplayName("Should calculate file hash")
        fun testCalculateFileHash() {
            val content = "Content for hashing"
            testFile.writeText(content)

            val hash = FileUtils.calculateFileHash(testFile.path)

            assertNotNull(hash)
            assertTrue(hash.isNotEmpty())

            val hash2 = FileUtils.calculateFileHash(testFile.path)
            assertEquals(hash, hash2)
        }

        @Test
        @DisplayName("Should calculate different hashes for different content")
        fun testCalculateFileHashDifferentContent() {
            val content1 = "Content 1"
            val content2 = "Content 2"
            testFile.writeText(content1)
            val otherFile = tempDir.resolve("other.txt").toFile()
            otherFile.writeText(content2)

            val hash1 = FileUtils.calculateFileHash(testFile.path)
            val hash2 = FileUtils.calculateFileHash(otherFile.path)

            assertNotEquals(hash1, hash2)
        }

        @Test
        @DisplayName("Should list files in directory")
        fun testListFiles() {
            val file1 = File(testDir, "file1.txt")
            val file2 = File(testDir, "file2.txt")
            val subDir = File(testDir, "subDir")

            file1.createNewFile()
            file2.createNewFile()
            subDir.mkdirs()

            val files = FileUtils.listFiles(testDir.path)

            assertEquals(3, files.size)
            assertTrue(files.contains(file1.name))
            assertTrue(files.contains(file2.name))
            assertTrue(files.contains(subDir.name))
        }

        @Test
        @DisplayName("Should filter files by extension")
        fun testListFilesByExtension() {
            val txtFile = File(testDir, "test.txt")
            val logFile = File(testDir, "test.log")
            val noExtFile = File(testDir, "test")

            txtFile.createNewFile()
            logFile.createNewFile()
            noExtFile.createNewFile()

            val txtFiles = FileUtils.listFilesByExtension(testDir.path, "txt")

            assertEquals(1, txtFiles.size)
            assertTrue(txtFiles.contains(txtFile.name))
        }

        @Test
        @DisplayName("Should create directory structure")
        fun testCreateDirectoryStructure() {
            val nestedPath = tempDir.resolve("level1/level2/level3").toString()

            val result = FileUtils.createDirectoryStructure(nestedPath)

            assertTrue(result)
            assertTrue(File(nestedPath).exists())
            assertTrue(File(nestedPath).isDirectory)
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCaseTests {

        @Test
        @DisplayName("Should handle null file path gracefully")
        fun testNullFilePath() {
            assertFalse(FileUtils.fileExists(null))
            assertFalse(FileUtils.createFile(null))
            assertFalse(FileUtils.deleteFile(null))
            assertNull(FileUtils.readFile(null))
            assertFalse(FileUtils.writeFile(null, "content"))
        }

        @Test
        @DisplayName("Should handle empty file path gracefully")
        fun testEmptyFilePath() {
            assertFalse(FileUtils.fileExists(""))
            assertFalse(FileUtils.createFile(""))
            assertFalse(FileUtils.deleteFile(""))
            assertNull(FileUtils.readFile(""))
            assertFalse(FileUtils.writeFile("", "content"))
        }

        @Test
        @DisplayName("Should handle whitespace-only file path")
        fun testWhitespaceFilePath() {
            val whitespacePath = "   \t\n   "
            assertFalse(FileUtils.fileExists(whitespacePath))
            assertFalse(FileUtils.createFile(whitespacePath))
        }

        @Test
        @DisplayName("Should handle very long file path")
        fun testVeryLongFilePath() {
            val longPath = "a".repeat(1000) + ".txt"
            val result = FileUtils.createFile(longPath)

            assertNotNull(result)
        }

        @Test
        @DisplayName("Should handle special characters in file path")
        fun testSpecialCharactersInPath() {
            val specialChars = "file with spaces & symbols!@#$%^&()_+.txt"
            val specialFile = tempDir.resolve(specialChars).toFile()

            val result = FileUtils.createFile(specialFile.path)

            assertNotNull(result)
        }

        @Test
        @DisplayName("Should handle concurrent file access")
        fun testConcurrentFileAccess() {
            val content = "Concurrent access test"
            testFile.writeText(content)

            val results = mutableListOf<String?>()
            val threads = mutableListOf<Thread>()

            repeat(10) {
                val thread = Thread {
                    results.add(FileUtils.readFile(testFile.path))
                }
                threads.add(thread)
                thread.start()
            }

            threads.forEach { it.join() }

            assertEquals(10, results.size)
            results.forEach { assertEquals(content, it) }
        }

        @Test
        @DisplayName("Should handle file system case sensitivity")
        fun testCaseSensitivity() {
            val lowerCaseFile = tempDir.resolve("test.txt").toFile()
            val upperCaseFile = tempDir.resolve("TEST.txt").toFile()

            lowerCaseFile.createNewFile()

            val lowerExists = FileUtils.fileExists(lowerCaseFile.path)
            val upperExists = FileUtils.fileExists(upperCaseFile.path)

            assertTrue(lowerExists)

            if (System.getProperty("os.name").lowercase().contains("windows")) {
                assertTrue(upperExists)
            } else {
                assertFalse(upperExists)
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle multiple file operations efficiently")
        fun testMultipleFileOperations() {
            val startTime = System.currentTimeMillis()

            repeat(100) { i ->
                val file = tempDir.resolve("perf_test_$i.txt").toFile()
                FileUtils.createFile(file.path)
                FileUtils.writeFile(file.path, "Performance test content $i")
                FileUtils.readFile(file.path)
                FileUtils.deleteFile(file.path)
            }

            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime

            assertTrue(duration < 5000, "Operations took too long: ${duration}ms")
        }

        @Test
        @DisplayName("Should handle large file operations")
        fun testLargeFileOperations() {
            val largeContent = "Large content line\n".repeat(10000)
            val startTime = System.currentTimeMillis()

            FileUtils.writeFile(testFile.path, largeContent)
            val readContent = FileUtils.readFile(testFile.path)

            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime

            assertEquals(largeContent, readContent)
            assertTrue(duration < 2000, "Large file operations took too long: ${duration}ms")
        }
    }

    @Nested
    @DisplayName("Security Tests")
    inner class SecurityTests {

        @Test
        @DisplayName("Should handle path traversal attempts")
        fun testPathTraversalSecurity() {
            val maliciousPath = "../../../etc/passwd"

            val result = FileUtils.readFile(maliciousPath)

            assertNull(result)
        }

        @Test
        @DisplayName("Should validate file permissions")
        fun testFilePermissionValidation() {
            if (System.getProperty("os.name").lowercase().contains("windows")) {
                return
            }

            testFile.createNewFile()
            testFile.setReadOnly()

            val isWritable = FileUtils.isWritable(testFile.path)

            assertFalse(isWritable)
        }

        @Test
        @DisplayName("Should handle symlink security")
        fun testSymlinkSecurity() {
            if (System.getProperty("os.name").lowercase().contains("windows")) {
                return
            }

            val targetFile = tempDir.resolve("target.txt").toFile()
            targetFile.writeText("Target content")

            val symlinkFile = tempDir.resolve("symlink.txt")

            try {
                Files.createSymbolicLink(symlinkFile, targetFile.toPath())

                val content = FileUtils.readFile(symlinkFile.toString())

                assertNotNull(content)
            } catch (e: Exception) {
                // Symlink creation may fail in some environments
            }
        }
    }
}