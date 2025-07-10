package dev.aurakai.auraframefx.system.utils

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir
import org.mockito.kotlin.*
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import kotlin.io.path.createTempDirectory
import kotlin.io.path.createTempFile
import kotlin.io.path.exists
import kotlin.io.path.writeText

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("FileUtils Tests")
class FileUtilsTest {

    @TempDir
    lateinit var tempDir: Path

    private lateinit var testFile: File
    private lateinit var testDirectory: File

    @BeforeEach
    fun setUp() {
        testFile = File(tempDir.toFile(), "test.txt")
        testDirectory = File(tempDir.toFile(), "testDir")
        testDirectory.mkdirs()
    }

    @AfterEach
    fun tearDown() {
        // Clean up any remaining files
        tempDir.toFile().deleteRecursively()
    }

    @Nested
    @DisplayName("File Existence Checks")
    inner class FileExistenceTests {

        @Test
        @DisplayName("Should return true when file exists")
        fun shouldReturnTrueWhenFileExists() {
            testFile.createNewFile()
            assertTrue(FileUtils.exists(testFile.absolutePath))
        }

        @Test
        @DisplayName("Should return false when file does not exist")
        fun shouldReturnFalseWhenFileDoesNotExist() {
            assertFalse(FileUtils.exists("nonexistent.txt"))
        }

        @Test
        @DisplayName("Should return true when directory exists")
        fun shouldReturnTrueWhenDirectoryExists() {
            assertTrue(FileUtils.exists(testDirectory.absolutePath))
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
    @DisplayName("File Creation Tests")
    inner class FileCreationTests {

        @Test
        @DisplayName("Should create new file successfully")
        fun shouldCreateNewFileSuccessfully() {
            val newFile = File(tempDir.toFile(), "newfile.txt")
            assertTrue(FileUtils.createFile(newFile.absolutePath))
            assertTrue(newFile.exists())
        }

        @Test
        @DisplayName("Should create parent directories when creating file")
        fun shouldCreateParentDirectoriesWhenCreatingFile() {
            val nestedFile = File(tempDir.toFile(), "nested/dir/file.txt")
            assertTrue(FileUtils.createFile(nestedFile.absolutePath))
            assertTrue(nestedFile.exists())
            assertTrue(nestedFile.parentFile.exists())
        }

        @Test
        @DisplayName("Should return false when file already exists")
        fun shouldReturnFalseWhenFileAlreadyExists() {
            testFile.createNewFile()
            assertFalse(FileUtils.createFile(testFile.absolutePath))
        }

        @Test
        @DisplayName("Should handle invalid file path gracefully")
        fun shouldHandleInvalidFilePathGracefully() {
            assertFalse(FileUtils.createFile(""))
        }

        @Test
        @DisplayName("Should handle null file path gracefully")
        fun shouldHandleNullFilePathGracefully() {
            assertFalse(FileUtils.createFile(null))
        }

        @Test
        @DisplayName("Should handle read-only directory gracefully")
        fun shouldHandleReadOnlyDirectoryGracefully() {
            val readOnlyDir = File(tempDir.toFile(), "readonly")
            readOnlyDir.mkdirs()
            readOnlyDir.setReadOnly()
            
            val fileInReadOnlyDir = File(readOnlyDir, "file.txt")
            assertFalse(FileUtils.createFile(fileInReadOnlyDir.absolutePath))
        }
    }

    @Nested
    @DisplayName("File Deletion Tests")
    inner class FileDeletionTests {

        @Test
        @DisplayName("Should delete existing file successfully")
        fun shouldDeleteExistingFileSuccessfully() {
            testFile.createNewFile()
            assertTrue(FileUtils.deleteFile(testFile.absolutePath))
            assertFalse(testFile.exists())
        }

        @Test
        @DisplayName("Should return false when deleting non-existent file")
        fun shouldReturnFalseWhenDeletingNonExistentFile() {
            assertFalse(FileUtils.deleteFile("nonexistent.txt"))
        }

        @Test
        @DisplayName("Should delete directory successfully")
        fun shouldDeleteDirectorySuccessfully() {
            assertTrue(FileUtils.deleteFile(testDirectory.absolutePath))
            assertFalse(testDirectory.exists())
        }

        @Test
        @DisplayName("Should delete directory with contents recursively")
        fun shouldDeleteDirectoryWithContentsRecursively() {
            val subFile = File(testDirectory, "subfile.txt")
            val subDir = File(testDirectory, "subdir")
            subDir.mkdirs()
            subFile.createNewFile()
            
            assertTrue(FileUtils.deleteFile(testDirectory.absolutePath))
            assertFalse(testDirectory.exists())
        }

        @Test
        @DisplayName("Should handle null path gracefully")
        fun shouldHandleNullPathGracefullyForDeletion() {
            assertFalse(FileUtils.deleteFile(null))
        }

        @Test
        @DisplayName("Should handle empty path gracefully")
        fun shouldHandleEmptyPathGracefullyForDeletion() {
            assertFalse(FileUtils.deleteFile(""))
        }
    }

    @Nested
    @DisplayName("File Copy Tests")
    inner class FileCopyTests {

        @Test
        @DisplayName("Should copy file successfully")
        fun shouldCopyFileSuccessfully() {
            testFile.createNewFile()
            testFile.writeText("Hello, World!")
            
            val targetFile = File(tempDir.toFile(), "copied.txt")
            assertTrue(FileUtils.copyFile(testFile.absolutePath, targetFile.absolutePath))
            assertTrue(targetFile.exists())
            assertEquals("Hello, World!", targetFile.readText())
        }

        @Test
        @DisplayName("Should overwrite existing target file")
        fun shouldOverwriteExistingTargetFile() {
            testFile.createNewFile()
            testFile.writeText("Original content")
            
            val targetFile = File(tempDir.toFile(), "target.txt")
            targetFile.createNewFile()
            targetFile.writeText("Target content")
            
            assertTrue(FileUtils.copyFile(testFile.absolutePath, targetFile.absolutePath))
            assertEquals("Original content", targetFile.readText())
        }

        @Test
        @DisplayName("Should create parent directories for target file")
        fun shouldCreateParentDirectoriesForTargetFile() {
            testFile.createNewFile()
            testFile.writeText("Content")
            
            val targetFile = File(tempDir.toFile(), "nested/deep/target.txt")
            assertTrue(FileUtils.copyFile(testFile.absolutePath, targetFile.absolutePath))
            assertTrue(targetFile.exists())
            assertEquals("Content", targetFile.readText())
        }

        @Test
        @DisplayName("Should return false when source file does not exist")
        fun shouldReturnFalseWhenSourceFileDoesNotExist() {
            val targetFile = File(tempDir.toFile(), "target.txt")
            assertFalse(FileUtils.copyFile("nonexistent.txt", targetFile.absolutePath))
        }

        @Test
        @DisplayName("Should handle null source path gracefully")
        fun shouldHandleNullSourcePathGracefully() {
            val targetFile = File(tempDir.toFile(), "target.txt")
            assertFalse(FileUtils.copyFile(null, targetFile.absolutePath))
        }

        @Test
        @DisplayName("Should handle null target path gracefully")
        fun shouldHandleNullTargetPathGracefully() {
            testFile.createNewFile()
            assertFalse(FileUtils.copyFile(testFile.absolutePath, null))
        }

        @Test
        @DisplayName("Should handle empty paths gracefully")
        fun shouldHandleEmptyPathsGracefully() {
            assertFalse(FileUtils.copyFile("", ""))
        }
    }

    @Nested
    @DisplayName("File Move Tests")
    inner class FileMoveTests {

        @Test
        @DisplayName("Should move file successfully")
        fun shouldMoveFileSuccessfully() {
            testFile.createNewFile()
            testFile.writeText("Move me!")
            
            val targetFile = File(tempDir.toFile(), "moved.txt")
            assertTrue(FileUtils.moveFile(testFile.absolutePath, targetFile.absolutePath))
            assertFalse(testFile.exists())
            assertTrue(targetFile.exists())
            assertEquals("Move me!", targetFile.readText())
        }

        @Test
        @DisplayName("Should rename file in same directory")
        fun shouldRenameFileInSameDirectory() {
            testFile.createNewFile()
            testFile.writeText("Rename me!")
            
            val renamedFile = File(tempDir.toFile(), "renamed.txt")
            assertTrue(FileUtils.moveFile(testFile.absolutePath, renamedFile.absolutePath))
            assertFalse(testFile.exists())
            assertTrue(renamedFile.exists())
            assertEquals("Rename me!", renamedFile.readText())
        }

        @Test
        @DisplayName("Should create parent directories for target")
        fun shouldCreateParentDirectoriesForTarget() {
            testFile.createNewFile()
            testFile.writeText("Content")
            
            val targetFile = File(tempDir.toFile(), "nested/target.txt")
            assertTrue(FileUtils.moveFile(testFile.absolutePath, targetFile.absolutePath))
            assertFalse(testFile.exists())
            assertTrue(targetFile.exists())
        }

        @Test
        @DisplayName("Should return false when source file does not exist")
        fun shouldReturnFalseWhenSourceFileDoesNotExistForMove() {
            val targetFile = File(tempDir.toFile(), "target.txt")
            assertFalse(FileUtils.moveFile("nonexistent.txt", targetFile.absolutePath))
        }

        @Test
        @DisplayName("Should handle null paths gracefully")
        fun shouldHandleNullPathsGracefullyForMove() {
            assertFalse(FileUtils.moveFile(null, null))
            assertFalse(FileUtils.moveFile(testFile.absolutePath, null))
            assertFalse(FileUtils.moveFile(null, "target.txt"))
        }
    }

    @Nested
    @DisplayName("File Reading Tests")
    inner class FileReadingTests {

        @Test
        @DisplayName("Should read file content successfully")
        fun shouldReadFileContentSuccessfully() {
            testFile.createNewFile()
            testFile.writeText("Hello, World!")
            
            val content = FileUtils.readFileContent(testFile.absolutePath)
            assertEquals("Hello, World!", content)
        }

        @Test
        @DisplayName("Should read empty file successfully")
        fun shouldReadEmptyFileSuccessfully() {
            testFile.createNewFile()
            
            val content = FileUtils.readFileContent(testFile.absolutePath)
            assertEquals("", content)
        }

        @Test
        @DisplayName("Should handle large file content")
        fun shouldHandleLargeFileContent() {
            testFile.createNewFile()
            val largeContent = "A".repeat(10000)
            testFile.writeText(largeContent)
            
            val content = FileUtils.readFileContent(testFile.absolutePath)
            assertEquals(largeContent, content)
        }

        @Test
        @DisplayName("Should handle non-existent file gracefully")
        fun shouldHandleNonExistentFileGracefully() {
            val content = FileUtils.readFileContent("nonexistent.txt")
            assertNull(content)
        }

        @Test
        @DisplayName("Should handle null path gracefully")
        fun shouldHandleNullPathGracefullyForReading() {
            val content = FileUtils.readFileContent(null)
            assertNull(content)
        }

        @Test
        @DisplayName("Should handle UTF-8 encoded content")
        fun shouldHandleUtf8EncodedContent() {
            testFile.createNewFile()
            val utf8Content = "Hello 世界! 🌍"
            testFile.writeText(utf8Content)
            
            val content = FileUtils.readFileContent(testFile.absolutePath)
            assertEquals(utf8Content, content)
        }

        @Test
        @DisplayName("Should handle files with different line endings")
        fun shouldHandleFilesWithDifferentLineEndings() {
            testFile.createNewFile()
            val contentWithCrlf = "Line 1\r\nLine 2\r\nLine 3"
            testFile.writeText(contentWithCrlf)
            
            val content = FileUtils.readFileContent(testFile.absolutePath)
            assertEquals(contentWithCrlf, content)
        }
    }

    @Nested
    @DisplayName("File Writing Tests")
    inner class FileWritingTests {

        @Test
        @DisplayName("Should write content to file successfully")
        fun shouldWriteContentToFileSuccessfully() {
            val content = "Hello, World!"
            assertTrue(FileUtils.writeFileContent(testFile.absolutePath, content))
            assertTrue(testFile.exists())
            assertEquals(content, testFile.readText())
        }

        @Test
        @DisplayName("Should overwrite existing file content")
        fun shouldOverwriteExistingFileContent() {
            testFile.createNewFile()
            testFile.writeText("Original content")
            
            val newContent = "New content"
            assertTrue(FileUtils.writeFileContent(testFile.absolutePath, newContent))
            assertEquals(newContent, testFile.readText())
        }

        @Test
        @DisplayName("Should create parent directories when writing")
        fun shouldCreateParentDirectoriesWhenWriting() {
            val nestedFile = File(tempDir.toFile(), "nested/dir/file.txt")
            val content = "Content"
            
            assertTrue(FileUtils.writeFileContent(nestedFile.absolutePath, content))
            assertTrue(nestedFile.exists())
            assertEquals(content, nestedFile.readText())
        }

        @Test
        @DisplayName("Should handle empty content")
        fun shouldHandleEmptyContent() {
            assertTrue(FileUtils.writeFileContent(testFile.absolutePath, ""))
            assertTrue(testFile.exists())
            assertEquals("", testFile.readText())
        }

        @Test
        @DisplayName("Should handle null content gracefully")
        fun shouldHandleNullContentGracefully() {
            assertFalse(FileUtils.writeFileContent(testFile.absolutePath, null))
        }

        @Test
        @DisplayName("Should handle null path gracefully")
        fun shouldHandleNullPathGracefullyForWriting() {
            assertFalse(FileUtils.writeFileContent(null, "content"))
        }

        @Test
        @DisplayName("Should handle UTF-8 content correctly")
        fun shouldHandleUtf8ContentCorrectly() {
            val utf8Content = "Hello 世界! 🌍"
            assertTrue(FileUtils.writeFileContent(testFile.absolutePath, utf8Content))
            assertEquals(utf8Content, testFile.readText())
        }

        @Test
        @DisplayName("Should handle large content")
        fun shouldHandleLargeContentForWriting() {
            val largeContent = "A".repeat(50000)
            assertTrue(FileUtils.writeFileContent(testFile.absolutePath, largeContent))
            assertEquals(largeContent, testFile.readText())
        }
    }

    @Nested
    @DisplayName("File Append Tests")
    inner class FileAppendTests {

        @Test
        @DisplayName("Should append content to existing file")
        fun shouldAppendContentToExistingFile() {
            testFile.createNewFile()
            testFile.writeText("Original content")
            
            assertTrue(FileUtils.appendFileContent(testFile.absolutePath, " appended"))
            assertEquals("Original content appended", testFile.readText())
        }

        @Test
        @DisplayName("Should create file if it doesn't exist when appending")
        fun shouldCreateFileIfItDoesNotExistWhenAppending() {
            val newFile = File(tempDir.toFile(), "new.txt")
            assertTrue(FileUtils.appendFileContent(newFile.absolutePath, "New content"))
            assertTrue(newFile.exists())
            assertEquals("New content", newFile.readText())
        }

        @Test
        @DisplayName("Should handle empty content when appending")
        fun shouldHandleEmptyContentWhenAppending() {
            testFile.createNewFile()
            testFile.writeText("Original")
            
            assertTrue(FileUtils.appendFileContent(testFile.absolutePath, ""))
            assertEquals("Original", testFile.readText())
        }

        @Test
        @DisplayName("Should handle null content gracefully when appending")
        fun shouldHandleNullContentGracefullyWhenAppending() {
            assertFalse(FileUtils.appendFileContent(testFile.absolutePath, null))
        }

        @Test
        @DisplayName("Should handle null path gracefully when appending")
        fun shouldHandleNullPathGracefullyWhenAppending() {
            assertFalse(FileUtils.appendFileContent(null, "content"))
        }
    }

    @Nested
    @DisplayName("File Size Tests")
    inner class FileSizeTests {

        @Test
        @DisplayName("Should return correct file size")
        fun shouldReturnCorrectFileSize() {
            testFile.createNewFile()
            val content = "Hello, World!"
            testFile.writeText(content)
            
            val size = FileUtils.getFileSize(testFile.absolutePath)
            assertEquals(content.toByteArray().size.toLong(), size)
        }

        @Test
        @DisplayName("Should return 0 for empty file")
        fun shouldReturnZeroForEmptyFile() {
            testFile.createNewFile()
            
            val size = FileUtils.getFileSize(testFile.absolutePath)
            assertEquals(0L, size)
        }

        @Test
        @DisplayName("Should return -1 for non-existent file")
        fun shouldReturnMinusOneForNonExistentFile() {
            val size = FileUtils.getFileSize("nonexistent.txt")
            assertEquals(-1L, size)
        }

        @Test
        @DisplayName("Should handle null path gracefully")
        fun shouldHandleNullPathGracefullyForSize() {
            val size = FileUtils.getFileSize(null)
            assertEquals(-1L, size)
        }

        @Test
        @DisplayName("Should handle directory size")
        fun shouldHandleDirectorySize() {
            val size = FileUtils.getFileSize(testDirectory.absolutePath)
            assertTrue(size >= 0)
        }
    }

    @Nested
    @DisplayName("File Extension Tests")
    inner class FileExtensionTests {

        @Test
        @DisplayName("Should extract file extension correctly")
        fun shouldExtractFileExtensionCorrectly() {
            val extension = FileUtils.getFileExtension("file.txt")
            assertEquals("txt", extension)
        }

        @Test
        @DisplayName("Should extract extension from path")
        fun shouldExtractExtensionFromPath() {
            val extension = FileUtils.getFileExtension("/path/to/file.java")
            assertEquals("java", extension)
        }

        @Test
        @DisplayName("Should handle file without extension")
        fun shouldHandleFileWithoutExtension() {
            val extension = FileUtils.getFileExtension("filename")
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should handle hidden files")
        fun shouldHandleHiddenFiles() {
            val extension = FileUtils.getFileExtension(".hidden")
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should handle hidden files with extension")
        fun shouldHandleHiddenFilesWithExtension() {
            val extension = FileUtils.getFileExtension(".hidden.txt")
            assertEquals("txt", extension)
        }

        @Test
        @DisplayName("Should handle multiple dots in filename")
        fun shouldHandleMultipleDotsInFilename() {
            val extension = FileUtils.getFileExtension("file.backup.txt")
            assertEquals("txt", extension)
        }

        @Test
        @DisplayName("Should handle null filename gracefully")
        fun shouldHandleNullFilenameGracefully() {
            val extension = FileUtils.getFileExtension(null)
            assertEquals("", extension)
        }

        @Test
        @DisplayName("Should handle empty filename gracefully")
        fun shouldHandleEmptyFilenameGracefully() {
            val extension = FileUtils.getFileExtension("")
            assertEquals("", extension)
        }
    }

    @Nested
    @DisplayName("File Permission Tests")
    inner class FilePermissionTests {

        @Test
        @DisplayName("Should check if file is readable")
        fun shouldCheckIfFileIsReadable() {
            testFile.createNewFile()
            assertTrue(FileUtils.isReadable(testFile.absolutePath))
        }

        @Test
        @DisplayName("Should check if file is writable")
        fun shouldCheckIfFileIsWritable() {
            testFile.createNewFile()
            assertTrue(FileUtils.isWritable(testFile.absolutePath))
        }

        @Test
        @DisplayName("Should check if file is executable")
        fun shouldCheckIfFileIsExecutable() {
            testFile.createNewFile()
            testFile.setExecutable(true)
            assertTrue(FileUtils.isExecutable(testFile.absolutePath))
        }

        @Test
        @DisplayName("Should return false for non-existent file permissions")
        fun shouldReturnFalseForNonExistentFilePermissions() {
            assertFalse(FileUtils.isReadable("nonexistent.txt"))
            assertFalse(FileUtils.isWritable("nonexistent.txt"))
            assertFalse(FileUtils.isExecutable("nonexistent.txt"))
        }

        @Test
        @DisplayName("Should handle null path gracefully for permissions")
        fun shouldHandleNullPathGracefullyForPermissions() {
            assertFalse(FileUtils.isReadable(null))
            assertFalse(FileUtils.isWritable(null))
            assertFalse(FileUtils.isExecutable(null))
        }
    }

    @Nested
    @DisplayName("Directory Tests")
    inner class DirectoryTests {

        @Test
        @DisplayName("Should create directory successfully")
        fun shouldCreateDirectorySuccessfully() {
            val newDir = File(tempDir.toFile(), "newdir")
            assertTrue(FileUtils.createDirectory(newDir.absolutePath))
            assertTrue(newDir.exists())
            assertTrue(newDir.isDirectory)
        }

        @Test
        @DisplayName("Should create nested directories")
        fun shouldCreateNestedDirectories() {
            val nestedDir = File(tempDir.toFile(), "nested/deep/dir")
            assertTrue(FileUtils.createDirectory(nestedDir.absolutePath))
            assertTrue(nestedDir.exists())
            assertTrue(nestedDir.isDirectory)
        }

        @Test
        @DisplayName("Should return true for existing directory")
        fun shouldReturnTrueForExistingDirectory() {
            assertTrue(FileUtils.createDirectory(testDirectory.absolutePath))
        }

        @Test
        @DisplayName("Should list directory contents")
        fun shouldListDirectoryContents() {
            val subFile1 = File(testDirectory, "file1.txt")
            val subFile2 = File(testDirectory, "file2.txt")
            val subDir = File(testDirectory, "subdir")
            
            subFile1.createNewFile()
            subFile2.createNewFile()
            subDir.mkdirs()
            
            val contents = FileUtils.listDirectoryContents(testDirectory.absolutePath)
            assertNotNull(contents)
            assertEquals(3, contents!!.size)
            assertTrue(contents.contains("file1.txt"))
            assertTrue(contents.contains("file2.txt"))
            assertTrue(contents.contains("subdir"))
        }

        @Test
        @DisplayName("Should return empty list for empty directory")
        fun shouldReturnEmptyListForEmptyDirectory() {
            val contents = FileUtils.listDirectoryContents(testDirectory.absolutePath)
            assertNotNull(contents)
            assertTrue(contents!!.isEmpty())
        }

        @Test
        @DisplayName("Should return null for non-existent directory")
        fun shouldReturnNullForNonExistentDirectory() {
            val contents = FileUtils.listDirectoryContents("nonexistent")
            assertNull(contents)
        }

        @Test
        @DisplayName("Should return null when listing file as directory")
        fun shouldReturnNullWhenListingFileAsDirectory() {
            testFile.createNewFile()
            val contents = FileUtils.listDirectoryContents(testFile.absolutePath)
            assertNull(contents)
        }

        @Test
        @DisplayName("Should check if path is directory")
        fun shouldCheckIfPathIsDirectory() {
            assertTrue(FileUtils.isDirectory(testDirectory.absolutePath))
        }

        @Test
        @DisplayName("Should return false for file when checking if directory")
        fun shouldReturnFalseForFileWhenCheckingIfDirectory() {
            testFile.createNewFile()
            assertFalse(FileUtils.isDirectory(testFile.absolutePath))
        }

        @Test
        @DisplayName("Should return false for non-existent path when checking if directory")
        fun shouldReturnFalseForNonExistentPathWhenCheckingIfDirectory() {
            assertFalse(FileUtils.isDirectory("nonexistent"))
        }
    }

    @Nested
    @DisplayName("File Comparison Tests")
    inner class FileComparisonTests {

        @Test
        @DisplayName("Should return true for identical files")
        fun shouldReturnTrueForIdenticalFiles() {
            testFile.createNewFile()
            testFile.writeText("Same content")
            
            val otherFile = File(tempDir.toFile(), "other.txt")
            otherFile.createNewFile()
            otherFile.writeText("Same content")
            
            assertTrue(FileUtils.filesEqual(testFile.absolutePath, otherFile.absolutePath))
        }

        @Test
        @DisplayName("Should return false for different files")
        fun shouldReturnFalseForDifferentFiles() {
            testFile.createNewFile()
            testFile.writeText("Content A")
            
            val otherFile = File(tempDir.toFile(), "other.txt")
            otherFile.createNewFile()
            otherFile.writeText("Content B")
            
            assertFalse(FileUtils.filesEqual(testFile.absolutePath, otherFile.absolutePath))
        }

        @Test
        @DisplayName("Should return false when one file doesn't exist")
        fun shouldReturnFalseWhenOneFileDoesNotExist() {
            testFile.createNewFile()
            assertFalse(FileUtils.filesEqual(testFile.absolutePath, "nonexistent.txt"))
        }

        @Test
        @DisplayName("Should return false when both files don't exist")
        fun shouldReturnFalseWhenBothFilesDoNotExist() {
            assertFalse(FileUtils.filesEqual("nonexistent1.txt", "nonexistent2.txt"))
        }

        @Test
        @DisplayName("Should handle null paths gracefully")
        fun shouldHandleNullPathsGracefullyForComparison() {
            assertFalse(FileUtils.filesEqual(null, null))
            assertFalse(FileUtils.filesEqual(testFile.absolutePath, null))
            assertFalse(FileUtils.filesEqual(null, testFile.absolutePath))
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandlingTests {

        @Test
        @DisplayName("Should handle very long file paths")
        fun shouldHandleVeryLongFilePaths() {
            val longPath = "a".repeat(200) + ".txt"
            val longFile = File(tempDir.toFile(), longPath)
            
            // This may fail on some systems due to path length limits
            // but should handle gracefully
            val result = FileUtils.createFile(longFile.absolutePath)
            // Don't assert the result since it's system-dependent
        }

        @Test
        @DisplayName("Should handle special characters in file names")
        fun shouldHandleSpecialCharactersInFileNames() {
            val specialFile = File(tempDir.toFile(), "file with spaces & symbols!.txt")
            assertTrue(FileUtils.createFile(specialFile.absolutePath))
            assertTrue(specialFile.exists())
        }

        @Test
        @DisplayName("Should handle concurrent file operations")
        fun shouldHandleConcurrentFileOperations() {
            val file1 = File(tempDir.toFile(), "concurrent1.txt")
            val file2 = File(tempDir.toFile(), "concurrent2.txt")
            
            val thread1 = Thread {
                FileUtils.createFile(file1.absolutePath)
                FileUtils.writeFileContent(file1.absolutePath, "Thread 1 content")
            }
            
            val thread2 = Thread {
                FileUtils.createFile(file2.absolutePath)
                FileUtils.writeFileContent(file2.absolutePath, "Thread 2 content")
            }
            
            thread1.start()
            thread2.start()
            
            thread1.join()
            thread2.join()
            
            assertTrue(file1.exists())
            assertTrue(file2.exists())
            assertEquals("Thread 1 content", file1.readText())
            assertEquals("Thread 2 content", file2.readText())
        }

        @Test
        @DisplayName("Should handle IO exceptions gracefully")
        fun shouldHandleIOExceptionsGracefully() {
            // Try to write to a read-only location (this should fail gracefully)
            val readOnlyDir = File(tempDir.toFile(), "readonly")
            readOnlyDir.mkdirs()
            readOnlyDir.setReadOnly()
            
            val fileInReadOnly = File(readOnlyDir, "file.txt")
            assertFalse(FileUtils.writeFileContent(fileInReadOnly.absolutePath, "content"))
        }

        @Test
        @DisplayName("Should handle files with no extension in various operations")
        fun shouldHandleFilesWithNoExtensionInVariousOperations() {
            val noExtFile = File(tempDir.toFile(), "noextension")
            assertTrue(FileUtils.createFile(noExtFile.absolutePath))
            assertTrue(FileUtils.writeFileContent(noExtFile.absolutePath, "content"))
            assertEquals("content", FileUtils.readFileContent(noExtFile.absolutePath))
            assertEquals("", FileUtils.getFileExtension(noExtFile.absolutePath))
        }

        @Test
        @DisplayName("Should handle binary file operations")
        fun shouldHandleBinaryFileOperations() {
            val binaryFile = File(tempDir.toFile(), "binary.bin")
            val binaryData = byteArrayOf(0x00, 0x01, 0x02, 0x03, 0xFF.toByte())
            
            binaryFile.writeBytes(binaryData)
            
            assertTrue(FileUtils.exists(binaryFile.absolutePath))
            assertTrue(FileUtils.getFileSize(binaryFile.absolutePath) > 0)
            
            val copiedFile = File(tempDir.toFile(), "binary_copy.bin")
            assertTrue(FileUtils.copyFile(binaryFile.absolutePath, copiedFile.absolutePath))
            
            assertArrayEquals(binaryData, copiedFile.readBytes())
        }
    }
}