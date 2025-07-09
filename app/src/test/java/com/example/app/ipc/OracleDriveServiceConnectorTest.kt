package com.example.app.ipc

import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.extension.ExtendWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.jupiter.MockitoExtension
import org.mockito.kotlin.*
import java.io.IOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeoutException

@ExtendWith(MockitoExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class OracleDriveServiceConnectorTest {

    @Mock
    private lateinit var mockServiceClient: OracleDriveServiceClient
    
    @Mock
    private lateinit var mockConnectionManager: ConnectionManager
    
    @Mock
    private lateinit var mockAuthProvider: AuthProvider
    
    private lateinit var connector: OracleDriveServiceConnector
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        connector = OracleDriveServiceConnector(
            serviceClient = mockServiceClient,
            connectionManager = mockConnectionManager,
            authProvider = mockAuthProvider
        )
    }
    
    @AfterEach
    fun tearDown() {
        connector.close()
    }
    
    @Nested
    @DisplayName("Connection Management Tests")
    inner class ConnectionManagementTests {
        
        @Test
        @DisplayName("Should successfully establish connection with valid credentials")
        fun testSuccessfulConnection() = runTest {
            // Given
            val validCredentials = Credentials("valid_token", "valid_endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(validCredentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When
            val result = connector.connect()
            
            // Then
            assertTrue(result)
            verify(mockAuthProvider).getCredentials()
            verify(mockConnectionManager).connect(validCredentials)
        }
        
        @Test
        @DisplayName("Should fail to connect with invalid credentials")
        fun testConnectionFailureWithInvalidCredentials() = runTest {
            // Given
            val invalidCredentials = Credentials("invalid_token", "invalid_endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(invalidCredentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(false)
            
            // When
            val result = connector.connect()
            
            // Then
            assertFalse(result)
            verify(mockAuthProvider).getCredentials()
            verify(mockConnectionManager).connect(invalidCredentials)
        }
        
        @Test
        @DisplayName("Should handle connection timeout gracefully")
        fun testConnectionTimeout() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials()).thenThrow(TimeoutException("Connection timeout"))
            
            // When & Then
            assertThrows<TimeoutException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should retry connection on temporary failures")
        fun testConnectionRetry() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any()))
                .thenReturn(false)
                .thenReturn(false)
                .thenReturn(true)
            
            // When
            val result = connector.connectWithRetry(maxRetries = 3)
            
            // Then
            assertTrue(result)
            verify(mockConnectionManager, times(3)).connect(credentials)
        }
        
        @Test
        @DisplayName("Should properly close connection and clean up resources")
        fun testProperConnectionCleanup() {
            // Given
            connector.connect()
            
            // When
            connector.close()
            
            // Then
            verify(mockConnectionManager).close()
            verify(mockServiceClient).shutdown()
        }
    }
    
    @Nested
    @DisplayName("Data Operation Tests")
    inner class DataOperationTests {
        
        @Test
        @DisplayName("Should successfully upload file")
        fun testSuccessfulFileUpload() = runTest {
            // Given
            val fileData = "test file content".toByteArray()
            val fileName = "test.txt"
            val expectedUploadId = "upload_123"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(expectedUploadId))
            
            // When
            val result = connector.uploadFile(fileName, fileData)
            
            // Then
            assertEquals(expectedUploadId, result.get())
            verify(mockServiceClient).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle file upload failure")
        fun testFileUploadFailure() = runTest {
            // Given
            val fileData = "test file content".toByteArray()
            val fileName = "test.txt"
            val exception = IOException("Upload failed")
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(exception))
            
            // When & Then
            val result = connector.uploadFile(fileName, fileData)
            assertThrows<IOException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should successfully download file")
        fun testSuccessfulFileDownload() = runTest {
            // Given
            val fileId = "file_123"
            val expectedData = "downloaded content".toByteArray()
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(expectedData))
            
            // When
            val result = connector.downloadFile(fileId)
            
            // Then
            assertArrayEquals(expectedData, result.get())
            verify(mockServiceClient).downloadFile(fileId)
        }
        
        @Test
        @DisplayName("Should handle file download failure")
        fun testFileDownloadFailure() = runTest {
            // Given
            val fileId = "non_existent_file"
            val exception = IOException("File not found")
            
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.failedFuture(exception))
            
            // When & Then
            val result = connector.downloadFile(fileId)
            assertThrows<IOException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should successfully delete file")
        fun testSuccessfulFileDelete() = runTest {
            // Given
            val fileId = "file_123"
            
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.completedFuture(true))
            
            // When
            val result = connector.deleteFile(fileId)
            
            // Then
            assertTrue(result.get())
            verify(mockServiceClient).deleteFile(fileId)
        }
        
        @Test
        @DisplayName("Should handle file delete failure")
        fun testFileDeleteFailure() = runTest {
            // Given
            val fileId = "protected_file"
            val exception = SecurityException("Access denied")
            
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.failedFuture(exception))
            
            // When & Then
            val result = connector.deleteFile(fileId)
            assertThrows<SecurityException> {
                result.get()
            }
        }
    }
    
    @Nested
    @DisplayName("Error Handling Tests")
    inner class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle authentication failure")
        fun testAuthenticationFailure() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials())
                .thenThrow(SecurityException("Authentication failed"))
            
            // When & Then
            assertThrows<SecurityException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle network connectivity issues")
        fun testNetworkConnectivityIssues() = runTest {
            // Given
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(IOException("Network unreachable"))
            
            // When & Then
            assertThrows<IOException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle service unavailable errors")
        fun testServiceUnavailableError() = runTest {
            // Given
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                    ServiceUnavailableException("Service temporarily unavailable")))
            
            // When & Then
            val result = connector.uploadFile("test.txt", "content".toByteArray())
            assertThrows<ServiceUnavailableException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle invalid input parameters")
        fun testInvalidInputParameters() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("", "content".toByteArray())
            }
            
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("test.txt", byteArrayOf())
            }
            
            assertThrows<IllegalArgumentException> {
                connector.downloadFile("")
            }
            
            assertThrows<IllegalArgumentException> {
                connector.deleteFile("")
            }
        }
    }
    
    @Nested
    @DisplayName("State Management Tests")
    inner class StateManagementTests {
        
        @Test
        @DisplayName("Should maintain connection state correctly")
        fun testConnectionStateManagement() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials()).thenReturn(Credentials("token", "endpoint"))
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)
            
            // When
            connector.connect()
            
            // Then
            assertTrue(connector.isConnected())
            
            // When
            connector.close()
            
            // Then
            assertFalse(connector.isConnected())
        }
        
        @Test
        @DisplayName("Should prevent operations on disconnected service")
        fun testOperationsOnDisconnectedService() = runTest {
            // Given
            whenever(mockConnectionManager.isConnected()).thenReturn(false)
            
            // When & Then
            assertThrows<IllegalStateException> {
                connector.uploadFile("test.txt", "content".toByteArray())
            }
            
            assertThrows<IllegalStateException> {
                connector.downloadFile("file_123")
            }
            
            assertThrows<IllegalStateException> {
                connector.deleteFile("file_123")
            }
        }
    }
    
    @Nested
    @DisplayName("Performance and Concurrency Tests")
    inner class PerformanceAndConcurrencyTests {
        
        @Test
        @DisplayName("Should handle concurrent operations safely")
        fun testConcurrentOperations() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_id"))
            
            // When
            val futures = (1..10).map { i ->
                connector.uploadFile("file_$i.txt", fileData)
            }
            
            // Then
            futures.forEach { future ->
                assertEquals("upload_id", future.get())
            }
            
            verify(mockServiceClient, times(10)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should handle large file uploads")
        fun testLargeFileUpload() = runTest {
            // Given
            val largeFileData = ByteArray(1024 * 1024) // 1MB
            val fileName = "large_file.dat"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(fileName, largeFileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile(fileName, largeFileData)
        }
        
        @Test
        @DisplayName("Should handle timeout on slow operations")
        fun testSlowOperationTimeout() = runTest {
            // Given
            val slowFuture = CompletableFuture<String>()
            whenever(mockServiceClient.uploadFile(any(), any())).thenReturn(slowFuture)
            
            // When & Then
            assertThrows<TimeoutException> {
                connector.uploadFileWithTimeout("test.txt", "content".toByteArray(), 1000)
            }
        }
    }
    
    @Nested
    @DisplayName("Edge Cases Tests")
    inner class EdgeCasesTests {
        
        @Test
        @DisplayName("Should handle empty file upload")
        fun testEmptyFileUpload() = runTest {
            // Given
            val emptyFileData = byteArrayOf()
            val fileName = "empty.txt"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(fileName, emptyFileData)
            }
        }
        
        @Test
        @DisplayName("Should handle file with special characters in name")
        fun testFileWithSpecialCharacters() = runTest {
            // Given
            val fileName = "file with spaces & special-chars.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(fileName, fileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile(fileName, fileData)
        }
        
        @Test
        @DisplayName("Should handle null parameters gracefully")
        fun testNullParameters() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(null, "content".toByteArray())
            }
            
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("test.txt", null)
            }
            
            assertThrows<IllegalArgumentException> {
                connector.downloadFile(null)
            }
            
            assertThrows<IllegalArgumentException> {
                connector.deleteFile(null)
            }
        }
        
        @Test
        @DisplayName("Should handle multiple consecutive connections")
        fun testMultipleConsecutiveConnections() = runTest {
            // Given
            val credentials = Credentials("token", "endpoint")
            whenever(mockAuthProvider.getCredentials()).thenReturn(credentials)
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            
            // When
            connector.connect()
            connector.connect()
            connector.connect()
            
            // Then
            verify(mockConnectionManager, times(3)).connect(credentials)
        }
    }
    
    @Nested
    @DisplayName("Integration-like Tests")
    inner class IntegrationLikeTests {
        
        @Test
        @DisplayName("Should complete full file lifecycle")
        fun testFullFileLifecycle() = runTest {
            // Given
            val fileName = "lifecycle_test.txt"
            val fileData = "test content".toByteArray()
            val uploadId = "upload_123"
            val fileId = "file_123"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture(uploadId))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(fileData))
            whenever(mockServiceClient.deleteFile(any()))
                .thenReturn(CompletableFuture.completedFuture(true))
            
            // When & Then
            // Upload
            val uploadResult = connector.uploadFile(fileName, fileData)
            assertEquals(uploadId, uploadResult.get())
            
            // Download
            val downloadResult = connector.downloadFile(fileId)
            assertArrayEquals(fileData, downloadResult.get())
            
            // Delete
            val deleteResult = connector.deleteFile(fileId)
            assertTrue(deleteResult.get())
            
            // Verify call sequence
            val inOrder = inOrder(mockServiceClient)
            inOrder.verify(mockServiceClient).uploadFile(fileName, fileData)
            inOrder.verify(mockServiceClient).downloadFile(fileId)
            inOrder.verify(mockServiceClient).deleteFile(fileId)
        }
        
        @Test
        @DisplayName("Should handle authentication refresh during operations")
        fun testAuthenticationRefreshDuringOperations() = runTest {
            // Given
            val expiredCredentials = Credentials("expired_token", "endpoint")
            val refreshedCredentials = Credentials("new_token", "endpoint")
            
            whenever(mockAuthProvider.getCredentials())
                .thenReturn(expiredCredentials)
                .thenReturn(refreshedCredentials)
            
            whenever(mockConnectionManager.connect(expiredCredentials))
                .thenThrow(SecurityException("Token expired"))
            whenever(mockConnectionManager.connect(refreshedCredentials))
                .thenReturn(true)
            
            // When
            val result = connector.connectWithAuthRefresh()
            
            // Then
            assertTrue(result)
            verify(mockAuthProvider, times(2)).getCredentials()
            verify(mockConnectionManager).connect(expiredCredentials)
            verify(mockConnectionManager).connect(refreshedCredentials)
        }
    }
}

// Helper data classes for testing
data class Credentials(val token: String, val endpoint: String)
class ServiceUnavailableException(message: String) : Exception(message)
    @Nested
    @DisplayName("Configuration and Initialization Tests")
    inner class ConfigurationAndInitializationTests {
        
        @Test
        @DisplayName("Should handle configuration with null service client")
        fun testConfigurationWithNullServiceClient() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = null,
                    connectionManager = mockConnectionManager,
                    authProvider = mockAuthProvider
                )
            }
        }
        
        @Test
        @DisplayName("Should handle configuration with null connection manager")
        fun testConfigurationWithNullConnectionManager() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = null,
                    authProvider = mockAuthProvider
                )
            }
        }
        
        @Test
        @DisplayName("Should handle configuration with null auth provider")
        fun testConfigurationWithNullAuthProvider() {
            // When & Then
            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = mockConnectionManager,
                    authProvider = null
                )
            }
        }
        
        @Test
        @DisplayName("Should initialize with default configuration values")
        fun testInitializationWithDefaultValues() {
            // Given
            val connector = OracleDriveServiceConnector(
                serviceClient = mockServiceClient,
                connectionManager = mockConnectionManager,
                authProvider = mockAuthProvider
            )
            
            // When & Then
            assertNotNull(connector)
            assertFalse(connector.isConnected())
            assertEquals("OracleDriveServiceConnector", connector.getComponentName())
        }
        
        @Test
        @DisplayName("Should validate configuration parameters during initialization")
        fun testConfigurationValidation() {
            // Given
            val invalidConfig = mapOf(
                "timeout" to -1,
                "retryCount" to -5,
                "bufferSize" to 0
            )
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                OracleDriveServiceConnector(
                    serviceClient = mockServiceClient,
                    connectionManager = mockConnectionManager,
                    authProvider = mockAuthProvider,
                    configuration = invalidConfig
                )
            }
        }
    }
    
    @Nested
    @DisplayName("Advanced Error Handling Tests")
    inner class AdvancedErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle interrupted operations gracefully")
        fun testInterruptedOperations() = runTest {
            // Given
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(InterruptedException("Operation interrupted")))
            
            // When & Then
            val result = connector.uploadFile("test.txt", "content".toByteArray())
            assertThrows<InterruptedException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle memory allocation failures")
        fun testMemoryAllocationFailures() = runTest {
            // Given
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(OutOfMemoryError("Insufficient memory")))
            
            // When & Then
            val result = connector.uploadFile("test.txt", "content".toByteArray())
            assertThrows<OutOfMemoryError> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle quota exceeded errors")
        fun testQuotaExceededErrors() = runTest {
            // Given
            val quotaException = QuotaExceededException("Storage quota exceeded")
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(quotaException))
            
            // When & Then
            val result = connector.uploadFile("test.txt", "content".toByteArray())
            assertThrows<QuotaExceededException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle rate limiting errors with exponential backoff")
        fun testRateLimitingWithBackoff() = runTest {
            // Given
            val rateLimitException = RateLimitExceededException("Rate limit exceeded")
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(rateLimitException))
                .thenReturn(CompletableFuture.failedFuture(rateLimitException))
                .thenReturn(CompletableFuture.completedFuture("success"))
            
            // When
            val result = connector.uploadFileWithBackoff("test.txt", "content".toByteArray())
            
            // Then
            assertEquals("success", result.get())
            verify(mockServiceClient, times(3)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should handle corrupted file data errors")
        fun testCorruptedFileDataErrors() = runTest {
            // Given
            val corruptedData = "corrupted".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(DataCorruptionException("File data is corrupted")))
            
            // When & Then
            val result = connector.uploadFile("test.txt", corruptedData)
            assertThrows<DataCorruptionException> {
                result.get()
            }
        }
    }
    
    @Nested
    @DisplayName("Security and Authentication Tests")
    inner class SecurityAndAuthenticationTests {
        
        @Test
        @DisplayName("Should handle token refresh during long operations")
        fun testTokenRefreshDuringLongOperations() = runTest {
            // Given
            val expiredCredentials = Credentials("expired_token", "endpoint")
            val newCredentials = Credentials("new_token", "endpoint")
            
            whenever(mockAuthProvider.getCredentials())
                .thenReturn(expiredCredentials)
                .thenReturn(newCredentials)
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(SecurityException("Token expired")))
                .thenReturn(CompletableFuture.completedFuture("success"))
            
            // When
            val result = connector.uploadFileWithTokenRefresh("test.txt", "content".toByteArray())
            
            // Then
            assertEquals("success", result.get())
            verify(mockAuthProvider, times(2)).getCredentials()
        }
        
        @Test
        @DisplayName("Should validate SSL certificates during connection")
        fun testSSLCertificateValidation() = runTest {
            // Given
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(SSLException("Certificate validation failed"))
            
            // When & Then
            assertThrows<SSLException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle permission denied errors")
        fun testPermissionDeniedErrors() = runTest {
            // Given
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(SecurityException("Permission denied")))
            
            // When & Then
            val result = connector.uploadFile("protected.txt", "content".toByteArray())
            assertThrows<SecurityException> {
                result.get()
            }
        }
        
        @Test
        @DisplayName("Should handle credential expiration gracefully")
        fun testCredentialExpirationHandling() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials())
                .thenThrow(SecurityException("Credentials expired"))
            
            // When & Then
            assertThrows<SecurityException> {
                connector.connect()
            }
            
            verify(mockAuthProvider).getCredentials()
        }
    }
    
    @Nested
    @DisplayName("Data Integrity and Validation Tests")
    inner class DataIntegrityAndValidationTests {
        
        @Test
        @DisplayName("Should validate file checksums after upload")
        fun testFileChecksumValidation() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            val expectedChecksum = "abc123"
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            whenever(mockServiceClient.getFileChecksum(any()))
                .thenReturn(CompletableFuture.completedFuture(expectedChecksum))
            
            // When
            val result = connector.uploadFileWithValidation("test.txt", fileData, expectedChecksum)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile("test.txt", fileData)
            verify(mockServiceClient).getFileChecksum("upload_123")
        }
        
        @Test
        @DisplayName("Should handle file size validation")
        fun testFileSizeValidation() = runTest {
            // Given
            val oversizedData = ByteArray(100 * 1024 * 1024) // 100MB
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("oversized.dat", oversizedData)
            }
        }
        
        @Test
        @DisplayName("Should validate file extensions")
        fun testFileExtensionValidation() = runTest {
            // Given
            val validExtensions = listOf(".txt", ".pdf", ".doc")
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("virus.exe", "content".toByteArray())
            }
            
            // Should pass for valid extensions
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            val result = connector.uploadFile("document.txt", "content".toByteArray())
            assertEquals("upload_123", result.get())
        }
        
        @Test
        @DisplayName("Should handle binary data correctly")
        fun testBinaryDataHandling() = runTest {
            // Given
            val binaryData = byteArrayOf(
                0x00, 0x01, 0x02, 0x03, 0xFF.toByte(), 0xFE.toByte(), 0xFD.toByte()
            )
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            whenever(mockServiceClient.downloadFile(any()))
                .thenReturn(CompletableFuture.completedFuture(binaryData))
            
            // When
            val uploadResult = connector.uploadFile("binary.dat", binaryData)
            val downloadResult = connector.downloadFile("upload_123")
            
            // Then
            assertEquals("upload_123", uploadResult.get())
            assertArrayEquals(binaryData, downloadResult.get())
        }
    }
    
    @Nested
    @DisplayName("Monitoring and Metrics Tests")
    inner class MonitoringAndMetricsTests {
        
        @Test
        @DisplayName("Should track operation metrics")
        fun testOperationMetrics() = runTest {
            // Given
            val fileData = "test content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            connector.uploadFile("test.txt", fileData)
            
            // Then
            val metrics = connector.getOperationMetrics()
            assertEquals(1, metrics.uploadCount)
            assertEquals(0, metrics.downloadCount)
            assertEquals(0, metrics.deleteCount)
            assertTrue(metrics.averageUploadTime > 0)
        }
        
        @Test
        @DisplayName("Should track connection uptime")
        fun testConnectionUptimeTracking() = runTest {
            // Given
            whenever(mockAuthProvider.getCredentials()).thenReturn(Credentials("token", "endpoint"))
            whenever(mockConnectionManager.connect(any())).thenReturn(true)
            whenever(mockConnectionManager.isConnected()).thenReturn(true)
            
            // When
            connector.connect()
            Thread.sleep(100) // Simulate uptime
            
            // Then
            val uptime = connector.getConnectionUptime()
            assertTrue(uptime > 0)
        }
        
        @Test
        @DisplayName("Should track error rates")
        fun testErrorRateTracking() = runTest {
            // Given
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.failedFuture(IOException("Upload failed")))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            try {
                connector.uploadFile("test1.txt", "content".toByteArray()).get()
            } catch (e: Exception) { /* Expected */ }
            
            connector.uploadFile("test2.txt", "content".toByteArray()).get()
            
            // Then
            val errorRate = connector.getErrorRate()
            assertEquals(0.5, errorRate, 0.01)
        }
    }
    
    @Nested
    @DisplayName("Advanced Performance Tests")
    inner class AdvancedPerformanceTests {
        
        @Test
        @DisplayName("Should handle burst traffic efficiently")
        fun testBurstTrafficHandling() = runTest {
            // Given
            val fileData = "content".toByteArray()
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_id"))
            
            // When
            val startTime = System.currentTimeMillis()
            val futures = (1..100).map { i ->
                connector.uploadFile("file_$i.txt", fileData)
            }
            
            futures.forEach { it.get() }
            val endTime = System.currentTimeMillis()
            
            // Then
            val duration = endTime - startTime
            assertTrue(duration < 5000) // Should complete within 5 seconds
            verify(mockServiceClient, times(100)).uploadFile(any(), any())
        }
        
        @Test
        @DisplayName("Should handle connection pool exhaustion")
        fun testConnectionPoolExhaustion() = runTest {
            // Given
            whenever(mockConnectionManager.connect(any()))
                .thenThrow(IOException("Connection pool exhausted"))
            
            // When & Then
            assertThrows<IOException> {
                connector.connect()
            }
        }
        
        @Test
        @DisplayName("Should handle memory pressure during operations")
        fun testMemoryPressureHandling() = runTest {
            // Given
            val largeData = ByteArray(10 * 1024 * 1024) // 10MB
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val futures = (1..50).map { i ->
                connector.uploadFile("large_file_$i.dat", largeData)
            }
            
            // Then
            futures.forEach { future ->
                assertEquals("upload_123", future.get())
            }
        }
    }
    
    @Nested
    @DisplayName("Edge Case Boundary Tests")
    inner class EdgeCaseBoundaryTests {
        
        @Test
        @DisplayName("Should handle maximum filename length")
        fun testMaximumFilenameLength() = runTest {
            // Given
            val maxLengthFilename = "a".repeat(255) + ".txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(maxLengthFilename, fileData)
            
            // Then
            assertEquals("upload_123", result.get())
        }
        
        @Test
        @DisplayName("Should handle filename with only whitespace")
        fun testWhitespaceOnlyFilename() = runTest {
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile("   ", "content".toByteArray())
            }
        }
        
        @Test
        @DisplayName("Should handle extremely long file paths")
        fun testExtremelyLongFilePath() = runTest {
            // Given
            val longPath = "dir/".repeat(100) + "file.txt"
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(longPath, "content".toByteArray())
            }
        }
        
        @Test
        @DisplayName("Should handle file with zero bytes but valid name")
        fun testZeroByteFileWithValidName() = runTest {
            // Given
            val validFileName = "empty_but_valid.txt"
            val emptyData = byteArrayOf()
            
            // When & Then
            assertThrows<IllegalArgumentException> {
                connector.uploadFile(validFileName, emptyData)
            }
        }
        
        @Test
        @DisplayName("Should handle unicode characters in filename")
        fun testUnicodeCharactersInFilename() = runTest {
            // Given
            val unicodeFilename = "测试文件_🚀_файл.txt"
            val fileData = "content".toByteArray()
            
            whenever(mockServiceClient.uploadFile(any(), any()))
                .thenReturn(CompletableFuture.completedFuture("upload_123"))
            
            // When
            val result = connector.uploadFile(unicodeFilename, fileData)
            
            // Then
            assertEquals("upload_123", result.get())
            verify(mockServiceClient).uploadFile(unicodeFilename, fileData)
        }
    }
}

// Additional helper classes for the new tests
class QuotaExceededException(message: String) : Exception(message)
class RateLimitExceededException(message: String) : Exception(message)
class DataCorruptionException(message: String) : Exception(message)
class SSLException(message: String) : Exception(message)

data class OperationMetrics(
    val uploadCount: Int,
    val downloadCount: Int,
    val deleteCount: Int,
    val averageUploadTime: Long
)