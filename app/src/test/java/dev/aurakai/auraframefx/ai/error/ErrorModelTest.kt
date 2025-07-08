package dev.aurakai.auraframefx.ai.error

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.whenever
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@DisplayName("ErrorModel Tests")
class ErrorModelTest {
    
    private lateinit var errorModel: ErrorModel
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
    }
    
    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {
        
        @Test
        @DisplayName("Should create ErrorModel with all required fields")
        fun shouldCreateErrorModelWithAllRequiredFields() {
            // Given
            val message = "Test error message"
            val errorCode = "ERR_001"
            val timestamp = LocalDateTime.now()
            val severity = ErrorSeverity.ERROR
            
            // When
            errorModel = ErrorModel(
                message = message,
                errorCode = errorCode,
                timestamp = timestamp,
                severity = severity
            )
            
            // Then
            assertEquals(message, errorModel.message)
            assertEquals(errorCode, errorModel.errorCode)
            assertEquals(timestamp, errorModel.timestamp)
            assertEquals(severity, errorModel.severity)
        }
        
        @Test
        @DisplayName("Should create ErrorModel with default values")
        fun shouldCreateErrorModelWithDefaultValues() {
            // Given
            val message = "Test error message"
            
            // When
            errorModel = ErrorModel(message = message)
            
            // Then
            assertEquals(message, errorModel.message)
            assertNotNull(errorModel.timestamp)
            assertEquals(ErrorSeverity.ERROR, errorModel.severity)
            assertNull(errorModel.errorCode)
        }
        
        @Test
        @DisplayName("Should throw exception for empty message")
        fun shouldThrowExceptionForEmptyMessage() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                ErrorModel(message = "")
            }
        }
        
        @Test
        @DisplayName("Should throw exception for blank message")
        fun shouldThrowExceptionForBlankMessage() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                ErrorModel(message = "   ")
            }
        }
    }
    
    @Nested
    @DisplayName("Message Validation Tests")
    inner class MessageValidationTests {
        
        @Test
        @DisplayName("Should accept valid non-empty message")
        fun shouldAcceptValidNonEmptyMessage() {
            // Given
            val validMessages = listOf(
                "Simple error",
                "Error with numbers 123",
                "Error with special chars !@#$%",
                "Very long error message that spans multiple lines and contains various characters",
                "Unicode message: 你好世界"
            )
            
            // When & Then
            validMessages.forEach { message ->
                assertDoesNotThrow {
                    ErrorModel(message = message)
                }
            }
        }
        
        @Test
        @DisplayName("Should reject null message")
        fun shouldRejectNullMessage() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                ErrorModel(message = null as String?)
            }
        }
        
        @Test
        @DisplayName("Should trim whitespace from message")
        fun shouldTrimWhitespaceFromMessage() {
            // Given
            val messageWithWhitespace = "  Error message  "
            
            // When
            errorModel = ErrorModel(message = messageWithWhitespace)
            
            // Then
            assertEquals("Error message", errorModel.message)
        }
    }
    
    @Nested
    @DisplayName("Error Code Tests")
    inner class ErrorCodeTests {
        
        @Test
        @DisplayName("Should accept valid error codes")
        fun shouldAcceptValidErrorCodes() {
            // Given
            val validErrorCodes = listOf(
                "ERR_001",
                "WARN_002",
                "INFO_003",
                "E001",
                "ERROR_VALIDATION",
                "AUTH_FAILED"
            )
            
            // When & Then
            validErrorCodes.forEach { errorCode ->
                assertDoesNotThrow {
                    ErrorModel(
                        message = "Test message",
                        errorCode = errorCode
                    )
                }
            }
        }
        
        @Test
        @DisplayName("Should accept null error code")
        fun shouldAcceptNullErrorCode() {
            // Given & When & Then
            assertDoesNotThrow {
                ErrorModel(
                    message = "Test message",
                    errorCode = null
                )
            }
        }
        
        @Test
        @DisplayName("Should reject empty error code")
        fun shouldRejectEmptyErrorCode() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                ErrorModel(
                    message = "Test message",
                    errorCode = ""
                )
            }
        }
        
        @Test
        @DisplayName("Should reject blank error code")
        fun shouldRejectBlankErrorCode() {
            // Given & When & Then
            assertThrows<IllegalArgumentException> {
                ErrorModel(
                    message = "Test message",
                    errorCode = "   "
                )
            }
        }
    }
    
    @Nested
    @DisplayName("Timestamp Tests")
    inner class TimestampTests {
        
        @Test
        @DisplayName("Should use provided timestamp")
        fun shouldUseProvidedTimestamp() {
            // Given
            val specificTimestamp = LocalDateTime.of(2024, 1, 1, 12, 0, 0)
            
            // When
            errorModel = ErrorModel(
                message = "Test message",
                timestamp = specificTimestamp
            )
            
            // Then
            assertEquals(specificTimestamp, errorModel.timestamp)
        }
        
        @Test
        @DisplayName("Should use current timestamp when not provided")
        fun shouldUseCurrentTimestampWhenNotProvided() {
            // Given
            val beforeCreation = LocalDateTime.now()
            
            // When
            errorModel = ErrorModel(message = "Test message")
            
            // Then
            val afterCreation = LocalDateTime.now()
            assertTrue(errorModel.timestamp.isAfter(beforeCreation) || errorModel.timestamp.isEqual(beforeCreation))
            assertTrue(errorModel.timestamp.isBefore(afterCreation) || errorModel.timestamp.isEqual(afterCreation))
        }
    }
    
    @Nested
    @DisplayName("Severity Tests")
    inner class SeverityTests {
        
        @Test
        @DisplayName("Should accept all severity levels")
        fun shouldAcceptAllSeverityLevels() {
            // Given
            val severityLevels = ErrorSeverity.values()
            
            // When & Then
            severityLevels.forEach { severity ->
                assertDoesNotThrow {
                    ErrorModel(
                        message = "Test message",
                        severity = severity
                    )
                }
            }
        }
        
        @Test
        @DisplayName("Should use ERROR as default severity")
        fun shouldUseErrorAsDefaultSeverity() {
            // Given & When
            errorModel = ErrorModel(message = "Test message")
            
            // Then
            assertEquals(ErrorSeverity.ERROR, errorModel.severity)
        }
    }
    
    @Nested
    @DisplayName("Business Logic Tests")
    inner class BusinessLogicTests {
        
        @Test
        @DisplayName("Should determine if error is critical")
        fun shouldDetermineIfErrorIsCritical() {
            // Given
            val criticalError = ErrorModel(
                message = "Critical error",
                severity = ErrorSeverity.CRITICAL
            )
            val nonCriticalError = ErrorModel(
                message = "Non-critical error",
                severity = ErrorSeverity.WARNING
            )
            
            // When & Then
            assertTrue(criticalError.isCritical())
            assertFalse(nonCriticalError.isCritical())
        }
        
        @Test
        @DisplayName("Should format error message correctly")
        fun shouldFormatErrorMessageCorrectly() {
            // Given
            val errorModel = ErrorModel(
                message = "Test error",
                errorCode = "ERR_001",
                timestamp = LocalDateTime.of(2024, 1, 1, 12, 0, 0),
                severity = ErrorSeverity.ERROR
            )
            
            // When
            val formattedMessage = errorModel.getFormattedMessage()
            
            // Then
            assertTrue(formattedMessage.contains("ERR_001"))
            assertTrue(formattedMessage.contains("Test error"))
            assertTrue(formattedMessage.contains("ERROR"))
        }
        
        @Test
        @DisplayName("Should format error message without error code")
        fun shouldFormatErrorMessageWithoutErrorCode() {
            // Given
            val errorModel = ErrorModel(
                message = "Test error",
                errorCode = null,
                severity = ErrorSeverity.WARNING
            )
            
            // When
            val formattedMessage = errorModel.getFormattedMessage()
            
            // Then
            assertTrue(formattedMessage.contains("Test error"))
            assertTrue(formattedMessage.contains("WARNING"))
            assertFalse(formattedMessage.contains("null"))
        }
    }
    
    @Nested
    @DisplayName("Utility Method Tests")
    inner class UtilityMethodTests {
        
        @Test
        @DisplayName("Should convert to JSON representation")
        fun shouldConvertToJsonRepresentation() {
            // Given
            val errorModel = ErrorModel(
                message = "Test error",
                errorCode = "ERR_001",
                severity = ErrorSeverity.ERROR
            )
            
            // When
            val json = errorModel.toJson()
            
            // Then
            assertTrue(json.contains("\"message\":\"Test error\""))
            assertTrue(json.contains("\"errorCode\":\"ERR_001\""))
            assertTrue(json.contains("\"severity\":\"ERROR\""))
        }
        
        @Test
        @DisplayName("Should create error model from JSON")
        fun shouldCreateErrorModelFromJson() {
            // Given
            val json = """
                {
                    "message": "Test error",
                    "errorCode": "ERR_001",
                    "severity": "ERROR"
                }
            """.trimIndent()
            
            // When
            val errorModel = ErrorModel.fromJson(json)
            
            // Then
            assertEquals("Test error", errorModel.message)
            assertEquals("ERR_001", errorModel.errorCode)
            assertEquals(ErrorSeverity.ERROR, errorModel.severity)
        }
        
        @Test
        @DisplayName("Should handle malformed JSON gracefully")
        fun shouldHandleMalformedJsonGracefully() {
            // Given
            val malformedJson = "{ invalid json }"
            
            // When & Then
            assertThrows<JsonParseException> {
                ErrorModel.fromJson(malformedJson)
            }
        }
    }
    
    @Nested
    @DisplayName("Equality and Hash Code Tests")
    inner class EqualityAndHashCodeTests {
        
        @Test
        @DisplayName("Should be equal when all fields match")
        fun shouldBeEqualWhenAllFieldsMatch() {
            // Given
            val timestamp = LocalDateTime.now()
            val error1 = ErrorModel(
                message = "Test error",
                errorCode = "ERR_001",
                timestamp = timestamp,
                severity = ErrorSeverity.ERROR
            )
            val error2 = ErrorModel(
                message = "Test error",
                errorCode = "ERR_001",
                timestamp = timestamp,
                severity = ErrorSeverity.ERROR
            )
            
            // When & Then
            assertEquals(error1, error2)
            assertEquals(error1.hashCode(), error2.hashCode())
        }
        
        @Test
        @DisplayName("Should not be equal when messages differ")
        fun shouldNotBeEqualWhenMessagesDiffer() {
            // Given
            val error1 = ErrorModel(message = "Error 1")
            val error2 = ErrorModel(message = "Error 2")
            
            // When & Then
            assertNotEquals(error1, error2)
        }
        
        @Test
        @DisplayName("Should not be equal when error codes differ")
        fun shouldNotBeEqualWhenErrorCodesDiffer() {
            // Given
            val error1 = ErrorModel(
                message = "Test error",
                errorCode = "ERR_001"
            )
            val error2 = ErrorModel(
                message = "Test error",
                errorCode = "ERR_002"
            )
            
            // When & Then
            assertNotEquals(error1, error2)
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle very long error messages")
        fun shouldHandleVeryLongErrorMessages() {
            // Given
            val longMessage = "Error ".repeat(1000)
            
            // When & Then
            assertDoesNotThrow {
                ErrorModel(message = longMessage)
            }
        }
        
        @Test
        @DisplayName("Should handle special characters in message")
        fun shouldHandleSpecialCharactersInMessage() {
            // Given
            val specialMessage = "Error with special chars: \n\t\r\"\\'\\/"
            
            // When & Then
            assertDoesNotThrow {
                ErrorModel(message = specialMessage)
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent access gracefully")
        fun shouldHandleConcurrentAccessGracefully() {
            // Given
            val errorModel = ErrorModel(message = "Concurrent test")
            
            // When
            val threads = (1..10).map { threadId ->
                Thread {
                    repeat(100) {
                        errorModel.getFormattedMessage()
                        errorModel.isCritical()
                    }
                }
            }
            
            // Then
            assertDoesNotThrow {
                threads.forEach { it.start() }
                threads.forEach { it.join() }
            }
        }
    }
    
    @Nested
    @DisplayName("Integration Tests")
    inner class IntegrationTests {
        
        @Test
        @DisplayName("Should work with error handling pipeline")
        fun shouldWorkWithErrorHandlingPipeline() {
            // Given
            val errors = listOf(
                ErrorModel(message = "Validation error", severity = ErrorSeverity.WARNING),
                ErrorModel(message = "Database error", severity = ErrorSeverity.ERROR),
                ErrorModel(message = "System crash", severity = ErrorSeverity.CRITICAL)
            )
            
            // When
            val criticalErrors = errors.filter { it.isCritical() }
            val formattedErrors = errors.map { it.getFormattedMessage() }
            
            // Then
            assertEquals(1, criticalErrors.size)
            assertEquals(3, formattedErrors.size)
            assertTrue(formattedErrors.all { it.isNotEmpty() })
        }
    }
}

// Supporting enums and classes for comprehensive testing
enum class ErrorSeverity {
    INFO, WARNING, ERROR, CRITICAL
}

class JsonParseException(message: String) : Exception(message)