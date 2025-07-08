package dev.aurakai.auraframefx.ai.error

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.jupiter.MockitoExtension
import org.junit.jupiter.api.extension.ExtendWith
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@ExtendWith(MockitoExtension::class)
@DisplayName("ErrorModel Tests")
class ErrorModelTest {
    
    private lateinit var errorModel: ErrorModel
    private val testMessage = "Test error message"
    private val testCode = "ERR_001"
    private val testTimestamp = LocalDateTime.now()
    private val testStackTrace = "Stack trace information"
    
    @BeforeEach
    fun setUp() {
        MockitoAnnotations.openMocks(this)
    }
    
    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {
        
        @Test
        @DisplayName("Should create ErrorModel with all parameters")
        fun shouldCreateErrorModelWithAllParameters() {
            // Given & When
            errorModel = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
            
            // Then
            assertEquals(testMessage, errorModel.message)
            assertEquals(testCode, errorModel.code)
            assertEquals(testTimestamp, errorModel.timestamp)
            assertEquals(testStackTrace, errorModel.stackTrace)
        }
        
        @Test
        @DisplayName("Should create ErrorModel with minimal parameters")
        fun shouldCreateErrorModelWithMinimalParameters() {
            // Given & When
            errorModel = ErrorModel(message = testMessage)
            
            // Then
            assertEquals(testMessage, errorModel.message)
            assertNull(errorModel.code)
            assertNotNull(errorModel.timestamp)
            assertNull(errorModel.stackTrace)
        }
        
        @Test
        @DisplayName("Should create ErrorModel with empty message")
        fun shouldCreateErrorModelWithEmptyMessage() {
            // Given & When
            errorModel = ErrorModel(message = "")
            
            // Then
            assertEquals("", errorModel.message)
            assertNotNull(errorModel.timestamp)
        }
        
        @Test
        @DisplayName("Should auto-generate timestamp when not provided")
        fun shouldAutoGenerateTimestamp() {
            // Given
            val beforeCreation = LocalDateTime.now()
            
            // When
            errorModel = ErrorModel(message = testMessage)
            val afterCreation = LocalDateTime.now()
            
            // Then
            assertNotNull(errorModel.timestamp)
            assertTrue(errorModel.timestamp.isAfter(beforeCreation) || errorModel.timestamp.isEqual(beforeCreation))
            assertTrue(errorModel.timestamp.isBefore(afterCreation) || errorModel.timestamp.isEqual(afterCreation))
        }
    }
    
    @Nested
    @DisplayName("Property Tests")
    inner class PropertyTests {
        
        @BeforeEach
        fun setUp() {
            errorModel = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
        }
        
        @Test
        @DisplayName("Should get message property")
        fun shouldGetMessageProperty() {
            assertEquals(testMessage, errorModel.message)
        }
        
        @Test
        @DisplayName("Should get code property")
        fun shouldGetCodeProperty() {
            assertEquals(testCode, errorModel.code)
        }
        
        @Test
        @DisplayName("Should get timestamp property")
        fun shouldGetTimestampProperty() {
            assertEquals(testTimestamp, errorModel.timestamp)
        }
        
        @Test
        @DisplayName("Should get stackTrace property")
        fun shouldGetStackTraceProperty() {
            assertEquals(testStackTrace, errorModel.stackTrace)
        }
        
        @Test
        @DisplayName("Should handle null code gracefully")
        fun shouldHandleNullCodeGracefully() {
            val errorModelWithNullCode = ErrorModel(message = testMessage, code = null)
            assertNull(errorModelWithNullCode.code)
        }
        
        @Test
        @DisplayName("Should handle null stackTrace gracefully")
        fun shouldHandleNullStackTraceGracefully() {
            val errorModelWithNullStackTrace = ErrorModel(message = testMessage, stackTrace = null)
            assertNull(errorModelWithNullStackTrace.stackTrace)
        }
    }
    
    @Nested
    @DisplayName("Utility Method Tests")
    inner class UtilityMethodTests {
        
        @BeforeEach
        fun setUp() {
            errorModel = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
        }
        
        @Test
        @DisplayName("Should format timestamp correctly")
        fun shouldFormatTimestampCorrectly() {
            val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
            val expectedFormat = testTimestamp.format(formatter)
            
            val formattedTimestamp = errorModel.getFormattedTimestamp()
            assertEquals(expectedFormat, formattedTimestamp)
        }
        
        @Test
        @DisplayName("Should return summary with all fields")
        fun shouldReturnSummaryWithAllFields() {
            val summary = errorModel.getSummary()
            
            assertTrue(summary.contains(testMessage))
            assertTrue(summary.contains(testCode))
            assertTrue(summary.contains(testTimestamp.toString()))
        }
        
        @Test
        @DisplayName("Should return summary with minimal fields")
        fun shouldReturnSummaryWithMinimalFields() {
            val minimalErrorModel = ErrorModel(message = testMessage)
            val summary = minimalErrorModel.getSummary()
            
            assertTrue(summary.contains(testMessage))
            assertFalse(summary.contains("null"))
        }
        
        @Test
        @DisplayName("Should check if error is critical")
        fun shouldCheckIfErrorIsCritical() {
            val criticalErrorModel = ErrorModel(message = testMessage, code = "CRITICAL_001")
            assertTrue(criticalErrorModel.isCritical())
            
            val nonCriticalErrorModel = ErrorModel(message = testMessage, code = "INFO_001")
            assertFalse(nonCriticalErrorModel.isCritical())
        }
        
        @Test
        @DisplayName("Should handle null code when checking criticality")
        fun shouldHandleNullCodeWhenCheckingCriticality() {
            val errorModelWithNullCode = ErrorModel(message = testMessage, code = null)
            assertFalse(errorModelWithNullCode.isCritical())
        }
    }
    
    @Nested
    @DisplayName("Comparison Tests")
    inner class ComparisonTests {
        
        @Test
        @DisplayName("Should be equal when all properties match")
        fun shouldBeEqualWhenAllPropertiesMatch() {
            val errorModel1 = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
            val errorModel2 = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
            
            assertEquals(errorModel1, errorModel2)
            assertEquals(errorModel1.hashCode(), errorModel2.hashCode())
        }
        
        @Test
        @DisplayName("Should not be equal when messages differ")
        fun shouldNotBeEqualWhenMessagesDiffer() {
            val errorModel1 = ErrorModel(message = testMessage)
            val errorModel2 = ErrorModel(message = "Different message")
            
            assertNotEquals(errorModel1, errorModel2)
        }
        
        @Test
        @DisplayName("Should not be equal when codes differ")
        fun shouldNotBeEqualWhenCodesDiffer() {
            val errorModel1 = ErrorModel(message = testMessage, code = testCode)
            val errorModel2 = ErrorModel(message = testMessage, code = "DIFFERENT_CODE")
            
            assertNotEquals(errorModel1, errorModel2)
        }
        
        @Test
        @DisplayName("Should not be equal when timestamps differ")
        fun shouldNotBeEqualWhenTimestampsDiffer() {
            val errorModel1 = ErrorModel(message = testMessage, timestamp = testTimestamp)
            val errorModel2 = ErrorModel(message = testMessage, timestamp = testTimestamp.plusMinutes(1))
            
            assertNotEquals(errorModel1, errorModel2)
        }
        
        @Test
        @DisplayName("Should handle comparison with null")
        fun shouldHandleComparisonWithNull() {
            val errorModel1 = ErrorModel(message = testMessage)
            assertNotEquals(errorModel1, null)
        }
        
        @Test
        @DisplayName("Should handle comparison with different type")
        fun shouldHandleComparisonWithDifferentType() {
            val errorModel1 = ErrorModel(message = testMessage)
            assertNotEquals(errorModel1, "Not an ErrorModel")
        }
    }
    
    @Nested
    @DisplayName("toString Tests")
    inner class ToStringTests {
        
        @Test
        @DisplayName("Should generate readable string representation")
        fun shouldGenerateReadableStringRepresentation() {
            errorModel = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
            
            val stringRepresentation = errorModel.toString()
            
            assertTrue(stringRepresentation.contains(testMessage))
            assertTrue(stringRepresentation.contains(testCode))
            assertTrue(stringRepresentation.contains(testTimestamp.toString()))
        }
        
        @Test
        @DisplayName("Should handle null values in string representation")
        fun shouldHandleNullValuesInStringRepresentation() {
            val errorModelWithNulls = ErrorModel(message = testMessage, code = null, stackTrace = null)
            val stringRepresentation = errorModelWithNulls.toString()
            
            assertTrue(stringRepresentation.contains(testMessage))
            assertFalse(stringRepresentation.contains("null"))
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandling {
        
        @Test
        @DisplayName("Should handle very long error messages")
        fun shouldHandleVeryLongErrorMessages() {
            val longMessage = "A".repeat(10000)
            val errorModel = ErrorModel(message = longMessage)
            
            assertEquals(longMessage, errorModel.message)
            assertTrue(errorModel.toString().contains(longMessage))
        }
        
        @Test
        @DisplayName("Should handle special characters in message")
        fun shouldHandleSpecialCharactersInMessage() {
            val specialMessage = "Error with special chars: \n\t\r\u0000\u001F"
            val errorModel = ErrorModel(message = specialMessage)
            
            assertEquals(specialMessage, errorModel.message)
        }
        
        @Test
        @DisplayName("Should handle Unicode characters in message")
        fun shouldHandleUnicodeCharactersInMessage() {
            val unicodeMessage = "Error with Unicode: 🔥 💥 ⚠️ 中文 العربية"
            val errorModel = ErrorModel(message = unicodeMessage)
            
            assertEquals(unicodeMessage, errorModel.message)
        }
        
        @Test
        @DisplayName("Should handle empty stackTrace")
        fun shouldHandleEmptyStackTrace() {
            val errorModel = ErrorModel(message = testMessage, stackTrace = "")
            assertEquals("", errorModel.stackTrace)
        }
        
        @Test
        @DisplayName("Should handle very long stack traces")
        fun shouldHandleVeryLongStackTraces() {
            val longStackTrace = "Stack trace line\n".repeat(1000)
            val errorModel = ErrorModel(message = testMessage, stackTrace = longStackTrace)
            
            assertEquals(longStackTrace, errorModel.stackTrace)
        }
    }
    
    @Nested
    @DisplayName("Factory Method Tests")
    inner class FactoryMethodTests {
        
        @Test
        @DisplayName("Should create ErrorModel from Exception")
        fun shouldCreateErrorModelFromException() {
            val exception = RuntimeException("Test exception")
            val errorModel = ErrorModel.fromException(exception)
            
            assertEquals("Test exception", errorModel.message)
            assertNotNull(errorModel.stackTrace)
            assertTrue(errorModel.stackTrace!!.contains("RuntimeException"))
        }
        
        @Test
        @DisplayName("Should create ErrorModel from Exception with custom code")
        fun shouldCreateErrorModelFromExceptionWithCustomCode() {
            val exception = IllegalArgumentException("Invalid argument")
            val errorModel = ErrorModel.fromException(exception, "ARG_001")
            
            assertEquals("Invalid argument", errorModel.message)
            assertEquals("ARG_001", errorModel.code)
            assertNotNull(errorModel.stackTrace)
        }
        
        @Test
        @DisplayName("Should handle null Exception message")
        fun shouldHandleNullExceptionMessage() {
            val exception = RuntimeException(null as String?)
            val errorModel = ErrorModel.fromException(exception)
            
            assertEquals("RuntimeException", errorModel.message)
        }
    }
    
    @Nested
    @DisplayName("Copy and Modification Tests")
    inner class CopyAndModificationTests {
        
        @Test
        @DisplayName("Should create copy with different message")
        fun shouldCreateCopyWithDifferentMessage() {
            val originalError = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
            
            val newMessage = "Updated error message"
            val copiedError = originalError.copy(message = newMessage)
            
            assertEquals(newMessage, copiedError.message)
            assertEquals(testCode, copiedError.code)
            assertEquals(testTimestamp, copiedError.timestamp)
            assertEquals(testStackTrace, copiedError.stackTrace)
        }
        
        @Test
        @DisplayName("Should create copy with different code")
        fun shouldCreateCopyWithDifferentCode() {
            val originalError = ErrorModel(message = testMessage, code = testCode)
            val newCode = "NEW_CODE"
            val copiedError = originalError.copy(code = newCode)
            
            assertEquals(testMessage, copiedError.message)
            assertEquals(newCode, copiedError.code)
        }
        
        @Test
        @DisplayName("Should create exact copy when no parameters changed")
        fun shouldCreateExactCopyWhenNoParametersChanged() {
            val originalError = ErrorModel(
                message = testMessage,
                code = testCode,
                timestamp = testTimestamp,
                stackTrace = testStackTrace
            )
            
            val copiedError = originalError.copy()
            
            assertEquals(originalError, copiedError)
            assertNotSame(originalError, copiedError)
        }
    }
}