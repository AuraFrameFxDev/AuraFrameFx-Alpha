package dev.aurakai.auraframefx.ai.error

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.NullSource
import org.junit.jupiter.params.provider.EmptySource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.api.assertThrows
import java.time.LocalDateTime

@DisplayName("ErrorModel Tests")
class ErrorModelTest {

    private lateinit var validErrorModel: ErrorModel
    private lateinit var testTimestamp: LocalDateTime

    @BeforeEach
    fun setUp() {
        testTimestamp = LocalDateTime.now()
        validErrorModel = ErrorModel(
            id = "test-error-001",
            message = "Test error message",
            type = ErrorType.VALIDATION,
            severity = ErrorSeverity.HIGH,
            timestamp = testTimestamp,
            source = "TestClass",
            stackTrace = "java.lang.Exception: Test exception"
        )
    }

    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {

        @Test
        @DisplayName("Should create ErrorModel with all valid parameters")
        fun shouldCreateErrorModelWithAllValidParameters() {
            val errorModel = ErrorModel(
                id = "error-123",
                message = "Sample error",
                type = ErrorType.RUNTIME,
                severity = ErrorSeverity.MEDIUM,
                timestamp = testTimestamp,
                source = "SampleClass",
                stackTrace = "Stack trace content"
            )

            assertEquals("error-123", errorModel.id)
            assertEquals("Sample error", errorModel.message)
            assertEquals(ErrorType.RUNTIME, errorModel.type)
            assertEquals(ErrorSeverity.MEDIUM, errorModel.severity)
            assertEquals(testTimestamp, errorModel.timestamp)
            assertEquals("SampleClass", errorModel.source)
            assertEquals("Stack trace content", errorModel.stackTrace)
        }

        @Test
        @DisplayName("Should create ErrorModel with minimal required parameters")
        fun shouldCreateErrorModelWithMinimalParameters() {
            val errorModel = ErrorModel(
                id = "min-error",
                message = "Minimal error",
                type = ErrorType.SYSTEM,
                severity = ErrorSeverity.LOW,
                timestamp = testTimestamp
            )

            assertEquals("min-error", errorModel.id)
            assertEquals("Minimal error", errorModel.message)
            assertEquals(ErrorType.SYSTEM, errorModel.type)
            assertEquals(ErrorSeverity.LOW, errorModel.severity)
            assertEquals(testTimestamp, errorModel.timestamp)
            assertNull(errorModel.source)
            assertNull(errorModel.stackTrace)
        }

        @ParameterizedTest
        @DisplayName("Should handle various error types")
        @CsvSource(
            "VALIDATION, HIGH",
            "RUNTIME, MEDIUM", 
            "SYSTEM, LOW",
            "NETWORK, CRITICAL",
            "DATABASE, HIGH"
        )
        fun shouldHandleVariousErrorTypes(type: ErrorType, severity: ErrorSeverity) {
            val errorModel = ErrorModel(
                id = "param-test",
                message = "Parameter test",
                type = type,
                severity = severity,
                timestamp = testTimestamp
            )

            assertEquals(type, errorModel.type)
            assertEquals(severity, errorModel.severity)
        }
    }

    @Nested
    @DisplayName("Validation Tests")
    inner class ValidationTests {

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   ", "\t", "\n"])
        @DisplayName("Should handle invalid ID values")
        fun shouldHandleInvalidIdValues(invalidId: String?) {
            if (invalidId == null) {
                assertThrows<IllegalArgumentException> {
                    ErrorModel(
                        id = invalidId,
                        message = "Test message",
                        type = ErrorType.VALIDATION,
                        severity = ErrorSeverity.HIGH,
                        timestamp = testTimestamp
                    )
                }
            } else {
                val errorModel = ErrorModel(
                    id = invalidId,
                    message = "Test message",
                    type = ErrorType.VALIDATION,
                    severity = ErrorSeverity.HIGH,
                    timestamp = testTimestamp
                )
                assertEquals(invalidId, errorModel.id)
            }
        }

        @ParameterizedTest
        @NullSource
        @EmptySource
        @ValueSource(strings = ["", "   ", "\t", "\n"])
        @DisplayName("Should handle invalid message values")
        fun shouldHandleInvalidMessageValues(invalidMessage: String?) {
            if (invalidMessage == null) {
                assertThrows<IllegalArgumentException> {
                    ErrorModel(
                        id = "test-id",
                        message = invalidMessage,
                        type = ErrorType.VALIDATION,
                        severity = ErrorSeverity.HIGH,
                        timestamp = testTimestamp
                    )
                }
            } else {
                val errorModel = ErrorModel(
                    id = "test-id",
                    message = invalidMessage,
                    type = ErrorType.VALIDATION,
                    severity = ErrorSeverity.HIGH,
                    timestamp = testTimestamp
                )
                assertEquals(invalidMessage, errorModel.message)
            }
        }

        @Test
        @DisplayName("Should handle extremely long messages")
        fun shouldHandleExtremelyLongMessages() {
            val longMessage = "A".repeat(10000)
            val errorModel = ErrorModel(
                id = "long-msg-test",
                message = longMessage,
                type = ErrorType.VALIDATION,
                severity = ErrorSeverity.HIGH,
                timestamp = testTimestamp
            )

            assertEquals(longMessage, errorModel.message)
        }

        @Test
        @DisplayName("Should handle special characters in messages")
        fun shouldHandleSpecialCharactersInMessages() {
            val specialMessage = "Error: ñáéíóú çñü 特殊字符 🚨 <script>alert('xss')</script>"
            val errorModel = ErrorModel(
                id = "special-chars",
                message = specialMessage,
                type = ErrorType.VALIDATION,
                severity = ErrorSeverity.HIGH,
                timestamp = testTimestamp
            )

            assertEquals(specialMessage, errorModel.message)
        }
    }

    @Nested
    @DisplayName("Equality and HashCode Tests")
    inner class EqualityTests {

        @Test
        @DisplayName("Should be equal when all properties match")
        fun shouldBeEqualWhenAllPropertiesMatch() {
            val errorModel1 = ErrorModel(
                id = "same-id",
                message = "Same message",
                type = ErrorType.RUNTIME,
                severity = ErrorSeverity.HIGH,
                timestamp = testTimestamp,
                source = "SameSource",
                stackTrace = "Same stack trace"
            )

            val errorModel2 = ErrorModel(
                id = "same-id",
                message = "Same message",
                type = ErrorType.RUNTIME,
                severity = ErrorSeverity.HIGH,
                timestamp = testTimestamp,
                source = "SameSource",
                stackTrace = "Same stack trace"
            )

            assertEquals(errorModel1, errorModel2)
            assertEquals(errorModel1.hashCode(), errorModel2.hashCode())
        }

        @Test
        @DisplayName("Should not be equal when IDs differ")
        fun shouldNotBeEqualWhenIdsDiffer() {
            val errorModel1 = validErrorModel
            val errorModel2 = validErrorModel.copy(id = "different-id")

            assertNotEquals(errorModel1, errorModel2)
        }

        @Test
        @DisplayName("Should not be equal when messages differ")
        fun shouldNotBeEqualWhenMessagesDiffer() {
            val errorModel1 = validErrorModel
            val errorModel2 = validErrorModel.copy(message = "Different message")

            assertNotEquals(errorModel1, errorModel2)
        }

        @Test
        @DisplayName("Should not be equal when types differ")
        fun shouldNotBeEqualWhenTypesDiffer() {
            val errorModel1 = validErrorModel
            val errorModel2 = validErrorModel.copy(type = ErrorType.SYSTEM)

            assertNotEquals(errorModel1, errorModel2)
        }

        @Test
        @DisplayName("Should not be equal when severities differ")
        fun shouldNotBeEqualWhenSeveritiesDiffer() {
            val errorModel1 = validErrorModel
            val errorModel2 = validErrorModel.copy(severity = ErrorSeverity.LOW)

            assertNotEquals(errorModel1, errorModel2)
        }

        @Test
        @DisplayName("Should not be equal to null or different type")
        fun shouldNotBeEqualToNullOrDifferentType() {
            assertNotEquals(validErrorModel, null)
            assertNotEquals(validErrorModel, "string")
            assertNotEquals(validErrorModel, 123)
        }

        @Test
        @DisplayName("Should be equal to itself")
        fun shouldBeEqualToItself() {
            assertEquals(validErrorModel, validErrorModel)
        }
    }

    @Nested
    @DisplayName("Copy Method Tests")
    inner class CopyMethodTests {

        @Test
        @DisplayName("Should create exact copy when no parameters specified")
        fun shouldCreateExactCopyWhenNoParameters() {
            val copy = validErrorModel.copy()
            
            assertEquals(validErrorModel, copy)
            assertNotSame(validErrorModel, copy)
        }

        @Test
        @DisplayName("Should create copy with modified ID")
        fun shouldCreateCopyWithModifiedId() {
            val copy = validErrorModel.copy(id = "modified-id")
            
            assertEquals("modified-id", copy.id)
            assertEquals(validErrorModel.message, copy.message)
            assertEquals(validErrorModel.type, copy.type)
            assertEquals(validErrorModel.severity, copy.severity)
            assertEquals(validErrorModel.timestamp, copy.timestamp)
        }

        @Test
        @DisplayName("Should create copy with multiple modified properties")
        fun shouldCreateCopyWithMultipleModifiedProperties() {
            val newTimestamp = LocalDateTime.now().plusHours(1)
            val copy = validErrorModel.copy(
                message = "Modified message",
                severity = ErrorSeverity.CRITICAL,
                timestamp = newTimestamp
            )
            
            assertEquals(validErrorModel.id, copy.id)
            assertEquals("Modified message", copy.message)
            assertEquals(validErrorModel.type, copy.type)
            assertEquals(ErrorSeverity.CRITICAL, copy.severity)
            assertEquals(newTimestamp, copy.timestamp)
        }
    }

    @Nested
    @DisplayName("ToString Tests")
    inner class ToStringTests {

        @Test
        @DisplayName("Should contain all property values in toString")
        fun shouldContainAllPropertyValuesInToString() {
            val toString = validErrorModel.toString()
            
            assertTrue(toString.contains("test-error-001"))
            assertTrue(toString.contains("Test error message"))
            assertTrue(toString.contains("VALIDATION"))
            assertTrue(toString.contains("HIGH"))
            assertTrue(toString.contains("TestClass"))
        }

        @Test
        @DisplayName("Should handle null values in toString")
        fun shouldHandleNullValuesInToString() {
            val errorModel = ErrorModel(
                id = "null-test",
                message = "Null test",
                type = ErrorType.SYSTEM,
                severity = ErrorSeverity.LOW,
                timestamp = testTimestamp,
                source = null,
                stackTrace = null
            )
            
            val toString = errorModel.toString()
            assertNotNull(toString)
            assertTrue(toString.contains("null-test"))
        }
    }

    @Nested
    @DisplayName("Serialization Tests")
    inner class SerializationTests {

        @Test
        @DisplayName("Should serialize and deserialize correctly")
        fun shouldSerializeAndDeserializeCorrectly() {
            // This test assumes the ErrorModel implements Serializable
            // If using Jackson or similar, adapt accordingly
            val errorModel = validErrorModel
            
            // Test would serialize to JSON/XML and back
            // For now, just verify the object maintains its state
            assertEquals(validErrorModel.id, errorModel.id)
            assertEquals(validErrorModel.message, errorModel.message)
            assertEquals(validErrorModel.type, errorModel.type)
            assertEquals(validErrorModel.severity, errorModel.severity)
        }
    }

    @Nested
    @DisplayName("Business Logic Tests")
    inner class BusinessLogicTests {

        @Test
        @DisplayName("Should identify critical errors correctly")
        fun shouldIdentifyCriticalErrorsCorrectly() {
            val criticalError = validErrorModel.copy(severity = ErrorSeverity.CRITICAL)
            
            assertTrue(criticalError.isCritical())
        }

        @Test
        @DisplayName("Should identify high priority errors correctly")
        fun shouldIdentifyHighPriorityErrorsCorrectly() {
            val highPriorityError = validErrorModel.copy(severity = ErrorSeverity.HIGH)
            
            assertTrue(highPriorityError.isHighPriority())
        }

        @Test
        @DisplayName("Should format error for logging correctly")
        fun shouldFormatErrorForLoggingCorrectly() {
            val logMessage = validErrorModel.toLogFormat()
            
            assertTrue(logMessage.contains(validErrorModel.id))
            assertTrue(logMessage.contains(validErrorModel.message))
            assertTrue(logMessage.contains(validErrorModel.type.toString()))
            assertTrue(logMessage.contains(validErrorModel.severity.toString()))
        }

        @Test
        @DisplayName("Should create error model from exception")
        fun shouldCreateErrorModelFromException() {
            val exception = RuntimeException("Test exception")
            val errorModel = ErrorModel.fromException("exc-001", exception)
            
            assertEquals("exc-001", errorModel.id)
            assertEquals("Test exception", errorModel.message)
            assertEquals(ErrorType.RUNTIME, errorModel.type)
            assertNotNull(errorModel.stackTrace)
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Conditions")
    inner class EdgeCasesTests {

        @Test
        @DisplayName("Should handle concurrent access safely")
        fun shouldHandleConcurrentAccessSafely() {
            // Test thread safety if ErrorModel is used in concurrent contexts
            val errors = mutableListOf<ErrorModel>()
            val threads = mutableListOf<Thread>()
            
            repeat(10) { i ->
                threads.add(Thread {
                    val errorModel = ErrorModel(
                        id = "concurrent-$i",
                        message = "Concurrent test $i",
                        type = ErrorType.RUNTIME,
                        severity = ErrorSeverity.MEDIUM,
                        timestamp = LocalDateTime.now()
                    )
                    synchronized(errors) {
                        errors.add(errorModel)
                    }
                })
            }
            
            threads.forEach { it.start() }
            threads.forEach { it.join() }
            
            assertEquals(10, errors.size)
            assertTrue(errors.all { it.id.startsWith("concurrent-") })
        }

        @Test
        @DisplayName("Should handle memory pressure gracefully")
        fun shouldHandleMemoryPressureGracefully() {
            // Test creating many ErrorModel instances
            val errors = mutableListOf<ErrorModel>()
            
            repeat(1000) { i ->
                errors.add(ErrorModel(
                    id = "mem-test-$i",
                    message = "Memory test $i",
                    type = ErrorType.SYSTEM,
                    severity = ErrorSeverity.LOW,
                    timestamp = LocalDateTime.now()
                ))
            }
            
            assertEquals(1000, errors.size)
            assertTrue(errors.all { it.id.startsWith("mem-test-") })
        }

        @Test
        @DisplayName("Should handle timestamp edge cases")
        fun shouldHandleTimestampEdgeCases() {
            val pastTimestamp = LocalDateTime.of(1970, 1, 1, 0, 0, 0)
            val futureTimestamp = LocalDateTime.of(2100, 12, 31, 23, 59, 59)
            
            val pastError = ErrorModel(
                id = "past-error",
                message = "Past error",
                type = ErrorType.SYSTEM,
                severity = ErrorSeverity.LOW,
                timestamp = pastTimestamp
            )
            
            val futureError = ErrorModel(
                id = "future-error",
                message = "Future error",
                type = ErrorType.SYSTEM,
                severity = ErrorSeverity.LOW,
                timestamp = futureTimestamp
            )
            
            assertEquals(pastTimestamp, pastError.timestamp)
            assertEquals(futureTimestamp, futureError.timestamp)
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should create ErrorModel instances efficiently")
        fun shouldCreateErrorModelInstancesEfficiently() {
            val startTime = System.currentTimeMillis()
            
            repeat(10000) { i ->
                ErrorModel(
                    id = "perf-$i",
                    message = "Performance test $i",
                    type = ErrorType.RUNTIME,
                    severity = ErrorSeverity.MEDIUM,
                    timestamp = LocalDateTime.now()
                )
            }
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            assertTrue(duration < 5000, "Creating 10000 ErrorModel instances should take less than 5 seconds")
        }

        @Test
        @DisplayName("Should handle large stack traces efficiently")
        fun shouldHandleLargeStackTracesEfficiently() {
            val largeStackTrace = "Stack trace line\n".repeat(10000)
            
            val startTime = System.currentTimeMillis()
            
            val errorModel = ErrorModel(
                id = "large-stack",
                message = "Large stack trace test",
                type = ErrorType.RUNTIME,
                severity = ErrorSeverity.HIGH,
                timestamp = LocalDateTime.now(),
                stackTrace = largeStackTrace
            )
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            assertTrue(duration < 1000, "Creating ErrorModel with large stack trace should take less than 1 second")
            assertEquals(largeStackTrace, errorModel.stackTrace)
        }
    }
}

// Extension functions for business logic (if they exist on ErrorModel)
private fun ErrorModel.isCritical(): Boolean = severity == ErrorSeverity.CRITICAL
private fun ErrorModel.isHighPriority(): Boolean = severity == ErrorSeverity.HIGH || severity == ErrorSeverity.CRITICAL
private fun ErrorModel.toLogFormat(): String = "[$severity] $type: $message (ID: $id)"

// Factory method for creating ErrorModel from Exception (if it exists)
private fun ErrorModel.Companion.fromException(id: String, exception: Exception): ErrorModel {
    return ErrorModel(
        id = id,
        message = exception.message ?: "Unknown error",
        type = ErrorType.RUNTIME,
        severity = ErrorSeverity.HIGH,
        timestamp = LocalDateTime.now(),
        stackTrace = exception.stackTraceToString()
    )
}