package dev.aurakai.auraframefx.ai.error

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.EnumSource
import org.junit.jupiter.params.provider.NullAndEmptySource
import java.time.LocalDateTime

@DisplayName("ErrorModel Tests")
class ErrorModelTest {

    private lateinit var basicErrorModel: ErrorModel
    private lateinit var detailedErrorModel: ErrorModel

    @BeforeEach
    fun setUp() {
        basicErrorModel = ErrorModel(
            type = ErrorType.VALIDATION_ERROR,
            message = "Test error message"
        )
        
        detailedErrorModel = ErrorModel(
            type = ErrorType.NETWORK_ERROR,
            message = "Network connection failed",
            cause = "Connection timeout",
            timestamp = LocalDateTime.now(),
            code = "NET_001"
        )
    }

    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {

        @Test
        @DisplayName("Should create ErrorModel with required parameters")
        fun shouldCreateWithRequiredParameters() {
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message"
            )
            
            assertThat(errorModel.type).isEqualTo(ErrorType.VALIDATION_ERROR)
            assertThat(errorModel.message).isEqualTo("Test message")
        }

        @Test
        @DisplayName("Should create ErrorModel with all parameters")
        fun shouldCreateWithAllParameters() {
            val timestamp = LocalDateTime.now()
            val errorModel = ErrorModel(
                type = ErrorType.NETWORK_ERROR,
                message = "Network error",
                cause = "Connection failed",
                timestamp = timestamp,
                code = "NET_001"
            )
            
            assertThat(errorModel.type).isEqualTo(ErrorType.NETWORK_ERROR)
            assertThat(errorModel.message).isEqualTo("Network error")
            assertThat(errorModel.cause).isEqualTo("Connection failed")
            assertThat(errorModel.timestamp).isEqualTo(timestamp)
            assertThat(errorModel.code).isEqualTo("NET_001")
        }

        @ParameterizedTest
        @EnumSource(ErrorType::class)
        @DisplayName("Should accept all ErrorType values")
        fun shouldAcceptAllErrorTypes(errorType: ErrorType) {
            val errorModel = ErrorModel(
                type = errorType,
                message = "Test message"
            )
            
            assertThat(errorModel.type).isEqualTo(errorType)
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = [" ", "   "])
        @DisplayName("Should handle empty and whitespace messages")
        fun shouldHandleEmptyAndWhitespaceMessages(message: String?) {
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = message ?: ""
            )
            
            assertThat(errorModel.message).isEqualTo(message ?: "")
        }
    }

    @Nested
    @DisplayName("Property Tests")
    inner class PropertyTests {

        @Test
        @DisplayName("Should have correct type property")
        fun shouldHaveCorrectType() {
            assertThat(basicErrorModel.type).isEqualTo(ErrorType.VALIDATION_ERROR)
            assertThat(detailedErrorModel.type).isEqualTo(ErrorType.NETWORK_ERROR)
        }

        @Test
        @DisplayName("Should have correct message property")
        fun shouldHaveCorrectMessage() {
            assertThat(basicErrorModel.message).isEqualTo("Test error message")
            assertThat(detailedErrorModel.message).isEqualTo("Network connection failed")
        }

        @Test
        @DisplayName("Should handle null cause properly")
        fun shouldHandleNullCause() {
            assertThat(basicErrorModel.cause).isNull()
            assertThat(detailedErrorModel.cause).isEqualTo("Connection timeout")
        }

        @Test
        @DisplayName("Should handle null timestamp properly")
        fun shouldHandleNullTimestamp() {
            assertThat(basicErrorModel.timestamp).isNull()
            assertThat(detailedErrorModel.timestamp).isNotNull()
        }

        @Test
        @DisplayName("Should handle null code properly")
        fun shouldHandleNullCode() {
            assertThat(basicErrorModel.code).isNull()
            assertThat(detailedErrorModel.code).isEqualTo("NET_001")
        }
    }

    @Nested
    @DisplayName("Equality Tests")
    inner class EqualityTests {

        @Test
        @DisplayName("Should be equal when all properties match")
        fun shouldBeEqualWhenAllPropertiesMatch() {
            val errorModel1 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message"
            )
            val errorModel2 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message"
            )
            
            assertThat(errorModel1).isEqualTo(errorModel2)
            assertThat(errorModel1.hashCode()).isEqualTo(errorModel2.hashCode())
        }

        @Test
        @DisplayName("Should not be equal when types differ")
        fun shouldNotBeEqualWhenTypesDiffer() {
            val errorModel1 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message"
            )
            val errorModel2 = ErrorModel(
                type = ErrorType.NETWORK_ERROR,
                message = "Test message"
            )
            
            assertThat(errorModel1).isNotEqualTo(errorModel2)
        }

        @Test
        @DisplayName("Should not be equal when messages differ")
        fun shouldNotBeEqualWhenMessagesDiffer() {
            val errorModel1 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message 1"
            )
            val errorModel2 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message 2"
            )
            
            assertThat(errorModel1).isNotEqualTo(errorModel2)
        }

        @Test
        @DisplayName("Should not be equal when causes differ")
        fun shouldNotBeEqualWhenCausesDiffer() {
            val errorModel1 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                cause = "Cause 1"
            )
            val errorModel2 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                cause = "Cause 2"
            )
            
            assertThat(errorModel1).isNotEqualTo(errorModel2)
        }

        @Test
        @DisplayName("Should be equal when comparing to itself")
        fun shouldBeEqualWhenComparingToItself() {
            assertThat(basicErrorModel).isEqualTo(basicErrorModel)
        }

        @Test
        @DisplayName("Should not be equal to null")
        fun shouldNotBeEqualToNull() {
            assertThat(basicErrorModel).isNotEqualTo(null)
        }

        @Test
        @DisplayName("Should not be equal to different type")
        fun shouldNotBeEqualToDifferentType() {
            assertThat(basicErrorModel).isNotEqualTo("Not an ErrorModel")
        }
    }

    @Nested
    @DisplayName("ToString Tests")
    inner class ToStringTests {

        @Test
        @DisplayName("Should contain type and message in toString")
        fun shouldContainTypeAndMessageInToString() {
            val toString = basicErrorModel.toString()
            
            assertThat(toString).contains("VALIDATION_ERROR")
            assertThat(toString).contains("Test error message")
        }

        @Test
        @DisplayName("Should contain all properties in toString when present")
        fun shouldContainAllPropertiesInToStringWhenPresent() {
            val toString = detailedErrorModel.toString()
            
            assertThat(toString).contains("NETWORK_ERROR")
            assertThat(toString).contains("Network connection failed")
            assertThat(toString).contains("Connection timeout")
            assertThat(toString).contains("NET_001")
        }

        @Test
        @DisplayName("Should handle null values gracefully in toString")
        fun shouldHandleNullValuesGracefullyInToString() {
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                cause = null,
                timestamp = null,
                code = null
            )
            
            val toString = errorModel.toString()
            
            assertThat(toString).isNotNull()
            assertThat(toString).contains("VALIDATION_ERROR")
            assertThat(toString).contains("Test message")
        }
    }

    @Nested
    @DisplayName("Copy Tests")
    inner class CopyTests {

        @Test
        @DisplayName("Should create copy with modified type")
        fun shouldCreateCopyWithModifiedType() {
            val copied = basicErrorModel.copy(type = ErrorType.NETWORK_ERROR)
            
            assertThat(copied.type).isEqualTo(ErrorType.NETWORK_ERROR)
            assertThat(copied.message).isEqualTo(basicErrorModel.message)
            assertThat(copied.cause).isEqualTo(basicErrorModel.cause)
            assertThat(copied.timestamp).isEqualTo(basicErrorModel.timestamp)
            assertThat(copied.code).isEqualTo(basicErrorModel.code)
        }

        @Test
        @DisplayName("Should create copy with modified message")
        fun shouldCreateCopyWithModifiedMessage() {
            val newMessage = "Modified message"
            val copied = basicErrorModel.copy(message = newMessage)
            
            assertThat(copied.message).isEqualTo(newMessage)
            assertThat(copied.type).isEqualTo(basicErrorModel.type)
        }

        @Test
        @DisplayName("Should create copy with additional cause")
        fun shouldCreateCopyWithAdditionalCause() {
            val newCause = "New cause"
            val copied = basicErrorModel.copy(cause = newCause)
            
            assertThat(copied.cause).isEqualTo(newCause)
            assertThat(copied.type).isEqualTo(basicErrorModel.type)
            assertThat(copied.message).isEqualTo(basicErrorModel.message)
        }

        @Test
        @DisplayName("Should create copy with timestamp")
        fun shouldCreateCopyWithTimestamp() {
            val timestamp = LocalDateTime.now()
            val copied = basicErrorModel.copy(timestamp = timestamp)
            
            assertThat(copied.timestamp).isEqualTo(timestamp)
            assertThat(copied.type).isEqualTo(basicErrorModel.type)
            assertThat(copied.message).isEqualTo(basicErrorModel.message)
        }

        @Test
        @DisplayName("Should create copy with code")
        fun shouldCreateCopyWithCode() {
            val code = "ERR_001"
            val copied = basicErrorModel.copy(code = code)
            
            assertThat(copied.code).isEqualTo(code)
            assertThat(copied.type).isEqualTo(basicErrorModel.type)
            assertThat(copied.message).isEqualTo(basicErrorModel.message)
        }

        @Test
        @DisplayName("Should create exact copy when no parameters changed")
        fun shouldCreateExactCopyWhenNoParametersChanged() {
            val copied = detailedErrorModel.copy()
            
            assertThat(copied).isEqualTo(detailedErrorModel)
            assertThat(copied).isNotSameAs(detailedErrorModel)
        }
    }

    @Nested
    @DisplayName("Validation Tests")
    inner class ValidationTests {

        @Test
        @DisplayName("Should validate message is not blank")
        fun shouldValidateMessageIsNotBlank() {
            assertThat(basicErrorModel.message).isNotBlank()
            assertThat(detailedErrorModel.message).isNotBlank()
        }

        @Test
        @DisplayName("Should handle very long messages")
        fun shouldHandleVeryLongMessages() {
            val longMessage = "x".repeat(10000)
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = longMessage
            )
            
            assertThat(errorModel.message).hasSize(10000)
            assertThat(errorModel.message).isEqualTo(longMessage)
        }

        @Test
        @DisplayName("Should handle special characters in message")
        fun shouldHandleSpecialCharactersInMessage() {
            val specialMessage = "Error: àáâäæãåāčėęīłńšūżž™©®"
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = specialMessage
            )
            
            assertThat(errorModel.message).isEqualTo(specialMessage)
        }

        @Test
        @DisplayName("Should handle newlines and tabs in message")
        fun shouldHandleNewlinesAndTabsInMessage() {
            val messageWithWhitespace = "Line 1\nLine 2\tTabbed"
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = messageWithWhitespace
            )
            
            assertThat(errorModel.message).isEqualTo(messageWithWhitespace)
        }
    }

    @Nested
    @DisplayName("Edge Case Tests")
    inner class EdgeCaseTests {

        @Test
        @DisplayName("Should handle empty string message")
        fun shouldHandleEmptyStringMessage() {
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = ""
            )
            
            assertThat(errorModel.message).isEmpty()
        }

        @Test
        @DisplayName("Should handle timestamp in past")
        fun shouldHandleTimestampInPast() {
            val pastTimestamp = LocalDateTime.of(2020, 1, 1, 0, 0, 0)
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                timestamp = pastTimestamp
            )
            
            assertThat(errorModel.timestamp).isEqualTo(pastTimestamp)
            assertThat(errorModel.timestamp).isBefore(LocalDateTime.now())
        }

        @Test
        @DisplayName("Should handle timestamp in future")
        fun shouldHandleTimestampInFuture() {
            val futureTimestamp = LocalDateTime.of(2030, 12, 31, 23, 59, 59)
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                timestamp = futureTimestamp
            )
            
            assertThat(errorModel.timestamp).isEqualTo(futureTimestamp)
            assertThat(errorModel.timestamp).isAfter(LocalDateTime.now())
        }

        @Test
        @DisplayName("Should handle very long cause")
        fun shouldHandleVeryLongCause() {
            val longCause = "cause".repeat(1000)
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                cause = longCause
            )
            
            assertThat(errorModel.cause).isEqualTo(longCause)
        }

        @Test
        @DisplayName("Should handle very long code")
        fun shouldHandleVeryLongCode() {
            val longCode = "ERR_" + "X".repeat(1000)
            val errorModel = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message",
                code = longCode
            )
            
            assertThat(errorModel.code).isEqualTo(longCode)
        }
    }

    @Nested
    @DisplayName("Immutability Tests")
    inner class ImmutabilityTests {

        @Test
        @DisplayName("Should be immutable after creation")
        fun shouldBeImmutableAfterCreation() {
            val originalType = detailedErrorModel.type
            val originalMessage = detailedErrorModel.message
            val originalCause = detailedErrorModel.cause
            val originalTimestamp = detailedErrorModel.timestamp
            val originalCode = detailedErrorModel.code
            
            // Properties should remain unchanged
            assertThat(detailedErrorModel.type).isEqualTo(originalType)
            assertThat(detailedErrorModel.message).isEqualTo(originalMessage)
            assertThat(detailedErrorModel.cause).isEqualTo(originalCause)
            assertThat(detailedErrorModel.timestamp).isEqualTo(originalTimestamp)
            assertThat(detailedErrorModel.code).isEqualTo(originalCode)
        }

        @Test
        @DisplayName("Should create new instance on copy")
        fun shouldCreateNewInstanceOnCopy() {
            val copied = detailedErrorModel.copy()
            
            assertThat(copied).isNotSameAs(detailedErrorModel)
            assertThat(copied).isEqualTo(detailedErrorModel)
        }
    }

    @Nested
    @DisplayName("Utility Method Tests")
    inner class UtilityMethodTests {

        @Test
        @DisplayName("Should have consistent hashCode for equal objects")
        fun shouldHaveConsistentHashCodeForEqualObjects() {
            val errorModel1 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message"
            )
            val errorModel2 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message"
            )
            
            assertThat(errorModel1.hashCode()).isEqualTo(errorModel2.hashCode())
        }

        @Test
        @DisplayName("Should have different hashCode for different objects")
        fun shouldHaveDifferentHashCodeForDifferentObjects() {
            val errorModel1 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message 1"
            )
            val errorModel2 = ErrorModel(
                type = ErrorType.VALIDATION_ERROR,
                message = "Test message 2"
            )
            
            assertThat(errorModel1.hashCode()).isNotEqualTo(errorModel2.hashCode())
        }

        @Test
        @DisplayName("Should maintain hashCode consistency across multiple calls")
        fun shouldMaintainHashCodeConsistencyAcrossMultipleCalls() {
            val originalHashCode = basicErrorModel.hashCode()
            
            // Multiple calls should return same hashCode
            assertThat(basicErrorModel.hashCode()).isEqualTo(originalHashCode)
            assertThat(basicErrorModel.hashCode()).isEqualTo(originalHashCode)
        }
    }
}