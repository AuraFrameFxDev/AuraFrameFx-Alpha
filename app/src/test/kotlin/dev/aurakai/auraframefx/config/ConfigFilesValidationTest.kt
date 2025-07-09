package dev.aurakai.auraframefx.config

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.assertThrows
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.io.File
import kotlin.io.path.writeText

/**
 * Comprehensive unit tests for configuration file validation.
 * Testing Framework: JUnit 5
 * 
 * This test suite covers:
 * - Valid configuration files
 * - Invalid configuration files
 * - Edge cases and error conditions
 * - File system interactions
 * - Schema validation
 */
class ConfigFilesValidationTest {

    @TempDir
    lateinit var tempDir: Path

    private lateinit var configValidator: ConfigValidator
    private lateinit var testConfigFile: Path

    @BeforeEach
    fun setUp() {
        configValidator = ConfigValidator()
        testConfigFile = tempDir.resolve("test-config.json")
    }

    @AfterEach
    fun tearDown() {
        // Clean up any temporary files if needed
        if (Files.exists(testConfigFile)) {
            Files.deleteIfExists(testConfigFile)
        }
    }

    @Nested
    @DisplayName("Valid Configuration Tests")
    inner class ValidConfigurationTests {

        @Test
        @DisplayName("Should validate minimal valid configuration")
        fun shouldValidateMinimalValidConfiguration() {
            // Given
            val validConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(validConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertTrue(result.isValid, "Valid configuration should pass validation")
            assertTrue(result.errors.isEmpty(), "Valid configuration should have no errors")
        }

        @Test
        @DisplayName("Should validate complete configuration with all optional fields")
        fun shouldValidateCompleteConfiguration() {
            // Given
            val completeConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "description": "Test application",
                    "database": {
                        "host": "localhost",
                        "port": 5432,
                        "username": "testuser",
                        "password": "testpass",
                        "ssl": true
                    },
                    "logging": {
                        "level": "INFO",
                        "file": "app.log"
                    },
                    "features": {
                        "enableMetrics": true,
                        "enableTracing": false
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(completeConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertTrue(result.isValid, "Complete configuration should pass validation")
            assertTrue(result.errors.isEmpty(), "Complete configuration should have no errors")
        }

        @Test
        @DisplayName("Should validate configuration with array values")
        fun shouldValidateConfigurationWithArrays() {
            // Given
            val configWithArrays = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    },
                    "allowedHosts": ["localhost", "127.0.0.1", "example.com"],
                    "supportedFormats": ["json", "xml", "yaml"]
                }
            """.trimIndent()
            
            testConfigFile.writeText(configWithArrays)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertTrue(result.isValid, "Configuration with arrays should pass validation")
            assertTrue(result.errors.isEmpty(), "Configuration with arrays should have no errors")
        }
    }

    @Nested
    @DisplayName("Invalid Configuration Tests")
    inner class InvalidConfigurationTests {

        @Test
        @DisplayName("Should reject configuration with missing required fields")
        fun shouldRejectConfigurationWithMissingRequiredFields() {
            // Given
            val invalidConfig = """
                {
                    "version": "1.0.0"
                }
            """.trimIndent()
            
            testConfigFile.writeText(invalidConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with missing required fields should fail validation")
            assertFalse(result.errors.isEmpty(), "Configuration with missing required fields should have errors")
            assertTrue(result.errors.any { it.contains("appName") }, "Should report missing appName")
        }

        @Test
        @DisplayName("Should reject configuration with invalid JSON syntax")
        fun shouldRejectConfigurationWithInvalidJson() {
            // Given
            val invalidJsonConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost"
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(invalidJsonConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with invalid JSON should fail validation")
            assertFalse(result.errors.isEmpty(), "Configuration with invalid JSON should have errors")
            assertTrue(result.errors.any { it.contains("JSON") || it.contains("syntax") }, 
                "Should report JSON syntax error")
        }

        @Test
        @DisplayName("Should reject configuration with invalid data types")
        fun shouldRejectConfigurationWithInvalidDataTypes() {
            // Given
            val invalidTypesConfig = """
                {
                    "appName": 123,
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": "invalid_port"
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(invalidTypesConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with invalid data types should fail validation")
            assertFalse(result.errors.isEmpty(), "Configuration with invalid data types should have errors")
            assertTrue(result.errors.any { it.contains("appName") }, "Should report invalid appName type")
            assertTrue(result.errors.any { it.contains("port") }, "Should report invalid port type")
        }

        @Test
        @DisplayName("Should reject configuration with invalid port range")
        fun shouldRejectConfigurationWithInvalidPortRange() {
            // Given
            val invalidPortConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 99999
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(invalidPortConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with invalid port range should fail validation")
            assertFalse(result.errors.isEmpty(), "Configuration with invalid port range should have errors")
            assertTrue(result.errors.any { it.contains("port") && it.contains("range") }, 
                "Should report port range error")
        }

        @Test
        @DisplayName("Should reject configuration with invalid version format")
        fun shouldRejectConfigurationWithInvalidVersionFormat() {
            // Given
            val invalidVersionConfig = """
                {
                    "appName": "TestApp",
                    "version": "invalid.version.format",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(invalidVersionConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with invalid version format should fail validation")
            assertFalse(result.errors.isEmpty(), "Configuration with invalid version format should have errors")
            assertTrue(result.errors.any { it.contains("version") }, "Should report version format error")
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Conditions")
    inner class EdgeCasesAndErrorConditions {

        @Test
        @DisplayName("Should handle empty configuration file")
        fun shouldHandleEmptyConfigurationFile() {
            // Given
            testConfigFile.writeText("")

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Empty configuration file should fail validation")
            assertFalse(result.errors.isEmpty(), "Empty configuration file should have errors")
        }

        @Test
        @DisplayName("Should handle non-existent configuration file")
        fun shouldHandleNonExistentConfigurationFile() {
            // Given
            val nonExistentFile = tempDir.resolve("non-existent-config.json")

            // When & Then
            assertThrows<ConfigurationException> {
                configValidator.validate(nonExistentFile)
            }
        }

        @Test
        @DisplayName("Should handle configuration file with only whitespace")
        fun shouldHandleConfigurationFileWithOnlyWhitespace() {
            // Given
            testConfigFile.writeText("   \n\t  \n   ")

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration file with only whitespace should fail validation")
            assertFalse(result.errors.isEmpty(), "Configuration file with only whitespace should have errors")
        }

        @Test
        @DisplayName("Should handle very large configuration file")
        fun shouldHandleVeryLargeConfigurationFile() {
            // Given
            val largeConfig = StringBuilder()
            largeConfig.append("{\n")
            largeConfig.append("  \"appName\": \"TestApp\",\n")
            largeConfig.append("  \"version\": \"1.0.0\",\n")
            largeConfig.append("  \"database\": {\n")
            largeConfig.append("    \"host\": \"localhost\",\n")
            largeConfig.append("    \"port\": 5432\n")
            largeConfig.append("  },\n")
            largeConfig.append("  \"data\": [\n")
            
            // Generate a large array to test performance
            for (i in 1..10000) {
                largeConfig.append("    {\"id\": $i, \"value\": \"data_$i\"}")
                if (i < 10000) largeConfig.append(",")
                largeConfig.append("\n")
            }
            
            largeConfig.append("  ]\n")
            largeConfig.append("}")
            
            testConfigFile.writeText(largeConfig.toString())

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertTrue(result.isValid, "Large but valid configuration should pass validation")
            assertTrue(result.errors.isEmpty(), "Large but valid configuration should have no errors")
        }

        @Test
        @DisplayName("Should handle configuration with deeply nested objects")
        fun shouldHandleConfigurationWithDeeplyNestedObjects() {
            // Given
            val deeplyNestedConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    },
                    "level1": {
                        "level2": {
                            "level3": {
                                "level4": {
                                    "level5": {
                                        "value": "deep_value"
                                    }
                                }
                            }
                        }
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(deeplyNestedConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertTrue(result.isValid, "Configuration with deeply nested objects should pass validation")
            assertTrue(result.errors.isEmpty(), "Configuration with deeply nested objects should have no errors")
        }

        @Test
        @DisplayName("Should handle configuration with special characters in strings")
        fun shouldHandleConfigurationWithSpecialCharacters() {
            // Given
            val specialCharsConfig = """
                {
                    "appName": "Test App with Special chars: @#$%^&*()",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    },
                    "description": "This is a test with unicode: 你好世界 and emojis: 🚀🎉",
                    "paths": {
                        "windows": "C:\\Program Files\\App",
                        "unix": "/usr/local/bin/app"
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(specialCharsConfig)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertTrue(result.isValid, "Configuration with special characters should pass validation")
            assertTrue(result.errors.isEmpty(), "Configuration with special characters should have no errors")
        }
    }

    @Nested
    @DisplayName("Schema Validation Tests")
    inner class SchemaValidationTests {

        @Test
        @DisplayName("Should validate against JSON schema if available")
        fun shouldValidateAgainstJsonSchema() {
            // Given
            val validConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(validConfig)

            // When
            val result = configValidator.validateWithSchema(testConfigFile)

            // Then
            assertTrue(result.isValid, "Valid configuration should pass schema validation")
            assertTrue(result.errors.isEmpty(), "Valid configuration should have no schema errors")
        }

        @Test
        @DisplayName("Should report schema validation errors")
        fun shouldReportSchemaValidationErrors() {
            // Given
            val invalidSchemaConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    },
                    "unknownField": "should not be allowed"
                }
            """.trimIndent()
            
            testConfigFile.writeText(invalidSchemaConfig)

            // When
            val result = configValidator.validateWithSchema(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with unknown fields should fail schema validation")
            assertFalse(result.errors.isEmpty(), "Configuration with unknown fields should have schema errors")
            assertTrue(result.errors.any { it.contains("unknownField") }, 
                "Should report unknown field error")
        }
    }

    @Nested
    @DisplayName("Performance and Resource Tests")
    inner class PerformanceAndResourceTests {

        @Test
        @DisplayName("Should handle multiple concurrent validations")
        fun shouldHandleMultipleConcurrentValidations() {
            // Given
            val validConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            val configFiles = mutableListOf<Path>()
            for (i in 1..10) {
                val configFile = tempDir.resolve("test-config-$i.json")
                configFile.writeText(validConfig)
                configFiles.add(configFile)
            }

            // When
            val results = configFiles.parallelStream()
                .map { configValidator.validate(it) }
                .toList()

            // Then
            assertTrue(results.all { it.isValid }, "All concurrent validations should pass")
            assertTrue(results.all { it.errors.isEmpty() }, "All concurrent validations should have no errors")
        }

        @Test
        @DisplayName("Should validate configuration within reasonable time")
        fun shouldValidateConfigurationWithinReasonableTime() {
            // Given
            val validConfig = """
                {
                    "appName": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(validConfig)

            // When
            val startTime = System.currentTimeMillis()
            val result = configValidator.validate(testConfigFile)
            val endTime = System.currentTimeMillis()

            // Then
            assertTrue(result.isValid, "Valid configuration should pass validation")
            assertTrue(endTime - startTime < 5000, "Validation should complete within 5 seconds")
        }
    }

    @Nested
    @DisplayName("Error Message Quality Tests")
    inner class ErrorMessageQualityTests {

        @Test
        @DisplayName("Should provide clear error messages for missing fields")
        fun shouldProvideClearErrorMessagesForMissingFields() {
            // Given
            val configWithMissingFields = """
                {
                    "version": "1.0.0"
                }
            """.trimIndent()
            
            testConfigFile.writeText(configWithMissingFields)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with missing fields should fail validation")
            assertTrue(result.errors.any { error ->
                error.contains("appName") && error.contains("required")
            }, "Should provide clear error message for missing appName")
        }

        @Test
        @DisplayName("Should provide helpful suggestions in error messages")
        fun shouldProvideHelpfulSuggestionsInErrorMessages() {
            // Given
            val configWithTypo = """
                {
                    "appname": "TestApp",
                    "version": "1.0.0",
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }
            """.trimIndent()
            
            testConfigFile.writeText(configWithTypo)

            // When
            val result = configValidator.validate(testConfigFile)

            // Then
            assertFalse(result.isValid, "Configuration with field typo should fail validation")
            assertTrue(result.errors.any { error ->
                error.contains("appName") && (error.contains("suggestion") || error.contains("did you mean"))
            }, "Should provide helpful suggestion for field typo")
        }
    }
}

// Supporting classes and data classes for testing
data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>
)

class ConfigValidator {
    fun validate(configFile: Path): ValidationResult {
        // Implementation would go here
        return ValidationResult(true, emptyList())
    }
    
    fun validateWithSchema(configFile: Path): ValidationResult {
        // Implementation would go here
        return ValidationResult(true, emptyList())
    }
}

class ConfigurationException(message: String) : Exception(message)