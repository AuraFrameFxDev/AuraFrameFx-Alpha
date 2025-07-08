package dev.aurakai.auraframefx.config

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.junit.jupiter.params.provider.CsvSource
import org.junit.jupiter.params.provider.NullSource
import org.junit.jupiter.params.provider.EmptySource
import org.junit.jupiter.params.provider.NullAndEmptySource
import java.io.File
import java.nio.file.Path
import java.util.Properties
import kotlin.test.assertFailsWith

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("GradleProperties Tests")
class GradlePropertiesTest {

    @TempDir
    lateinit var tempDir: Path

    private lateinit var gradleProperties: GradleProperties
    private lateinit var propertiesFile: File

    @BeforeEach
    fun setUp() {
        propertiesFile = tempDir.resolve("gradle.properties").toFile()
        gradleProperties = GradleProperties(propertiesFile.absolutePath)
    }

    @AfterEach
    fun tearDown() {
        if (propertiesFile.exists()) {
            propertiesFile.delete()
        }
    }

    @Nested
    @DisplayName("Constructor Tests")
    inner class ConstructorTests {

        @Test
        @DisplayName("Should create GradleProperties with valid file path")
        fun shouldCreateWithValidFilePath() {
            // Given
            val validPath = tempDir.resolve("valid.properties").toString()
            
            // When
            val result = GradleProperties(validPath)
            
            // Then
            assertNotNull(result)
            assertEquals(validPath, result.filePath)
        }

        @Test
        @DisplayName("Should handle non-existent file path gracefully")
        fun shouldHandleNonExistentFile() {
            // Given
            val nonExistentPath = tempDir.resolve("non_existent.properties").toString()
            
            // When & Then
            assertDoesNotThrow { GradleProperties(nonExistentPath) }
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("Should handle null or empty file path")
        fun shouldHandleInvalidFilePath(path: String?) {
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                GradleProperties(path ?: "")
            }
        }
    }

    @Nested
    @DisplayName("Property Loading Tests")
    inner class PropertyLoadingTests {

        @Test
        @DisplayName("Should load properties from valid file")
        fun shouldLoadPropertiesFromValidFile() {
            // Given
            propertiesFile.writeText("""
                app.name=AuraFrameFX
                app.version=1.0.0
                app.description=A modern JavaFX application
            """.trimIndent())
            
            // When
            gradleProperties.loadProperties()
            
            // Then
            assertEquals("AuraFrameFX", gradleProperties.getProperty("app.name"))
            assertEquals("1.0.0", gradleProperties.getProperty("app.version"))
            assertEquals("A modern JavaFX application", gradleProperties.getProperty("app.description"))
        }

        @Test
        @DisplayName("Should handle empty properties file")
        fun shouldHandleEmptyPropertiesFile() {
            // Given
            propertiesFile.writeText("")
            
            // When
            gradleProperties.loadProperties()
            
            // Then
            assertTrue(gradleProperties.isEmpty())
        }

        @Test
        @DisplayName("Should handle malformed properties file")
        fun shouldHandleMalformedPropertiesFile() {
            // Given
            propertiesFile.writeText("""
                invalid line without equals
                valid.property=value
                another=invalid=line=with=multiple=equals
            """.trimIndent())
            
            // When & Then
            assertDoesNotThrow { gradleProperties.loadProperties() }
            assertEquals("value", gradleProperties.getProperty("valid.property"))
        }

        @Test
        @DisplayName("Should handle properties with special characters")
        fun shouldHandleSpecialCharacters() {
            // Given
            propertiesFile.writeText("""
                special.chars=!@#$%^&*()
                unicode.property=éñümlaut
                spaces.property=value with spaces
                empty.property=
            """.trimIndent())
            
            // When
            gradleProperties.loadProperties()
            
            // Then
            assertEquals("!@#$%^&*()", gradleProperties.getProperty("special.chars"))
            assertEquals("éñümlaut", gradleProperties.getProperty("unicode.property"))
            assertEquals("value with spaces", gradleProperties.getProperty("spaces.property"))
            assertEquals("", gradleProperties.getProperty("empty.property"))
        }

        @Test
        @DisplayName("Should handle file IO errors gracefully")
        fun shouldHandleFileIOErrors() {
            // Given
            val readOnlyDir = tempDir.resolve("readonly").toFile()
            readOnlyDir.mkdirs()
            readOnlyDir.setReadOnly()
            val inaccessibleFile = readOnlyDir.resolve("inaccessible.properties")
            val gradlePropsWithInaccessibleFile = GradleProperties(inaccessibleFile.absolutePath)
            
            // When & Then
            assertDoesNotThrow { gradlePropsWithInaccessibleFile.loadProperties() }
            
            // Cleanup
            readOnlyDir.setWritable(true)
        }
    }

    @Nested
    @DisplayName("Property Retrieval Tests")
    inner class PropertyRetrievalTests {

        @BeforeEach
        fun setUpProperties() {
            propertiesFile.writeText("""
                string.property=test_value
                number.property=12345
                boolean.property=true
                empty.property=
                null.property
            """.trimIndent())
            gradleProperties.loadProperties()
        }

        @Test
        @DisplayName("Should retrieve existing property")
        fun shouldRetrieveExistingProperty() {
            // When
            val result = gradleProperties.getProperty("string.property")
            
            // Then
            assertEquals("test_value", result)
        }

        @Test
        @DisplayName("Should return default value for non-existent property")
        fun shouldReturnDefaultValueForNonExistentProperty() {
            // When
            val result = gradleProperties.getProperty("non.existent", "default_value")
            
            // Then
            assertEquals("default_value", result)
        }

        @Test
        @DisplayName("Should return null for non-existent property without default")
        fun shouldReturnNullForNonExistentProperty() {
            // When
            val result = gradleProperties.getProperty("non.existent")
            
            // Then
            assertNull(result)
        }

        @ParameterizedTest
        @ValueSource(strings = ["string.property", "number.property", "boolean.property"])
        @DisplayName("Should retrieve various property types")
        fun shouldRetrieveVariousPropertyTypes(propertyName: String) {
            // When
            val result = gradleProperties.getProperty(propertyName)
            
            // Then
            assertNotNull(result)
            assertTrue(result.isNotEmpty())
        }

        @Test
        @DisplayName("Should handle empty property values")
        fun shouldHandleEmptyPropertyValues() {
            // When
            val result = gradleProperties.getProperty("empty.property")
            
            // Then
            assertEquals("", result)
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("Should handle invalid property keys")
        fun shouldHandleInvalidPropertyKeys(key: String?) {
            // When
            val result = gradleProperties.getProperty(key ?: "")
            
            // Then
            assertNull(result)
        }
    }

    @Nested
    @DisplayName("Property Setting Tests")
    inner class PropertySettingTests {

        @Test
        @DisplayName("Should set new property")
        fun shouldSetNewProperty() {
            // Given
            val key = "new.property"
            val value = "new_value"
            
            // When
            gradleProperties.setProperty(key, value)
            
            // Then
            assertEquals(value, gradleProperties.getProperty(key))
        }

        @Test
        @DisplayName("Should update existing property")
        fun shouldUpdateExistingProperty() {
            // Given
            gradleProperties.setProperty("existing.property", "old_value")
            
            // When
            gradleProperties.setProperty("existing.property", "new_value")
            
            // Then
            assertEquals("new_value", gradleProperties.getProperty("existing.property"))
        }

        @ParameterizedTest
        @CsvSource(
            "'string.key', 'string_value'",
            "'number.key', '123'",
            "'boolean.key', 'true'",
            "'empty.key', ''",
            "'special.key', '!@#$%^&*()'"
        )
        @DisplayName("Should set properties with various value types")
        fun shouldSetPropertiesWithVariousValueTypes(key: String, value: String) {
            // When
            gradleProperties.setProperty(key, value)
            
            // Then
            assertEquals(value, gradleProperties.getProperty(key))
        }

        @Test
        @DisplayName("Should handle null values gracefully")
        fun shouldHandleNullValues() {
            // When & Then
            assertDoesNotThrow { gradleProperties.setProperty("null.key", null) }
            assertNull(gradleProperties.getProperty("null.key"))
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("Should handle invalid property keys when setting")
        fun shouldHandleInvalidPropertyKeysWhenSetting(key: String?) {
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                gradleProperties.setProperty(key ?: "", "value")
            }
        }
    }

    @Nested
    @DisplayName("Property Removal Tests")
    inner class PropertyRemovalTests {

        @BeforeEach
        fun setUpProperties() {
            propertiesFile.writeText("""
                removable.property=value
                another.property=another_value
            """.trimIndent())
            gradleProperties.loadProperties()
        }

        @Test
        @DisplayName("Should remove existing property")
        fun shouldRemoveExistingProperty() {
            // Given
            assertTrue(gradleProperties.containsProperty("removable.property"))
            
            // When
            gradleProperties.removeProperty("removable.property")
            
            // Then
            assertFalse(gradleProperties.containsProperty("removable.property"))
            assertNull(gradleProperties.getProperty("removable.property"))
        }

        @Test
        @DisplayName("Should handle removal of non-existent property")
        fun shouldHandleRemovalOfNonExistentProperty() {
            // When & Then
            assertDoesNotThrow { gradleProperties.removeProperty("non.existent") }
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("Should handle invalid property keys when removing")
        fun shouldHandleInvalidPropertyKeysWhenRemoving(key: String?) {
            // When & Then
            assertFailsWith<IllegalArgumentException> {
                gradleProperties.removeProperty(key ?: "")
            }
        }
    }

    @Nested
    @DisplayName("Property Existence Tests")
    inner class PropertyExistenceTests {

        @BeforeEach
        fun setUpProperties() {
            propertiesFile.writeText("""
                existing.property=value
                empty.property=
            """.trimIndent())
            gradleProperties.loadProperties()
        }

        @Test
        @DisplayName("Should return true for existing property")
        fun shouldReturnTrueForExistingProperty() {
            // When
            val result = gradleProperties.containsProperty("existing.property")
            
            // Then
            assertTrue(result)
        }

        @Test
        @DisplayName("Should return true for existing property with empty value")
        fun shouldReturnTrueForExistingPropertyWithEmptyValue() {
            // When
            val result = gradleProperties.containsProperty("empty.property")
            
            // Then
            assertTrue(result)
        }

        @Test
        @DisplayName("Should return false for non-existent property")
        fun shouldReturnFalseForNonExistentProperty() {
            // When
            val result = gradleProperties.containsProperty("non.existent")
            
            // Then
            assertFalse(result)
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("Should handle invalid property keys when checking existence")
        fun shouldHandleInvalidPropertyKeysWhenCheckingExistence(key: String?) {
            // When
            val result = gradleProperties.containsProperty(key ?: "")
            
            // Then
            assertFalse(result)
        }
    }

    @Nested
    @DisplayName("File Operations Tests")
    inner class FileOperationsTests {

        @Test
        @DisplayName("Should save properties to file")
        fun shouldSavePropertiesToFile() {
            // Given
            gradleProperties.setProperty("save.test", "saved_value")
            
            // When
            gradleProperties.saveProperties()
            
            // Then
            val loadedProperties = GradleProperties(propertiesFile.absolutePath)
            loadedProperties.loadProperties()
            assertEquals("saved_value", loadedProperties.getProperty("save.test"))
        }

        @Test
        @DisplayName("Should handle save errors gracefully")
        fun shouldHandleSaveErrorsGracefully() {
            // Given
            val readOnlyDir = tempDir.resolve("readonly").toFile()
            readOnlyDir.mkdirs()
            readOnlyDir.setReadOnly()
            val readOnlyFile = readOnlyDir.resolve("readonly.properties")
            val gradlePropsWithReadOnlyFile = GradleProperties(readOnlyFile.absolutePath)
            
            // When & Then
            assertDoesNotThrow { gradlePropsWithReadOnlyFile.saveProperties() }
            
            // Cleanup
            readOnlyDir.setWritable(true)
        }

        @Test
        @DisplayName("Should create parent directories if they don't exist")
        fun shouldCreateParentDirectoriesIfTheyDontExist() {
            // Given
            val newDir = tempDir.resolve("new").resolve("nested").resolve("dir")
            val nestedFile = newDir.resolve("nested.properties").toFile()
            val gradlePropsWithNestedFile = GradleProperties(nestedFile.absolutePath)
            
            // When
            gradlePropsWithNestedFile.setProperty("nested.test", "value")
            gradlePropsWithNestedFile.saveProperties()
            
            // Then
            assertTrue(nestedFile.exists())
            assertTrue(nestedFile.parentFile.exists())
        }
    }

    @Nested
    @DisplayName("Utility Methods Tests")
    inner class UtilityMethodsTests {

        @BeforeEach
        fun setUpProperties() {
            propertiesFile.writeText("""
                prop1=value1
                prop2=value2
                prop3=value3
            """.trimIndent())
            gradleProperties.loadProperties()
        }

        @Test
        @DisplayName("Should return correct size")
        fun shouldReturnCorrectSize() {
            // When
            val size = gradleProperties.size()
            
            // Then
            assertEquals(3, size)
        }

        @Test
        @DisplayName("Should return false for isEmpty when properties exist")
        fun shouldReturnFalseForIsEmptyWhenPropertiesExist() {
            // When
            val isEmpty = gradleProperties.isEmpty()
            
            // Then
            assertFalse(isEmpty)
        }

        @Test
        @DisplayName("Should return true for isEmpty when no properties exist")
        fun shouldReturnTrueForIsEmptyWhenNoPropertiesExist() {
            // Given
            val emptyGradleProps = GradleProperties(tempDir.resolve("empty.properties").toString())
            
            // When
            val isEmpty = emptyGradleProps.isEmpty()
            
            // Then
            assertTrue(isEmpty)
        }

        @Test
        @DisplayName("Should clear all properties")
        fun shouldClearAllProperties() {
            // Given
            assertFalse(gradleProperties.isEmpty())
            
            // When
            gradleProperties.clear()
            
            // Then
            assertTrue(gradleProperties.isEmpty())
            assertEquals(0, gradleProperties.size())
        }

        @Test
        @DisplayName("Should return all property keys")
        fun shouldReturnAllPropertyKeys() {
            // When
            val keys = gradleProperties.getAllKeys()
            
            // Then
            assertEquals(3, keys.size)
            assertTrue(keys.contains("prop1"))
            assertTrue(keys.contains("prop2"))
            assertTrue(keys.contains("prop3"))
        }

        @Test
        @DisplayName("Should return all property values")
        fun shouldReturnAllPropertyValues() {
            // When
            val values = gradleProperties.getAllValues()
            
            // Then
            assertEquals(3, values.size)
            assertTrue(values.contains("value1"))
            assertTrue(values.contains("value2"))
            assertTrue(values.contains("value3"))
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    inner class EdgeCasesAndErrorHandlingTests {

        @Test
        @DisplayName("Should handle concurrent access gracefully")
        fun shouldHandleConcurrentAccessGracefully() {
            // Given
            propertiesFile.writeText("concurrent.test=initial")
            gradleProperties.loadProperties()
            
            // When & Then
            assertDoesNotThrow {
                val threads = (1..10).map { threadId ->
                    Thread {
                        repeat(100) { iteration ->
                            gradleProperties.setProperty("thread.$threadId.iteration", iteration.toString())
                            gradleProperties.getProperty("concurrent.test")
                        }
                    }
                }
                threads.forEach { it.start() }
                threads.forEach { it.join() }
            }
        }

        @Test
        @DisplayName("Should handle large property values")
        fun shouldHandleLargePropertyValues() {
            // Given
            val largeValue = "x".repeat(10000)
            
            // When
            gradleProperties.setProperty("large.property", largeValue)
            
            // Then
            assertEquals(largeValue, gradleProperties.getProperty("large.property"))
        }

        @Test
        @DisplayName("Should handle many properties")
        fun shouldHandleManyProperties() {
            // Given
            val propertyCount = 1000
            
            // When
            repeat(propertyCount) { index ->
                gradleProperties.setProperty("property.$index", "value$index")
            }
            
            // Then
            assertEquals(propertyCount, gradleProperties.size())
            assertEquals("value500", gradleProperties.getProperty("property.500"))
        }

        @Test
        @DisplayName("Should handle property names with special characters")
        fun shouldHandlePropertyNamesWithSpecialCharacters() {
            // Given
            val specialKeys = listOf(
                "prop.with.dots",
                "prop-with-hyphens",
                "prop_with_underscores",
                "prop123with456numbers"
            )
            
            // When & Then
            specialKeys.forEach { key ->
                assertDoesNotThrow { gradleProperties.setProperty(key, "value") }
                assertEquals("value", gradleProperties.getProperty(key))
            }
        }

        @Test
        @DisplayName("Should handle system properties interaction")
        fun shouldHandleSystemPropertiesInteraction() {
            // Given
            val systemProp = "java.version"
            val systemValue = System.getProperty(systemProp)
            
            // When
            gradleProperties.setProperty(systemProp, "custom_value")
            
            // Then
            assertEquals("custom_value", gradleProperties.getProperty(systemProp))
            assertEquals(systemValue, System.getProperty(systemProp)) // System properties should be unchanged
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    inner class PerformanceTests {

        @Test
        @DisplayName("Should handle rapid property access efficiently")
        fun shouldHandleRapidPropertyAccessEfficiently() {
            // Given
            repeat(100) { index ->
                gradleProperties.setProperty("perf.prop.$index", "value$index")
            }
            
            // When
            val startTime = System.currentTimeMillis()
            repeat(10000) { index ->
                gradleProperties.getProperty("perf.prop.${index % 100}")
            }
            val endTime = System.currentTimeMillis()
            
            // Then
            val duration = endTime - startTime
            assertTrue(duration < 1000, "Property access should be efficient, took ${duration}ms")
        }

        @Test
        @DisplayName("Should handle batch property operations efficiently")
        fun shouldHandleBatchPropertyOperationsEfficiently() {
            // Given
            val batchSize = 500
            
            // When
            val startTime = System.currentTimeMillis()
            repeat(batchSize) { index ->
                gradleProperties.setProperty("batch.prop.$index", "value$index")
            }
            val endTime = System.currentTimeMillis()
            
            // Then
            val duration = endTime - startTime
            assertTrue(duration < 2000, "Batch property operations should be efficient, took ${duration}ms")
            assertEquals(batchSize, gradleProperties.size())
        }
    }
}