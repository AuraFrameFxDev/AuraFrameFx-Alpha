package dev.gradle;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.MockitoAnnotations;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

/**
 * Comprehensive unit tests for GradleProperties class.
 * Tests cover happy paths, edge cases, and failure conditions.
 */
@DisplayName("GradleProperties Tests")
class GradlePropertiesTest {

    @TempDir
    Path tempDir;
    
    private GradleProperties gradleProperties;
    private File propertiesFile;
    private AutoCloseable closeable;

    @BeforeEach
    void setUp() throws IOException {
        closeable = MockitoAnnotations.openMocks(this);
        propertiesFile = tempDir.resolve("gradle.properties").toFile();
        gradleProperties = new GradleProperties(propertiesFile);
    }

    @AfterEach
    void tearDown() throws Exception {
        if (closeable != null) {
            closeable.close();
        }
    }

    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {

        @Test
        @DisplayName("Should create GradleProperties with valid file")
        void shouldCreateWithValidFile() throws IOException {
            File validFile = tempDir.resolve("valid.properties").toFile();
            validFile.createNewFile();
            
            assertDoesNotThrow(() -> new GradleProperties(validFile));
        }

        @Test
        @DisplayName("Should create GradleProperties with non-existent file")
        void shouldCreateWithNonExistentFile() {
            File nonExistentFile = tempDir.resolve("nonexistent.properties").toFile();
            
            assertDoesNotThrow(() -> new GradleProperties(nonExistentFile));
        }

        @Test
        @DisplayName("Should throw exception for null file")
        void shouldThrowExceptionForNullFile() {
            assertThrows(IllegalArgumentException.class, () -> new GradleProperties(null));
        }

        @Test
        @DisplayName("Should create GradleProperties with directory as file")
        void shouldHandleDirectoryAsFile() throws IOException {
            File directory = tempDir.resolve("directory").toFile();
            directory.mkdir();
            
            assertDoesNotThrow(() -> new GradleProperties(directory));
        }
    }

    @Nested
    @DisplayName("Property Reading Tests")
    class PropertyReadingTests {

        @Test
        @DisplayName("Should read existing property")
        void shouldReadExistingProperty() throws IOException {
            writePropertiesFile("key1=value1\nkey2=value2");
            
            String result = gradleProperties.getProperty("key1");
            
            assertEquals("value1", result);
        }

        @Test
        @DisplayName("Should return null for non-existent property")
        void shouldReturnNullForNonExistentProperty() throws IOException {
            writePropertiesFile("key1=value1");
            
            String result = gradleProperties.getProperty("nonexistent");
            
            assertNull(result);
        }

        @Test
        @DisplayName("Should return default value for non-existent property")
        void shouldReturnDefaultValueForNonExistentProperty() throws IOException {
            writePropertiesFile("key1=value1");
            
            String result = gradleProperties.getProperty("nonexistent", "default");
            
            assertEquals("default", result);
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("Should handle null and empty keys")
        void shouldHandleNullAndEmptyKeys(String key) throws IOException {
            writePropertiesFile("key1=value1");
            
            String result = gradleProperties.getProperty(key);
            
            assertNull(result);
        }

        @Test
        @DisplayName("Should read properties with special characters")
        void shouldReadPropertiesWithSpecialCharacters() throws IOException {
            writePropertiesFile("key.with.dots=value\nkey-with-dashes=value2\nkey_with_underscores=value3");
            
            assertEquals("value", gradleProperties.getProperty("key.with.dots"));
            assertEquals("value2", gradleProperties.getProperty("key-with-dashes"));
            assertEquals("value3", gradleProperties.getProperty("key_with_underscores"));
        }

        @Test
        @DisplayName("Should handle properties with empty values")
        void shouldHandlePropertiesWithEmptyValues() throws IOException {
            writePropertiesFile("emptyKey=\nspaceKey= \ntabKey=\t");
            
            assertEquals("", gradleProperties.getProperty("emptyKey"));
            assertEquals(" ", gradleProperties.getProperty("spaceKey"));
            assertEquals("\t", gradleProperties.getProperty("tabKey"));
        }

        @Test
        @DisplayName("Should handle properties with unicode characters")
        void shouldHandlePropertiesWithUnicodeCharacters() throws IOException {
            writePropertiesFile("unicode.key=café\nrussian.key=привет");
            
            assertEquals("café", gradleProperties.getProperty("unicode.key"));
            assertEquals("привет", gradleProperties.getProperty("russian.key"));
        }
    }

    @Nested
    @DisplayName("Property Writing Tests")
    class PropertyWritingTests {

        @Test
        @DisplayName("Should write new property")
        void shouldWriteNewProperty() throws IOException {
            gradleProperties.setProperty("newKey", "newValue");
            
            assertEquals("newValue", gradleProperties.getProperty("newKey"));
        }

        @Test
        @DisplayName("Should overwrite existing property")
        void shouldOverwriteExistingProperty() throws IOException {
            writePropertiesFile("existingKey=oldValue");
            
            gradleProperties.setProperty("existingKey", "newValue");
            
            assertEquals("newValue", gradleProperties.getProperty("existingKey"));
        }

        @Test
        @DisplayName("Should handle null value")
        void shouldHandleNullValue() throws IOException {
            gradleProperties.setProperty("key", null);
            
            assertNull(gradleProperties.getProperty("key"));
        }

        @Test
        @DisplayName("Should throw exception for null key")
        void shouldThrowExceptionForNullKey() {
            assertThrows(IllegalArgumentException.class, 
                () -> gradleProperties.setProperty(null, "value"));
        }

        @Test
        @DisplayName("Should handle empty string value")
        void shouldHandleEmptyStringValue() throws IOException {
            gradleProperties.setProperty("emptyKey", "");
            
            assertEquals("", gradleProperties.getProperty("emptyKey"));
        }

        @Test
        @DisplayName("Should handle special characters in values")
        void shouldHandleSpecialCharactersInValues() throws IOException {
            String specialValue = "!@#$%^&*()_+{}|:<>?[];',./~`";
            gradleProperties.setProperty("specialKey", specialValue);
            
            assertEquals(specialValue, gradleProperties.getProperty("specialKey"));
        }
    }

    @Nested
    @DisplayName("File Operations Tests")
    class FileOperationsTests {

        @Test
        @DisplayName("Should save properties to file")
        void shouldSavePropertiesToFile() throws IOException {
            gradleProperties.setProperty("key1", "value1");
            gradleProperties.setProperty("key2", "value2");
            
            gradleProperties.save();
            
            assertTrue(propertiesFile.exists());
            String content = Files.readString(propertiesFile.toPath());
            assertTrue(content.contains("key1=value1"));
            assertTrue(content.contains("key2=value2"));
        }

        @Test
        @DisplayName("Should load properties from file")
        void shouldLoadPropertiesFromFile() throws IOException {
            writePropertiesFile("loadKey1=loadValue1\nloadKey2=loadValue2");
            
            gradleProperties.load();
            
            assertEquals("loadValue1", gradleProperties.getProperty("loadKey1"));
            assertEquals("loadValue2", gradleProperties.getProperty("loadKey2"));
        }

        @Test
        @DisplayName("Should handle file not found during load")
        void shouldHandleFileNotFoundDuringLoad() {
            File nonExistentFile = tempDir.resolve("nonexistent.properties").toFile();
            GradleProperties props = new GradleProperties(nonExistentFile);
            
            assertDoesNotThrow(() -> props.load());
        }

        @Test
        @DisplayName("Should handle IOException during save")
        void shouldHandleIOExceptionDuringSave() throws IOException {
            File readOnlyFile = tempDir.resolve("readonly.properties").toFile();
            readOnlyFile.createNewFile();
            readOnlyFile.setReadOnly();
            
            GradleProperties props = new GradleProperties(readOnlyFile);
            props.setProperty("key", "value");
            
            assertThrows(IOException.class, () -> props.save());
        }

        @Test
        @DisplayName("Should preserve file permissions after save")
        void shouldPreserveFilePermissionsAfterSave() throws IOException {
            propertiesFile.createNewFile();
            propertiesFile.setExecutable(true);
            boolean wasExecutable = propertiesFile.canExecute();
            
            gradleProperties.setProperty("key", "value");
            gradleProperties.save();
            
            assertEquals(wasExecutable, propertiesFile.canExecute());
        }
    }

    @Nested
    @DisplayName("Property Validation Tests")
    class PropertyValidationTests {

        @ParameterizedTest
        @ValueSource(strings = {"true", "TRUE", "True", "1", "yes", "YES", "Yes"})
        @DisplayName("Should parse boolean true values")
        void shouldParseBooleanTrueValues(String value) throws IOException {
            writePropertiesFile("booleanKey=" + value);
            
            assertTrue(gradleProperties.getBooleanProperty("booleanKey"));
        }

        @ParameterizedTest
        @ValueSource(strings = {"false", "FALSE", "False", "0", "no", "NO", "No"})
        @DisplayName("Should parse boolean false values")
        void shouldParseBooleanFalseValues(String value) throws IOException {
            writePropertiesFile("booleanKey=" + value);
            
            assertFalse(gradleProperties.getBooleanProperty("booleanKey"));
        }

        @Test
        @DisplayName("Should return default boolean for non-existent property")
        void shouldReturnDefaultBooleanForNonExistentProperty() throws IOException {
            writePropertiesFile("key=value");
            
            assertTrue(gradleProperties.getBooleanProperty("nonexistent", true));
            assertFalse(gradleProperties.getBooleanProperty("nonexistent", false));
        }

        @ParameterizedTest
        @CsvSource({
            "123, 123",
            "-456, -456",
            "0, 0",
            "2147483647, 2147483647",
            "-2147483648, -2147483648"
        })
        @DisplayName("Should parse integer values")
        void shouldParseIntegerValues(String value, int expected) throws IOException {
            writePropertiesFile("intKey=" + value);
            
            assertEquals(expected, gradleProperties.getIntProperty("intKey"));
        }

        @Test
        @DisplayName("Should throw exception for invalid integer")
        void shouldThrowExceptionForInvalidInteger() throws IOException {
            writePropertiesFile("intKey=notAnInteger");
            
            assertThrows(NumberFormatException.class, 
                () -> gradleProperties.getIntProperty("intKey"));
        }

        @Test
        @DisplayName("Should return default integer for non-existent property")
        void shouldReturnDefaultIntegerForNonExistentProperty() throws IOException {
            writePropertiesFile("key=value");
            
            assertEquals(42, gradleProperties.getIntProperty("nonexistent", 42));
        }
    }

    @Nested
    @DisplayName("Property Listing Tests")
    class PropertyListingTests {

        @Test
        @DisplayName("Should list all property names")
        void shouldListAllPropertyNames() throws IOException {
            writePropertiesFile("key1=value1\nkey2=value2\nkey3=value3");
            
            var propertyNames = gradleProperties.getPropertyNames();
            
            assertEquals(3, propertyNames.size());
            assertTrue(propertyNames.contains("key1"));
            assertTrue(propertyNames.contains("key2"));
            assertTrue(propertyNames.contains("key3"));
        }

        @Test
        @DisplayName("Should return empty list for no properties")
        void shouldReturnEmptyListForNoProperties() throws IOException {
            writePropertiesFile("");
            
            var propertyNames = gradleProperties.getPropertyNames();
            
            assertTrue(propertyNames.isEmpty());
        }

        @Test
        @DisplayName("Should list properties with filter")
        void shouldListPropertiesWithFilter() throws IOException {
            writePropertiesFile("app.name=MyApp\napp.version=1.0\ndb.url=jdbc:mysql");
            
            var appProperties = gradleProperties.getPropertiesWithPrefix("app.");
            
            assertEquals(2, appProperties.size());
            assertTrue(appProperties.containsKey("app.name"));
            assertTrue(appProperties.containsKey("app.version"));
            assertFalse(appProperties.containsKey("db.url"));
        }
    }

    @Nested
    @DisplayName("Property Removal Tests")
    class PropertyRemovalTests {

        @Test
        @DisplayName("Should remove existing property")
        void shouldRemoveExistingProperty() throws IOException {
            writePropertiesFile("key1=value1\nkey2=value2");
            
            gradleProperties.removeProperty("key1");
            
            assertNull(gradleProperties.getProperty("key1"));
            assertEquals("value2", gradleProperties.getProperty("key2"));
        }

        @Test
        @DisplayName("Should handle removal of non-existent property")
        void shouldHandleRemovalOfNonExistentProperty() throws IOException {
            writePropertiesFile("key1=value1");
            
            assertDoesNotThrow(() -> gradleProperties.removeProperty("nonexistent"));
        }

        @Test
        @DisplayName("Should clear all properties")
        void shouldClearAllProperties() throws IOException {
            writePropertiesFile("key1=value1\nkey2=value2\nkey3=value3");
            
            gradleProperties.clear();
            
            assertTrue(gradleProperties.getPropertyNames().isEmpty());
        }
    }

    @Nested
    @DisplayName("Concurrent Access Tests")
    class ConcurrentAccessTests {

        @Test
        @DisplayName("Should handle concurrent read operations")
        void shouldHandleConcurrentReadOperations() throws IOException, InterruptedException {
            writePropertiesFile("key1=value1\nkey2=value2");
            
            Thread[] threads = new Thread[10];
            for (int i = 0; i < threads.length; i++) {
                threads[i] = new Thread(() -> {
                    for (int j = 0; j < 100; j++) {
                        assertEquals("value1", gradleProperties.getProperty("key1"));
                    }
                });
            }
            
            for (Thread thread : threads) {
                thread.start();
            }
            
            for (Thread thread : threads) {
                thread.join();
            }
        }

        @Test
        @DisplayName("Should handle concurrent write operations")
        void shouldHandleConcurrentWriteOperations() throws InterruptedException {
            Thread[] threads = new Thread[10];
            for (int i = 0; i < threads.length; i++) {
                int threadId = i;
                threads[i] = new Thread(() -> {
                    for (int j = 0; j < 10; j++) {
                        gradleProperties.setProperty("key" + threadId, "value" + j);
                    }
                });
            }
            
            for (Thread thread : threads) {
                thread.start();
            }
            
            for (Thread thread : threads) {
                thread.join();
            }
            
            assertEquals(10, gradleProperties.getPropertyNames().size());
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {

        @Test
        @DisplayName("Should handle corrupted properties file")
        void shouldHandleCorruptedPropertiesFile() throws IOException {
            writePropertiesFile("validKey=validValue\n\u0000\u0001\u0002corrupted data");
            
            assertDoesNotThrow(() -> gradleProperties.load());
        }

        @Test
        @DisplayName("Should handle very long property values")
        void shouldHandleVeryLongPropertyValues() throws IOException {
            String longValue = "a".repeat(10000);
            gradleProperties.setProperty("longKey", longValue);
            
            assertEquals(longValue, gradleProperties.getProperty("longKey"));
        }

        @Test
        @DisplayName("Should handle properties with equals signs in values")
        void shouldHandlePropertiesWithEqualsSignsInValues() throws IOException {
            writePropertiesFile("urlKey=http://example.com?param=value&other=value2");
            
            assertEquals("http://example.com?param=value&other=value2", 
                gradleProperties.getProperty("urlKey"));
        }

        @Test
        @DisplayName("Should handle properties with line breaks in values")
        void shouldHandlePropertiesWithLineBreaksInValues() throws IOException {
            writePropertiesFile("multilineKey=line1\\\n    line2\\\n    line3");
            
            String result = gradleProperties.getProperty("multilineKey");
            assertNotNull(result);
            assertTrue(result.contains("line1"));
        }
    }

    // Helper method to write properties to test file
    private void writePropertiesFile(String content) throws IOException {
        try (FileWriter writer = new FileWriter(propertiesFile)) {
            writer.write(content);
        }
    }
}