package dev.aurakai.auraframefx.core

class NativeLib {

    companion object {
        // Used to load the 'native-lib' library on application startup.
        init {
            try {
                System.loadLibrary("aura-native-lib") // Corrected to match CMakeLists.txt target name
            } catch (e: UnsatisfiedLinkError) {
                // Log error or handle exception if library fails to load
                // This is crucial for debugging JNI issues.
                System.err.println("Native library failed to load: aura-native-lib\n$e")
                // Potentially re-throw or provide a fallback if critical
            }
        }

        /**
 * Returns a string from the native 'aura-native-lib' library.
 *
 * The content of the returned string is determined by the native implementation.
 * @return A string provided by the native library.
 */
        external fun stringFromJNI(): String

        // --- Language Identification JNI Functions ---

        /**
 * Initializes the native language identifier with the specified model path.
 *
 * @param modelPath The file system path to the language model. This parameter may be logged but is not currently used by the detector.
 * @return A string indicating the version or status of the native language identifier, such as "1.2.0".
 */
        external fun nativeInitialize(modelPath: String): String

        /**
 * Identifies the language of the specified text using the native language detection library.
 *
 * @param handle The native handle for the language identifier, typically obtained from `nativeInitialize`. May be unused depending on native implementation.
 * @param text The text to analyze for language detection.
 * @return The detected language code as a string (e.g., "en", "es").
 */
        external fun nativeDetectLanguage(handle: Long, text: String): String // Or (text: String) if handle is not used

        /**
 * Releases resources associated with the specified language identifier handle.
 *
 * @param handle The handle referencing the native language identifier instance to be released.
 */
        external fun nativeRelease(handle: Long) // Or remove if no handle/resources

        /**
 * Returns the version string of the native language detection library.
 *
 * @return The version string of the native library.
 */
        external fun nativeGetVersion(): String
    }
}
