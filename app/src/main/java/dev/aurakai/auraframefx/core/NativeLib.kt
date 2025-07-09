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
         * A native method that is implemented by the 'aura-native-lib' native library,
         * which is packaged with this application.
         */
        external fun stringFromJNI(): String

        // --- Language Identification JNI Functions ---

        /**
         * Initializes the native language identifier.
         * @param modelPath Path to the language model (currently logged, not used by detector).
         * @return A version string or status indicator. For example, "1.2.0".
         */
        external fun nativeInitialize(modelPath: String): String

        /**
         * Detects the language of the given text.
         * @param handle A handle obtained from nativeInitialize (if applicable, current C++ code doesn't use it).
         *               Consider removing or redesigning if nativeInitialize doesn't provide a meaningful handle.
         * @param text The text to analyze.
         * @return A string representing the detected language code (e.g., "en", "es").
         */
        external fun nativeDetectLanguage(handle: Long, text: String): String // Or (text: String) if handle is not used

        /**
         * Releases any resources associated with the language identifier.
         * @param handle A handle obtained from nativeInitialize (if applicable).
         */
        external fun nativeRelease(handle: Long) // Or remove if no handle/resources

        /**
         * Gets the version of the native language detection library.
         * @return The version string.
         */
        external fun nativeGetVersion(): String
    }
}
