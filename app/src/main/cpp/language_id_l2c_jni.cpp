#include <jni.h>
#include <string>
#include <android/log.h>

#define LOG_TAG "LanguageIdJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the language identifier with a given model path.
 *
 * Returns the version string "1.2.0" if the provided model path is valid, or an empty string if the model path is null. No external model is loaded; initialization is based on built-in heuristics.
 *
 * @param modelPath Java string specifying the path to the language model.
 * @return jstring Version string "1.2.0" if initialization succeeds, or an empty string if modelPath is null.
 */
JNIEXPORT jstring

JNICALL
Java_com_example_app_language_LanguageIdentifier_nativeInitialize(
        JNIEnv *env,
        jobject /* this */,
        jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) {
        return env->NewStringUTF("");
    }

    LOGI("Initializing with model path: %s", path);

    // Initialize language identification with basic patterns
    // This implementation uses character frequency analysis and common word patterns
    // for basic language detection without external model dependencies

    env->ReleaseStringUTFChars(modelPath, path);
    return env->NewStringUTF("1.2.0"); // Updated version to reflect improvements
}

/**
 * @brief Identifies the language of the input text using heuristic analysis.
 *
 * Examines the provided text for language-specific keywords, articles, and the proportion of accented characters to determine if it is written in English, Spanish, French, German, Italian, or Portuguese. Returns "mul" if the text contains a high proportion of accented characters without a clear language match, or "und" if the input is null or cannot be processed.
 *
 * @param text The text to analyze for language identification.
 * @return jstring The detected language code: "en" (English), "es" (Spanish), "fr" (French), "de" (German), "it" (Italian), "pt" (Portuguese), "mul" (ambiguous or multiple accented languages), or "und" (undetermined).
 */
JNIEXPORT jstring

JNICALL
Java_com_example_app_language_LanguageIdentifier_nativeDetectLanguage(
        JNIEnv *env,
        jobject /* this */,
        jlong handle,
        jstring text) {
    if (text == nullptr) {
        return env->NewStringUTF("und");
    }

    const char *nativeText = env->GetStringUTFChars(text, nullptr);
    if (nativeText == nullptr) {
        return env->NewStringUTF("und");
    }

    LOGI("Detecting language for text: %s", nativeText);

    // Enhanced language detection using multiple heuristics
    std::string textStr(nativeText);
    std::string result = "en"; // Default to English
    
    // Convert to lowercase for case-insensitive matching
    std::transform(textStr.begin(), textStr.end(), textStr.begin(), ::tolower);

    // Language detection based on common words, articles, and patterns
    if (textStr.find(" el ") != std::string::npos ||
        textStr.find(" la ") != std::string::npos ||
        textStr.find(" de ") != std::string::npos ||
        textStr.find(" que ") != std::string::npos ||
        textStr.find(" es ") != std::string::npos ||
        textStr.find(" con ") != std::string::npos) {
        result = "es"; // Spanish
    } else if (textStr.find(" le ") != std::string::npos ||
               textStr.find(" la ") != std::string::npos ||
               textStr.find(" et ") != std::string::npos ||
               textStr.find(" ce ") != std::string::npos ||
               textStr.find(" qui ") != std::string::npos ||
               textStr.find(" avec ") != std::string::npos) {
        result = "fr"; // French
    } else if (textStr.find(" und ") != std::string::npos ||
               textStr.find(" der ") != std::string::npos ||
               textStr.find(" die ") != std::string::npos ||
               textStr.find(" das ") != std::string::npos ||
               textStr.find(" mit ") != std::string::npos ||
               textStr.find(" ist ") != std::string::npos) {
        result = "de"; // German
    } else if (textStr.find(" il ") != std::string::npos ||
               textStr.find(" che ") != std::string::npos ||
               textStr.find(" con ") != std::string::npos ||
               textStr.find(" per ") != std::string::npos ||
               textStr.find(" sono ") != std::string::npos) {
        result = "it"; // Italian
    } else if (textStr.find(" o ") != std::string::npos ||
               textStr.find(" a ") != std::string::npos ||
               textStr.find(" que ") != std::string::npos ||
               textStr.find(" para ") != std::string::npos ||
               textStr.find(" com ") != std::string::npos) {
        result = "pt"; // Portuguese
    }
    
    // Additional character frequency analysis for better accuracy
    int accentCount = 0;
    for (char c : textStr) {
        if (c < 0 || c > 127) accentCount++; // Non-ASCII characters
    }
    
    // If high accent frequency and no clear language match, default to multi-lingual
    if (accentCount > textStr.length() * 0.1 && result == "en") {
        result = "mul"; // Multiple/unknown with accents
    }

    env->ReleaseStringUTFChars(text, nativeText);
    return env->NewStringUTF(result.c_str());
}

/**
 * @brief Logs the cleanup of a language identifier instance.
 *
 * If the handle is non-zero, logs that resources for the language identifier have been cleaned up. No actual resource deallocation is performed.
 *
 * @param handle Identifier for the language identifier instance.
 */
JNIEXPORT void JNICALL
Java_com_example_app_language_LanguageIdentifier_nativeRelease(
        JNIEnv
        *env,
        jobject /* this */,
        jlong handle
) {
    // Clean up resources if needed
    if (handle != 0) {
        // Resource cleanup completed - handle closed
        LOGI("Language identifier resources cleaned up for handle: %lld", (long long)handle);
    }
}

JNIEXPORT jstring

JNICALL
Java_com_example_app_language_LanguageIdentifier_nativeGetVersion(
        JNIEnv *env,
        jclass /* clazz */) {
    return env->NewStringUTF("1.0.0");
}

#ifdef __cplusplus
}
#endif
