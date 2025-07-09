// app/src/main/cpp/native-lib.cpp

#include <jni.h>
#include <string>

// This is a sample JNI function.
// The name MUST follow the pattern: Java_PackageName_ClassName_MethodName
// For Kotlin, underscores in package names are kept as underscores.
// If the Kotlin class is a companion object, the class name part includes 'Companion'.
// However, for methods defined in `companion object {}` and called as `NativeLib.stringFromJNI()`,
// the JNI signature usually refers to the enclosing class `NativeLib`.
extern "C" /**
 * @brief Returns a greeting string from the native C++ core to the Java/Kotlin layer.
 *
 * This JNI function is called from the `NativeLib` class in the `dev.aurakai.auraframefx.core` package.
 *
 * @return jstring A UTF-8 encoded greeting message for the Java/Kotlin caller.
 */
JNIEXPORT jstring JNICALL
Java_dev_aurakai_auraframefx_core_NativeLib_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) { // For companion object methods, 'this' refers to the class instance.
    std::string hello = "Hello from AuraFrameFX Native C++ Core (via NativeLib)";
    return env->NewStringUTF(hello.c_str());
}

// If you had a method inside a regular object within NativeLib (not companion), the signature changes.
// For example, if NativeLib was an object instance and stringFromJNI was its method:
// Java_dev_aurakai_auraframefx_core_NativeLib_stringFromJNI(JNIEnv* env, jobject instance)