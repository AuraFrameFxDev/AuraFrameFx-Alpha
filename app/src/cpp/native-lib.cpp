// app/src/main/cpp/native-lib.cpp

#include <jni.h>
#include <string>

// This is a sample JNI function.
// The name MUST follow the pattern: Java_your_package_name_YourActivityName_yourMethodName
// Note that underscores in the package name are replaced with _1.
extern "C" /**
 * @brief Returns a greeting string from native C++ code to Java.
 *
 * Converts a C++ greeting message to a Java UTF string and returns it to the caller.
 *
 * @return jstring A Java string containing "Hello from Genesis C++ Core".
 */
JNIEXPORT jstring JNICALL
Java_com_auraframes_fx_MainActivity_stringFromJNI(JNIEnv* env, jobject /* this */) {
    std::string hello = "Hello from Genesis C++ Core";
    return env->NewStringUTF(hello.c_str());
}