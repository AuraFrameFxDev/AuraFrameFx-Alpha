// app/src/main/cpp/native-lib.cpp

#include <jni.h>
#include <string>

// This is a sample JNI function.
// The name MUST follow the pattern: Java_your_package_name_YourActivityName_yourMethodName
// Note that underscores in the package name are replaced with _1.
extern "C" /**
 * @brief Returns a greeting string from native C++ code to Java via JNI.
 *
 * Creates a new Java UTF-8 string with the content "Hello from Genesis C++ Core" and returns it to the calling Java environment.
 *
 * @return jstring A new Java string containing the greeting message.
 */
JNIEXPORT jstring JNICALL
Java_com_auraframes_fx_MainActivity_stringFromJNI(JNIEnv *env, jobject /* this */) {
    std::string hello = "Hello from Genesis C++ Core";
    return env->NewStringUTF(hello.c_str());
}