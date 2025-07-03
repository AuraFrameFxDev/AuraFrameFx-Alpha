// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.androidApplication) apply false
    alias(libs.plugins.kotlin.android) apply false    // Accessor for 'kotlin-android' from TOML
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.hilt) apply false
    alias(libs.plugins.kotlin.serialization) apply false // Accessor for 'kotlin-serialization' from TOML
    alias(libs.plugins.openapi.generator) apply false  // Accessor for 'openapi-generator' from TOML
    alias(libs.plugins.googleServices) apply false     // Accessor for 'google-services' from TOML
    alias(libs.plugins.firebase.crashlytics) apply false // Accessor for 'firebase-crashlytics' from TOML
    alias(libs.plugins.firebase.perf) apply false      // Accessor for 'firebase-perf' from TOML
    alias(libs.plugins.kotlinCompose) apply false      // Accessor for 'kotlin-compose' from TOML
}
