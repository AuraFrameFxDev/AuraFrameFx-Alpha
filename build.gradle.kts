// Top-level build file where you can add configuration options common to all sub-projects/modules.
// Plugin versions are managed in settings.gradle.kts
plugins {
    alias(libs.plugins.androidApplication) apply false // Was com.android.application
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.hilt) apply false // Was com.google.dagger.hilt.android
    alias(libs.plugins.google.services) apply false // Was com.google.gms.google-services
    alias(libs.plugins.firebase.crashlytics) apply false
    alias(libs.plugins.firebase.perf) apply false
    alias(libs.plugins.kotlin.serialization) apply false // Was org.jetbrains.kotlin.plugin.serialization
    alias(libs.plugins.kotlin.compose) apply false // Was org.jetbrains.kotlin.plugin.compose
    alias(libs.plugins.ksp) apply false // Was com.google.devtools.ksp with version
    alias(libs.plugins.openapi.generator) apply false // Was org.openapi.generator
}

tasks.register("clean", Delete::class) {
    delete(rootProject.layout.buildDirectory)
}

// All repositories are configured in settings.gradle.kts
