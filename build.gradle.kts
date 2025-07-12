// build.gradle.kts (PROJECT LEVEL)
plugins {
    alias(libs.plugins.androidApplication) version "8.11.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.0" apply false
    id("org.jetbrains.kotlin.compose") version "2.2.0" apply false
    id("com.google.devtools.ksp") version "2.2.0-1.0.21" apply false
    alias(libs.plugins.hilt) apply false
}
