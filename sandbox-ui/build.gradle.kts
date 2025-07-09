plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.0"
    id("kotlin-kapt")
    id("dagger.hilt.android.plugin")
    id("kotlin-parcelize")
}

android {
    namespace = "dev.aurakai.auraframefx.sandbox.ui"
    compileSdk = 36

    defaultConfig {
        minSdk = 33
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_24
        targetCompatibility = JavaVersion.VERSION_24
    }

    kotlinOptions {

    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "2.2.0"
    }
}

dependencies {
    // Core project dependency
    implementation(project(":app"))

    // AndroidX Core
    implementation(libs.androidxCoreKtx)
    implementation(libs.androidxLifecycleRuntimeKtx)
    implementation(libs.androidxActivityCompose)

    // Compose BOM
    implementation(platform(libs.composeBom))
    implementation(libs.ui)
    implementation(libs.uiToolingPreview)
    implementation(libs.material3)
    implementation(libs.animation)
    implementation(libs.foundation)

    // Navigation
    implementation(libs.androidxNavigationCompose)

    // Hilt
    implementation(libs.hiltAndroid)
    kapt(libs.hiltCompiler)
    implementation(libs.hiltNavigationCompose)

    // Debug tools
    debugImplementation(libs.uiTooling)
    debugImplementation(libs.composeUiTestManifest)

    // Testing
    testImplementation(libs.testJunit)
    androidTestImplementation(libs.androidxTestExtJunit)
    androidTestImplementation(libs.espressoCore)
    androidTestImplementation(platform(libs.composeBom))
    androidTestImplementation(libs.uiTestJunit4)
}
