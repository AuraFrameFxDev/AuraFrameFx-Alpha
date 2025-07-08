import org.openapitools.generator.gradle.plugin.tasks.GenerateTask

plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.kotlinCompose)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
    alias(libs.plugins.googleServices)
    alias(libs.plugins.openapiGenerator)
}

android {
    namespace = "dev.aurakai.auraframefx"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.aurakai.auraframefx"
        minSdk = 33
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags += ""
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = libs.versions.composeCompiler.get()
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
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
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        freeCompilerArgs.addAll(
            "-opt-in=kotlin.RequiresOptIn",
            "-opt-in=androidx.compose.material3.ExperimentalMaterial3Api",
            "-opt-in=androidx.compose.foundation.ExperimentalFoundationApi",
            "-opt-in=androidx.compose.animation.ExperimentalAnimationApi",
            "-opt-in=androidx.compose.ui.ExperimentalComposeUiApi",
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
            "-opt-in=kotlin.time.ExperimentalTime",
            "-opt-in=kotlinx.serialization.ExperimentalSerializationApi"
        )
    }
}

// OpenAPI Generator: Generate Kotlin client
tasks.register<GenerateTask>("generateKotlinClient") {
    generatorName.set("kotlin")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/kotlin")
    apiPackage.set("dev.aurakai.auraframefx.api.client.apis")
    modelPackage.set("dev.aurakai.auraframefx.api.client.models")
    invokerPackage.set("dev.aurakai.auraframefx.api.client.infrastructure")
    configOptions.set(
        mapOf(
            "dateLibrary" to "kotlinx-datetime",
            "serializationLibrary" to "kotlinx_serialization"
        )
    )
}

// Ensure KSP and compilation tasks depend on the code generation
tasks.named("preBuild") {
    dependsOn("generateKotlinClient")
}

dependencies {
    // Xposed
    compileOnly(files("Libs/api-82.jar")) // Assuming Libs folder is in app/

    // Hilt
    implementation(libs.hiltAndroid)
    ksp(libs.hiltCompiler)
    implementation(libs.hiltNavigationCompose)
    implementation(libs.androidxHiltWork)

    // Hilt Testing
    androidTestImplementation(libs.daggerHiltAndroidTesting)
    kspAndroidTest(libs.daggerHiltAndroidCompiler)
    testImplementation(libs.daggerHiltAndroidTesting)
    kspTest(libs.daggerHiltAndroidCompiler)

    // Time and Date
    implementation(libs.kotlinxDatetime)

    // AndroidX & Compose
    implementation(libs.androidxCoreKtx)
    implementation(libs.androidxAppcompat)
    implementation(libs.androidxLifecycleRuntimeKtx)
    implementation(libs.androidxActivityCompose)
    implementation(platform(libs.composeBom))
    implementation(libs.androidxUi)
    implementation(libs.androidxUiGraphics)
    implementation(libs.androidxUiToolingPreview)
    implementation(libs.androidxMaterial3)
    implementation(libs.androidxNavigationCompose)

    // Animation
    implementation(libs.androidxComposeAnimation)
    debugImplementation(libs.composeUiTooling)

    // Lifecycle
    implementation(libs.lifecycleViewmodelCompose)
    implementation(libs.androidxLifecycleRuntimeCompose)
    implementation(libs.androidxLifecycleViewmodelKtx)
    implementation(libs.androidxLifecycleLivedataKtx)
    implementation(libs.lifecycleCommonJava8)
    implementation(libs.androidxLifecycleProcess)
    implementation(libs.androidxLifecycleService)

    // Room
    implementation(libs.androidxRoomRuntime)
    implementation(libs.androidxRoomKtx)
    ksp(libs.androidxRoomCompiler)

    // Security
    implementation(libs.androidxSecurityCrypto)

    // Google AI
    implementation(libs.guava)

    // Firebase
    implementation(platform(libs.firebaseBom))
    implementation(libs.firebaseAnalyticsKtx)
    implementation(libs.firebaseCrashlyticsKtx)
    implementation(libs.firebasePerfKtx)
    implementation(libs.firebaseConfigKtx)
    implementation(libs.firebaseStorageKtx)
    implementation(libs.firebaseMessagingKtx)

    // Kotlin
    implementation(libs.kotlinxCoroutinesAndroid)
    implementation(libs.kotlinxCoroutinesPlayServices)
    implementation(libs.kotlinxSerializationJson)

    // Network
    implementation(libs.retrofit)
    implementation(libs.converterGson)
    implementation(libs.okhttp)
    implementation(libs.okhttpLoggingInterceptor)
    implementation(libs.retrofitKotlinxSerializationConverter)

    // DataStore
    implementation(libs.androidxDatastorePreferences)
    implementation(libs.androidxDatastoreCore)

    // Security
    implementation(libs.androidxSecurityCrypto)

    // UI Utilities
    implementation(libs.coilCompose)
    implementation(libs.timber)
    implementation(libs.guava)

    // Accompanist
    implementation(libs.accompanistSystemuicontroller)
    implementation(libs.accompanistPermissions)
    implementation(libs.accompanistPager)
    implementation(libs.accompanistPagerIndicators)

    // WorkManager
    implementation(libs.androidxWorkRuntimeKtx)

    // Testing
    testImplementation(libs.testJunit)
    testImplementation(libs.kotlinxCoroutinesTest)
    testImplementation(libs.mockkAgent)
    androidTestImplementation(libs.androidxTestExtJunit)
    androidTestImplementation(libs.espressoCore)
    androidTestImplementation(platform(libs.composeBom))
    androidTestImplementation(libs.composeUiTestJunit4)
    androidTestImplementation(libs.mockkAndroid)
    debugImplementation(libs.composeUiTooling)
    debugImplementation(libs.composeUiTestManifest)
}