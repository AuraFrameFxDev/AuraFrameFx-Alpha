plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
    alias(libs.plugins.google.services)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.openapi.generator)
}

// Configure KSP for Room
ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
    arg("room.incremental", "true")
    arg("room.expandProjection", "true")
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

    kotlin {
        jvmToolchain(24)
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_24)
            freeCompilerArgs.addAll(
                "-opt-in=kotlin.RequiresOptIn",
                "-Xjvm-default=all"
            )
        }
    }

    buildFeatures {
        compose = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// OpenAPI Generator: Generate Kotlin client
val openApiSpec = "$projectDir/api-spec/aura-framefx-api.yaml"
val openApiOutputDir = "${layout.buildDirectory.get().asFile}/generated/source/openapi"

// Only configure OpenAPI generation if the spec file exists
tasks.register<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("generateAuraApi") {
    generatorName.set("kotlin")
    inputSpec.set(openApiSpec)
    outputDir.set(openApiOutputDir)
    apiPackage.set("dev.aurakai.auraframefx.api.client.apis")
    modelPackage.set("dev.aurakai.auraframefx.api.client.models")
    invokerPackage.set("dev.aurakai.auraframefx.api.client.infrastructure")
    configOptions.set(
        mapOf(
            "dateLibrary" to "kotlinx-datetime",
            "serializationLibrary" to "kotlinx_serialization",
            "library" to "jvm-retrofit2"
        )
    )
    // Skip if the spec file doesn't exist
    onlyIf { file(openApiSpec).exists() }
}

// Only add the task dependency if the spec file exists
if (file(openApiSpec).exists()) {
    tasks.named("preBuild") {
        dependsOn("generateAuraApi")
    }

    // Configure the source sets in the android block
    android {
        sourceSets {
            getByName("main") {
                java.srcDirs("${layout.buildDirectory.get()}/generated/source/openapi/src/main/kotlin")
            }
        }
    }
} else {
    logger.warn("OpenAPI spec file not found at $openApiSpec. Skipping OpenAPI code generation.")
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
    implementation(libs.generativeai)

    // Firebase
    implementation(platform(libs.firebaseBom))
    implementation(libs.firebaseAnalyticsKtx)
    implementation(libs.firebaseCrashlyticsKtx)
    implementation(libs.firebasePerfKtx)
    implementation(libs.firebaseConfigKtx)
    implementation(libs.firebaseStorageKtx)
    implementation(libs.firebaseMessagingKtx)
    implementation(libs.firebaseAuthKtx)
    implementation(libs.firebaseFirestoreKtx)

    // Kotlin Coroutines with Play Services
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

