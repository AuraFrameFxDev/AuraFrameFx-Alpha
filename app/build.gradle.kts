plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
    id("com.google.devtools.ksp")
    id("com.google.dagger.hilt.android")
    id("com.google.gms.google-services")
    id("com.google.firebase.crashlytics")
    id("com.google.firebase.firebase-perf")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.openapi.generator")
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
        vectorDrawables {
            useSupportLibrary = true
        }
        signingConfig = signingConfigs.getByName("debug")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        jvmToolchain(17)
    }

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(17))
        }
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    sourceSets {
        getByName("main") {
            aidl.srcDirs("src/main/aidl")
            java.setSrcDirs(listOf(
                "src/main/java",
                "${layout.buildDirectory.get().asFile}/generated/kotlin/src/main/kotlin",
                "${layout.buildDirectory.get().asFile}/generated/kotlin/src/main/java",
                "${layout.buildDirectory.get().asFile}/generated/ksp/debug/java",
                "${layout.buildDirectory.get().asFile}/generated/ksp/release/java",
                "${layout.buildDirectory.get().asFile}/generated/aidl_source_output_dir/debug/compileDebugAidl/out",
                "${layout.buildDirectory.get().asFile}/generated/aidl_source_output_dir/release/compileReleaseAidl/out"
            ))
            kotlin.setSrcDirs(listOf(
                "src/main/kotlin",
                "${layout.buildDirectory.get().asFile}/generated/ksp/debug/kotlin",
                "${layout.buildDirectory.get().asFile}/generated/ksp/release/kotlin"
            ))
        }
    }
    ndkVersion = "26.2.11394342"
    kotlin {
        compilerOptions {
            freeCompilerArgs.addAll(
                "-opt-in=kotlin.RequiresOptIn",
                "-opt-in=kotlinx.serialization.ExperimentalSerializationApi",
                "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
                "-opt-in=kotlinx.coroutines.FlowPreview",
                "-opt-in=kotlinx.coroutines.InternalCoroutinesApi"
            )
        }
    }
}

tasks.register<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("generateTypeScriptClient") {
    generatorName.set("typescript-fetch")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/typescript")

    configOptions.set(mapOf(
        "npmName" to "@auraframefx/api-client",
        "npmVersion" to "1.0.0",
        "supportsES6" to "true",
        "withInterfaces" to "true",
        "typescriptThreePlus" to "true"
    ))
}

tasks.register<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("generateJavaClient") {
    generatorName.set("java")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/java")

    configOptions.set(mapOf(
        "library" to "retrofit2",
        "serializationLibrary" to "gson",
        "dateLibrary" to "java8",
        "java8" to "true",
        "useRxJava2" to "false"
    ))

    apiPackage.set("dev.aurakai.auraframefx.java.api")
    modelPackage.set("dev.aurakai.auraframefx.java.model")
    invokerPackage.set("dev.aurakai.auraframefx.java.client")
}

tasks.named<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("openApiGenerate") {
    generatorName.set("kotlin")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/kotlin")
    apiPackage.set("dev.aurakai.auraframefx.api")
    modelPackage.set("dev.aurakai.auraframefx.api.model")
    invokerPackage.set("dev.aurakai.auraframefx.api.invoker")
    configOptions.set(mapOf(
        "dateLibrary" to "kotlinx-datetime",
        "serializationLibrary" to "kotlinx_serialization",
        "importMappings" to "Instant=kotlinx.datetime.Instant"
    ))

    globalProperties.set(mapOf(
        "library" to "kotlin",
        "serializationLibrary" to "kotlinx_serialization"
    ))
    dependsOn("generateTypeScriptClient", "generateJavaClient")
}

val generatePythonClient by tasks.registering(org.openapitools.generator.gradle.plugin.tasks.GenerateTask::class) {
    generatorName.set("python")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/python")
    configOptions.set(mapOf
