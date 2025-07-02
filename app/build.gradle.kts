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
            aidl {
                srcDirs("src/main/aidl")
            }
            java {
                // Using setSrcDirs should be fine, the error is puzzling.
                // This explicitly sets these as the source directories, replacing defaults.
                setSrcDirs(listOf(
                    "src/main/java",
                    "${layout.buildDirectory.get().asFile}/generated/kotlin/src/main/kotlin",
                    "${layout.buildDirectory.get().asFile}/generated/kotlin/src/main/java",
                    "${layout.buildDirectory.get().asFile}/generated/ksp/debug/java",
                    "${layout.buildDirectory.get().asFile}/generated/ksp/release/java"
                ))
            }
            kotlin {
                // Using setSrcDirs should be fine.
                setSrcDirs(listOf(
                    "src/main/kotlin",
                    "${layout.buildDirectory.get().asFile}/generated/ksp/debug/kotlin",
                    "${layout.buildDirectory.get().asFile}/generated/ksp/release/kotlin"
                ))
            }
            // Removed redundant aidl.setSrcDirs from original line 75
        }
    }
    ndkVersion = "26.2.11394342"
    // JVM target is now configured in the kotlin compiler options block above
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
        "serializationLibrary" to "kotlinx_serialization"
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
    configOptions.set(mapOf(
        "packageName" to "auraframefx_api_client"
    ))
}

tasks.register("generateOpenApiContract") {
    group = "OpenAPI tools"
    description = "Generates all OpenAPI client artifacts (Kotlin, TypeScript, Java, Python)."
    dependsOn(
        "openApiGenerate",
        "generateTypeScriptClient",
        "generateJavaClient",
        generatePythonClient // Use the val here
    )
}

// Ensure codegen runs before build
// Ensure OpenAPI generation runs before KSP and compilation
project.afterEvaluate {
    // Make KSP tasks depend on OpenAPI generation
    tasks.named("kspDebugKotlin") {
        dependsOn("openApiGenerate")
    }
    tasks.named("kspReleaseKotlin") {
        dependsOn("openApiGenerate")
    }
    
    // Make compile tasks depend on OpenAPI generation
    tasks.named("compileDebugKotlin") {
        dependsOn("openApiGenerate")
    }
    tasks.named("compileReleaseKotlin") {
        dependsOn("openApiGenerate")
    }
    
    // Make sure KSP can see the generated sources
    tasks.named("kspDebugKotlin") {
        mustRunAfter("openApiGenerate")
    }
    tasks.named("kspReleaseKotlin") {
        mustRunAfter("openApiGenerate")
    }
}

tasks.named("preBuild") {
    dependsOn(
        "generateTypeScriptClient",
        "generateJavaClient",
        "openApiGenerate",
        "generatePythonClient"
    )
}

// Configure KSP
ksp {
    // Output directories
    arg("ksp.class.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/kotlin/main")
    arg("ksp.java.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/java/main")
    arg("ksp.resources.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/resources/main")
    arg("ksp.kotlin.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/kotlin")
    
    // Include generated OpenAPI code in the classpath
    arg("classOutputDir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/kotlin/main")
    arg("javaOutputDir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/java/main")
    
    // Add source directories to the classpath
    arg("project.buildDir", layout.buildDirectory.get().asFile.absolutePath)
    
    // Enable incremental processing
    arg("ksp.incremental", "true")
    
    // The source directories are already configured in the sourceSets block above
}

dependencies {
    // Hilt
    implementation(libs.hilt.android)
    ksp(libs.hilt.android.compiler)
    implementation(libs.hilt.navigation.compose)
    
    // For WorkManager with Hilt
    implementation(libs.androidx.work.runtime.ktx)
    implementation(libs.androidx.hilt.work)
    ksp(libs.androidx.hilt.hilt.compiler)

    // JNDI API for the missing javax.naming classes

    // AndroidX Core
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)

    // Compose
    val composeBom = platform(libs.androidx.compose.bom)
    implementation(composeBom)
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.hilt.navigation.compose)
    implementation(libs.androidx.compose.material3)

    // Lifecycle
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.lifecycle.livedata.ktx)
    implementation(libs.lifecycle.common.java8)
    implementation(libs.androidx.lifecycle.process)
    implementation(libs.androidx.lifecycle.service)
    implementation(libs.androidx.lifecycle.extensions)

    // Room
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    // Firebase
    implementation(platform(libs.firebase.bom))
    implementation(libs.firebase.analytics.ktx)
    implementation(libs.firebase.crashlytics.ktx)
    implementation(libs.firebase.perf.ktx)
    implementation(libs.firebase.config.ktx)
    implementation(libs.firebase.storage.ktx)
    implementation(libs.firebase.messaging.ktx)

    // Kotlin
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.play.services)
    implementation(libs.jetbrains.kotlinx.serialization.json)
    implementation(libs.jetbrains.kotlinx.datetime) // Added kotlinx-datetime

    // Network
    implementation(libs.squareup.retrofit2.retrofit)
    implementation(libs.converter.gson)
    implementation(libs.squareup.okhttp3.okhttp)
    implementation(libs.squareup.okhttp3.logging.interceptor)
    implementation(libs.jakewharton.retrofit2.kotlinx.serialization.converter)
    
    // DataStore
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.datastore.core)
    
    // Kotlinx Serialization
    implementation(libs.jetbrains.kotlinx.serialization.json)
    implementation(libs.jakewharton.retrofit2.kotlinx.serialization.converter)

    // UI
    implementation(libs.coil.compose)
    implementation(libs.google.accompanist.systemuicontroller)
    implementation(libs.google.accompanist.permissions)
    implementation(libs.accompanist.pager)



    implementation(libs.accompanist.pager.indicators)
    implementation(libs.accompanist.flowlayout)

    // Testing
    testImplementation(libs.test.junit)
    androidTestImplementation(libs.androidTest.androidx.test.ext.junit)
    androidTestImplementation(libs.androidTest.espresso.core)
    androidTestImplementation(composeBom)
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    // Compose Animation Tooling
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.animationTooling) // Corrected alias to match TOML conversion (kebab-case to camelCase)
}