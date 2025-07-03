plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
    alias(libs.plugins.googleServices)
    alias(libs.plugins.firebaseCrashlytics)
    alias(libs.plugins.firebasePerf)
    alias(libs.plugins.kotlinCompose)
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
        vectorDrawables.useSupportLibrary = true
        signingConfig = signingConfigs.getByName("debug")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    kotlin {
        jvmToolchain(17)
        compilerOptions {
            freeCompilerArgs.addAll(
                "-opt-in=kotlin.RequiresOptIn",
                "-opt-in=kotlinx.serialization.ExperimentalSerializationApi",
                "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
                "-opt-in=kotlinx.coroutines.FlowPreview",
                "-opt-in=kotlinx.coroutines.InternalCoroutinesApi"
            )
        }
    } // <- This closing brace was missing

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    // Only include sourceSets if you have custom dirs, otherwise remove or keep minimal.
    // Remove if unnecessary; Gradle will use defaults.
    /*
    sourceSets {
        getByName("main") {
            java.srcDir("src/main/java")
            kotlin.srcDir("src/main/kotlin")
            aidl.srcDir("src/main/aidl")
            // Add generated dirs ONLY if you have them
        }
    }
    */

    ndkVersion = "26.2.11394342"
}

tasks.register<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("generateTypeScriptClient") {
    generatorName.set("typescript-fetch")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/typescript")
    configOptions.set(
        mapOf(
            "npmName" to "@auraframefx/api-client",
            "npmVersion" to "1.0.0",
            "supportsES6" to "true",
            "withInterfaces" to "true",
            "typescriptThreePlus" to "true"
        )
    )
}

tasks.register<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("generateJavaClient") {
    generatorName.set("java")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/java")
    configOptions.set(
        mapOf(
            "library" to "retrofit2",
            "serializationLibrary" to "gson",
            "dateLibrary" to "java8",
            "java8" to "true",
            "useRxJava2" to "false"
        )
    )
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
    configOptions.set(
        mapOf(
            "dateLibrary" to "kotlinx-datetime",
            "serializationLibrary" to "kotlinx_serialization"
        )
    )
    globalProperties.set(
        mapOf(
            "library" to "kotlin",
            "serializationLibrary" to "kotlinx_serialization"
        )
    )
    dependsOn("generateTypeScriptClient", "generateJavaClient")
}

val generatePythonClient by tasks.registering(org.openapitools.generator.gradle.plugin.tasks.GenerateTask::class) {
    generatorName.set("python")
    inputSpec.set("$projectDir/api-spec/aura-framefx-api.yaml")
    outputDir.set("${layout.buildDirectory.get().asFile}/generated/python")
    configOptions.set(mapOf("packageName" to "auraframefx_api_client"))
}

tasks.register("generateOpenApiContract") {
    group = "OpenAPI tools"
    description = "Generates all OpenAPI client artifacts (Kotlin, TypeScript, Java, Python)."
    dependsOn(
        "openApiGenerate",
        "generateTypeScriptClient",
        "generateJavaClient",
        generatePythonClient
    )
}

project.afterEvaluate {
    tasks.named("kspDebugKotlin") { dependsOn("openApiGenerate") }
    tasks.named("kspReleaseKotlin") { dependsOn("openApiGenerate") }
    tasks.named("compileDebugKotlin") { dependsOn("openApiGenerate") }
    tasks.named("compileReleaseKotlin") { dependsOn("openApiGenerate") }
    tasks.named("kspDebugKotlin") { mustRunAfter("openApiGenerate") }
    tasks.named("kspReleaseKotlin") { mustRunAfter("openApiGenerate") }
}

tasks.named("preBuild") {
    dependsOn(
        "generateTypeScriptClient",
        "generateJavaClient",
        "openApiGenerate",
        "generatePythonClient"
    )
}

ksp {
    arg("ksp.class.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/kotlin/main")
    arg("ksp.java.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/java/main")
    arg("ksp.resources.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/resources/main")
    arg("ksp.kotlin.output.dir", "${layout.buildDirectory.get().asFile}/generated/ksp/kotlin")
    arg("classOutputDir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/kotlin/main")
    arg("javaOutputDir", "${layout.buildDirectory.get().asFile}/generated/ksp/classes/java/main")
    arg("project.buildDir", layout.buildDirectory.get().asFile.absolutePath)
    arg("ksp.incremental", "true")
}

dependencies {
    // Hilt
    implementation(libs.hiltAndroid)
    ksp(libs.hiltAndroidCompiler)
    implementation(libs.hiltNavigationCompose)
    implementation(libs.androidxWorkRuntimeKtx)
    implementation(libs.androidxHiltWork)

    // AndroidX Core & Compose
    implementation(libs.androidxCoreKtx)
    implementation(libs.androidxAppcompat)
    implementation(libs.androidxLifecycleRuntimeKtx)
    implementation(libs.androidxActivityCompose)
    implementation(platform(libs.composeBom)) // Apply the Compose BOM platform directly
    implementation(libs.androidxUi) // Version will come from composeBom
    implementation(libs.androidxUiGraphics) // Version will come from composeBom
    implementation(libs.androidxUiToolingPreview) // Version will come from composeBom
    implementation(libs.androidxMaterial3) // Has its own version defined in TOML
    implementation(libs.androidxNavigationCompose) // Has its own version
    implementation(libs.hiltNavigationCompose) // Has its own version
    // implementation(libs.androidxMaterial3) // This was a duplicate line

    // Animation
    implementation(libs.animationTooling) // Version will come from composeBom

    // Lifecycle
    implementation(libs.lifecycleViewmodelCompose)
    implementation(libs.androidxLifecycleRuntimeCompose)
    implementation(libs.androidxLifecycleViewmodelKtx)
    implementation(libs.androidxLifecycleLivedataKtx)
    implementation(libs.lifecycleCommonJava8)
    implementation(libs.androidxLifecycleProcess)
    implementation(libs.androidxLifecycleService)
    implementation(libs.androidxLifecycleExtensions)

    // Room
    implementation(libs.androidxRoomRuntime)
    implementation(libs.androidxRoomKtx)
    ksp(libs.androidxRoomCompiler)

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
    implementation(libs.kotlinxDatetime)

    // Network
    implementation(libs.retrofit2Retrofit)
    implementation(libs.converterGson)
    implementation(libs.okhttp3Okhttp)
    implementation(libs.okhttp3LoggingInterceptor)
    implementation(libs.retrofit2KotlinxSerializationConverter)
    
    // DataStore
    implementation(libs.androidxDatastorePreferences)
    implementation(libs.androidxDatastoreCore)
    
    // UI
    implementation(libs.coilCompose)
    implementation(libs.accompanistSystemuicontroller)
    implementation(libs.accompanistPermissions)
    implementation(libs.accompanistPager)
    implementation(libs.accompanistPagerIndicators)
    implementation(libs.accompanistFlowlayout)

    // Testing
    testImplementation(libs.testJunit)
    androidTestImplementation(libs.androidxTestExtJunit)
    androidTestImplementation(libs.espressoCore)
    androidTestImplementation(platform(libs.composeBom)) // Apply BOM for test dependencies too
    androidTestImplementation(libs.composeUiTestJunit4) // Version from composeBom
    debugImplementation(libs.composeUiTestManifest) // Version from composeBom
    debugImplementation(libs.composeUiTooling) // Version from composeBom
    debugImplementation(libs.animationTooling) // Version from composeBom

    // Xposed API - local JARs from app/Libs
    compileOnly(files("app/Libs/api-82.jar"))

    // Logging
    implementation(libs.timber)
}