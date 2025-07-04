// app/build.gradle.kts

plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
    alias(libs.plugins.googleServices)
    alias(libs.plugins.kotlinCompose)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.firebaseCrashlytics) // Added based on error log
    alias(libs.plugins.firebasePerf)        // Added based on error log
    alias(libs.plugins.openapiGenerator)    // Added based on error log
}

android {
    namespace = "dev.aurakai.auraframefx"
    compileSdk = 34 // As requested

    defaultConfig {
        applicationId = "dev.aurakai.auraframefx"
        minSdk = 33
        targetSdk = 34 // As requested
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf("-opt-in=kotlin.RequiresOptIn")
    }

    buildFeatures {
        compose = true
        aidl = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = libs.versions.composeCompiler.get()
    }
}

dependencies {
    // Xposed
    compileOnly(files("Libs/api-82.jar")) // Changed to local file dependency

    // Hilt - Already in new base, using new aliases
    implementation(libs.hiltAndroid)
    ksp(libs.hiltCompiler)
    implementation(libs.hiltNavigationCompose) // From old, uses new TOML alias
    implementation(libs.androidxHiltWork)      // From old, uses new TOML alias (depends on hilt version)

    // AndroidX Core & Compose
    implementation(libs.androidxCoreKtx)
    implementation(libs.androidxAppcompat)
    implementation(libs.androidxLifecycleRuntimeKtx)
    implementation(libs.androidxActivityCompose)
    implementation(platform(libs.composeBom)) // Platform import for Compose
    implementation(libs.androidxUi)
    implementation(libs.androidxUiGraphics)
    implementation(libs.androidxUiToolingPreview)
    implementation(libs.androidxMaterial3) // Version managed by Compose BOM
    implementation(libs.androidxNavigationCompose)

    // Animation (version managed by Compose BOM)
    implementation(libs.androidxComposeAnimation) // Using the new specific animation library alias
    // For debug/preview features related to animation:
    debugImplementation(libs.animationTooling) // Alias for androidx.compose.animation:animation-tooling

    // Lifecycle
    implementation(libs.lifecycleViewmodelCompose)
    implementation(libs.androidxLifecycleRuntimeCompose)
    implementation(libs.androidxLifecycleViewmodelKtx)
    implementation(libs.androidxLifecycleLivedataKtx)
    implementation(libs.lifecycleCommonJava8)
    implementation(libs.androidxLifecycleProcess)
    implementation(libs.androidxLifecycleService)
    // androidxLifecycleExtensions is deprecated and removed

    // Room
    implementation(libs.androidxRoomRuntime)
    implementation(libs.androidxRoomKtx)
    ksp(libs.androidxRoomCompiler) // Ensure Room compiler uses KSP

    // Firebase
    implementation(platform(libs.firebaseBom)) // Platform import for Firebase
    implementation(libs.firebaseAnalyticsKtx)
    implementation(libs.firebaseCrashlyticsKtx)
    implementation(libs.firebasePerfKtx)
    implementation(libs.firebaseMessagingKtx)
    implementation(libs.firebaseConfigKtx)  // Explicit version from TOML
    implementation(libs.firebaseStorageKtx) // Explicit version from TOML

    // Kotlin Coroutines & Serialization & DateTime
    implementation(libs.kotlinxCoroutinesAndroid)
    implementation(libs.kotlinxCoroutinesPlayServices)
    implementation(libs.kotlinxSerializationJson)
    implementation(libs.kotlinxDatetime)

    // Network
    implementation(libs.retrofit) // Using new alias
    implementation(libs.converterGson)
    implementation(libs.okhttp) // Using new alias
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
    implementation(libs.guava) // Using new alias

    // Accompanist (review if still needed, versions from new TOML)
    implementation(libs.accompanistSystemuicontroller)
    implementation(libs.accompanistPermissions)
    implementation(libs.accompanistPager)
    implementation(libs.accompanistPagerIndicators)
    // implementation(libs.accompanistFlowlayout) // Assuming covered by pager or not strictly needed for now

    // WorkManager (already included via androidxHiltWork which pulls in workManager)
    implementation(libs.androidxWorkRuntimeKtx)


    // Testing
    testImplementation(libs.testJunit)
    testImplementation(libs.kotlinxCoroutinesTest) // Added from old TOML's list
    testImplementation(libs.mockkAgent) // For local unit tests

    // Hilt testing dependencies
    testImplementation("com.google.dagger:hilt-android-testing:2.56.2")
    kspTest("com.google.dagger:hilt-compiler:2.56.2")

    androidTestImplementation(libs.androidxTestExtJunit)
    androidTestImplementation(libs.espressoCore)
    androidTestImplementation(platform(libs.composeBom)) // Compose BOM for tests
    androidTestImplementation(libs.composeUiTestJunit4)
    androidTestImplementation(libs.mockkAndroid) // For instrumented tests

    // Hilt instrumentation testing dependencies
    androidTestImplementation("com.google.dagger:hilt-android-testing:2.56.2")
    kspAndroidTest("com.google.dagger:hilt-compiler:2.56.2")
    // androidTestImplementation(libs.kotlinxCoroutinesTest) // Already in testImplementation

    debugImplementation(libs.composeUiTooling) // For debug builds
    debugImplementation(libs.composeUiTestManifest) // For debug builds
}

// Hilt configuration for better incremental builds
hilt {
    enableAggregatingTask = true
}
