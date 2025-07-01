
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    
    // Define plugin versions (aligned with Android Studio Giraffe+ requirements)
    val agpVersion = "8.11.0"  // Updated to latest stable version
    val kotlinVersion = "2.2.0"
    val hiltVersion = "2.56.2"
    val googleServicesVersion = "4.4.3"
    val firebaseCrashlyticsVersion = "2.9.9"
    val firebasePerfVersion = "1.4.2"
    val kspVersion = "2.2.0-2.0.2"
    val openApiGeneratorVersion = "7.6.0"

    plugins {
        id("com.android.application") version agpVersion apply false
        id("com.android.library") version agpVersion apply false
        id("org.jetbrains.kotlin.android") version kotlinVersion apply false
        id("com.google.dagger.hilt.android") version hiltVersion apply false
        id("com.google.gms.google-services") version googleServicesVersion apply false
        id("com.google.firebase.crashlytics") version firebaseCrashlyticsVersion apply false
        id("com.google.firebase.firebase-perf") version firebasePerfVersion apply false
        id("com.google.devtools.ksp") version kspVersion apply false
        id("org.jetbrains.kotlin.plugin.serialization") version kotlinVersion apply false
        id("org.jetbrains.kotlin.plugin.compose") version kotlinVersion apply false
        id("org.openapi.generator") version openApiGeneratorVersion apply false
        id("org.gradle.toolchains.foojay-resolver") version "0.8.0" apply false
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver")
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
        maven { url = uri("https://androidx.dev/storage/compose-compiler/repository/") }
    }
}

rootProject.name = "AuraFrameFXAlpha"
include(":app")

toolchainManagement {
    jvm {
        javaRepositories {
            repository("foojay") {
                resolverClass.set(org.gradle.toolchains.foojay.FoojayToolchainResolver::class.java)
            }
        }
    }
}
