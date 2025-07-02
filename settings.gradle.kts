
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    // Hardcoded plugin versions removed from here.
    // Versions will be sourced from the version catalog (libs.versions.toml)
    // when plugins are applied in build.gradle.kts files.
}

// This top-level plugins block is for plugins applied to the settings script itself.
plugins {
    id(libs.plugins.org.gradle.toolchains.foojay.resolver)
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
