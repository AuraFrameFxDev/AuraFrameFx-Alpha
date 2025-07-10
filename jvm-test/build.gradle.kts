// Apply plugins using the older syntax to avoid version conflicts
buildscript {
    repositories {
        mavenCentral()
        google()
    }
    
    dependencies {
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.2.0")
    }
}

apply(plugin = "org.jetbrains.kotlin.jvm")
apply(plugin = "java-library")

// Kotlin version
val kotlinVersion = "2.2.0"

group = "dev.aurakai.auraframefx.jvmtest"
version = "1.0.0"

repositories {
    mavenCentral()
    google()
}

dependencies {
    // Kotlin standard library
    implementation("org.jetbrains.kotlin:kotlin-stdlib:$kotlinVersion")
    
    // JUnit 5
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.9.2")
    
    // MockK for mocking
    testImplementation("io.mockk:mockk:1.13.4")
    
    // Kotlin coroutines for testing
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    
    // Add the main module as a dependency
    implementation(project(":app"))
    
    // Ensure all Kotlin dependencies use the same version
    configurations.all {
        resolutionStrategy {
            force("org.jetbrains.kotlin:kotlin-stdlib:$kotlinVersion")
            force("org.jetbrains.kotlin:kotlin-stdlib-common:$kotlinVersion")
            force("org.jetbrains.kotlin:kotlin-reflect:$kotlinVersion")
            force("org.jetbrains.kotlin:kotlin-stdlib-jdk7:$kotlinVersion")
            force("org.jetbrains.kotlin:kotlin-stdlib-jdk8:$kotlinVersion")
        }
    }
}

// Configure the Kotlin compiler
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        jvmTarget = "17"
        apiVersion = "1.9"
        languageVersion = "1.9"
        freeCompilerArgs = listOf("-Xjsr305=strict")
    }
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
}

// Configure Java compatibility
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
