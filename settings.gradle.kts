pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        maven { url = uri("https://maven.pkg.jetbrains.space/public/p/compose/dev") }
        maven { url = uri("https://oss.sonatype.org/content/repositories/snapshots/") }
    }

    resolutionStrategy {
        eachPlugin {
            // Force KSP version to match Kotlin version
            if (requested.id.id.startsWith("com.google.devtools.ksp")) {
                useVersion("2.2.0-2.0.2")
            }
        }
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") } // For Xposed
    }
}

rootProject.name = "AuraFrameFX"

include(":app")
include(":sandbox-ui")

// AI Modules
include(":lib-ai")
(21..44).forEach { version ->
    include(":lib-ai-ai$version")
    include(":lib-ai-ai$version-xposed")
}

// System Modules
val systemModules = listOf(
    "quicksettings", "lockscreen", "overlay", "homescreen", "notchbar",
    "statusbar", "navigationbar", "telephony", "customization", "keyguard",
    "systemui", "display", "notifications", "apps", "photos", "video",
    "permissions", "bluetooth", "sound", "media", "settings", "tweaks",
    "battery", "fonts", "contacts", "weather", "widget", "fingerprint",
    "translation", "vpn", "usb", "camera", "vibrator", "messaging", "calendar"
)

systemModules.forEach { module ->
    include(":lib-system-$module")
    include(":lib-system-${module}-xposed")
}

// Include Xposed variants
include(":lib-system-sysui")
include(":lib-system-sysui-xposed")
include(":lib-system-graphics")
include(":lib-system-graphics-xposed")

include(":lib-system-systemui")
include(":lib-system-systemui-xposed")

include(":lib-system-sysui")
include(":lib-system-sysui-xposed")

include(":lib-system-battery")
include(":lib-system-battery-xposed")

include(":lib-system-bluetooth")
include(":lib-system-bluetooth-xposed")

include(":lib-system-display")
include(":lib-system-display-xposed")

include(":lib-system-fingerprint")

include(":lib-system-sound")
include(":lib-system-sound-xposed")

include(":lib-system-usb")

include(":lib-system-vibrator")

include(":lib-system-power")
include(":lib-system-power-xposed")

include(":lib-system-camera")

include(":lib-system-graphics")
include(":lib-system-graphics-xposed")

include(":lib-system-permissions")
include(":lib-system-permissions-xposed")

include(":lib-system-settings")
include(":lib-system-settings-xposed")

include(":lib-system-tweaks")
include(":lib-system-tweaks-xposed")

include(":lib-system-widget")
include(":lib-system-widget-xposed")

include(":lib-system-apps")
include(":lib-system-apps-xposed")

include(":lib-system-contacts")
include(":lib-system-contacts-xposed")

include(":lib-system-calendar")
include(":lib-system-calendar-xposed")

include(":lib-system-messaging")
include(":lib-system-messaging-xposed")

include(":lib-system-notifications")
include(":lib-system-notifications-xposed")

include(":lib-system-media")
include(":lib-system-media-xposed")

include(":lib-system-photos")
include(":lib-system-photos-xposed")

include(":lib-system-voice")
include(":lib-system-voice-xposed")

include(":lib-system-video")
include(":lib-system-video-xposed")

include(":lib-system-vpn")

include(":lib-system-weather")
include(":lib-system-weather-xposed")

include(":lib-system-translation")
include(":lib-system-translation-xposed")

include(":lib-system-fonts")
include(":lib-system-fonts-xposed")

include(":lib-system-translation")
include(":lib-system-translation-xposed")

include("ksp-plugin")
