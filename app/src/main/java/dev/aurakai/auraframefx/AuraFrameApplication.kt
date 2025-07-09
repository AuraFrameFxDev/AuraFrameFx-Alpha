package dev.aurakai.auraframefx

import android.app.Application
import android.util.Log
import androidx.work.Configuration
import com.highcapable.yukihookapi.YukiHookAPI
import com.highcapable.yukihookapi.hook.xposed.bridge.YukiHookBridge
import com.highcapable.yukihookapi.hook.xposed.proxy.IYukiHookXposedInit
import dagger.hilt.android.HiltAndroidApp
import de.robv.android.xposed.XposedBridge
import dev.aurakai.auraframefx.system.lockscreen.model.LockScreenConfig
import dev.aurakai.auraframefx.system.overlay.model.NotchBarConfig
import dev.aurakai.auraframefx.system.quicksettings.model.QuickSettingsConfig
import dev.aurakai.auraframefx.xposed.hooks.LockScreenHooker
import dev.aurakai.auraframefx.xposed.hooks.NotchBarHooker
import dev.aurakai.auraframefx.xposed.hooks.QuickSettingsHooker

@HiltAndroidApp
class AuraFrameApplication : Application(), Configuration.Provider, IYukiHookXposedInit {
    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setMinimumLoggingLevel(Log.INFO)
            .build()

    override fun onInit() {
        XposedBridge.log("AuraFrameFX: YukiHookAPI onInit in Application")
        YukiHookAPI.configs {
            debugLog {
                isEnable = true // Enable debug logging for YukiHookAPI
                tag = "AuraFrameFX_Yuki"
            }
            isEnableModuleAppResourcesCache = true
            isEnableDataChannel = true // If you plan to use data channels
        }
        // Initialize preferences if needed, e.g.:
        // val prefs = YukiHookPrefsBridge.from("dev.aurakai.auraframefx_preferences")
    }

    override fun onHook() {
        XposedBridge.log("AuraFrameFX: YukiHookAPI onHook in Application")
        YukiHookAPI.encase {
            // Load hooks for specific packages, e.g., SystemUI
            // You might want to target specific package names like "com.android.systemui"
            loadPackage(isExcludeSelf = true) { // Using loadPackage without specific name hooks all (except self)
                XposedBridge.log("AuraFrameFX: Encasing hooks for package: $packageName, ClassLoader: $classLoader")

                // For now, creating default configs. In a real scenario, these would be loaded from prefs or DI.
                val notchBarConfig = NotchBarConfig(hideNotch = false, customIconStyle = false, notchBarColor = 0)
                val quickSettingsConfig = QuickSettingsConfig(enableTheming = false, tileLayout = "default", customBackgroundColor = 0, showCustomQuickAccessButton = false, customTileColor = 0)
                val lockScreenConfig = LockScreenConfig(
                    customClockEnabled = false,
                    customShortcutsEnabled = false,
                    hideStatusBarOnLockScreen = false,
                    customClockStyle = null,
                    customClockLayoutId = 0,
                    enableCustomAnimations = false,
                    customUnlockAnimation = null
                )

                NotchBarHooker(classLoader, notchBarConfig).applyNotchBarHooks()
                QuickSettingsHooker(classLoader, quickSettingsConfig).applyQuickSettingsHooks()
                LockScreenHooker(classLoader, lockScreenConfig).applyLockScreenHooks()

                XposedBridge.log("AuraFrameFX: All hookers applied for $packageName.")
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        if (YukiHookBridge.isXposedEnvironment) {
            XposedBridge.log("AuraFrameFX: Application onCreate, Xposed Environment detected.")
            // It's generally recommended to initialize hooks in onInit/onHook via IYukiHookXposedInit
            // rather than directly in onCreate for Xposed modules.
            // YukiHookAPI.initiate(this) // This might be needed if not using IYukiHookXposedInit, but we are.
        } else {
            Log.d("AuraFrameFX_App", "Application onCreate, Not an Xposed Environment.")
        }
    }
}