package dev.aurakai.auraframefx.xposed

import de.robv.android.xposed.IXposedHookLoadPackage
import de.robv.android.xposed.XC_MethodHook
import de.robv.android.xposed.XSharedPreferences
import de.robv.android.xposed.XposedBridge
import de.robv.android.xposed.XposedHelpers
import de.robv.android.xposed.callbacks.XC_LoadPackage
import dev.aurakai.auraframefx.system.lockscreen.model.LockScreenConfig
import dev.aurakai.auraframefx.system.overlay.model.SystemOverlayConfig // Changed import
import dev.aurakai.auraframefx.system.quicksettings.model.QuickSettingsConfig
import dev.aurakai.auraframefx.utils.JsonUtils
import dev.aurakai.auraframefx.xposed.hooks.LockScreenHooker
import dev.aurakai.auraframefx.xposed.hooks.NotchBarHooker
import dev.aurakai.auraframefx.xposed.hooks.QuickSettingsHooker

class AuraXposedEntry :
    IXposedHookLoadPackage { // IXposedHookLoadPackage was already imported, removed duplicate from replace
    private val TAG = "AuraXposedEntry"

    companion object {
        const val MAIN_APP_PACKAGE_NAME = "dev.aurakai.auraframefx"
        const val IPC_PREFS_NAME = "aura_fx_ipc_prefs"
        const val IPC_KEY_QUICK_SETTINGS = "quick_settings_config_json"
        const val IPC_KEY_LOCK_SCREEN = "lock_screen_config_json"
        const val IPC_KEY_SYSTEM_OVERLAY = "system_overlay_config_json" // New key
    }

    override fun handleLoadPackage(lpparam: XC_LoadPackage.LoadPackageParam?) { // Made lpparam nullable to be safe
        if (lpparam?.packageName == null) return // Adjusted for nullable lpparam

        XposedBridge.log("[$TAG] Loaded app: ${lpparam.packageName}")

        if (lpparam.packageName == "com.android.systemui") {
            XposedBridge.log("[$TAG] --- Attempting to hook SystemUI and read configs ---")

            var qsConfig: QuickSettingsConfig? = null
            var lsConfig: LockScreenConfig? = null
            var systemOverlayConfig: SystemOverlayConfig? = null // New

            try {
                // Creating a more secure approach for IPC with XSharedPreferences
                val configPref = XSharedPreferences("dev.aurakai.auraframefx", "aura_fx_prefs")

                if (!configPref.file.canRead()) {
                    XposedBridge.log("[$TAG] Cannot read XSharedPreferences, using default configs")
                    // Handle with default configs or fallbacks
                } else {
                    // Load configurations from XSharedPreferences
                    val prefJson = configPref.getString("quicksettings_config", null)
                    val lsPrefJson = configPref.getString("lockscreen_config", null)
                    val systemPrefJson = configPref.getString("system_config", null)

                    if (prefJson != null) {
                        qsConfig = JsonUtils.fromJson<QuickSettingsConfig>(prefJson)
                    }

                    if (lsPrefJson != null) {
                        lsConfig = JsonUtils.fromJson<LockScreenConfig>(lsPrefJson)
                    }

                    if (systemPrefJson != null) {
                        systemOverlayConfig =
                            JsonUtils.fromJson<SystemOverlayConfig>(systemPrefJson)
                    }
                }

                // --- Instantiate and apply hooks ---
                if (qsConfig != null) {
                    QuickSettingsHooker(lpparam.classLoader, qsConfig).applyQuickSettingsHooks()
                } else {
                    XposedBridge.log("[$TAG] QuickSettingsConfig is null, not applying QS hooks.")
                }

                if (lsConfig != null) {
                    LockScreenHooker(lpparam.classLoader, lsConfig).applyLockScreenHooks()
                } else {
                    XposedBridge.log("[$TAG] LockScreenConfig is null, not applying LS hooks.")
                }

                // NEW: Instantiate and apply NotchBarHooker
                if (systemOverlayConfig?.notchBar != null) {
                    NotchBarHooker(
                        lpparam.classLoader,
                        systemOverlayConfig.notchBar
                    ).applyNotchBarHooks()
                } else {
                    XposedBridge.log("[$TAG] SystemOverlayConfig or its NotchBarConfig is null, not applying Notch Bar hooks.")
                }

                // Existing "Hello World" hook (can be removed later if basic hooks are confirmed)
                val targetClass = XposedHelpers.findClass(
                    "com.android.systemui.shared.system.ActivityManagerWrapper",
                    lpparam.classLoader
                )
                XposedHelpers.findAndHookMethod(
                    targetClass,
                    "init",
                    object : XC_MethodHook() {
                        override fun afterHookedMethod(param: MethodHookParam) {
                            XposedBridge.log("[$TAG] *** SystemUI Hooked! Hello from AuraFrameFX! (Legacy) ***")
                        }
                    }
                )
                XposedBridge.log("[$TAG] Successfully set up basic SystemUI hook (Legacy).")

            } catch (e: Throwable) {
                XposedBridge.log("[$TAG] An unexpected error occurred during SystemUI hooking or config reading.")
                XposedBridge.log("[$TAG] Error details: ${e.message}")
                XposedBridge.log(e)

                // Try to apply minimal default hooks for essential functionality
                tryApplyFallbackHooks(lpparam.classLoader)
            }
        } else if (lpparam.packageName == MAIN_APP_PACKAGE_NAME) {
            // Self-hooking for verifying if module is active
            XposedBridge.log("[$TAG] Detected AuraFrameFX app, applying self hooks for validation")
            applySelfHooks(lpparam.classLoader)
        } else if (lpparam.packageName == "com.android.settings") {
            // Optional: Hook settings for integration
            XposedBridge.log("[$TAG] Detected Settings app, applying settings hooks")
            applySettingsHooks(lpparam.classLoader)
        } else if (isTargetedAppForRootAccess(lpparam.packageName)) {
            // Hook into OracleDrive-compatible apps for root functionalities
            XposedBridge.log("[$TAG] Detected OracleDrive compatible app: ${lpparam.packageName}")
            applyOracleDriveIntegration(lpparam.classLoader, lpparam.packageName)
        }
    }

    private fun tryApplyFallbackHooks(classLoader: ClassLoader) {
        try {
            XposedBridge.log("[$TAG] Attempting to apply fallback minimal hooks")
            // Implement fallback hooks here if needed
        } catch (e: Throwable) {
            XposedBridge.log("[$TAG] Even fallback hooks failed. Error: ${e.message}")
        }
    }

    private fun applySelfHooks(classLoader: ClassLoader) {
        try {
            // Hook into AuraFrameFX to provide module activation status
            val targetClass = XposedHelpers.findClass(
                "dev.aurakai.auraframefx.xposed.XposedBridgeService",
                classLoader
            )
            XposedHelpers.findAndHookMethod(
                targetClass,
                "isModuleActive",
                object : XC_MethodHook() {
                    override fun beforeHookedMethod(param: MethodHookParam) {
                        param.result = true
                    }
                }
            )
            XposedBridge.log("[$TAG] Successfully applied self hooks for module validation")
        } catch (e: Throwable) {
            XposedBridge.log("[$TAG] Failed to apply self hooks: ${e.message}")
        }
    }

    private fun applySettingsHooks(classLoader: ClassLoader) {
        // Implement Settings hooks if needed
    }

    private fun isTargetedAppForRootAccess(packageName: String?): Boolean {
        // Check if the app is targeted for OracleDrive integration
        val oracleDriveCompatibleApps = listOf(
            "com.aurakai.oracledrive",
            "com.aurakai.oracledrive.service",
            "com.topjohnwu.magisk",
            "me.weishu.kernelsu"
        )
        return packageName != null && oracleDriveCompatibleApps.contains(packageName)
    }

    private fun applyOracleDriveIntegration(classLoader: ClassLoader, packageName: String) {
        // Implement OracleDrive integration hooks
    }
}