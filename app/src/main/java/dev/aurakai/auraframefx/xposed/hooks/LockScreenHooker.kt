package dev.aurakai.auraframefx.xposed.hooks

import android.view.View
import de.robv.android.xposed.XC_MethodHook
import de.robv.android.xposed.XposedHelpers
import dev.aurakai.auraframefx.system.lockscreen.model.LockScreenConfig
import dev.aurakai.auraframefx.system.utils.XposedLogger

/**
 * Xposed hook implementation for customizing the Android lock screen.
 *
 * @property classLoader The class loader for the target package
 * @property config Configuration for lock screen customizations
 */
class LockScreenHooker(
    private val classLoader: ClassLoader,
    private val config: LockScreenConfig
) {
    private val TAG = "LockScreenHooker"

    /**
     * Applies all lock screen hooks based on the provided configuration.
     */
    fun applyLockScreenHooks() {
        try {
            hookKeyguardViewMediator()
            hookKeyguardHostView()
            hookKeyguardSecurityContainer()
            XposedLogger.log(TAG, "Lock screen hooks applied successfully")
        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to apply lock screen hooks", e)
        }
    }

    private fun hookKeyguardViewMediator() {
        try {
            val keyguardViewMediatorClass = XposedHelpers.findClass(
                "com.android.systemui.keyguard.KeyguardViewMediator",
                classLoader
            )

            // Hook into keyguard visibility changes
            XposedHelpers.findAndHookMethod(
                keyguardViewMediatorClass,
                "setKeyguardEnabled",
                Boolean::class.javaPrimitiveType,
                object : XC_MethodHook() {
                    override fun beforeHookedMethod(param: MethodHookParam) {
                        val enabled = param.args[0] as Boolean
                        XposedLogger.log(TAG, "Keyguard enabled: $enabled")

                        // Apply custom behavior based on config
                        if (config.forceShow) {
                            param.args[0] = true
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook KeyguardViewMediator", e)
        }
    }

    private fun hookKeyguardHostView() {
        try {
            val keyguardHostViewClass = XposedHelpers.findClass(
                "com.android.keyguard.KeyguardHostView",
                classLoader
            )

            // Customize keyguard host view appearance
            XposedHelpers.findAndHookMethod(
                keyguardHostViewClass,
                "onFinishInflate",
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        val hostView = param.thisObject as View

                        // Apply custom background if configured
                        config.backgroundResId?.let { resId ->
                            try {
                                val resources = hostView.context.resources
                                hostView.background = resources.getDrawable(resId, null)
                            } catch (e: Exception) {
                                XposedLogger.logError(TAG, "Failed to set keyguard background", e)
                            }
                        }

                        // Apply custom alpha
                        if (config.alpha != null) {
                            hostView.alpha = config.alpha
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook KeyguardHostView", e)
        }
    }

    private fun hookKeyguardSecurityContainer() {
        try {
            val securityContainerClass = XposedHelpers.findClass(
                "com.android.keyguard.KeyguardSecurityContainer",
                classLoader
            )

            // Customize security view (PIN/pattern/password)
            XposedHelpers.findAndHookMethod(
                securityContainerClass,
                "showSecurityScreen",
                XposedHelpers.findClass(
                    "com.android.keyguard.KeyguardSecurityModel.SecurityMode",
                    classLoader
                ),
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        // Customize security view appearance here
                        if (config.disableCamera) {
                            try {
                                val cameraView = XposedHelpers.getObjectField(
                                    param.thisObject,
                                    "mSecurityViewFlipper"
                                )
                                XposedHelpers.callMethod(cameraView, "setCameraDisabled", true)
                            } catch (e: Exception) {
                                XposedLogger.logError(
                                    TAG,
                                    "Failed to disable camera on keyguard",
                                    e
                                )
                            }
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook KeyguardSecurityContainer", e)
        }
    }
}
