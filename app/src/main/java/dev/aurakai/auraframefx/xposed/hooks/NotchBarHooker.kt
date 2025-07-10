package dev.aurakai.auraframefx.xposed.hooks

import android.content.res.Configuration
import android.graphics.Color
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import de.robv.android.xposed.XC_MethodHook
import de.robv.android.xposed.XposedHelpers
import dev.aurakai.auraframefx.system.overlay.model.NotchBarConfig
import dev.aurakai.auraframefx.system.utils.XposedLogger

/**
 * Xposed hook implementation for customizing the status bar and notch area.
 *
 * @property classLoader The class loader for the target package
 * @property config Configuration for notch and status bar customizations
 */
class NotchBarHooker(
    private val classLoader: ClassLoader,
    private val config: NotchBarConfig
) {
    private val TAG = "NotchBarHooker"
    private var customNotchView: View? = null

    /**
     * Applies all notch and status bar hooks based on the provided configuration.
     */
    fun applyNotchBarHooks() {
        try {
            hookStatusBarWindowView()
            hookStatusBarWindowManager()
            hookSystemUITheme()
            XposedLogger.log(TAG, "Notch bar hooks applied successfully")
        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to apply notch bar hooks", e)
        }
    }

    private fun hookStatusBarWindowView() {
        try {
            val statusBarWindowViewClass = XposedHelpers.findClass(
                "com.android.systemui.statusbar.phone.NotificationPanelViewController",
                classLoader
            )

            // Hook into status bar window view creation
            XposedHelpers.findAndHookMethod(
                statusBarWindowViewClass,
                "onFinishInflate",
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        val statusBarView = param.thisObject as ViewGroup

                        // Apply custom padding for notch
                        if (config.notchWidth > 0 && config.notchHeight > 0) {
                            try {
                                val resources = statusBarView.context.resources
                                val statusBarHeight = resources.getDimensionPixelSize(
                                    resources.getIdentifier("status_bar_height", "dimen", "android")
                                )

                                // Calculate notch position (centered at top)
                                val displayMetrics = resources.displayMetrics
                                (displayMetrics.widthPixels - config.notchWidth) / 2

                                // Create custom notch view
                                val windowManager = statusBarView.context.getSystemService(
                                    Context.WINDOW_SERVICE
                                ) as WindowManager

                                val layoutParams = WindowManager.LayoutParams(
                                    config.notchWidth,
                                    config.notchHeight,
                                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                                            or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                                            or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                                    PixelFormat.TRANSLUCENT
                                )

                                layoutParams.gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                                layoutParams.y = statusBarHeight - config.notchHeight / 2

                                // Create and add the custom notch view
                                customNotchView = View(statusBarView.context).apply {
                                    setBackgroundColor(Color.TRANSPARENT)
                                    alpha = 0f // Start hidden, will be shown if enabled in config
                                }

                                windowManager.addView(customNotchView, layoutParams)

                                // Apply corner radius if specified
                                if (config.cornerRadius > 0) {
                                    customNotchView?.apply {
                                        outlineProvider = object : ViewOutlineProvider() {
                                            override fun getOutline(view: View, outline: Outline) {
                                                outline.setRoundRect(
                                                    0, 0, view.width, view.height,
                                                    config.cornerRadius.toFloat()
                                                )
                                            }
                                        }
                                        clipToOutline = true
                                    }
                                }

                            } catch (e: Exception) {
                                XposedLogger.logError(TAG, "Failed to create custom notch view", e)
                            }
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook status bar window view", e)
        }
    }

    private fun hookStatusBarWindowManager() {
        try {
            val statusBarWindowManagerClass = XposedHelpers.findClass(
                "com.android.systemui.statusbar.phone.StatusBarWindowController",
                classLoader
            )

            // Hook into status bar window layout changes
            XposedHelpers.findAndHookMethod(
                statusBarWindowManagerClass,
                "apply",
                XposedHelpers.findClass(
                    "com.android.systemui.statusbar.phone.StatusBarWindowController.State",
                    classLoader
                ),
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        // Update notch view visibility based on configuration
                        customNotchView?.visibility =
                            if (config.showNotch) View.VISIBLE else View.GONE

                        // Apply custom background color if specified
                        config.backgroundColor?.let { color ->
                            customNotchView?.setBackgroundColor(color)
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook status bar window manager", e)
        }
    }

    private fun hookSystemUITheme() {
        try {
            val statusBarClass = XposedHelpers.findClass(
                "com.android.systemui.statusbar.phone.PhoneStatusBarView",
                classLoader
            )

            // Hook into status bar theme changes
            XposedHelpers.findAndHookMethod(
                statusBarClass,
                "onConfigurationChanged",
                Configuration::class.java,
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        val statusBar = param.thisObject as View

                        // Apply custom theme settings
                        if (config.forceLightStatusBar) {
                            statusBar.systemUiVisibility = statusBar.systemUiVisibility or
                                    View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
                        }

                        // Apply custom background alpha
                        if (config.backgroundAlpha != null) {
                            statusBar.alpha = config.backgroundAlpha
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook SystemUI theme", e)
        }
    }
}
