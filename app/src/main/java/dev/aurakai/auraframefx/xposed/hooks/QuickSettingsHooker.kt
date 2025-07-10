package dev.aurakai.auraframefx.xposed.hooks

import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Rect
import android.view.View
import android.view.ViewGroup
import de.robv.android.xposed.XC_MethodHook
import de.robv.android.xposed.XposedHelpers
import dev.aurakai.auraframefx.system.quicksettings.model.QuickSettingsConfig
import dev.aurakai.auraframefx.system.utils.XposedLogger

/**
 * Xposed hook implementation for customizing the Quick Settings panel.
 *
 * @property classLoader The class loader for the target package
 * @property config Configuration for Quick Settings customizations
 */
class QuickSettingsHooker(
    private val classLoader: ClassLoader,
    private val config: QuickSettingsConfig
) {
    private val TAG = "QuickSettingsHooker"

    /**
     * Applies all Quick Settings hooks based on the provided configuration.
     */
    fun applyQuickSettingsHooks() {
        try {
            hookQSPanel()
            hookQSTileHost()
            hookQSContainerImpl()
            XposedLogger.log(TAG, "Quick Settings hooks applied successfully")
        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to apply Quick Settings hooks", e)
        }
    }

    private fun hookQSPanel() {
        try {
            val qsPanelClass = XposedHelpers.findClass(
                "com.android.systemui.qs.QSPanel",
                classLoader
            )

            // Hook into QS panel inflation
            XposedHelpers.findAndHookMethod(
                qsPanelClass,
                "onFinishInflate",
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        val qsPanel = param.thisObject as ViewGroup

                        // Apply custom background from config if available
                        config.background?.let { background ->
                            // Apply background color if specified
                            if (background.color != null) {
                                qsPanel.setBackgroundColor(Color.parseColor(background.color))
                            }
                            
                            // Apply alpha if specified
                            background.alpha?.let { alpha ->
                                qsPanel.alpha = alpha
                            }
                            
                            // Apply padding if specified
                            background.padding?.let { padding ->
                                val paddingPx = qsPanel.resources.getDimensionPixelSize(padding)
                                qsPanel.setPadding(paddingPx, paddingPx, paddingPx, paddingPx)
                            }
                        }
                    }
                }
            )

            // Hook into tile click handling
            XposedHelpers.findAndHookMethod(
                qsPanelClass,
                "handleShowDetail",
                XposedHelpers.findClass(
                    "com.android.systemui.qs.QSPanel\$Record",
                    classLoader
                ),
                Boolean::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                Int::class.javaPrimitiveType,
                Boolean::class.javaPrimitiveType,
                object : XC_MethodHook() {
                    override fun beforeHookedMethod(param: MethodHookParam) {
                        // Check if we should disable tile clicks for all tiles
                        if (config.tiles.none { it.enableClicks }) {
                            param.result = null // Prevent default behavior
                            return
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook QSPanel", e)
        }
    }

    private fun hookQSTileHost() {
        try {
            val qsTileHostClass = XposedHelpers.findClass(
                "com.android.systemui.qs.tileimpl.QSTileImpl",
                classLoader
            )

            // Hook into tile state changes
            XposedHelpers.findAndHookMethod(
                qsTileHostClass,
                "refreshState",
                object : XC_MethodHook() {
                    override fun beforeHookedMethod(param: MethodHookParam) {
                        // Get tile spec/ID
                        val tileSpec = XposedHelpers.getObjectField(
                            param.thisObject,
                            "mTileSpec"
                        ) as? String ?: return

                        // Find matching tile config
                        val tileConfig = config.tiles.find { it.id == tileSpec } ?: return

                        try {
                            val state = XposedHelpers.getObjectField(
                                param.thisObject,
                                "mState"
                            )
                            
                            // Apply tile state from config
                            XposedHelpers.setBooleanField(
                                state, 
                                "value", 
                                tileConfig.enabled
                            )
                            
                            // Apply any additional tile state customizations here
                            
                        } catch (e: Exception) {
                            XposedLogger.logError(TAG, "Failed to update tile state for $tileSpec", e)
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook QSTileHost", e)
        }
    }

    private fun hookQSContainerImpl() {
        try {
            val qsContainerClass = XposedHelpers.findClass(
                "com.android.systemui.qs.QSContainerImpl",
                classLoader
            )

            // Hook into container layout changes
            XposedHelpers.findAndHookMethod(
                qsContainerClass,
                "onConfigurationChanged",
                Configuration::class.java,
                object : XC_MethodHook() {
                    override fun afterHookedMethod(param: MethodHookParam) {
                        val container = param.thisObject as View

                        // Apply background from config if available
                        config.background?.let { background ->
                            // Apply background color if specified
                            background.color?.let { color ->
                                container.setBackgroundColor(Color.parseColor(color))
                            }
                            
                            // Apply custom height if specified
                            background.heightDp?.let { heightDp ->
                                val density = container.resources.displayMetrics.density
                                val heightPx = (heightDp * density + 0.5f).toInt()
                                val layoutParams = container.layoutParams
                                layoutParams.height = heightPx
                                container.layoutParams = layoutParams
                            }
                        }
                    }
                }
            )

        } catch (e: Throwable) {
            XposedLogger.logError(TAG, "Failed to hook QSContainerImpl", e)
        }
    }
}
