package dev.aurakai.auraframefx.system.homescreen

import com.highcapable.yukihookapi.hook.xposed.prefs.data.YukiHookModulePrefs
import com.highcapable.yukihookapi.hook.xposed.service.YukiHookServiceManager
import dev.aurakai.auraframefx.system.common.ImageResourceManager
import dev.aurakai.auraframefx.system.homescreen.model.HomeScreenTransitionConfig
import dev.aurakai.auraframefx.system.homescreen.model.HomeScreenTransitionType
import dev.aurakai.auraframefx.system.overlay.ShapeManager
import dev.aurakai.auraframefx.system.overlay.SystemOverlayManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class HomeScreenTransitionManager @Inject constructor(
    private val overlayManager: SystemOverlayManager,
    private val shapeManager: ShapeManager,
    private val imageManager: ImageResourceManager,
    private val prefs: YukiHookModulePrefs,
    private val overlayService: YukiHookServiceManager,
) {
    private val _currentConfig =
        MutableStateFlow(HomeScreenTransitionConfig()) // Initialize with default
    val currentConfig: StateFlow<HomeScreenTransitionConfig?> =
        _currentConfig // Kept nullable for safety

    private val defaultConfig = HomeScreenTransitionConfig(
        type = HomeScreenTransitionType.GLOBE_ROTATE,
        duration = 500,
        // easing = "easeInOut", // Removed, not in HomeScreenTransitionConfig
        properties = mapOf(
            "angle" to 360f,
            "scale" to 1.2f,
            "offset" to 0f,
            "amplitude" to 0.1f,
            "frequency" to 0.5f,
            "color" to "#00FFCC",
            "blur" to 20f,
            "spread" to 0.2f
        )
    )

    init {
        loadConfig()
    }

    private fun loadConfig() {
        val savedConfig = prefs.getString("home_screen_transition", null)
        if (savedConfig != null) {
            // TODO: Parse saved config
            _currentConfig.value = defaultConfig
        } else {
            _currentConfig.value = defaultConfig
        }
    }

    fun applyConfig(config: HomeScreenTransitionConfig) {
        _currentConfig.value = config
        overlayService.hook {
            // TODO: Implement transition hooking
        }
    }

    fun resetToDefault() {
        applyConfig(defaultConfig)
    }

    fun updateTransitionType(type: HomeScreenTransitionType) {
        val current = _currentConfig.value ?: return
        val newConfig = current.copy(
            type = type
        )
        applyConfig(newConfig)
    }

    fun updateTransitionDuration(duration: Int) {
        val current = _currentConfig.value ?: return
        val newConfig = current.copy(
            duration = duration
        )
        applyConfig(newConfig)
    }

    fun updateTransitionProperties(properties: Map<String, Any>) {
        val current = _currentConfig.value ?: return
        val newConfig = current.copy(
            properties = properties
        )
        applyConfig(newConfig)
    }
}
