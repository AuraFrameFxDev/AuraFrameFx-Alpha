package dev.aurakai.auraframefx.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.system.lockscreen.LockScreenCustomizer
import dev.aurakai.auraframefx.system.lockscreen.model.LockScreenAnimation
import dev.aurakai.auraframefx.system.lockscreen.model.LockScreenConfig
import dev.aurakai.auraframefx.system.lockscreen.model.LockScreenElementType
import dev.aurakai.auraframefx.system.overlay.model.OverlayShape
import dev.aurakai.auraframefx.system.quicksettings.QuickSettingsCustomizer
import dev.aurakai.auraframefx.system.quicksettings.model.QuickSettingsAnimation
import dev.aurakai.auraframefx.system.quicksettings.model.QuickSettingsConfig
import dev.aurakai.auraframefx.ui.model.ImageResource
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SystemCustomizationViewModel @Inject constructor(
    private val quickSettingsCustomizer: QuickSettingsCustomizer,
    private val lockScreenCustomizer: LockScreenCustomizer,
) : ViewModel() {
    private val _quickSettingsConfig =
        MutableStateFlow(QuickSettingsConfig()) // Initialized with default
    val quickSettingsConfig: StateFlow<QuickSettingsConfig?> =
        _quickSettingsConfig // Kept nullable for safety

    private val _lockScreenConfig = MutableStateFlow(LockScreenConfig()) // Initialized with default
    val lockScreenConfig: StateFlow<LockScreenConfig?> =
        _lockScreenConfig // Kept nullable for safety

    init {
        viewModelScope.launch {
            quickSettingsCustomizer.currentConfig.collect { config ->
                _quickSettingsConfig.value = config
            }

            lockScreenCustomizer.currentConfig.collect { config ->
                _lockScreenConfig.value = config
            }
        }
    }

    fun updateQuickSettingsTileShape(tileId: String, shape: OverlayShape) {
        viewModelScope.launch {
            quickSettingsCustomizer.updateTileShape(tileId, shape)
        }
    }

    fun updateQuickSettingsTileAnimation(tileId: String, animation: QuickSettingsAnimation) {
        viewModelScope.launch {
            quickSettingsCustomizer.updateTileAnimation(tileId, animation)
        }
    }

    fun updateQuickSettingsBackground(image: ImageResource?) {
        viewModelScope.launch {
            quickSettingsCustomizer.updateBackground(image)
        }
    }

    fun updateLockScreenElementShape(elementType: LockScreenElementType, shape: OverlayShape) {
        viewModelScope.launch {
            lockScreenCustomizer.updateElementShape(elementType, shape)
        }
    }

    fun updateLockScreenElementAnimation(
        elementType: LockScreenElementType,
        animation: LockScreenAnimation,
    ) {
        viewModelScope.launch {
            lockScreenCustomizer.updateElementAnimation(elementType, animation)
        }
    }

    fun updateLockScreenBackground(image: ImageResource?) {
        viewModelScope.launch {
            lockScreenCustomizer.updateBackground(image)
        }
    }

    fun resetToDefaults() {
        viewModelScope.launch {
            quickSettingsCustomizer.resetToDefault()
            lockScreenCustomizer.resetToDefault()
        }
    }
}
