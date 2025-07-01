package dev.aurakai.auraframefx.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.aurakai.auraframefx.system.homescreen.HomeScreenTransitionManager
import dev.aurakai.auraframefx.system.homescreen.model.HomeScreenTransitionConfig
import dev.aurakai.auraframefx.system.homescreen.model.HomeScreenTransitionType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HomeScreenTransitionViewModel @Inject constructor(
    private val transitionManager: HomeScreenTransitionManager,
) : ViewModel() {
    private val _currentConfig =
        MutableStateFlow(HomeScreenTransitionConfig()) // Initialized with default
    val currentConfig: StateFlow<HomeScreenTransitionConfig?> =
        _currentConfig // Kept nullable for safety, though now initialized

    init {
        viewModelScope.launch {
            transitionManager.currentConfig.collect { config ->
                _currentConfig.value = config
            }
        }
    }

    fun updateTransitionType(type: HomeScreenTransitionType) {
        viewModelScope.launch {
            transitionManager.updateTransitionType(type)
        }
    }

    fun updateTransitionDuration(duration: Int) {
        viewModelScope.launch {
            transitionManager.updateTransitionDuration(duration)
        }
    }

    fun updateTransitionProperties(properties: Map<String, Any>) {
        viewModelScope.launch {
            transitionManager.updateTransitionProperties(properties)
        }
    }

    fun resetToDefault() {
        viewModelScope.launch {
            transitionManager.resetToDefault()
        }
    }
}
