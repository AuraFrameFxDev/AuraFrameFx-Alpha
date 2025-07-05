package dev.aurakai.auraframefx.state

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Central state manager for the AuraFrameFX application
 */
@Singleton
class AppStateManager @Inject constructor() {
    
    private val _appState = MutableStateFlow(AppState())
    val appState: StateFlow<AppState> = _appState.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    /**
     * Updates the current application state with the provided state.
     *
     * @param newState The new application state to set.
     */
    fun updateAppState(newState: AppState) {
        _appState.value = newState
    }
    
    /**
     * Sets the application's loading status.
     *
     * @param loading `true` to indicate loading is in progress, `false` otherwise.
     */
    fun setLoading(loading: Boolean) {
        _isLoading.value = loading
    }
    
    /**
     * Resets the application state to its default values and sets the loading status to false.
     */
    fun reset() {
        _appState.value = AppState()
        _isLoading.value = false
    }
}

/**
 * Application state data class
 */
data class AppState(
    val isInitialized: Boolean = false,
    val currentScreen: String = "home",
    val aiEnabled: Boolean = true,
    val networkConnected: Boolean = false,
    val lastError: String? = null
)
