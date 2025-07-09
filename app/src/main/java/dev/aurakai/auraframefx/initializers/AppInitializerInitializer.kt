package dev.aurakai.auraframefx.initializers

import android.content.Context
import androidx.startup.Initializer

// Assuming this is meant to be an Initializer for the App Startup library.
// Replace 'Unit' with the actual type this initializer provides if different.
import android.util.Log // Added import for Log

class AppInitializerInitializer : Initializer<Unit> {

    /**
     * Initializes core components during app startup.
     *
     * Called on the main thread when the application starts. Intended for AuraFrameFX core initialization; currently performs no operations beyond logging.
     */
    override fun create(context: Context) {
        Log.d("AppInitializer", "AppInitializerInitializer - create called.")
        // This method is called on the main thread during app startup.
        // Actual initialization logic for AuraFrameFX core components would go here.
        // For now, just logging.
    }

    /**
     * Returns an empty list, indicating that this initializer has no dependencies on other initializers.
     *
     * @return An empty list of initializer dependencies.
     */
    override fun dependencies(): List<Class<out Initializer<*>>> {
        // No explicit dependencies for now.
        return emptyList()
    }
}
