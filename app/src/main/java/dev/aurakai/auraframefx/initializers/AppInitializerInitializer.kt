package dev.aurakai.auraframefx.initializers

import android.content.Context
import androidx.startup.Initializer

// Assuming this is meant to be an Initializer for the App Startup library.
// Replace 'Unit' with the actual type this initializer provides if different.
import android.util.Log // Added import for Log

class AppInitializerInitializer : Initializer<Unit> {

    /**
     * Invoked on the main thread during app startup to perform application initialization.
     *
     * Currently, this method logs a debug message and does not execute any additional initialization logic.
     */
    override fun create(context: Context) {
        Log.d("AppInitializer", "AppInitializerInitializer - create called.")
        // This method is called on the main thread during app startup.
        // Actual initialization logic for AuraFrameFX core components would go here.
        // For now, just logging.
    }

    /**
     * Returns an empty list to indicate that this initializer does not depend on any other initializers.
     *
     * @return An empty list of initializer dependencies.
     */
    override fun dependencies(): List<Class<out Initializer<*>>> {
        // No explicit dependencies for now.
        return emptyList()
    }
}
