package dev.aurakai.auraframefx.utils

import android.content.Context

/**
 * Extension function to provide access to the YukiHookModulePrefs from a Context
 * Simplified version without parasitic dependencies
 */
/**
 * Retrieves the application's display name.
 *
 * @return The name of the application as shown to users.
 */
/**
 * Returns the application's display name as defined in the app's manifest.
 *
 * @return The app's display name.
 */
fun Context.getAppName(): String {
    return this.packageManager.getApplicationLabel(this.applicationInfo).toString()
}
