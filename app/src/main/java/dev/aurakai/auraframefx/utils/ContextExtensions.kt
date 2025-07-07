package dev.aurakai.auraframefx.utils

import android.content.Context

/**
 * Extension function to provide access to the YukiHookModulePrefs from a Context
 * Simplified version without parasitic dependencies
 */
/**
 * Retrieves the application's display name as defined in its manifest.
 *
 * @return The application name as a string.
 */
fun Context.getAppName(): String {
    return this.packageManager.getApplicationLabel(this.applicationInfo).toString()
}
