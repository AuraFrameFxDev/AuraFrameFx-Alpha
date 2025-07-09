package dev.aurakai.auraframefx.services

import android.app.Service
import android.content.Intent
import android.os.IBinder
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class AmbientMusicService @Inject constructor() : Service() {
    /**
     * Returns null to indicate that binding to this service is not supported.
     *
     * @return Always returns null, as this service does not allow binding.
     */

    override fun onBind(_intent: Intent?): IBinder? { // intent -> _intent
        // TODO: Implement binding if needed, otherwise this service cannot be bound.
        // TODO: Parameter _intent reported as unused.
        return null
    }

    /**
     * Handles requests to start the service.
     *
     * Returns `START_NOT_STICKY`, indicating the service should not be recreated if it is killed by the system.
     *
     * @return The start mode for the service.
     */
    override fun onStartCommand(_intent: Intent?, _flags: Int, _startId: Int): Int {
        // TODO: Implement service logic for starting the service.
        // TODO: Utilize parameters (_intent, _flags, _startId) or remove if not needed by actual implementation.
        return START_NOT_STICKY
    }

    /**
     * Initializes the service when it is first created.
     *
     * Override this method to set up resources or perform setup tasks needed for the service lifecycle.
     */
    override fun onCreate() {
        super.onCreate()
        // TODO: Initialization code for the service.
    }

    /**
     * Called when the service is being destroyed.
     *
     * Override this method to perform cleanup before the service is terminated.
     */
    override fun onDestroy() {
        super.onDestroy()
        // TODO: Cleanup code for the service.
    }

    /**
     * Pauses music playback.
     */
    fun pause() {
        // TODO: Implement pause logic. Reported as unused. Implement or remove.
    }

    fun resume() {
        // TODO: Implement resume logic. Reported as unused. Implement or remove.
    }

    fun setVolume(_volume: Float) {
        // TODO: Reported as unused. Implement or remove.
    }

    fun setShuffling(_isShuffling: Boolean) {
        // TODO: Reported as unused. Implement or remove.
    }

    fun getCurrentTrack(): Any? { // Return type Any? as placeholder
        // TODO: Reported as unused. Implement or remove.
        return null
    }

    fun getTrackHistory(): List<Any> { // Return type List<Any> as placeholder
        // TODO: Reported as unused. Implement or remove.
        return emptyList()
    }

    fun skipToNextTrack() {
        // TODO: Reported as unused. Implement or remove.
    }

    fun skipToPreviousTrack() {
        // TODO: Reported as unused. Implement or remove.
    }
}
