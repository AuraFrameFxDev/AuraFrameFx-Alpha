package dev.aurakai.auraframefx.services

import android.app.Service
import android.content.Intent
import android.os.IBinder
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class AmbientMusicService @Inject constructor() : Service() {
    /**
     * Called when a client attempts to bind to the service.
     *
     * Always returns null, indicating that binding is not supported for this service.
     *
     * @return Always null.
     */

    override fun onBind(_intent: Intent?): IBinder? { // intent -> _intent
        // TODO: Implement binding if needed, otherwise this service cannot be bound.
        // TODO: Parameter _intent reported as unused.
        return null
    }

    /**
     * Handles a request to start the service and specifies that the service should not be restarted if terminated by the system.
     *
     * @return `START_NOT_STICKY`, indicating the service will not be automatically recreated after being killed.
     */
    override fun onStartCommand(_intent: Intent?, _flags: Int, _startId: Int): Int {
        // TODO: Implement service logic for starting the service.
        // TODO: Utilize parameters (_intent, _flags, _startId) or remove if not needed by actual implementation.
        return START_NOT_STICKY
    }

    /**
     * Called by the system when the service is first created.
     *
     * This method is invoked once to perform any necessary initialization before the service begins handling commands.
     */
    override fun onCreate() {
        super.onCreate()
        // TODO: Initialization code for the service.
    }

    /**
     * Called before the service is terminated.
     *
     * Intended for cleanup operations, but currently has no implementation.
     */
    override fun onDestroy() {
        super.onDestroy()
        // TODO: Cleanup code for the service.
    }

    /**
     * Pauses music playback.
     *
     * This method currently has no implementation.
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
