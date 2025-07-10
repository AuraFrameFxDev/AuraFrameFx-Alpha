package dev.aurakai.auraframefx

import android.app.Application
import android.util.Log
import androidx.work.Configuration
import dagger.hilt.android.HiltAndroidApp
import dev.aurakai.auraframefx.core.logging.TimberInitializer
import timber.log.Timber
import javax.inject.Inject

@HiltAndroidApp
class AuraFrameApplication : Application(), Configuration.Provider {

    @Inject
    lateinit var timberInitializer: TimberInitializer

    override fun onCreate() {
        super.onCreate()

        // Initialize Timber for logging
        timberInitializer.initialize(this)

        // Log application start
        Timber.tag("AuraFrameFX").i("AuraFrameFX Application started")
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setMinimumLoggingLevel(if (BuildConfig.DEBUG) Log.DEBUG else Log.INFO)
            .build()
}

// Extension function to get the application instance
val Application.timber: Timber.Tree
    get() = Timber.asTree()