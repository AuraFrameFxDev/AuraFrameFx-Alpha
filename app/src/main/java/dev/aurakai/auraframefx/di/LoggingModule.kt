package dev.aurakai.auraframefx.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.data.logging.AuraFxLogger as AuraFxLoggerImpl
import dev.aurakai.auraframefx.ai.services.KaiAIService
import javax.inject.Singleton

/**
 * Hilt Module for providing logging dependencies.
 */
@Module
@InstallIn(SingletonComponent::class)
object LoggingModule {

    /**
     * Returns a singleton instance of `AuraFxLogger` for application-wide logging.
     *
     * The logger is constructed using the application context and the provided `KaiAIService`.
     *
     * @return A singleton `AuraFxLogger` implementation.
     */
    @Provides
    @Singleton
    fun provideAuraFxLogger(
        @ApplicationContext context: Context,
        kaiService: KaiAIService
    ): AuraFxLogger {
        return AuraFxLoggerImpl(context, kaiService)
    }
}
