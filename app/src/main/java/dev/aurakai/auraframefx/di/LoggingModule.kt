package dev.aurakai.auraframefx.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import javax.inject.Singleton
import dev.aurakai.auraframefx.data.logging.AuraFxLogger as AuraFxLoggerImpl

/**
 * Hilt Module for providing logging dependencies.
 */
@Module
@InstallIn(SingletonComponent::class)
object LoggingModule {

    /**
     * Returns a singleton instance of `AuraFxLogger` configured with the application context and KaiAIService.
     *
     * @param context The application context for logger configuration.
     * @param kaiService The KaiAIService used by the logger.
     * @return A singleton `AuraFxLogger` implementation for application-wide logging.
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
