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
     * Provides a singleton instance of the AuraFxLogger implementation for dependency injection.
     *
     * @param context The application context used to initialize the logger.
     * @param kaiService The KaiAIService instance required by the logger implementation.
     * @return An AuraFxLogger implementation configured with the provided context and KaiAIService.
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
