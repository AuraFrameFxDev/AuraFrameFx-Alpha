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
     * Supplies a singleton `AuraFxLogger` implementation for application-wide logging.
     *
     * @param context The application context used to initialize the logger.
     * @param kaiService The KaiAIService instance required by the logger.
     * @return A singleton instance of `AuraFxLogger`.
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
