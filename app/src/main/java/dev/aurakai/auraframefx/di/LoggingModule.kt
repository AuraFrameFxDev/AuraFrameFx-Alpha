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
     * Supplies a singleton instance of `AuraFxLogger` for use throughout the application.
     *
     * @param context The application context required by the logger implementation.
     * @param kaiService The KaiAIService dependency used by the logger.
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
