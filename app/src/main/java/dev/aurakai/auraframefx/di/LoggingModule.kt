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
     * Provides a singleton instance of the AuraFxLogger implementation for application-wide logging.
     *
     * @param context Application context used by the logger.
     * @param kaiService Service dependency required by the logger implementation.
     * @return A singleton AuraFxLogger instance.
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
