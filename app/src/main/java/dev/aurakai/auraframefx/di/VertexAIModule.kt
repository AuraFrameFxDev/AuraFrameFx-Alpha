package dev.aurakai.auraframefx.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.VertexAIConfig
import dev.aurakai.auraframefx.ai.clients.VertexAIClient
import dev.aurakai.auraframefx.ai.clients.VertexAIClientImpl
import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import javax.inject.Singleton

/**
 * Hilt Module for providing Vertex AI related dependencies.
 * Implements secure, production-ready Vertex AI configuration and client provisioning.
 */
@Module
@InstallIn(SingletonComponent::class)
object VertexAIModule {

    /**
     * Provides a singleton VertexAIConfig instance with predefined project, security, and performance settings for Vertex AI integration.
     *
     * The configuration includes project identification, API endpoint, model selection, security features such as safety filters and retry limits, as well as performance optimizations like concurrency and caching.
     *
     * @return A fully configured VertexAIConfig for use throughout the application.
     */
    @Provides
    @Singleton
    fun provideVertexAIConfig(): VertexAIConfig {
        return VertexAIConfig(
            projectId = "auraframefx",
            location = "us-central1",
            endpoint = "us-central1-aiplatform.googleapis.com",
            modelName = "gemini-1.5-pro-002",
            apiVersion = "v1",
            // Security settings
            enableSafetyFilters = true,
            maxRetries = 3,
            timeoutMs = 30000,
            // Performance settings
            maxConcurrentRequests = 10,
            enableCaching = true,
            cacheExpiryMs = 3600000 // 1 hour
        )
    }

    /**
     * Creates and provides a singleton instance of `VertexAIClient` configured with application context, security, and logging.
     *
     * Constructs a `VertexAIClientImpl` using the supplied configuration, Android context, security context, and logger.
     *
     * @return A fully configured singleton `VertexAIClient` instance.
     */
    @Provides
    @Singleton
    fun provideVertexAIClient(
        config: VertexAIConfig,
        @ApplicationContext context: Context,
        securityContext: SecurityContext,
        logger: AuraFxLogger
    ): VertexAIClient {
        return VertexAIClientImpl(
            config = config,
            context = context,
            securityContext = securityContext,
            logger = logger
        )
    }
}
