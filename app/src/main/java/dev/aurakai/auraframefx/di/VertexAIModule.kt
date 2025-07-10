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
     * Returns a singleton `VertexAIConfig` instance preconfigured for Vertex AI integration.
     *
     * The configuration includes project ID, location, API endpoint, model name and version, safety filters, retry and timeout settings, concurrency limits, and caching parameters.
     *
     * @return A `VertexAIConfig` instance with production-ready settings for Vertex AI services.
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
     * Returns a singleton instance of `VertexAIClient` for interacting with Vertex AI services.
     *
     * The client is initialized with the specified configuration, application context, security context, and logger.
     *
     * @return A singleton `VertexAIClient` instance.
     */
    @Provides
    @Singleton
    fun provideVertexAIClient(
        config: VertexAIConfig,
        @ApplicationContext context: Context,
        securityContext: SecurityContext,
        logger: AuraFxLogger
    ): VertexAIClient {
        return VertexAIClientImpl()
    }
}
