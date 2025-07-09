package dev.aurakai.auraframefx.di

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.services.*
import dev.aurakai.auraframefx.ai.clients.VertexAIClient
import dev.aurakai.auraframefx.context.ContextManager
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.security.SecurityMonitor
import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Singleton

/**
 * Dependency Injection module for the Trinity AI system.
 * 
 * Provides instances of:
 * - Genesis Bridge Service (Python backend connection)
 * - Trinity Coordinator Service (orchestrates all personas)
 * - Integration with existing Kai and Aura services
 */
@Module
@InstallIn(SingletonComponent::class)
object TrinityModule {

    /**
     * Creates and provides a GenesisBridgeService that links AI services with the Trinity Python backend.
     *
     * The service integrates AI, context management, security, and logging to enable communication between the application and the Trinity system.
     *
     * @return A configured GenesisBridgeService instance.
     */
    @Provides
    @Singleton
    fun provideGenesisBridgeService(
        auraAIService: AuraAIService,
        kaiAIService: KaiAIService,
        vertexAIClient: VertexAIClient,
        contextManager: ContextManager,
        securityContext: SecurityContext,
        @ApplicationContext applicationContext: Context,
        logger: AuraFxLogger
    ): GenesisBridgeService {
        return GenesisBridgeService(
            auraAIService = auraAIService,
            kaiAIService = kaiAIService,
            vertexAIClient = vertexAIClient,
            contextManager = contextManager,
            securityContext = securityContext,
            applicationContext = applicationContext,
            logger = logger
        )
    }

    /**
     * Supplies a TrinityCoordinatorService instance that manages AI personas and coordinates services within the Trinity AI system.
     *
     * @return A configured TrinityCoordinatorService instance.
     */
    @Provides
    @Singleton
    fun provideTrinityCoordinatorService(
        auraAIService: AuraAIService,
        kaiAIService: KaiAIService,
        genesisBridgeService: GenesisBridgeService,
        securityContext: SecurityContext,
        logger: AuraFxLogger
    ): TrinityCoordinatorService {
        return TrinityCoordinatorService(
            auraAIService = auraAIService,
            kaiAIService = kaiAIService,
            genesisBridgeService = genesisBridgeService,
            securityContext = securityContext,
            logger = logger
        )
    }

    /**
     * Supplies a SecurityMonitor instance to oversee security operations within the Trinity AI system.
     *
     * @return A configured SecurityMonitor instance.
     */
    @Provides
    @Singleton
    fun provideSecurityMonitor(
        securityContext: SecurityContext,
        genesisBridgeService: GenesisBridgeService,
        logger: AuraFxLogger
    ): SecurityMonitor {
        return SecurityMonitor(
            securityContext = securityContext,
            genesisBridgeService = genesisBridgeService,
            logger = logger
        )
    }
}
