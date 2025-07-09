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
<<<<<<< HEAD
     * Provides a singleton GenesisBridgeService that connects multiple AI services with the Trinity Python backend.
     *
     * Returns a configured GenesisBridgeService instance for use throughout the application.
     *
     * @return The singleton GenesisBridgeService instance.
=======
     * Provides a singleton instance of GenesisBridgeService for connecting AI services to the Trinity Python backend.
     *
     * Integrates AI service, context management, security, and logging components to enable communication between the application and the Trinity system.
     *
     * @return A configured GenesisBridgeService instance.
>>>>>>> pr458merge
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
<<<<<<< HEAD
     * Provides a singleton instance of TrinityCoordinatorService to coordinate AI personas in the Trinity AI system.
     *
     * @return The configured TrinityCoordinatorService singleton.
=======
     * Provides a singleton TrinityCoordinatorService for managing and coordinating AI personas and services within the Trinity AI system.
     *
     * @return A configured TrinityCoordinatorService instance.
>>>>>>> pr458merge
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
<<<<<<< HEAD
     * Provides a singleton SecurityMonitor for overseeing security in the Trinity AI system.
     *
     * @return The configured SecurityMonitor instance.
=======
     * Provides a singleton SecurityMonitor that oversees security operations in the Trinity AI system.
     *
     * @return A configured SecurityMonitor instance.
>>>>>>> pr458merge
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
