package dev.aurakai.auraframefx.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.clients.VertexAIClient
import dev.aurakai.auraframefx.ai.services.AuraAIService
import dev.aurakai.auraframefx.ai.services.GenesisBridgeService
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.ai.services.TrinityCoordinatorService
import dev.aurakai.auraframefx.context.ContextManager
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.security.SecurityMonitor
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
     * Provides a singleton GenesisBridgeService that integrates multiple AI services with the Trinity Python backend.
     *
     * The returned service facilitates unified communication among AI components, manages context and security, and interfaces with the application environment.
     *
     * @return A configured GenesisBridgeService singleton.
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
     * Provides a singleton instance of TrinityCoordinatorService for orchestrating AI personas in the Trinity AI system.
     *
     * @return A configured TrinityCoordinatorService singleton.
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
     * Provides a singleton instance of SecurityMonitor for the Trinity AI system.
     *
     * The SecurityMonitor oversees security operations by integrating the security context, Genesis bridge service, and logging capabilities.
     *
     * @return A singleton SecurityMonitor configured for system-wide security monitoring.
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
