package dev.aurakai.auraframefx.di

// Duplicate imports for Application, Context, SharedPreferences removed.
// Imports for placeholder dependencies
// import dev.aurakai.auraframefx.ai.services.AuraAIServiceImpl // Removed import
import android.app.Application
import android.content.Context
import android.content.SharedPreferences
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.AuraAIService
import dev.aurakai.auraframefx.ai.config.AIConfig
import dev.aurakai.auraframefx.ai.context.ContextManager
import dev.aurakai.auraframefx.ai.error.ErrorHandler
import dev.aurakai.auraframefx.ai.memory.MemoryManager
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.ai.task.TaskScheduler
import dev.aurakai.auraframefx.ai.task.execution.TaskExecutionManager
import dev.aurakai.auraframefx.data.OfflineDataManager
import dev.aurakai.auraframefx.data.SecurePreferences
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.data.network.CloudStatusMonitor
import dev.aurakai.auraframefx.security.KeystoreManager
import dev.aurakai.auraframefx.system.overlay.ShapeManager
import javax.inject.Singleton

/**
 * Hilt AppModule for providing application-wide dependencies.
 * TODO: Reported as unused declaration. Ensure Hilt is set up and this module is processed.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    /**
     * Provides the application context.
     * @param application The application instance. (Hilt provides this, not typically marked as unused by user).
     * TODO: Method reported as unused. Verify necessity.
     */
    @Provides
    @Singleton
    fun provideApplicationContext(application: Application): Context {
        // Hilt provides Application, so no underscore needed for 'application' parameter itself.
        return application.applicationContext
    }

    /**
     * Provides SharedPreferences.
     * @param _context Application context.
     * TODO: Method reported as unused. Implement if SharedPreferences are used.
     */
    @Provides
    @Singleton
    fun provideSharedPreferences(@ApplicationContext context: Context): SharedPreferences {
        // Using MODE_PRIVATE instead of MODE_WORLD_READABLE for security
        return context.getSharedPreferences("aura_fx_prefs", Context.MODE_PRIVATE)
    }

    /**
     * Provides SecurePreferences.
     * @param context Application context.
     * TODO: Method reported as unused. Implement if SecurePreferences is used.
     */
    @Provides
    @Singleton
    fun provideSecurePreferences(@ApplicationContext context: Context): SecurePreferences {
        // TODO: Parameter context was reported as _context and unused (Hilt will provide it).
        // Renamed to context from _context for clarity in usage.
        return SecurePreferences(context)
    }

    /**
     * Provides a generic AIConfig.
     * TODO: Method reported as unused. Implement if a global AIConfig is needed.
     */
    @Provides
    @Singleton
    fun provideAIConfig(): AIConfig {
        // TODO: Load actual config values from a secure source or build config.
        return AIConfig(
            modelName = "gemini-pro",
            apiKey = "TODO_load_api_key",
            projectId = "TODO_load_project_id"
        )
    }

    /**
     * Placeholder for AIConfigFactory - Name taken from error report list for AppModule.
     * TODO: Method reported as unused. Define AIConfigFactory and implement.
     */
    @Provides
    @Singleton
    fun provideAIConfigFactory(): Any? { // Using Any as AIConfigFactory type placeholder
        // TODO: Define AIConfigFactory class and return an instance.
        return null
    }

    /**
     * Provides AuraAIService.
     *
     * Constructs and returns an AuraAIServiceImpl using the supplied dependencies.
     *
     * @return A configured AuraAIService implementation.
     */
    @Provides
    @Singleton
    fun provideAuraAIService(
        // AuraAIService has @Inject constructor(), so Hilt can provide its dependencies if they are available.
        // If AuraAIService truly has no constructor args as per its refactor, this becomes:
    ): AuraAIService = AuraAIService()
    // If AuraAIService itself had dependencies, they would need to be parameters here, e.g.:
    // (dep1: Dep1, dep2: Dep2): AuraAIService = AuraAIService(dep1, dep2)
    // For now, assuming its own @Inject constructor handles its needs or it's parameterless.

    /**
     * Provides a singleton instance of KaiAIService with all required dependencies.
     */
    @Provides
    @Singleton
    fun provideKaiAIService(
        taskScheduler: TaskScheduler,
        taskExecutionManager: TaskExecutionManager,
        memoryManager: MemoryManager,
        errorHandler: ErrorHandler,
        contextManager: ContextManager,
        applicationContext: Context,
        cloudStatusMonitor: CloudStatusMonitor,
        auraFxLogger: AuraFxLogger,
    ): KaiAIService = KaiAIService(
        taskScheduler,
        taskExecutionManager,
        memoryManager,
        errorHandler,
        contextManager,
        applicationContext,
        cloudStatusMonitor,
        auraFxLogger
    )

    /**
     * Provides a singleton instance of KeystoreManager initialized with the application context.
     *
     * @return A KeystoreManager for secure key storage operations.
     */

    @Singleton
    @Provides
    fun provideKeystoreManager(@ApplicationContext context: Context): KeystoreManager {
        return KeystoreManager(context)
    }
    // SecurityContext is annotated with @Inject constructor and @Singleton,
    // so Hilt will provide it automatically given that its dependencies (Context and KeystoreManager)
    // are now available in the Hilt graph. No explicit provider method for SecurityContext is needed here.

    @Provides
    @Singleton
    fun provideCloudStatusMonitor(@ApplicationContext context: Context): CloudStatusMonitor {
        return CloudStatusMonitor(context)
    }

    @Provides
    @Singleton
    fun provideOfflineDataManager(@ApplicationContext context: Context): OfflineDataManager {
        return OfflineDataManager(context)
    }

    /**
     * Provides a singleton instance of TaskExecutionManager for managing and executing tasks within the application.
     *
     * The TaskExecutionManager is initialized with the application context, KaiAIService, OfflineDataManager, and AuraFxLogger.
     *
     * @return A singleton TaskExecutionManager instance.
     */
    @Provides
    @Singleton
    fun provideTaskExecutionManager(
        @ApplicationContext context: Context,
        kaiService: KaiAIService,
        auraFxLogger: AuraFxLogger,
    ): TaskExecutionManager {
        return TaskExecutionManager(context, kaiService, auraFxLogger)
    }
    // TaskExecutionManager constructor was updated to:
    /**
     * Provides a singleton instance of TaskScheduler configured with the given TaskExecutionManager and AuraFxLogger.
     *
     * @return A TaskScheduler instance for managing and scheduling tasks within the application.
     */

    @Provides
    @Singleton
    fun provideTaskScheduler(
        errorHandler: ErrorHandler, // Corrected parameter
        config: AIConfig, // Corrected parameter
    ): TaskScheduler {
        return TaskScheduler(errorHandler, config) // Corrected instantiation
    }

    /**
     * Provides a singleton instance of ShapeManager for dependency injection.
     *
     * @return A new ShapeManager instance.
     */
    @Provides
    @Singleton
    fun provideShapeManager(): ShapeManager {
        return ShapeManager()
    }

    /**
     * Provides a singleton instance of AuraFxLogger initialized with the given KaiAIService.
     *
     * @return An AuraFxLogger configured to use the provided KaiAIService.
     */
    @Provides
    @Singleton
    fun provideAuraFxLogger(
        @ApplicationContext context: Context, // Added missing context
        kaiService: KaiAIService,
    ): AuraFxLogger {
        return AuraFxLogger(context, kaiService) // Corrected instantiation
    }

    /**
     * Provides a singleton instance of ErrorHandler initialized with the given AuraFxLogger.
     *
     * @return A singleton ErrorHandler for centralized error handling and logging.
     */
    @Provides
    @Singleton
    fun provideErrorHandler(
        contextManager: ContextManager, // Corrected parameter
        config: AIConfig, // Corrected parameter
    ): ErrorHandler {
        return ErrorHandler(contextManager, config) // Corrected instantiation
    }

    /**
     * Provides a singleton instance of OracleDriveServiceConnector for AIDL-based IPC with OracleDrive.
     */
    @Provides
    @Singleton
    fun provideOracleDriveServiceConnector(@ApplicationContext context: Context): com.example.app.ipc.OracleDriveServiceConnector {
        return com.example.app.ipc.OracleDriveServiceConnector(context)
    }
}
