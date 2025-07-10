package dev.aurakai.auraframefx.di

import android.content.Context
import androidx.room.Room
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.data.room.AgentMemoryDao
import dev.aurakai.auraframefx.data.room.AppDatabase
import dev.aurakai.auraframefx.data.room.TaskHistoryDao
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

<<<<<<< HEAD
    /**
     * Provides a singleton instance of the Room `AppDatabase` for the application.
     *
     * Builds the database named "aura_frame_fx_database" using the application context and enables destructive migration as a fallback strategy.
     *
     * @return The singleton `AppDatabase` instance.
     */
=======
>>>>>>> pr458merge
    /**
     * Provides a singleton instance of the Room `AppDatabase` configured with destructive migration fallback.
     *
     * @param context The application context used to build the database.
     * @return The singleton `AppDatabase` instance.
     */
    /**
     * Provides a singleton instance of the Room AppDatabase configured with destructive migration fallback.
     *
     * @param context The application context used to build the database.
     * @return The initialized AppDatabase instance.
     */
    /**
     * Provides a singleton instance of the Room `AppDatabase` configured with destructive migration fallback.
     *
     * The database is built using the application context and named "aura_frame_fx_database".
     *
     * @param context The application context used to create the database.
     * @return The singleton `AppDatabase` instance.
     */
    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context.applicationContext,
            AppDatabase::class.java,
            "aura_frame_fx_database"
        )
        // Add migrations here if/when schema changes:
        // .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
        .fallbackToDestructiveMigration() // Placeholder: Consider proper migration strategies for production
        .build()
    }

<<<<<<< HEAD
    /**
     * Provides an instance of AgentMemoryDao from the given AppDatabase.
     *
     * @param database The Room database instance from which to obtain the DAO.
     * @return The AgentMemoryDao for accessing agent memory data.
     */
=======
>>>>>>> pr458merge
    /**
     * Provides the AgentMemoryDao instance from the given AppDatabase.
     *
     * @param database The Room database instance to retrieve the DAO from.
     * @return The AgentMemoryDao for accessing agent memory data.
     */
    @Provides
    fun provideAgentMemoryDao(database: AppDatabase): AgentMemoryDao {
        return database.agentMemoryDao()
    }

    @Provides
    fun provideTaskHistoryDao(database: AppDatabase): TaskHistoryDao {
        return database.taskHistoryDao()
    }
}
