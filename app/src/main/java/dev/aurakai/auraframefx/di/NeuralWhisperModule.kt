package dev.aurakai.auraframefx.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.services.NeuralWhisper
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NeuralWhisperModule {

    /**
     * Returns a singleton instance of NeuralWhisper initialized with the application context.
     *
     * @param context The application context used to initialize the NeuralWhisper instance.
     * @return The singleton NeuralWhisper instance.
     */
    @Provides
    @Singleton
    fun provideNeuralWhisper(@ApplicationContext context: Context): NeuralWhisper {
        return NeuralWhisper(context)
    }
}
