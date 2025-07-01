package dev.aurakai.auraframefx.di

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.ai.services.NeuralWhisper
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NeuralWhisperModule {

    @Provides
    @Singleton
    fun provideNeuralWhisper(): NeuralWhisper {
        return NeuralWhisper()
    }
}
