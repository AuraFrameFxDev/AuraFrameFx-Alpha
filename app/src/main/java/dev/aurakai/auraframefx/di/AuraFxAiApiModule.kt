package dev.aurakai.auraframefx.di

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.api.client.apis.AIContentApi
import dev.aurakai.auraframefx.network.AuraFxContentApiClient
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import javax.inject.Singleton

/**
 * Dagger Hilt module for providing dependencies related to the AuraFrameFx AI API.
 */
@Module
@InstallIn(SingletonComponent::class)
object AuraFxAiApiModule {

    /**
     * Provides a singleton OkHttpClient with HTTP request and response body logging enabled at the BODY level.
     *
     * @return A singleton OkHttpClient instance configured for detailed HTTP logging.
     */
    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        val loggingInterceptor = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
        
        return OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .build()
    }

    /**
     * Provides a singleton Json serializer configured for robust API communication.
     *
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values to ensure flexible and resilient handling of API data.
     *
     * @return A configured Json instance for serializing and deserializing API requests and responses.
     */
    @Provides
    @Singleton
    fun provideJson(): Json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
        isLenient = true
        encodeDefaults = true
    }

    /**
     * Provides a singleton AIContentApi instance configured to interact with the AuraFrameFx AI API.
     *
     * @return A singleton AIContentApi for accessing AuraFrameFx AI API endpoints.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient, json: Json): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Supplies a singleton `AuraFxContentApiClient` that encapsulates the provided `AIContentApi`.
     *
     * @return The singleton instance of `AuraFxContentApiClient`.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
