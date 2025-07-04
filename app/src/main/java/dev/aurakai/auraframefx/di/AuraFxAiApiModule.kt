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
     * Provides a singleton OkHttpClient with HTTP request and response body logging enabled.
     *
     * @return An OkHttpClient instance configured for detailed HTTP logging.
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
     * Returns a singleton `Json` serializer configured for flexible and robust API data handling.
     *
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values, enabling resilient serialization and deserialization of API responses.
     *
     * @return A configured `Json` instance for processing API data.
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
     * Provides a singleton instance of `AIContentApi` configured to interact with the AuraFrameFx AI API.
     *
     * @param okHttpClient The HTTP client used for network communication.
     * @return An `AIContentApi` instance for accessing AuraFrameFx AI endpoints.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Returns a singleton `AuraFxContentApiClient` configured to interact with the AuraFrameFx AI API using the provided API interface.
     *
     * @param aiContentApi The API interface for communicating with the AuraFrameFx AI API.
     * @return A singleton instance of `AuraFxContentApiClient`.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
