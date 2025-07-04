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
     * Provides a singleton OkHttpClient configured to log HTTP request and response bodies.
     *
     * @return An OkHttpClient instance with body-level logging enabled for detailed HTTP traffic inspection.
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
     * Provides a singleton Json serializer configured for resilient and flexible API data handling.
     *
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values to support robust serialization and deserialization of diverse API responses.
     *
     * @return A configured Json instance for processing API data.
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
     * Provides a singleton `AIContentApi` configured with the specified HTTP client for accessing AuraFrameFx AI API endpoints.
     *
     * @param okHttpClient The HTTP client used to perform API requests.
     * @return An `AIContentApi` instance set up for communication with the AuraFrameFx AI API.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Supplies a singleton `AuraFxContentApiClient` configured to interact with the AuraFrameFx AI API using the provided `AIContentApi` interface.
     *
     * @param aiContentApi The API interface for communicating with AuraFrameFx AI endpoints.
     * @return A singleton instance of `AuraFxContentApiClient`.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
