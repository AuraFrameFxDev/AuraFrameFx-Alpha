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
     * Creates a singleton OkHttpClient with HTTP request and response body logging enabled.
     *
     * @return An OkHttpClient instance configured for detailed network logging.
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
     * Provides a singleton Json serializer configured for resilient API data handling.
     *
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values to ensure robust serialization and deserialization of API responses.
     *
     * @return A configured Json instance for flexible API data processing.
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
     * Creates a singleton AIContentApi instance configured for the AuraFrameFx AI API.
     *
     * @param okHttpClient The OkHttpClient used for HTTP communication with the API.
     * @return An AIContentApi instance set up to interact with the AuraFrameFx AI API.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"

        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Returns a singleton instance of AuraFxContentApiClient configured with the provided AIContentApi.
     *
     * @param aiContentApi The API interface used for communication with AuraFrameFx AI endpoints.
     * @return A singleton AuraFxContentApiClient for accessing AuraFrameFx AI API features.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
