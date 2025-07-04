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
     * Supplies a singleton OkHttpClient configured with a logging interceptor at the BODY level.
     *
     * @return An OkHttpClient instance that logs full HTTP request and response bodies for debugging and inspection.
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
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values to support flexible serialization and deserialization of API responses.
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
     * Provides a singleton AIContentApi instance configured with the given OkHttpClient for communicating with the AuraFrameFx AI API.
     *
     * @param okHttpClient The HTTP client used for API requests.
     * @return An AIContentApi instance for accessing AuraFrameFx AI API endpoints.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Supplies a singleton AuraFxContentApiClient for accessing AuraFrameFx AI API features.
     *
     * @param aiContentApi The API interface used to communicate with AuraFrameFx AI endpoints.
     * @return A singleton AuraFxContentApiClient instance.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
