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
     * Returns a singleton OkHttpClient configured to log HTTP request and response bodies.
     *
     * @return An OkHttpClient instance with detailed HTTP body logging enabled.
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
     * Provides a singleton `Json` serializer configured for resilient API data processing.
     *
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values to ensure robust serialization and deserialization of API responses.
     *
     * @return A configured `Json` instance for handling API data.
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
     * Provides a singleton `AIContentApi` instance configured to communicate with the AuraFrameFx AI API.
     *
     * @param okHttpClient The HTTP client used for API requests.
     * @return An `AIContentApi` for accessing AuraFrameFx AI endpoints.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Provides a singleton instance of `AuraFxContentApiClient` for interacting with the AuraFrameFx AI API.
     *
     * @param aiContentApi The API interface used for communication with the AuraFrameFx AI API.
     * @return A singleton `AuraFxContentApiClient` configured with the provided API interface.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
