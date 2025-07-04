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
     * Returns a singleton OkHttpClient configured with an HTTP logging interceptor that logs request and response bodies.
     *
     * @return The configured OkHttpClient instance.
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
     * Returns a singleton `Json` serializer configured for lenient parsing, ignoring unknown keys, coercing input values, and encoding default values.
     *
     * @return A `Json` instance suitable for robust API serialization and deserialization.
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
     * Returns a singleton instance of `AIContentApi` configured to communicate with the AuraFrameFx AI API at the specified base URL.
     *
     * @param okHttpClient The HTTP client used for API requests.
     * @return A singleton `AIContentApi` for accessing AuraFrameFx AI endpoints.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Returns a singleton instance of `AuraFxContentApiClient` that wraps the specified `AIContentApi`.
     *
     * @param aiContentApi The API interface to be used by the client.
     * @return A singleton `AuraFxContentApiClient` configured to interact with the provided API.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
