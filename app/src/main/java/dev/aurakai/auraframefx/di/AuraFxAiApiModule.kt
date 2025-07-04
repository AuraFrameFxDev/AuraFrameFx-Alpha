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
     * @return An OkHttpClient instance configured to log HTTP bodies at the BODY level.
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
     * Provides a singleton `Json` serializer configured to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values.
     *
     * @return A configured `Json` instance for API serialization and deserialization.
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
     * Provides a singleton `AIContentApi` configured to communicate with the AuraFrameFx AI API.
     *
     * The returned instance uses the base URL "https://api.auraframefx.com/v1" and the supplied `OkHttpClient`.
     *
     * @return A configured `AIContentApi` instance.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient, json: Json): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Provides a singleton instance of `AuraFxContentApiClient` that wraps the given `AIContentApi`.
     *
     * @param aiContentApi The AI content API instance to wrap.
     * @return A singleton `AuraFxContentApiClient` configured with the specified API.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
