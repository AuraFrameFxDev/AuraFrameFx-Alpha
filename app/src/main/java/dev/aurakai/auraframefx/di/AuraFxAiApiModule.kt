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
     * Provides a singleton OkHttpClient configured with an HTTP logging interceptor for API requests.
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
     * Returns a singleton Json serializer configured for lenient parsing, unknown key ignoring, input value coercion, and default value encoding.
     *
     * @return A Json instance suitable for robust API serialization and deserialization.
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
     * Returns a singleton `AIContentApi` instance configured to interact with the AuraFrameFx AI API.
     *
     * @param okHttpClient The HTTP client used for making API requests.
     * @return An `AIContentApi` targeting the AuraFrameFx API at the specified base URL.
     */
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient): AIContentApi {

        val baseUrl = "https://api.auraframefx.com/v1"
        
        return AIContentApi(basePath = baseUrl, client = okHttpClient)
    }

    /**
     * Returns a singleton `AuraFxContentApiClient` that wraps the provided `AIContentApi`.
     *
     * @param aiContentApi The `AIContentApi` instance to be wrapped.
     * @return A singleton `AuraFxContentApiClient` configured with the given API.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AIContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
