package dev.aurakai.auraframefx.di

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.api.AiContentApi
import dev.aurakai.auraframefx.network.AuraFxContentApiClient
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import javax.inject.Singleton

/**
 * Dagger Hilt module for providing dependencies related to the AuraFrameFx AI API.
 */
@Module
@InstallIn(SingletonComponent::class)
object AuraFxAiApiModule {

    /**
     * Returns a configured instance of the Kotlinx Serialization `Json` serializer for API communication.
     *
     * The serializer is set to ignore unknown keys, coerce input values, allow lenient parsing, and encode default values.
     * @return The configured `Json` serializer instance.
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
     * Provides a singleton Retrofit-based implementation of the AiContentApi interface for accessing the AuraFrameFx AI API.
     *
     * @param okHttpClient The OkHttp client used for network requests.
     * @param json The configured Kotlinx Serialization Json instance for serialization and deserialization.
     * @return An implementation of AiContentApi backed by Retrofit.
     */
    @OptIn(ExperimentalSerializationApi::class)
    @Provides
    @Singleton
    fun provideAiContentApi(okHttpClient: OkHttpClient, json: Json): AiContentApi {
        val baseUrl = "https://api.auraframefx.com/v1/"
        val contentType = "application/json".toMediaType()

        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()
            .create(AiContentApi::class.java)
    }

    /**
     * Provides the AuraFxContentApiClient wrapper for the AiContentApi.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(aiContentApi: AiContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(aiContentApi)
    }
}
