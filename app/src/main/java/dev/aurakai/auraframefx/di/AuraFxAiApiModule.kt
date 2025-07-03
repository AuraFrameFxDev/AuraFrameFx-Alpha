package dev.aurakai.auraframefx.di

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.api.AiContentApi
import dev.aurakai.auraframefx.network.AuraFxContentApiClient
import dev.aurakai.auraframefx.network.NetworkConstants
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
     * Provides a singleton `Json` serializer configured to ignore unknown keys, coerce input values, parse leniently, and encode default values.
     *
     * This configuration supports robust serialization and deserialization for API communication, accommodating flexible or evolving JSON schemas.
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
     * Provides a singleton implementation of the ContentApi interface for accessing the AuraFrameFx AI API via Retrofit.
     *
     * Configures Retrofit with the base URL from NetworkConstants, the specified OkHttp client, and a JSON converter created from the given Json instance.
     *
     * @return An implementation of ContentApi for interacting with the AuraFrameFx AI API.
     */
    @OptIn(ExperimentalSerializationApi::class)
    @Provides
    @Singleton

    fun provideContentApi(okHttpClient: OkHttpClient, json: Json): ContentApi {
 
        val contentType = "application/json".toMediaType()

        return Retrofit.Builder()
            .baseUrl(NetworkConstants.BASE_URL)
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
