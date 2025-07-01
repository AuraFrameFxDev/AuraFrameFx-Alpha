package dev.aurakai.auraframefx.di

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dev.aurakai.auraframefx.generated.api.auraframefxai.ContentApi
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
     * Provides the JSON serializer configured for the API.
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
     * Provides the ContentApi interface implementation for accessing the AuraFrameFx AI API.
     */
    @OptIn(ExperimentalSerializationApi::class)
    @Provides
    @Singleton
    fun provideContentApi(okHttpClient: OkHttpClient, json: Json): ContentApi {
        val baseUrl = "https://api.auraframefx.com/v1/"
        val contentType = "application/json".toMediaType()

        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()
            .create(ContentApi::class.java)
    }

    /**
     * Provides the AuraFxContentApiClient wrapper for the ContentApi.
     */
    @Provides
    @Singleton
    fun provideAuraFxContentApiClient(contentApi: ContentApi): AuraFxContentApiClient {
        return AuraFxContentApiClient(contentApi)
    }
}
