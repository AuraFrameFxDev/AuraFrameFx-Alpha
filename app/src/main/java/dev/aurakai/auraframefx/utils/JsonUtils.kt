package dev.aurakai.auraframefx.utils

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

object JsonUtils {
    internal val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        prettyPrint = true
    }

    inline fun <reified T> toJson(obj: T): String? {
        return try {
            json.encodeToString(obj)
        } catch (e: Exception) {
            // Log the exception
            null
        }
    }

    inline fun <reified T> fromJson(jsonString: String): T? {
        return try {
            json.decodeFromString<T>(jsonString)
        } catch (e: Exception) {
            // Log the exception
            null
        }
    }
}
