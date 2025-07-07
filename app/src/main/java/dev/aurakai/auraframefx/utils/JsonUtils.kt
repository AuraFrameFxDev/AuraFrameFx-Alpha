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

    /**
     * Serializes an object to a JSON string using the provided serializer.
     *
     * @param obj The object to serialize.
     * @param serializer The serializer for the object's type.
     * @return The JSON string representation of the object, or null if serialization fails.
     */
    fun <T> toJson(obj: T, serializer: kotlinx.serialization.KSerializer<T>): String? {
        return try {
            json.encodeToString(serializer, obj)
        } catch (e: Exception) {
            // Log the exception
            null
        }
    }

    /**
     * Deserializes a JSON string into an object of the specified type using the provided serializer.
     *
     * @param jsonString The JSON string to deserialize.
     * @param serializer The serializer for the target type.
     * @return The deserialized object, or null if deserialization fails.
     */
    fun <T> fromJson(jsonString: String, serializer: kotlinx.serialization.KSerializer<T>): T? {
        return try {
            Json.decodeFromString(serializer, jsonString)
        } catch (e: Exception) {
            // Log the exception
            null
        }
    }
}
