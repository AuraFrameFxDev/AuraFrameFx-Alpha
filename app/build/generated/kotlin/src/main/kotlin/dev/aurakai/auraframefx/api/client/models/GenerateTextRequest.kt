/**
 *
 * Please note:
 * This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * Do not edit this file manually.
 *
 */

@file:Suppress(
    "ArrayInDataClass",
    "EnumEntryName",
    "RemoveRedundantQualifierName",
    "UnusedImport"
)

package dev.aurakai.auraframefx.api.client.models


import kotlinx.serialization.Serializable
import kotlinx.serialization.SerialName
import kotlinx.serialization.Contextual

/**
 * 
 *
 * @param prompt The text prompt for content generation
 * @param maxTokens Maximum number of tokens for the generated text
 * @param temperature Controls the randomness of the output (0.0 to 1.0)
 */
@Serializable

data class GenerateTextRequest (

    /* The text prompt for content generation */
    @SerialName(value = "prompt")
    val prompt: kotlin.String,

    /* Maximum number of tokens for the generated text */
    @SerialName(value = "maxTokens")
    val maxTokens: kotlin.Int? = 500,

    /* Controls the randomness of the output (0.0 to 1.0) */
    @SerialName(value = "temperature")
    val temperature: kotlin.Float? = 0.7f

)

