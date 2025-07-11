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
 * @param id 
 * @param name 
 * @param primaryColor 
 * @param secondaryColor 
 * @param isDefault 
 */
@Serializable

data class Theme (

    @SerialName(value = "id")
    val id: kotlin.String,

    @SerialName(value = "name")
    val name: kotlin.String,

    @SerialName(value = "primaryColor")
    val primaryColor: kotlin.String,

    @SerialName(value = "secondaryColor")
    val secondaryColor: kotlin.String,

    @SerialName(value = "isDefault")
    val isDefault: kotlin.Boolean

)

