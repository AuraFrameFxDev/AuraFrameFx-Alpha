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

import io.kotlintest.shouldBe
import io.kotlintest.specs.ShouldSpec

import dev.aurakai.auraframefx.api.client.models.AgentProcessRequest

class AgentProcessRequestTest : ShouldSpec() {
    init {
        // uncomment below to create an instance of AgentProcessRequest
        //val modelInstance = AgentProcessRequest()

        // to test the property `prompt` - The prompt/instruction for the AI agent
        should("test prompt") {
            // uncomment below to test the property
            //modelInstance.prompt shouldBe ("TODO")
        }

        // to test the property `context` - Additional context (e.g., previous messages, data references)
        should("test context") {
            // uncomment below to test the property
            //modelInstance.context shouldBe ("TODO")
        }

    }
}
