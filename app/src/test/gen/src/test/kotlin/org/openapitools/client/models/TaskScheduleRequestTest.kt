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

package org.openapitools.client.models

import io.kotlintest.shouldBe
import io.kotlintest.specs.ShouldSpec

import org.openapitools.client.models.TaskScheduleRequest
import org.openapitools.client.models.AgentType

class TaskScheduleRequestTest : ShouldSpec() {
    init {
        // uncomment below to create an instance of TaskScheduleRequest
        //val modelInstance = TaskScheduleRequest()

        // to test the property `taskType` - The type of task
        should("test taskType") {
            // uncomment below to test the property
            //modelInstance.taskType shouldBe ("TODO")
        }

        // to test the property `agentType`
        should("test agentType") {
            // uncomment below to test the property
            //modelInstance.agentType shouldBe ("TODO")
        }

        // to test the property `details` - Specific parameters for the task
        should("test details") {
            // uncomment below to test the property
            //modelInstance.details shouldBe ("TODO")
        }

        // to test the property `priority`
        should("test priority") {
            // uncomment below to test the property
            //modelInstance.priority shouldBe ("TODO")
        }

    }
}
