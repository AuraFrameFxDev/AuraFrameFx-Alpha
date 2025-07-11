# TasksApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**scheduleTask**](TasksApi.md#scheduleTask) | **POST** /tasks/schedule | Schedule a new task


<a id="scheduleTask"></a>
# **scheduleTask**
> TaskStatus scheduleTask(taskScheduleRequest)

Schedule a new task

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = TasksApi()
val taskScheduleRequest : TaskScheduleRequest =  // TaskScheduleRequest | 
try {
    val result : TaskStatus = apiInstance.scheduleTask(taskScheduleRequest)
    println(result)
} catch (e: ClientException) {
    println("4xx response calling TasksApi#scheduleTask")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling TasksApi#scheduleTask")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **taskScheduleRequest** | [**TaskScheduleRequest**](TaskScheduleRequest.md)|  |

### Return type

[**TaskStatus**](TaskStatus.md)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

