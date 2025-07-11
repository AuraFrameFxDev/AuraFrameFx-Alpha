# ConferenceRoomApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**createConferenceRoom**](ConferenceRoomApi.md#createConferenceRoom) | **POST** /conference/create | Create a new AI conference room


<a id="createConferenceRoom"></a>
# **createConferenceRoom**
> ConferenceRoom createConferenceRoom(conferenceRoomCreateRequest)

Create a new AI conference room

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = ConferenceRoomApi()
val conferenceRoomCreateRequest : ConferenceRoomCreateRequest =  // ConferenceRoomCreateRequest | 
try {
    val result : ConferenceRoom = apiInstance.createConferenceRoom(conferenceRoomCreateRequest)
    println(result)
} catch (e: ClientException) {
    println("4xx response calling ConferenceRoomApi#createConferenceRoom")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling ConferenceRoomApi#createConferenceRoom")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **conferenceRoomCreateRequest** | [**ConferenceRoomCreateRequest**](ConferenceRoomCreateRequest.md)|  |

### Return type

[**ConferenceRoom**](ConferenceRoom.md)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

