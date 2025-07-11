# UsersApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**getUser**](UsersApi.md#getUser) | **GET** /user | Get current user information
[**updateUserPreferences**](UsersApi.md#updateUserPreferences) | **PUT** /user/preferences | Update user preferences


<a id="getUser"></a>
# **getUser**
> User getUser()

Get current user information

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = UsersApi()
try {
    val result : User = apiInstance.getUser()
    println(result)
} catch (e: ClientException) {
    println("4xx response calling UsersApi#getUser")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling UsersApi#getUser")
    e.printStackTrace()
}
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**User**](User.md)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

<a id="updateUserPreferences"></a>
# **updateUserPreferences**
> updateUserPreferences(userPreferencesUpdate)

Update user preferences

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = UsersApi()
val userPreferencesUpdate : UserPreferencesUpdate =  // UserPreferencesUpdate | 
try {
    apiInstance.updateUserPreferences(userPreferencesUpdate)
} catch (e: ClientException) {
    println("4xx response calling UsersApi#updateUserPreferences")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling UsersApi#updateUserPreferences")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **userPreferencesUpdate** | [**UserPreferencesUpdate**](UserPreferencesUpdate.md)|  |

### Return type

null (empty response body)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

