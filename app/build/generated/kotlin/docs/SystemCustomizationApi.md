# SystemCustomizationApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**getSystemLockscreenConfig**](SystemCustomizationApi.md#getSystemLockscreenConfig) | **GET** /system/lockscreen-config | Get lock screen configuration
[**updateSystemLockscreenConfig**](SystemCustomizationApi.md#updateSystemLockscreenConfig) | **PUT** /system/lockscreen-config | Update lock screen configuration


<a id="getSystemLockscreenConfig"></a>
# **getSystemLockscreenConfig**
> LockScreenConfig getSystemLockscreenConfig()

Get lock screen configuration

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = SystemCustomizationApi()
try {
    val result : LockScreenConfig = apiInstance.getSystemLockscreenConfig()
    println(result)
} catch (e: ClientException) {
    println("4xx response calling SystemCustomizationApi#getSystemLockscreenConfig")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling SystemCustomizationApi#getSystemLockscreenConfig")
    e.printStackTrace()
}
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**LockScreenConfig**](LockScreenConfig.md)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

<a id="updateSystemLockscreenConfig"></a>
# **updateSystemLockscreenConfig**
> updateSystemLockscreenConfig(lockScreenConfig)

Update lock screen configuration

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = SystemCustomizationApi()
val lockScreenConfig : LockScreenConfig =  // LockScreenConfig | 
try {
    apiInstance.updateSystemLockscreenConfig(lockScreenConfig)
} catch (e: ClientException) {
    println("4xx response calling SystemCustomizationApi#updateSystemLockscreenConfig")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling SystemCustomizationApi#updateSystemLockscreenConfig")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **lockScreenConfig** | [**LockScreenConfig**](LockScreenConfig.md)|  |

### Return type

null (empty response body)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

