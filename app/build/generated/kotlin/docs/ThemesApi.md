# ThemesApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**applyTheme**](ThemesApi.md#applyTheme) | **PUT** /theme/apply | Apply a theme
[**getThemes**](ThemesApi.md#getThemes) | **GET** /themes | Get available themes


<a id="applyTheme"></a>
# **applyTheme**
> applyTheme(themeApplyRequest)

Apply a theme

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = ThemesApi()
val themeApplyRequest : ThemeApplyRequest =  // ThemeApplyRequest | 
try {
    apiInstance.applyTheme(themeApplyRequest)
} catch (e: ClientException) {
    println("4xx response calling ThemesApi#applyTheme")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling ThemesApi#applyTheme")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **themeApplyRequest** | [**ThemeApplyRequest**](ThemeApplyRequest.md)|  |

### Return type

null (empty response body)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

<a id="getThemes"></a>
# **getThemes**
> kotlin.collections.List&lt;Theme&gt; getThemes()

Get available themes

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = ThemesApi()
try {
    val result : kotlin.collections.List<Theme> = apiInstance.getThemes()
    println(result)
} catch (e: ClientException) {
    println("4xx response calling ThemesApi#getThemes")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling ThemesApi#getThemes")
    e.printStackTrace()
}
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**kotlin.collections.List&lt;Theme&gt;**](Theme.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

