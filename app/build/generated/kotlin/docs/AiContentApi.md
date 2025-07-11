# AiContentApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**generateText**](AiContentApi.md#generateText) | **POST** /generate-text | Generate text content


<a id="generateText"></a>
# **generateText**
> GenerateTextResponse generateText(generateTextRequest)

Generate text content

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = AiContentApi()
val generateTextRequest : GenerateTextRequest =  // GenerateTextRequest | 
try {
    val result : GenerateTextResponse = apiInstance.generateText(generateTextRequest)
    println(result)
} catch (e: ClientException) {
    println("4xx response calling AiContentApi#generateText")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling AiContentApi#generateText")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **generateTextRequest** | [**GenerateTextRequest**](GenerateTextRequest.md)|  |

### Return type

[**GenerateTextResponse**](GenerateTextResponse.md)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

