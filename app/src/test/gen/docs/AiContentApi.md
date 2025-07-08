# AiContentApi

All URIs are relative to *https://api.auraframefx.com/v1*

<<<<<<< HEAD
| Method                                                                               | HTTP request                            | Description                         |
|--------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------|
| [**aiGenerateImageDescriptionPost**](AiContentApi.md#aiGenerateImageDescriptionPost) | **POST** /ai/generate/image-description | Generate image description using AI |
| [**generateTextPost**](AiContentApi.md#generateTextPost)                             | **POST** /generate-text                 | Generate text content               |

<a id="aiGenerateImageDescriptionPost"></a>

# **aiGenerateImageDescriptionPost**

=======
| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**aiGenerateImageDescriptionPost**](AiContentApi.md#aiGenerateImageDescriptionPost) | **POST** /ai/generate/image-description | Generate image description using AI |
| [**generateTextPost**](AiContentApi.md#generateTextPost) | **POST** /generate-text | Generate text content |


<a id="aiGenerateImageDescriptionPost"></a>
# **aiGenerateImageDescriptionPost**
>>>>>>> origin/coderabbitai/docstrings/78f34ad
> GenerateImageDescriptionResponse aiGenerateImageDescriptionPost(generateImageDescriptionRequest)

Generate image description using AI

Generate a description for the provided image URL

### Example
<<<<<<< HEAD

=======
>>>>>>> origin/coderabbitai/docstrings/78f34ad
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.AiContentApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");

    AiContentApi apiInstance = new AiContentApi(defaultClient);
    GenerateImageDescriptionRequest generateImageDescriptionRequest = new GenerateImageDescriptionRequest(); // GenerateImageDescriptionRequest | 
    try {
      GenerateImageDescriptionResponse result = apiInstance.aiGenerateImageDescriptionPost(generateImageDescriptionRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling AiContentApi#aiGenerateImageDescriptionPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

<<<<<<< HEAD
| Name                                | Type                                                                      | Description | Notes |
|-------------------------------------|---------------------------------------------------------------------------|-------------|-------|
| **generateImageDescriptionRequest** | [**GenerateImageDescriptionRequest**](GenerateImageDescriptionRequest.md) |             |       |
=======
| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **generateImageDescriptionRequest** | [**GenerateImageDescriptionRequest**](GenerateImageDescriptionRequest.md)|  | |
>>>>>>> origin/coderabbitai/docstrings/78f34ad

### Return type

[**GenerateImageDescriptionResponse**](GenerateImageDescriptionResponse.md)

### Authorization

No authorization required

### HTTP request headers

<<<<<<< HEAD
- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description                              | Response headers |
|-------------|------------------------------------------|------------------|
| **200**     | Image description generated successfully | -                |
| **400**     | Invalid request parameters               | -                |
| **500**     | Internal server error                    | -                |

<a id="generateTextPost"></a>

# **generateTextPost**

=======
 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Image description generated successfully |  -  |
| **400** | Invalid request parameters |  -  |
| **500** | Internal server error |  -  |

<a id="generateTextPost"></a>
# **generateTextPost**
>>>>>>> origin/coderabbitai/docstrings/78f34ad
> GenerateTextResponse generateTextPost(generateTextRequest)

Generate text content

### Example
<<<<<<< HEAD

=======
>>>>>>> origin/coderabbitai/docstrings/78f34ad
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.AiContentApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    AiContentApi apiInstance = new AiContentApi(defaultClient);
    GenerateTextRequest generateTextRequest = new GenerateTextRequest(); // GenerateTextRequest | 
    try {
      GenerateTextResponse result = apiInstance.generateTextPost(generateTextRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling AiContentApi#generateTextPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

<<<<<<< HEAD
| Name                    | Type                                              | Description | Notes |
|-------------------------|---------------------------------------------------|-------------|-------|
| **generateTextRequest** | [**GenerateTextRequest**](GenerateTextRequest.md) |             |       |
=======
| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **generateTextRequest** | [**GenerateTextRequest**](GenerateTextRequest.md)|  | |
>>>>>>> origin/coderabbitai/docstrings/78f34ad

### Return type

[**GenerateTextResponse**](GenerateTextResponse.md)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

<<<<<<< HEAD
- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description                                          | Response headers |
|-------------|------------------------------------------------------|------------------|
| **200**     | Text generated successfully                          | -                |
| **400**     | Invalid request format or parameters                 | -                |
| **401**     | Authentication credentials were missing or incorrect | -                |
| **429**     | Rate limit exceeded                                  | -                |
=======
 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Text generated successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |
| **429** | Rate limit exceeded |  -  |
>>>>>>> origin/coderabbitai/docstrings/78f34ad

