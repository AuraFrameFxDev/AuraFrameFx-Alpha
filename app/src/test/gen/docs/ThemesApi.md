# ThemesApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**themeApplyPut**](ThemesApi.md#themeApplyPut) | **PUT** /theme/apply | Apply a theme |
| [**themesGet**](ThemesApi.md#themesGet) | **GET** /themes | Get available themes |


<a id="themeApplyPut"></a>
# **themeApplyPut**
> themeApplyPut(themeApplyRequest)

Apply a theme

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.ThemesApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    ThemesApi apiInstance = new ThemesApi(defaultClient);
    ThemeApplyRequest themeApplyRequest = new ThemeApplyRequest(); // ThemeApplyRequest | 
    try {
      apiInstance.themeApplyPut(themeApplyRequest);
    } catch (ApiException e) {
      System.err.println("Exception when calling ThemesApi#themeApplyPut");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **themeApplyRequest** | [**ThemeApplyRequest**](ThemeApplyRequest.md)|  | |

### Return type

null (empty response body)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Theme applied successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |
| **404** | Theme not found |  -  |

<a id="themesGet"></a>
# **themesGet**
> List&lt;Theme&gt; themesGet()

Get available themes

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.ThemesApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");

    ThemesApi apiInstance = new ThemesApi(defaultClient);
    try {
      List<Theme> result = apiInstance.themesGet();
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling ThemesApi#themesGet");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**List&lt;Theme&gt;**](Theme.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | List of available themes |  -  |

