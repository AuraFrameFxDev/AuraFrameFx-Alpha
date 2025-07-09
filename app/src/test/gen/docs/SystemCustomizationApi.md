# SystemCustomizationApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**systemLockscreenConfigGet**](SystemCustomizationApi.md#systemLockscreenConfigGet) | **GET** /system/lockscreen-config | Get lock screen configuration |
| [**systemLockscreenConfigPut**](SystemCustomizationApi.md#systemLockscreenConfigPut) | **PUT** /system/lockscreen-config | Update lock screen configuration |


<a id="systemLockscreenConfigGet"></a>
# **systemLockscreenConfigGet**
> LockScreenConfig systemLockscreenConfigGet()

Get lock screen configuration

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.SystemCustomizationApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    SystemCustomizationApi apiInstance = new SystemCustomizationApi(defaultClient);
    try {
      LockScreenConfig result = apiInstance.systemLockscreenConfigGet();
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling SystemCustomizationApi#systemLockscreenConfigGet");
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

[**LockScreenConfig**](LockScreenConfig.md)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Lock screen configuration retrieved successfully |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |

<a id="systemLockscreenConfigPut"></a>
# **systemLockscreenConfigPut**
> systemLockscreenConfigPut(lockScreenConfig)

Update lock screen configuration

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.SystemCustomizationApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    SystemCustomizationApi apiInstance = new SystemCustomizationApi(defaultClient);
    LockScreenConfig lockScreenConfig = new LockScreenConfig(); // LockScreenConfig | 
    try {
      apiInstance.systemLockscreenConfigPut(lockScreenConfig);
    } catch (ApiException e) {
      System.err.println("Exception when calling SystemCustomizationApi#systemLockscreenConfigPut");
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
| **lockScreenConfig** | [**LockScreenConfig**](LockScreenConfig.md)|  | |

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
| **200** | Lock screen configuration updated successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |

