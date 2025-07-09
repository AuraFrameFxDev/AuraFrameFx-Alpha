# AdlApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**adlRelayPost**](AdlApi.md#adlRelayPost) | **POST** /adl/relay | Relay a request to the Android Device Layer (ADL) |


<a id="adlRelayPost"></a>
# **adlRelayPost**
> ADLRelayResponse adlRelayPost(adLRelayRequest)

Relay a request to the Android Device Layer (ADL)

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.AdlApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure API key authorization: ApiKeyAuth
    ApiKeyAuth ApiKeyAuth = (ApiKeyAuth) defaultClient.getAuthentication("ApiKeyAuth");
    ApiKeyAuth.setApiKey("YOUR API KEY");
    // Uncomment the following line to set a prefix for the API key, e.g. "Token" (defaults to null)
    //ApiKeyAuth.setApiKeyPrefix("Token");

    AdlApi apiInstance = new AdlApi(defaultClient);
    ADLRelayRequest adLRelayRequest = new ADLRelayRequest(); // ADLRelayRequest | 
    try {
      ADLRelayResponse result = apiInstance.adlRelayPost(adLRelayRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling AdlApi#adlRelayPost");
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
| **adLRelayRequest** | [**ADLRelayRequest**](ADLRelayRequest.md)|  | |

### Return type

[**ADLRelayResponse**](ADLRelayResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | ADL relay successful |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |
| **500** | Internal server error |  -  |

