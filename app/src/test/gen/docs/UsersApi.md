# UsersApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**userGet**](UsersApi.md#userGet) | **GET** /user | Get current user information |
| [**userPreferencesPut**](UsersApi.md#userPreferencesPut) | **PUT** /user/preferences | Update user preferences |


<a id="userGet"></a>
# **userGet**
> User userGet()

Get current user information

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.UsersApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    UsersApi apiInstance = new UsersApi(defaultClient);
    try {
      User result = apiInstance.userGet();
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling UsersApi#userGet");
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

[**User**](User.md)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | User information retrieved successfully |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |

<a id="userPreferencesPut"></a>
# **userPreferencesPut**
> userPreferencesPut(userPreferencesUpdate)

Update user preferences

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.UsersApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    UsersApi apiInstance = new UsersApi(defaultClient);
    UserPreferencesUpdate userPreferencesUpdate = new UserPreferencesUpdate(); // UserPreferencesUpdate | 
    try {
      apiInstance.userPreferencesPut(userPreferencesUpdate);
    } catch (ApiException e) {
      System.err.println("Exception when calling UsersApi#userPreferencesPut");
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
| **userPreferencesUpdate** | [**UserPreferencesUpdate**](UserPreferencesUpdate.md)|  | |

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
| **200** | Preferences updated successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |

