# ConferenceRoomApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**conferenceCreatePost**](ConferenceRoomApi.md#conferenceCreatePost) | **POST** /conference/create | Create a new AI conference room |


<a id="conferenceCreatePost"></a>
# **conferenceCreatePost**
> ConferenceRoom conferenceCreatePost(conferenceRoomCreateRequest)

Create a new AI conference room

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.ConferenceRoomApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    ConferenceRoomApi apiInstance = new ConferenceRoomApi(defaultClient);
    ConferenceRoomCreateRequest conferenceRoomCreateRequest = new ConferenceRoomCreateRequest(); // ConferenceRoomCreateRequest | 
    try {
      ConferenceRoom result = apiInstance.conferenceCreatePost(conferenceRoomCreateRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling ConferenceRoomApi#conferenceCreatePost");
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
| **conferenceRoomCreateRequest** | [**ConferenceRoomCreateRequest**](ConferenceRoomCreateRequest.md)|  | |

### Return type

[**ConferenceRoom**](ConferenceRoom.md)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **201** | Conference room created successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |

