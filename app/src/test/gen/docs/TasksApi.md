# TasksApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**tasksSchedulePost**](TasksApi.md#tasksSchedulePost) | **POST** /tasks/schedule | Schedule a new task |


<a id="tasksSchedulePost"></a>
# **tasksSchedulePost**
> TaskStatus tasksSchedulePost(taskScheduleRequest)

Schedule a new task

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.TasksApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    TasksApi apiInstance = new TasksApi(defaultClient);
    TaskScheduleRequest taskScheduleRequest = new TaskScheduleRequest(); // TaskScheduleRequest | 
    try {
      TaskStatus result = apiInstance.tasksSchedulePost(taskScheduleRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling TasksApi#tasksSchedulePost");
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
| **taskScheduleRequest** | [**TaskScheduleRequest**](TaskScheduleRequest.md)|  | |

### Return type

[**TaskStatus**](TaskStatus.md)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **202** | Task scheduled successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |
| **429** | Rate limit exceeded |  -  |

