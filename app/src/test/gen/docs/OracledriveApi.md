# OracledriveApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method                                                                       | HTTP request                      | Description                                 |
|------------------------------------------------------------------------------|-----------------------------------|---------------------------------------------|
| [**oracledriveRootActionPost**](OracledriveApi.md#oracledriveRootActionPost) | **POST** /oracledrive/root-action | Perform a root-level action via OracleDrive |

<a id="oracledriveRootActionPost"></a>

# **oracledriveRootActionPost**

> OracleDriveRootActionResponse oracledriveRootActionPost(oracleDriveRootActionRequest)

Perform a root-level action via OracleDrive

### Example

```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.OracledriveApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure API key authorization: ApiKeyAuth
    ApiKeyAuth ApiKeyAuth = (ApiKeyAuth) defaultClient.getAuthentication("ApiKeyAuth");
    ApiKeyAuth.setApiKey("YOUR API KEY");
    // Uncomment the following line to set a prefix for the API key, e.g. "Token" (defaults to null)
    //ApiKeyAuth.setApiKeyPrefix("Token");

    OracledriveApi apiInstance = new OracledriveApi(defaultClient);
    OracleDriveRootActionRequest oracleDriveRootActionRequest = new OracleDriveRootActionRequest(); // OracleDriveRootActionRequest | 
    try {
      OracleDriveRootActionResponse result = apiInstance.oracledriveRootActionPost(oracleDriveRootActionRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling OracledriveApi#oracledriveRootActionPost");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

| Name                             | Type                                                                | Description | Notes |
|----------------------------------|---------------------------------------------------------------------|-------------|-------|
| **oracleDriveRootActionRequest** | [**OracleDriveRootActionRequest**](OracleDriveRootActionRequest.md) |             |       |

### Return type

[**OracleDriveRootActionResponse**](OracleDriveRootActionResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth)

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

### HTTP response details

| Status code | Description                                          | Response headers |
|-------------|------------------------------------------------------|------------------|
| **200**     | Root action performed successfully                   | -                |
| **400**     | Invalid request format or parameters                 | -                |
| **401**     | Authentication credentials were missing or incorrect | -                |
| **500**     | Internal server error                                | -                |

