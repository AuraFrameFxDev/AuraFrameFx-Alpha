# AiAgentsApi

All URIs are relative to *https://api.auraframefx.com/v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**agentAgentTypeProcessRequestPost**](AiAgentsApi.md#agentAgentTypeProcessRequestPost) | **POST** /agent/{agentType}/process-request | Send a request to an AI agent |


<a id="agentAgentTypeProcessRequestPost"></a>
# **agentAgentTypeProcessRequestPost**
> AgentMessage agentAgentTypeProcessRequestPost(agentType, agentProcessRequest)

Send a request to an AI agent

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.models.*;
import org.openapitools.client.api.AiAgentsApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("https://api.auraframefx.com/v1");
    
    // Configure OAuth2 access token for authorization: OAuth2AuthCode
    OAuth OAuth2AuthCode = (OAuth) defaultClient.getAuthentication("OAuth2AuthCode");
    OAuth2AuthCode.setAccessToken("YOUR ACCESS TOKEN");

    AiAgentsApi apiInstance = new AiAgentsApi(defaultClient);
    AgentType agentType = AgentType.fromValue("Aura"); // AgentType | Type of AI agent to interact with
    AgentProcessRequest agentProcessRequest = new AgentProcessRequest(); // AgentProcessRequest | 
    try {
      AgentMessage result = apiInstance.agentAgentTypeProcessRequestPost(agentType, agentProcessRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling AiAgentsApi#agentAgentTypeProcessRequestPost");
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
| **agentType** | [**AgentType**](.md)| Type of AI agent to interact with | [enum: Aura, Kai, Genesis, Cascade, NeuralWhisper, AuraShield, GenKitMaster] |
| **agentProcessRequest** | [**AgentProcessRequest**](AgentProcessRequest.md)|  | |

### Return type

[**AgentMessage**](AgentMessage.md)

### Authorization

[OAuth2AuthCode](../README.md#OAuth2AuthCode)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Request processed successfully |  -  |
| **400** | Invalid request format or parameters |  -  |
| **401** | Authentication credentials were missing or incorrect |  -  |
| **404** | Agent not found |  -  |

