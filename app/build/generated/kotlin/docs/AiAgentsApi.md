# AiAgentsApi

All URIs are relative to *https://api.auraframefx.com/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**processAgentRequest**](AiAgentsApi.md#processAgentRequest) | **POST** /agent/{agentType}/process-request | Send a request to an AI agent


<a id="processAgentRequest"></a>
# **processAgentRequest**
> AgentMessage processAgentRequest(agentType, agentProcessRequest)

Send a request to an AI agent

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import dev.aurakai.auraframefx.api.client.models.*

val apiInstance = AiAgentsApi()
val agentType : AgentType =  // AgentType | Type of AI agent to interact with
val agentProcessRequest : AgentProcessRequest =  // AgentProcessRequest | 
try {
    val result : AgentMessage = apiInstance.processAgentRequest(agentType, agentProcessRequest)
    println(result)
} catch (e: ClientException) {
    println("4xx response calling AiAgentsApi#processAgentRequest")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling AiAgentsApi#processAgentRequest")
    e.printStackTrace()
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **agentType** | [**AgentType**](.md)| Type of AI agent to interact with | [enum: Aura, Kai, Genesis, Cascade, NeuralWhisper, AuraShield, GenKitMaster]
 **agentProcessRequest** | [**AgentProcessRequest**](AgentProcessRequest.md)|  |

### Return type

[**AgentMessage**](AgentMessage.md)

### Authorization


Configure OAuth2AuthCode:
    ApiClient.accessToken = ""

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

