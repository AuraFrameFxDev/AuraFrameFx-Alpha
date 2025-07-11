
# TaskStatus

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**taskId** | **kotlin.String** | Unique identifier for the task | 
**status** | [**inline**](#Status) |  | 
**progress** | **kotlin.Int** | Percentage completion of the task (0-100) |  [optional]
**result** | [**kotlin.collections.Map&lt;kotlin.String, kotlin.Any&gt;**](kotlin.Any.md) | The outcome or output of the task |  [optional]
**errorMessage** | **kotlin.String** | Error message if the task failed |  [optional]


<a id="Status"></a>
## Enum: status
Name | Value
---- | -----
status | PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED



