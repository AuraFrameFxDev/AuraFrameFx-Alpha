

# TaskStatus


## Properties

| Name | Type | Description | Notes |
|------------ | ------------- | ------------- | -------------|
|**taskId** | **String** | Unique identifier for the task |  |
|**status** | [**StatusEnum**](#StatusEnum) |  |  |
|**progress** | **Integer** | Percentage completion of the task (0-100) |  [optional] |
|**result** | **Map&lt;String, Object&gt;** | The outcome or output of the task |  [optional] |
|**errorMessage** | **String** | Error message if the task failed |  [optional] |



## Enum: StatusEnum

| Name | Value |
|---- | -----|
| PENDING | &quot;PENDING&quot; |
| IN_PROGRESS | &quot;IN_PROGRESS&quot; |
| COMPLETED | &quot;COMPLETED&quot; |
| FAILED | &quot;FAILED&quot; |
| CANCELLED | &quot;CANCELLED&quot; |



