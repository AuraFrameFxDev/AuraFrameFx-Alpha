package dev.aurakai.auraframefx.ai.task.execution

import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.serialization.InstantSerializer
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable
// Removed import kotlinx.datetime.Clock.System as System is now java.lang.System
import java.lang.System // Added import for java.lang.System

@Serializable
data class TaskExecution(
    val id: String = "exec_${System.currentTimeMillis()}",
    val taskId: String,
    val agent: AgentType,
    val type: String,
    val data: Map<String, String> = emptyMap(),
    val priority: TaskPriority = TaskPriority.NORMAL,
    @Serializable(with = InstantSerializer::class) val startTime: Instant = Clock.System.now(),
    @Serializable(with = InstantSerializer::class) val endTime: Instant? = null,
    val startedAt: Long? = null,
    val completedAt: Long? = null,
    val errorMessage: String? = null,
    val status: ExecutionStatus = ExecutionStatus.PENDING,
    val progress: Float = 0.0f,
    val result: ExecutionResult? = null,
    val metadata: Map<String, String> = emptyMap(),
    val executionPlan: ExecutionPlan? = null,
    val checkpoints: List<Checkpoint> = emptyList(),
    val scheduledTime: Long = System.currentTimeMillis(),
    val agentPreference: String? = null,
    val createdAt: Long = System.currentTimeMillis()
)

@Serializable
data class ExecutionPlan(
    val id: String = "plan_${System.currentTimeMillis()}",
    val steps: List<ExecutionStep>,
    val estimatedDuration: Long,
    val requiredResources: Set<String>,
    val metadata: Map<String, String> = emptyMap(),
)

@Serializable
data class ExecutionStep(
    val id: String = "step_${System.currentTimeMillis()}",
    val description: String,
    val type: StepType,
    val priority: Float = 0.5f,
    val estimatedDuration: Long = 0,
    val dependencies: Set<String> = emptySet(),
    val metadata: Map<String, String> = emptyMap(),
)

@Serializable
data class Checkpoint(
    val id: String = "chk_${System.currentTimeMillis()}",
    @Serializable(with = InstantSerializer::class) val timestamp: Instant = Clock.System.now(),
    val stepId: String,
    val status: CheckpointStatus,
    val progress: Float = 0.0f,
    val metadata: Map<String, String> = emptyMap(),
)

@Serializable
enum class ExecutionStatus {
    PENDING,
    INITIALIZING,
    RUNNING,
    PAUSED,
    COMPLETED,
    FAILED,
    CANCELLED,
    TIMEOUT
}

@Serializable
enum class ExecutionResult {
    SUCCESS,
    PARTIAL_SUCCESS,
    FAILURE,
    CANCELLED,
    TIMEOUT,
    UNKNOWN
}

@Serializable
enum class StepType {
    COMPUTATION,
    COMMUNICATION,
    MEMORY,
    CONTEXT,
    DECISION,
    ACTION,
    MONITORING,
    REPORTING
}

@Serializable
enum class CheckpointStatus {
    PENDING,
    STARTED,
    COMPLETED,
    FAILED,
    SKIPPED
}

@Serializable
enum class TaskPriority {
    LOW,
    NORMAL,
    HIGH,
    URGENT;

    val value: Int get() = ordinal
}
