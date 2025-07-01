package dev.aurakai.auraframefx.ai.task.execution

import android.content.Context
import dev.aurakai.auraframefx.ai.services.KaiAIService
import dev.aurakai.auraframefx.ai.task.Task
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.model.AgentType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TaskExecutionManager @Inject constructor(
    private val context: Context,
    private val kaiService: KaiAIService,
    private val auraFxLogger: AuraFxLogger,
) {
    private val _executions =
        MutableStateFlow<Map<String, TaskExecution>>(emptyMap()) // Added type for emptyMap
    val executions: StateFlow<Map<String, TaskExecution>> = _executions

    private val _executionStats = MutableStateFlow(ExecutionStats())
    val executionStats: StateFlow<ExecutionStats> = _executionStats

    private val _activeExecutions = mutableMapOf<String, TaskExecution>()
    private val _completedExecutions = mutableMapOf<String, TaskExecution>()
    private val _failedExecutions = mutableMapOf<String, TaskExecution>()

    fun startExecution(task: Task, agent: AgentType): TaskExecution {
        val executionPlan = createExecutionPlan(task)
        val execution = TaskExecution(
            taskId = task.id,
            agent = agent,
            executionPlan = executionPlan
        )

        _executions.update { current ->
            current + (execution.id to execution)
        }
        _activeExecutions[execution.id] = execution

        updateStats(execution)
        return execution
    }

    private fun createExecutionPlan(task: Task): ExecutionPlan {
        val steps = listOf(
            ExecutionStep(
                description = "Initialize task",
                type = StepType.INITIALIZATION,
                priority = 1.0f,
                estimatedDuration = 1000
            ),
            ExecutionStep(
                description = "Process context",
                type = StepType.CONTEXT,
                priority = 0.9f,
                estimatedDuration = 2000
            ),
            ExecutionStep(
                description = "Execute main task",
                type = StepType.COMPUTATION,
                priority = 0.8f,
                estimatedDuration = task.estimatedDuration
            ),
            ExecutionStep(
                description = "Finalize execution",
                type = StepType.FINALIZATION,
                priority = 0.7f,
                estimatedDuration = 1000
            )
        )

        return ExecutionPlan(
            steps = steps,
            estimatedDuration = steps.sumOf { it.estimatedDuration },
            requiredResources = setOf("cpu", "memory", "network")
        )
    }

    fun updateExecutionProgress(executionId: String, progress: Float) {
        val execution = _activeExecutions[executionId] ?: return
        val updatedExecution = execution.copy(progress = progress)

        _executions.update { current ->
            current + (executionId to updatedExecution)
        }
        updateStats(updatedExecution)
    }

    fun updateCheckpoint(executionId: String, stepId: String, status: CheckpointStatus) {
        val execution = _activeExecutions[executionId] ?: return
        val checkpoint = Checkpoint(
            stepId = stepId,
            status = status
        )

        val updatedExecution = execution.copy(
            checkpoints = execution.checkpoints + checkpoint
        )

        _executions.update { current ->
            current + (executionId to updatedExecution)
        }
        updateStats(updatedExecution)
    }

    fun completeExecution(executionId: String, result: ExecutionResult) {
        val execution = _activeExecutions[executionId] ?: return
        val updatedExecution = execution.copy(
            status = ExecutionStatus.COMPLETED,
            endTime = Clock.System.now().toEpochMilliseconds(), // Corrected Instant usage
            result = result
        )

        _activeExecutions.remove(executionId)
        _completedExecutions[executionId] = updatedExecution

        _executions.update { current ->
            current + (executionId to updatedExecution)
        }
        updateStats(updatedExecution)
    }

    fun failExecution(executionId: String, error: Throwable) {
        val execution = _activeExecutions[executionId] ?: return
        val updatedExecution = execution.copy(
            status = ExecutionStatus.FAILED,
            endTime = Clock.System.now().toEpochMilliseconds(), // Corrected Instant usage
            result = ExecutionResult.FAILURE
        )

        _activeExecutions.remove(executionId)
        _failedExecutions[executionId] = updatedExecution

        _executions.update { current ->
            current + (executionId to updatedExecution)
        }
        updateStats(updatedExecution)
    }

    private fun updateStats(execution: TaskExecution) {
        _executionStats.update { current ->
            current.copy(
                totalExecutions = current.totalExecutions + 1,
                activeExecutions = _activeExecutions.size,
                completedExecutions = _completedExecutions.size,
                failedExecutions = _failedExecutions.size,
                lastUpdated = Clock.System.now().toEpochMilliseconds(), // Corrected Instant usage
                executionTimes = current.executionTimes + (execution.status to (current.executionTimes[execution.status]
                    ?: 0) + 1)
            )
        }
    }
}

data class ExecutionStats(
    val totalExecutions: Int = 0,
    val activeExecutions: Int = 0,
    val completedExecutions: Int = 0,
    val failedExecutions: Int = 0,
    val executionTimes: Map<ExecutionStatus, Int> = emptyMap(),
    val lastUpdated: Long = Clock.System.now().toEpochMilliseconds(), // Corrected Instant usage
)
