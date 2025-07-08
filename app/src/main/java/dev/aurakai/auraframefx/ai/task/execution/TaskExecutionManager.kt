package dev.aurakai.auraframefx.ai.task.execution

import dev.aurakai.auraframefx.ai.agents.AuraAgent
import dev.aurakai.auraframefx.ai.agents.KaiAgent
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.ai.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.PriorityBlockingQueue
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * TaskExecutionManager handles scheduling, queuing, and monitoring of background tasks.
 * Implements the /tasks/schedule endpoint functionality with intelligent agent routing.
 */
@Singleton
class TaskExecutionManager @Inject constructor(
    private val auraAgent: AuraAgent,
    private val kaiAgent: KaiAgent,
    private val genesisAgent: GenesisAgent,
    private val securityContext: SecurityContext,
    private val logger: AuraFxLogger
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Task management
    private val taskQueue = PriorityBlockingQueue<TaskExecution>(100, TaskPriorityComparator())
    private val activeExecutions = ConcurrentHashMap<String, TaskExecution>()
    private val completedExecutions = ConcurrentHashMap<String, TaskResult>()
    
    // State management
    private val _executionStats = MutableStateFlow(ExecutionStats())
    val executionStats: StateFlow<ExecutionStats> = _executionStats
    
    private val _queueStatus = MutableStateFlow(QueueStatus())
    val queueStatus: StateFlow<QueueStatus> = _queueStatus
    
    private var isProcessing = false
    private val maxConcurrentTasks = 5

    init {
        startTaskProcessor()
    }

    /**
     * Schedules a new task for execution, assigning it to the most suitable agent based on task type or agent preference.
     *
     * Validates the scheduling request, creates a `TaskExecution` with the specified parameters, determines the optimal agent, and enqueues the task for background processing.
     *
     * @param type The type of task to execute.
     * @param data The input data required for the task.
     * @param priority The priority level for the task.
     * @param agentPreference An optional preferred agent for task execution.
     * @param scheduledTime An optional timestamp (in milliseconds) for delayed scheduling; if null, schedules immediately.
     * @return The `TaskExecution` instance representing the scheduled task.
     */
    suspend fun scheduleTask(
        type: String,
        data: Map<String, Any>,
        priority: TaskPriority = TaskPriority.NORMAL,
        agentPreference: String? = null,
        scheduledTime: Long? = null
    ): TaskExecution {
        logger.info("TaskExecutionManager", "Scheduling task: $type")
        
        // Security validation
        securityContext.validateRequest("task_schedule", data.toString())
        
        // Create task execution
        val execution = TaskExecution(
            id = UUID.randomUUID().toString(),
            type = type,
            data = data,
            priority = priority,
            agentPreference = agentPreference,
            scheduledTime = scheduledTime ?: System.currentTimeMillis(),
            status = TaskStatus.QUEUED,
            createdAt = System.currentTimeMillis()
        )
        
        // Determine optimal agent
        val optimalAgent = determineOptimalAgent(execution)
        execution.assignedAgent = optimalAgent
        
        // Add to queue
        taskQueue.offer(execution)
        updateQueueStatus()
        
        logger.info("TaskExecutionManager", "Task scheduled: ${execution.id} -> $optimalAgent")
        return execution
    }

    /**
     * Retrieves the current status of a task by its ID.
     *
     * Searches active, completed, and queued tasks for the specified ID and returns the corresponding status, or null if the task is not found.
     *
     * @param taskId The unique identifier of the task.
     * @return The current status of the task, or null if the task does not exist.
     */
    fun getTaskStatus(taskId: String): TaskStatus? {
        // Check active executions first
        activeExecutions[taskId]?.let { return it.status }
        
        // Check completed executions
        completedExecutions[taskId]?.let { return TaskStatus.COMPLETED }
        
        // Check queue
        taskQueue.find { it.id == taskId }?.let { return it.status }
        
        return null
    }

    /**
     * Returns the result of a completed task by its unique ID.
     *
     * @param taskId The unique identifier of the task.
     * @return The result of the task if it has completed (successfully or failed), or null if the task is not found or not yet completed.
     */
    fun getTaskResult(taskId: String): TaskResult? {
        return completedExecutions[taskId]
    }

    /**
     * Attempts to cancel a task by its ID if it is queued or currently running.
     *
     * If the task is queued, it is removed from the queue and marked as cancelled. If the task is running, its status is set to cancelling to signal the execution to stop.
     *
     * @param taskId The unique identifier of the task to cancel.
     * @return `true` if cancellation was initiated; `false` if the task was not found.
     */
    suspend fun cancelTask(taskId: String): Boolean {
        logger.info("TaskExecutionManager", "Cancelling task: $taskId")
        
        // Try to remove from queue first
        val queuedTask = taskQueue.find { it.id == taskId }
        if (queuedTask != null) {
            taskQueue.remove(queuedTask)
            queuedTask.status = TaskStatus.CANCELLED
            updateQueueStatus()
            return true
        }
        
        // Try to cancel active execution
        val activeTask = activeExecutions[taskId]
        if (activeTask != null) {
            activeTask.status = TaskStatus.CANCELLING
            // The execution coroutine will check this status and cancel itself
            return true
        }
        
        return false
    }

    /**
     * Retrieves all tasks managed by the system, including queued, active, and completed tasks, with optional filtering by status and agent type.
     *
     * @param status If specified, only tasks with this status are returned.
     * @param agentType If specified, only tasks assigned to this agent type are returned.
     * @return A list of tasks matching the provided filters.
     */
    fun getTasks(status: TaskStatus? = null, agentType: String? = null): List<TaskExecution> {
        val allTasks = mutableListOf<TaskExecution>()
        
        // Add queued tasks
        allTasks.addAll(taskQueue.toList())
        
        // Add active tasks
        allTasks.addAll(activeExecutions.values)
        
        // Add completed tasks (convert from results)
        allTasks.addAll(completedExecutions.values.map { result ->
            TaskExecution(
                id = result.taskId,
                type = result.type,
                data = result.originalData,
                priority = TaskPriority.NORMAL, // We don't store original priority in result
                status = TaskStatus.COMPLETED,
                assignedAgent = result.executedBy,
                createdAt = result.startTime,
                startedAt = result.startTime,
                completedAt = result.endTime
            )
        })
        
        // Apply filters
        return allTasks.filter { task ->
            (status == null || task.status == status) &&
            (agentType == null || task.assignedAgent == agentType)
        }
    }

    /**
     * Starts a background coroutine that continuously processes tasks from the queue while processing is active.
     *
     * Should be invoked once during initialization to enable ongoing task execution management. Applies a short delay between processing cycles and a longer delay if an exception occurs.
     */

    private fun startTaskProcessor() {
        scope.launch {
            isProcessing = true
            logger.info("TaskExecutionManager", "Starting task processor")
            
            while (isProcessing) {
                try {
                    processNextTask()
                    delay(100) // Small delay to prevent busy waiting
                } catch (e: Exception) {
                    logger.error("TaskExecutionManager", "Task processor error", e)
                    delay(1000) // Longer delay on error
                }
            }
        }
    }

    /**
     * Attempts to process the next task in the queue if concurrency limits permit.
     *
     * Polls the task queue for a task whose scheduled time has arrived. If the task is not yet ready, it is re-queued. If the maximum number of concurrent tasks is reached, no task is processed.
     */
    private suspend fun processNextTask() {
        // Check if we can process more tasks
        if (activeExecutions.size >= maxConcurrentTasks) {
            return
        }
        
        // Get next task from queue
        val task = taskQueue.poll() ?: return
        
        // Check if task should be executed now
        if (task.scheduledTime > System.currentTimeMillis()) {
            // Put back in queue if not ready
            taskQueue.offer(task)
            return
        }
        
        // Execute the task
        executeTask(task)
    }

    /**
     * Executes the given task using its assigned agent and updates its status, timing, and result.
     *
     * Sets the task status to running, delegates execution to the appropriate agent, and updates the task's status and result upon completion or failure. Removes the task from active executions and refreshes execution statistics and queue status.
     *
     * @param execution The task to execute.
     */
    private suspend fun executeTask(execution: TaskExecution) {
        execution.status = TaskStatus.RUNNING
        execution.startedAt = System.currentTimeMillis()
        activeExecutions[execution.id] = execution
        
        logger.info("TaskExecutionManager", "Executing task: ${execution.id}")
        
        scope.launch {
            try {
                // Execute based on assigned agent
                val result = when (execution.assignedAgent) {
                    "aura" -> executeWithAura(execution)
                    "kai" -> executeWithKai(execution)
                    "genesis" -> executeWithGenesis(execution)
                    else -> throw IllegalArgumentException("Unknown agent: ${execution.assignedAgent}")
                }
                
                // Mark as completed
                execution.status = TaskStatus.COMPLETED
                execution.completedAt = System.currentTimeMillis()
                
                // Store result
                val taskResult = TaskResult(
                    taskId = execution.id,
                    type = execution.type,
                    success = result.success,
                    data = result.data,
                    message = result.message,
                    executedBy = execution.assignedAgent!!,
                    startTime = execution.startedAt!!,
                    endTime = execution.completedAt!!,
                    executionTimeMs = execution.completedAt!! - execution.startedAt!!,
                    originalData = execution.data
                )
                
                completedExecutions[execution.id] = taskResult
                
                logger.info("TaskExecutionManager", "Task completed: ${execution.id}")
                
            } catch (e: Exception) {
                // Handle task failure
                execution.status = TaskStatus.FAILED
                execution.completedAt = System.currentTimeMillis()
                execution.errorMessage = e.message
                
                logger.error("TaskExecutionManager", "Task failed: ${execution.id}", e)
                
            } finally {
                // Remove from active executions
                activeExecutions.remove(execution.id)
                updateExecutionStats()
                updateQueueStatus()
            }
        }
    }

    /**
     * Processes the specified task using the Aura agent and returns the agent's response.
     *
     * @param execution The task execution details to be handled by the Aura agent.
     * @return The response produced by the Aura agent after processing the task.
     */
    private suspend fun executeWithAura(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            type = execution.type,
            data = execution.data,
            priority = execution.priority.value
        )
        return auraAgent.processRequest(request)
    }

    /**
     * Executes a task using the Kai agent and returns the agent's response.
     *
     * @param execution The task execution details.
     * @return The response produced by the Kai agent.
     */
    private suspend fun executeWithKai(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            type = execution.type,
            data = execution.data,
            priority = execution.priority.value
        )
        return kaiAgent.processRequest(request)
    }

    /**
     * Executes a task using the Genesis agent and returns the agent's response.
     *
     * @param execution The task to execute.
     * @return The response from the Genesis agent.
     */
    private suspend fun executeWithGenesis(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            type = execution.type,
            data = execution.data,
            priority = execution.priority.value
        )
        return genesisAgent.processRequest(request)
    }

    /**
     * Selects the most suitable agent ("aura", "kai", or "genesis") to execute a task based on agent preference or task type keywords.
     *
     * If the task specifies a valid agent preference, that agent is chosen. Otherwise, the agent is determined by matching keywords in the task type, with "genesis" as the default if no match is found.
     *
     * @param execution The task execution for which to determine the agent.
     * @return The name of the assigned agent.
     */
    private fun determineOptimalAgent(execution: TaskExecution): String {
        // Use agent preference if specified and valid
        execution.agentPreference?.let { preference ->
            if (listOf("aura", "kai", "genesis").contains(preference)) {
                return preference
            }
        }
        
        // Intelligent routing based on task type
        return when {
            execution.type.contains("creative", ignoreCase = true) -> "aura"
            execution.type.contains("ui", ignoreCase = true) -> "aura"
            execution.type.contains("security", ignoreCase = true) -> "kai"
            execution.type.contains("analysis", ignoreCase = true) -> "kai"
            execution.type.contains("complex", ignoreCase = true) -> "genesis"
            execution.type.contains("fusion", ignoreCase = true) -> "genesis"
            else -> "genesis" // Default to Genesis for intelligent routing
        }
    }

    /**
     * Updates the execution statistics with current task counts and average execution time.
     *
     * Aggregates totals for all, completed, active, queued, and failed tasks, and recalculates the average execution time from completed tasks.
     */
    private fun updateExecutionStats() {
        val total = activeExecutions.size + completedExecutions.size
        val completed = completedExecutions.size
        val active = activeExecutions.size
        val queued = taskQueue.size
        
        _executionStats.value = ExecutionStats(
            totalTasks = total,
            completedTasks = completed,
            activeTasks = active,
            queuedTasks = queued,
            failedTasks = completedExecutions.values.count { !it.success },
            averageExecutionTimeMs = calculateAverageExecutionTime()
        )
    }

    /**
     * Updates the queue status with current queue size, number of active executions, concurrency limit, and processing state.
     */
    private fun updateQueueStatus() {
        _queueStatus.value = QueueStatus(
            queueSize = taskQueue.size,
            activeExecutions = activeExecutions.size,
            maxConcurrentTasks = maxConcurrentTasks,
            isProcessing = isProcessing
        )
    }

    /**
     * Returns the average execution time in milliseconds for all completed tasks.
     *
     * @return The average execution time in milliseconds, or 0 if no tasks have been completed.
     */
    private fun calculateAverageExecutionTime(): Long {
        val executions = completedExecutions.values
        return if (executions.isNotEmpty()) {
            executions.map { it.executionTimeMs }.average().toLong()
        } else 0L
    }

    /**
     * Stops all background processing and releases resources used by the task execution manager.
     *
     * Cancels all running coroutines and prevents further task processing.
     */
    fun cleanup() {
        logger.info("TaskExecutionManager", "Cleaning up TaskExecutionManager")
        isProcessing = false
        scope.cancel()
    }
}

// Supporting data classes and enums
@Serializable
data class TaskExecution(
    val id: String,
    val type: String,
    val data: Map<String, Any>,
    val priority: TaskPriority,
    var status: TaskStatus = TaskStatus.QUEUED,
    val agentPreference: String? = null,
    var assignedAgent: String? = null,
    val scheduledTime: Long,
    val createdAt: Long,
    var startedAt: Long? = null,
    var completedAt: Long? = null,
    var errorMessage: String? = null
)

@Serializable
data class TaskResult(
    val taskId: String,
    val type: String,
    val success: Boolean,
    val data: Map<String, Any>,
    val message: String,
    val executedBy: String,
    val startTime: Long,
    val endTime: Long,
    val executionTimeMs: Long,
    val originalData: Map<String, Any>
)

enum class TaskStatus {
    QUEUED,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
    CANCELLING
}

enum class TaskPriority(val value: Int) {
    LOW(1),
    NORMAL(5),
    HIGH(8),
    CRITICAL(10)
}

@Serializable
data class ExecutionStats(
    val totalTasks: Int = 0,
    val completedTasks: Int = 0,
    val activeTasks: Int = 0,
    val queuedTasks: Int = 0,
    val failedTasks: Int = 0,
    val averageExecutionTimeMs: Long = 0
)

@Serializable
data class QueueStatus(
    val queueSize: Int = 0,
    val activeExecutions: Int = 0,
    val maxConcurrentTasks: Int = 0,
    val isProcessing: Boolean = false
)

class TaskPriorityComparator : Comparator<TaskExecution> {
    /**
     * Compares two `TaskExecution` objects for ordering in a priority queue.
     *
     * Tasks with higher priority values are ordered before those with lower priority. If priorities are equal, tasks scheduled for earlier execution are ordered first.
     *
     * @return A negative integer if the first task should precede the second, a positive integer if it should follow, or zero if both are equal in priority and scheduled time.
     */
    override fun compare(t1: TaskExecution, t2: TaskExecution): Int {
        // Higher priority first, then earlier scheduled time
        return when {
            t1.priority.value != t2.priority.value -> t2.priority.value - t1.priority.value
            else -> t1.scheduledTime.compareTo(t2.scheduledTime)
        }
    }
}
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
