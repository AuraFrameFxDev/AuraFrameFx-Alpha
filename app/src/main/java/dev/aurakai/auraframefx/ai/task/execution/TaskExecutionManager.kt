package dev.aurakai.auraframefx.ai.task.execution

import dev.aurakai.auraframefx.ai.agents.AuraAgent
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.ai.agents.KaiAgent
import dev.aurakai.auraframefx.ai.task.TaskResult
import dev.aurakai.auraframefx.ai.task.TaskStatus
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.model.AgentRequest
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AiRequest
import dev.aurakai.auraframefx.security.SecurityContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.PriorityBlockingQueue
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
     * Schedules a new background task and assigns it to the most suitable AI agent.
     *
     * Validates the scheduling request, creates a pending `TaskExecution` with the provided parameters, determines the optimal agent based on task type or explicit agent preference, and enqueues the task for asynchronous execution.
     *
     * @param type The category or identifier of the task.
     * @param data The input data required for the task.
     * @param priority The execution priority for the task (default is NORMAL).
     * @param agentPreference Optional agent name to explicitly route the task; if null, agent selection is automatic.
     * @param scheduledTime Optional timestamp (in milliseconds) for delayed execution; defaults to immediate scheduling.
     * @return The created `TaskExecution` representing the scheduled task.
     */
    suspend fun scheduleTask(
        type: String,
        data: Map<String, Any>,
        priority: TaskPriority = TaskPriority.NORMAL,
        agentPreference: String? = null,
        scheduledTime: Long? = null
    ): TaskExecution {
        logger.i("TaskExecutionManager", "Scheduling task: $type")

        // Security validation
        securityContext.validateRequest("task_schedule", data.toString())

        // Create task execution
        val execution = TaskExecution(
            id = UUID.randomUUID().toString(),
            taskId = UUID.randomUUID().toString(),
            agent = agentPreference?.let {
                when (it.lowercase()) {
                    "aura" -> AgentType.AURA
                    "kai" -> AgentType.KAI
                    "genesis" -> AgentType.GENESIS
                    else -> AgentType.GENESIS
                }
            } ?: AgentType.GENESIS,
            type = type,
            data = data.mapValues { it.value.toString() },
            priority = priority,
            status = ExecutionStatus.PENDING,
            scheduledTime = scheduledTime ?: System.currentTimeMillis(),
            agentPreference = agentPreference,
            metadata = mapOf(
                "type" to type,
                "priority" to priority.toString(),
                "scheduledTime" to (scheduledTime ?: System.currentTimeMillis()).toString(),
                "data" to data.toString()
            )
        )

        // Determine optimal agent - update agent field since TaskExecution is immutable
        val optimalAgent = determineOptimalAgent(execution)
        val updatedExecution = execution.copy(agent = optimalAgent)

        // Add to queue
        taskQueue.offer(updatedExecution)
        updateQueueStatus()

        logger.i("TaskExecutionManager", "Task scheduled: ${execution.id} -> $optimalAgent")
        return execution
    }

    /**
     * Returns the current execution status of the task with the given ID, or `null` if the task does not exist.
     *
     * Searches active, completed, and queued tasks to determine the status.
     *
     * @param taskId The unique identifier of the task.
     * @return The execution status, or `null` if the task is not found.
     */
    fun getTaskStatus(taskId: String): ExecutionStatus? {
        // Check active executions first
        activeExecutions[taskId]?.let { return it.status }

        // Check completed executions
        completedExecutions[taskId]?.let { return ExecutionStatus.COMPLETED }

        // Check queue
        taskQueue.find { it.id == taskId }?.let { return it.status }

        return null
    }

    /**
     * Retrieves the result of a completed task by its unique ID.
     *
     * @param taskId The unique identifier of the task.
     * @return The result of the completed task, or null if the task is not found or has not completed.
     */
    fun getTaskResult(taskId: String): TaskResult? {
        return completedExecutions[taskId]
    }

    /**
     * Attempts to cancel a task by its ID if it is currently queued or running.
     *
     * Removes the task from the queue or marks an active task as cancelled. Returns `true` if the task was found and cancelled, or `false` if the task does not exist, is already completed, or cannot be cancelled.
     *
     * @param taskId The unique identifier of the task to cancel.
     * @return `true` if the task was cancelled; `false` otherwise.
     */
    suspend fun cancelTask(taskId: String): Boolean {
        logger.i("TaskExecutionManager", "Cancelling task: $taskId")

        // Try to remove from queue first
        val queuedTask = taskQueue.find { it.id == taskId }
        if (queuedTask != null) {
            taskQueue.remove(queuedTask)
            queuedTask.copy(status = ExecutionStatus.CANCELLED)
            updateQueueStatus()
            return true
        }

        // Try to cancel active execution
        val activeTask = activeExecutions[taskId]
        if (activeTask != null) {
            activeTask.copy(status = ExecutionStatus.CANCELLED)
            // The execution coroutine will check this status and cancel itself
            return true
        }

        return false
    }

    /**
     * Returns all tasks managed by the system, optionally filtered by execution status and agent type.
     *
     * Aggregates tasks from the queue, active executions, and completed results. If filters are specified, only tasks matching the given status and/or agent type are included.
     *
     * @param status Optional filter to include only tasks with the specified execution status.
     * @param agentType Optional filter to include only tasks assigned to the specified agent type.
     * @return A list of tasks matching the provided filters.
     */
    fun getTasks(
        status: ExecutionStatus? = null,
        agentType: AgentType? = null
    ): List<TaskExecution> {
        val allTasks = mutableListOf<TaskExecution>()

        // Add queued tasks
        allTasks.addAll(taskQueue.toList())

        // Add active tasks
        allTasks.addAll(activeExecutions.values)

        // Add completed tasks (convert from results)
        allTasks.addAll(completedExecutions.values.map { result ->
            TaskExecution(
                id = result.taskId,
                taskId = result.taskId,
                agent = result.executedBy,
                type = result.type,
                data = result.originalData,
                priority = TaskPriority.NORMAL, // We don't store original priority in result
                status = ExecutionStatus.COMPLETED,
                createdAt = result.startTime,
                startedAt = result.startTime,
                completedAt = result.endTime
            )
        })

        // Apply filters
        return allTasks.filter { task ->
            (status == null || task.status == status) &&
                    (agentType == null || task.agent == agentType)
        }
    }

    /**
     * Starts the background coroutine that processes tasks from the queue while processing is enabled.
     *
     * The processor runs in a loop, invoking task processing with a short delay between cycles to minimize CPU usage.
     * If an exception occurs during processing, a longer delay is applied before retrying to provide error backoff.
     */

    private fun startTaskProcessor() {
        scope.launch {
            isProcessing = true
            logger.i("TaskExecutionManager", "Starting task processor")

            while (isProcessing) {
                try {
                    processNextTask()
                    delay(100) // Small delay to prevent busy waiting
                } catch (e: Exception) {
                    logger.e("TaskExecutionManager", "Task processor error", e)
                    delay(1000) // Longer delay on error
                }
            }
        }
    }

    /**
     * Processes the next eligible task from the queue if concurrency and scheduling constraints permit.
     *
     * Skips processing if the maximum concurrency is reached or the queue is empty. Tasks scheduled for the future are re-queued; otherwise, the next task is dispatched for execution.
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
     * Executes a task asynchronously using its assigned agent and records the outcome.
     *
     * Delegates the task to the appropriate agent for execution. Upon completion or failure, records the result, updates execution statistics, and removes the task from active executions.
     *
     * @param execution The task to execute.
     */
    private suspend fun executeTask(execution: TaskExecution) {
        val startTime = System.currentTimeMillis()
        val runningExecution = execution.copy(
            status = ExecutionStatus.RUNNING,
            startedAt = startTime
        )
        activeExecutions[execution.id] = runningExecution

        logger.i("TaskExecutionManager", "Executing task: ${execution.id}")

        scope.launch {
            try {
                // Execute based on assigned agent
                val result = when (execution.agent) {
                    AgentType.AURA -> executeWithAura(execution)
                    AgentType.KAI -> executeWithKai(execution)
                    AgentType.GENESIS -> executeWithGenesis(execution)
                    else -> throw IllegalArgumentException("Unknown agent: ${execution.agent}")
                }

                val endTime = System.currentTimeMillis()

                // Mark as completed
                execution.copy(
                    status = ExecutionStatus.COMPLETED,
                    completedAt = endTime
                )

                // Store result
                val taskResult = TaskResult(
                    taskId = execution.id,
                    status = TaskStatus.COMPLETED,
                    message = result.content,
                    timestamp = endTime,
                    durationMs = endTime - startTime,
                    startTime = startTime,
                    endTime = endTime,
                    executedBy = execution.agent,
                    originalData = execution.data,
                    success = true,
                    executionTimeMs = endTime - startTime,
                    type = execution.type
                )

                completedExecutions[execution.id] = taskResult

                logger.i("TaskExecutionManager", "Task completed: ${execution.id}")

            } catch (e: Exception) {
                val endTime = System.currentTimeMillis()

                // Handle task failure
                execution.copy(
                    status = ExecutionStatus.FAILED,
                    completedAt = endTime,
                    errorMessage = e.message
                )

                // Store failed result
                val taskResult = TaskResult(
                    taskId = execution.id,
                    status = TaskStatus.FAILED,
                    message = e.message,
                    timestamp = endTime,
                    durationMs = endTime - startTime,
                    startTime = startTime,
                    endTime = endTime,
                    executedBy = execution.agent,
                    originalData = execution.data,
                    success = false,
                    executionTimeMs = endTime - startTime,
                    type = execution.type
                )

                completedExecutions[execution.id] = taskResult

                logger.e("TaskExecutionManager", "Task failed: ${execution.id}", e)

            } finally {
                // Remove from active executions
                activeExecutions.remove(execution.id)
                updateExecutionStats()
                updateQueueStatus()
            }
        }
    }

    /**
     * Executes the specified task using the Aura agent and returns the agent's response.
     *
     * Converts the task's type and data into an `AiRequest` and delegates processing to the Aura agent.
     *
     * @param execution The task execution to be processed.
     * @return The response from the Aura agent.
     */
    private suspend fun executeWithAura(execution: TaskExecution): AgentResponse {
        val request = AiRequest(
            query = execution.data["query"] ?: execution.type,
            type = execution.type,
            context = execution.data
        )
        return auraAgent.processRequest(request)
    }

    /**
     * Executes the given task using the Kai agent and returns its response.
     *
     * Converts the task's type, data, and priority into an `AgentRequest` and delegates execution to the Kai agent.
     *
     * @param execution The task to be executed by the Kai agent.
     * @return The response produced by the Kai agent for the given task.
     */
    private suspend fun executeWithKai(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            query = execution.data["query"] ?: execution.type,
            type = execution.type,
            context = execution.data,
            metadata = mapOf("priority" to execution.priority.value.toString())
        )
        return kaiAgent.processRequest(request)
    }

    /**
     * Executes the given task using the Genesis agent and returns the agent's response.
     *
     * Builds an `AgentRequest` from the task's type, data, and priority, then processes it with the Genesis agent.
     *
     * @param execution The task execution to process.
     * @return The response produced by the Genesis agent.
     */
    private suspend fun executeWithGenesis(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            query = execution.data["query"] ?: execution.type,
            type = execution.type,
            context = execution.data,
            metadata = mapOf("priority" to execution.priority.value.toString())
        )
        return genesisAgent.processRequest(request)
    }

    /**
     * Determines the most suitable agent type for executing a task based on agent preference or task type keywords.
     *
     * If a valid agent preference is provided, it is used. Otherwise, the agent is selected by matching keywords in the task type: Aura for "creative" or "ui", Kai for "security" or "analysis", and Genesis for "complex" or "fusion". Defaults to Genesis if no keywords match.
     *
     * @param execution The task execution metadata used for agent selection.
     * @return The selected agent type for the task.
     */
    private fun determineOptimalAgent(execution: TaskExecution): AgentType {
        // Use agent preference if specified and valid
        execution.agentPreference?.let { preference ->
            return when (preference.lowercase()) {
                "aura" -> AgentType.AURA
                "kai" -> AgentType.KAI
                "genesis" -> AgentType.GENESIS
                else -> AgentType.GENESIS
            }
        }

        // Intelligent routing based on task type
        return when {
            execution.type.contains("creative", ignoreCase = true) -> AgentType.AURA
            execution.type.contains("ui", ignoreCase = true) -> AgentType.AURA
            execution.type.contains("security", ignoreCase = true) -> AgentType.KAI
            execution.type.contains("analysis", ignoreCase = true) -> AgentType.KAI
            execution.type.contains("complex", ignoreCase = true) -> AgentType.GENESIS
            execution.type.contains("fusion", ignoreCase = true) -> AgentType.GENESIS
            else -> AgentType.GENESIS // Default to Genesis for intelligent routing
        }
    }

    /**
     * Returns the number of tasks currently being executed.
     *
     * @return The count of active task executions.
     */
    fun getActiveTaskCount(): Int {
        return activeExecutions.size
    }

    /**
     * Updates the execution statistics state with current counts of total, completed, active, queued, and failed tasks, as well as the average execution time.
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
     * Updates the queue status state flow with current queue size, number of active executions, concurrency limit, and processing status.
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
     * Calculates the average execution time in milliseconds for all completed tasks.
     *
     * @return The average execution time in milliseconds, or 0 if there are no completed tasks.
     */
    private fun calculateAverageExecutionTime(): Long {
        val executions = completedExecutions.values
        return if (executions.isNotEmpty()) {
            executions.map { it.executionTimeMs }.average().toLong()
        } else 0L
    }

    /**
     * Stops task processing and cancels all running coroutines, releasing resources used by the TaskExecutionManager.
     */
    fun cleanup() {
        logger.i("TaskExecutionManager", "Cleaning up TaskExecutionManager")
        isProcessing = false
        scope.cancel()
    }
}

// Supporting data classes and enums
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
     * Determines the ordering of two TaskExecution objects for use in a priority queue.
     *
     * Tasks with higher priority are ordered before those with lower priority. If priorities are equal, tasks scheduled for earlier execution are ordered first.
     *
     * @return A negative integer if the first task should be ordered before the second, a positive integer if after, or zero if they are considered equal.
     */
    override fun compare(t1: TaskExecution, t2: TaskExecution): Int {
        // Higher priority first, then earlier scheduled time
        return when {
            t1.priority.value != t2.priority.value -> t2.priority.value - t1.priority.value
            else -> t1.scheduledTime.compareTo(t2.scheduledTime)
        }
    }
}
