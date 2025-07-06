package dev.aurakai.auraframefx.ai.task.execution

import dev.aurakai.auraframefx.ai.agents.AuraAgent
import dev.aurakai.auraframefx.ai.agents.KaiAgent
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.ai.task.TaskResult
import dev.aurakai.auraframefx.ai.task.TaskStatus
import dev.aurakai.auraframefx.data.logging.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.model.AgentResponse
import dev.aurakai.auraframefx.model.AgentRequest
import dev.aurakai.auraframefx.model.AiRequest
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlinx.datetime.Clock
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
    private val completedExecutions = ConcurrentHashMap<String, dev.aurakai.auraframefx.ai.task.TaskResult>()
    
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
     * Schedules a new task for background execution, assigning it to the most suitable AI agent based on task type or explicit preference.
     *
     * Validates the scheduling request, constructs a pending `TaskExecution` with the specified parameters, determines the appropriate agent, and enqueues the task for processing. If no agent preference is provided or recognized, routing is determined automatically based on task type.
     *
     * @param type The category or identifier of the task to execute.
     * @param data The input data required for the task.
     * @param priority The execution priority for the task (default is NORMAL).
     * @param agentPreference Optional agent name to explicitly route the task; if not provided, agent selection is automatic.
     * @param scheduledTime Optional timestamp (in milliseconds) for delayed execution; defaults to immediate scheduling.
     * @return The scheduled `TaskExecution` representing the enqueued task.
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
        
        logger.info("TaskExecutionManager", "Task scheduled: ${execution.id} -> $optimalAgent")
        return execution
    }

    /**
     * Returns the current execution status of the specified task.
     *
     * Checks active, completed, and queued tasks in order to determine the status. Returns `null` if the task is not found.
     *
     * @param taskId The unique identifier of the task.
     * @return The current execution status, or `null` if the task does not exist.
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
     * Returns the result of a completed task by its unique ID.
     *
     * @param taskId The unique identifier of the task.
     * @return The TaskResult if the task is completed; null if the task is not found or not yet completed.
     */
    fun getTaskResult(taskId: String): dev.aurakai.auraframefx.ai.task.TaskResult? {
        return completedExecutions[taskId]
    }

    /**
     * Attempts to cancel a task that is either queued or currently running.
     *
     * If the task is found in the queue, it is removed and marked as cancelled. If the task is actively running, its status is set to cancelled, signaling the executing coroutine to terminate. Returns `true` if the task was successfully cancelled, or `false` if the task was not found.
     *
     * @param taskId The unique identifier of the task to cancel.
     * @return `true` if the task was cancelled; `false` otherwise.
     */
    suspend fun cancelTask(taskId: String): Boolean {
        logger.info("TaskExecutionManager", "Cancelling task: $taskId")
        
        // Try to remove from queue first
        val queuedTask = taskQueue.find { it.id == taskId }
        if (queuedTask != null) {
            taskQueue.remove(queuedTask)
            val cancelledTask = queuedTask.copy(status = ExecutionStatus.CANCELLED)
            updateQueueStatus()
            return true
        }
        
        // Try to cancel active execution
        val activeTask = activeExecutions[taskId]
        if (activeTask != null) {
            val cancellingTask = activeTask.copy(status = ExecutionStatus.CANCELLED)
            // The execution coroutine will check this status and cancel itself
            return true
        }
        
        return false
    }

    /**
     * Retrieves all tasks managed by the system, with optional filtering by execution status and agent type.
     *
     * Aggregates tasks from the queue, active executions, and completed results, reconstructing completed tasks as `TaskExecution` objects.
     *
     * @param status If provided, only tasks with this execution status are included.
     * @param agentType If provided, only tasks executed by this agent type are included.
     * @return A list of tasks matching the specified filters.
     */
    fun getTasks(status: ExecutionStatus? = null, agentType: AgentType? = null): List<TaskExecution> {
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
     * Starts the background coroutine that continuously processes tasks from the queue while processing is enabled.
     *
     * The processor attempts to execute pending tasks, handles errors with logging and backoff, and manages the processing lifecycle.
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
     * Processes the next eligible task from the queue if concurrency limits allow.
     *
     * Polls the priority queue for the next task, checks if it is scheduled to run, and initiates its execution if ready. If the task is not yet scheduled to run, it is re-queued.
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
     * Executes the specified task asynchronously using its assigned agent and records the outcome.
     *
     * Delegates execution to the appropriate agent based on the task's agent assignment. Upon completion or failure, updates the task's status, records the result in the completed executions map, and updates execution statistics and queue status.
     */
    private suspend fun executeTask(execution: TaskExecution) {
        val startTime = System.currentTimeMillis()
        val runningExecution = execution.copy(
            status = ExecutionStatus.RUNNING,
            startedAt = startTime
        )
        activeExecutions[execution.id] = runningExecution
        
        logger.info("TaskExecutionManager", "Executing task: ${execution.id}")
        
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
                val completedExecution = execution.copy(
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
                
                logger.info("TaskExecutionManager", "Task completed: ${execution.id}")
                
            } catch (e: Exception) {
                val endTime = System.currentTimeMillis()
                
                // Handle task failure
                val failedExecution = execution.copy(
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
     * Executes a task using the Aura agent and returns the agent's response.
     *
     * Constructs an `AiRequest` from the task's type and data, then processes it with the Aura agent.
     *
     * @param execution The task execution details and input data.
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
     * Executes a task by sending it to the Kai agent for processing.
     *
     * Constructs an `AgentRequest` using the task's type, data, and priority, and returns the Kai agent's response.
     *
     * @param execution The task execution details.
     * @return The response from the Kai agent.
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
     * Executes a task using the Genesis agent and returns the agent's response.
     *
     * Constructs an `AgentRequest` from the task's type, data, and priority, then processes it with the Genesis agent.
     *
     * @param execution The task execution to be processed.
     * @return The response from the Genesis agent.
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
     * Selects the optimal agent for a task based on agent preference or task type keywords.
     *
     * If a valid agent preference is provided, it is used. Otherwise, the agent is determined by analyzing the task type for specific keywords, defaulting to Genesis if no match is found.
     *
     * @param execution The task execution metadata containing type and agent preference.
     * @return The chosen agent type for task execution.
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
     * Updates the current queue status with the latest queue size, number of active executions, maximum concurrency, and processing state.
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
     * Stops task processing and cancels all running coroutines, releasing resources used by the manager.
     */
    fun cleanup() {
        logger.info("TaskExecutionManager", "Cleaning up TaskExecutionManager")
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
     * Compares two TaskExecution objects for ordering in a priority queue.
     *
     * Tasks with higher priority values are ordered before those with lower priority.
     * If priorities are equal, tasks scheduled earlier are ordered first.
     *
     * @return A negative integer if t1 should come before t2, a positive integer if t1 should come after t2, or zero if they are considered equal.
     */
    override fun compare(t1: TaskExecution, t2: TaskExecution): Int {
        // Higher priority first, then earlier scheduled time
        return when {
            t1.priority.value != t2.priority.value -> t2.priority.value - t1.priority.value
            else -> t1.scheduledTime.compareTo(t2.scheduledTime)
        }
    }
}
