package dev.aurakai.auraframefx.ai.task.execution

import dev.aurakai.auraframefx.ai.agents.AuraAgent
import dev.aurakai.auraframefx.ai.agents.KaiAgent
import dev.aurakai.auraframefx.ai.agents.GenesisAgent
import dev.aurakai.auraframefx.utils.AuraFxLogger
import dev.aurakai.auraframefx.security.SecurityContext
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.ai.*
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
     * Schedule a task for execution with intelligent agent routing.
     * Implements the core logic for the /tasks/schedule endpoint.
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
            agent = agentPreference ?: AgentType.GENESIS,
            status = ExecutionStatus.PENDING,
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
     * Get status of a specific task execution.
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
     * Get detailed task execution result.
     */
    fun getTaskResult(taskId: String): TaskResult? {
        return completedExecutions[taskId]
    }

    /**
     * Cancel a queued or running task.
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
     * Get all tasks with optional filtering.
     */
    fun getTasks(status: ExecutionStatus? = null, agentType: String? = null): List<TaskExecution> {
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
                status = ExecutionStatus.COMPLETED,
                agent = result.executedBy,
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

    // Private task processing methods

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

    private suspend fun executeTask(execution: TaskExecution) {
        val runningExecution = execution.copy(status = ExecutionStatus.RUNNING)
        execution.startedAt = System.currentTimeMillis()
        activeExecutions[execution.id] = execution
        
        logger.info("TaskExecutionManager", "Executing task: ${execution.id}")
        
        scope.launch {
            try {
                // Execute based on assigned agent
                val result = when (execution.agent) {
                    "aura" -> executeWithAura(execution)
                    "kai" -> executeWithKai(execution)
                    "genesis" -> executeWithGenesis(execution)
                    else -> throw IllegalArgumentException("Unknown agent: ${execution.agent}")
                }
                
                // Mark as completed
                val completedExecution = execution.copy(status = ExecutionStatus.COMPLETED)
                execution.completedAt = System.currentTimeMillis()
                
                // Store result
                val taskResult = TaskResult(
                    taskId = execution.id,
                    type = execution.type,
                    success = result.success,
                    data = result.data,
                    message = result.message,
                    executedBy = execution.agent,
                    startTime = execution.startedAt!!,
                    endTime = execution.completedAt!!,
                    executionTimeMs = execution.completedAt!! - execution.startedAt!!,
                    originalData = execution.data
                )
                
                completedExecutions[execution.id] = taskResult
                
                logger.info("TaskExecutionManager", "Task completed: ${execution.id}")
                
            } catch (e: Exception) {
                // Handle task failure
                val failedExecution = execution.copy(status = ExecutionStatus.FAILED)
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

    private suspend fun executeWithAura(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            type = execution.type,
            data = execution.data,
            priority = execution.priority.value
        )
        return auraAgent.processRequest(request)
    }

    private suspend fun executeWithKai(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            type = execution.type,
            data = execution.data,
            priority = execution.priority.value
        )
        return kaiAgent.processRequest(request)
    }

    private suspend fun executeWithGenesis(execution: TaskExecution): AgentResponse {
        val request = AgentRequest(
            type = execution.type,
            data = execution.data,
            priority = execution.priority.value
        )
        return genesisAgent.processRequest(request)
    }

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

    private fun updateQueueStatus() {
        _queueStatus.value = QueueStatus(
            queueSize = taskQueue.size,
            activeExecutions = activeExecutions.size,
            maxConcurrentTasks = maxConcurrentTasks,
            isProcessing = isProcessing
        )
    }

    private fun calculateAverageExecutionTime(): Long {
        val executions = completedExecutions.values
        return if (executions.isNotEmpty()) {
            executions.map { it.executionTimeMs }.average().toLong()
        } else 0L
    }

    /**
     * Cleanup resources when manager is destroyed.
     */
    fun cleanup() {
        logger.info("TaskExecutionManager", "Cleaning up TaskExecutionManager")
        isProcessing = false
        scope.cancel()
    }
}

// Supporting data classes and enums
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
    override fun compare(t1: TaskExecution, t2: TaskExecution): Int {
        // Higher priority first, then earlier scheduled time
        return when {
            t1.priority.value != t2.priority.value -> t2.priority.value - t1.priority.value
            else -> t1.scheduledTime.compareTo(t2.scheduledTime)
        }
    }
}
