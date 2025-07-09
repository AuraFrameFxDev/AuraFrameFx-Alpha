package dev.aurakai.auraframefx.ai.task

import dev.aurakai.auraframefx.ai.error.ErrorHandler
import dev.aurakai.auraframefx.ai.pipeline.AIPipelineConfig
import dev.aurakai.auraframefx.model.AgentType
import dev.aurakai.auraframefx.serialization.InstantSerializer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import java.lang.System // Added import
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TaskScheduler @Inject constructor(
    private val errorHandler: ErrorHandler,
    private val config: AIPipelineConfig,
) {
    private val _tasks = MutableStateFlow(mapOf<String, Task>())
    val tasks: StateFlow<Map<String, Task>> = _tasks

    private val _taskStats = MutableStateFlow(TaskStats())
    val taskStats: StateFlow<TaskStats> = _taskStats

    private val _taskQueue = mutableListOf<Task>()
    private val _activeTasks = mutableMapOf<String, Task>()
    private val _completedTasks = mutableMapOf<String, Task>()

    fun createTask(
        content: String,
        context: String,
        priority: TaskPriority = TaskPriority.NORMAL,
        urgency: TaskUrgency = TaskUrgency.MEDIUM,
        importance: TaskImportance = TaskImportance.MEDIUM,
        requiredAgents: Set<AgentType> = emptySet(),
        dependencies: Set<String> = emptySet(),
        metadata: Map<String, String> = emptyMap(),
    ): Task {
        val task = Task(
            content = content,
            context = context,
            priority = priority,
            urgency = urgency,
            importance = importance,
            requiredAgents = requiredAgents,
            dependencies = dependencies,
            metadata = metadata
        )

        _tasks.update { current ->
            current + (task.id to task)
        }

        updateStats(task)
        scheduleTask(task)
        return task
    }

    private fun scheduleTask(task: Task) {
        try {
            val priorityScore = calculatePriorityScore(task)
            val urgencyScore = calculateUrgencyScore(task)
            val importanceScore = calculateImportanceScore(task)

            val totalScore = (priorityScore * config.priorityWeight) +
                    (urgencyScore * config.urgencyWeight) +
                    (importanceScore * config.importanceWeight)

            _taskQueue.add(
                task.copy(
                    metadata = task.metadata + mapOf(
                        "priority_score" to priorityScore.toString(),
                        "urgency_score" to urgencyScore.toString(),
                        "importance_score" to importanceScore.toString(),
                        "total_score" to totalScore.toString()
                    )
                )
            )

            _taskQueue.sortByDescending { it.metadata["total_score"]?.toFloatOrNull() ?: 0f }
            processQueue()
        } catch (e: Exception) {
            errorHandler.handleError(
                error = e,
                agent = AgentType.GENESIS,
                context = "Task scheduling error",
                metadata = mapOf("taskId" to task.id)
            )
        }
    }

    private fun processQueue() {
        while (_taskQueue.isNotEmpty() && _activeTasks.size < config.maxActiveTasks) {
            val nextTask = _taskQueue.first()
            if (canExecuteTask(nextTask)) {
                executeTask(nextTask)
                _taskQueue.remove(nextTask)
            } else {
                break
            }
        }
    }

    private fun canExecuteTask(task: Task): Boolean {
        // Check dependencies
        val dependencies = task.dependencies.mapNotNull { _tasks.value[it] }
        if (dependencies.any { it.status != TaskStatus.COMPLETED }) {
            return false
        }

        // Check agent availability
        val requiredAgents = task.requiredAgents
        if (requiredAgents.isNotEmpty()) {
            // TODO: Implement agent availability check
            return true
        }

        return true
    }

    /**
     * Marks the given task as in progress, assigns required agents, and updates internal tracking for active and scheduled tasks.
     *
     * The task's status is set to `IN_PROGRESS`, and it is added to the active tasks map and updated in the overall task state.
     */
    private fun executeTask(task: Task) {
        val updatedTask = task.copy(
            status = TaskStatus.IN_PROGRESS,
            assignedAgents = task.requiredAgents
        )

        _activeTasks[task.id] = updatedTask
        _tasks.update { current ->
            current + (task.id to updatedTask)
        }
    }

    /**
     * Updates the status of a task and manages its transition between active, completed, and failed states.
     *
     * If the task is completed, it is moved from active to completed tasks. If the task fails, it is removed from active tasks and the failure is reported. The task's status is updated in the internal map, task statistics are refreshed, and pending tasks are processed.
     *
     * @param taskId The unique identifier of the task to update.
     * @param status The new status to assign to the task.
     */
    fun updateTaskStatus(taskId: String, status: TaskStatus) {
        val task = _tasks.value[taskId] ?: return
        val updatedTask = task.copy(status = status)

        when (status) {
            TaskStatus.COMPLETED -> {
                _activeTasks.remove(taskId)
                _completedTasks[taskId] = updatedTask
            }

            TaskStatus.FAILED -> {
                _activeTasks.remove(taskId)
                errorHandler.handleError(
                    error = Exception("Task execution failed"),
                    agent = task.assignedAgents.firstOrNull() ?: AgentType.GENESIS,
                    context = task.content,
                    metadata = mapOf("taskId" to taskId)
                )
            }

            else -> {}
        }

        _tasks.update { current ->
            current + (taskId to updatedTask)
        }
        updateStats(updatedTask)
        processQueue()
    }

    /**
     * Computes the weighted priority score for the given task using its priority value and the configured priority weight.
     *
     * @return The weighted priority score.
     */
    private fun calculatePriorityScore(task: Task): Float {
        return task.priority.value * config.priorityWeight
    }

    /**
     * Computes the weighted urgency score for a task using its urgency value and the configured urgency weight.
     *
     * @return The calculated urgency score.
     */
    private fun calculateUrgencyScore(task: Task): Float {
        return task.urgency.value * config.urgencyWeight
    }

    /**
     * Computes the weighted importance score for the given task using the configured importance weight.
     *
     * @return The weighted importance score.
     */
    private fun calculateImportanceScore(task: Task): Float {
        return task.importance.value * config.importanceWeight
    }

    /**
     * Updates the task statistics to reflect the current state after a task is created or its status changes.
     *
     * Increments the total task count, updates the counts for active, completed, and pending tasks, refreshes per-status counts, and sets the last updated timestamp.
     */
    private fun updateStats(task: Task) {
        _taskStats.update { current ->
            current.copy(
                totalTasks = current.totalTasks + 1,
                activeTasks = _activeTasks.size,
                completedTasks = _completedTasks.size,
                pendingTasks = _taskQueue.size,
                lastUpdated = Clock.System.now(),
                taskCounts = current.taskCounts + (task.status to (current.taskCounts[task.status]
                    ?: 0) + 1)
            )
        }
    }
}

@kotlinx.serialization.Serializable
data class TaskStats(
    val totalTasks: Int = 0,
    val activeTasks: Int = 0,
    val completedTasks: Int = 0,
    val pendingTasks: Int = 0,
    val taskCounts: Map<TaskStatus, Int> = emptyMap(),
    @kotlinx.serialization.Serializable(with = InstantSerializer::class) val lastUpdated: Instant = Clock.System.now(),
)
