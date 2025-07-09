package dev.aurakai.auraframefx.ai.task

import dev.aurakai.auraframefx.ai.error.ErrorHandler
import dev.aurakai.auraframefx.ai.pipeline.AIPipelineConfig
import dev.aurakai.auraframefx.model.AgentType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
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

    /**
     * Creates and registers a new task with the specified parameters, updates task statistics, and schedules it for execution.
     *
     * @param content The main content or description of the task.
     * @param context The context or environment in which the task should be executed.
     * @param priority The priority level assigned to the task.
     * @param urgency The urgency level assigned to the task.
     * @param importance The importance level assigned to the task.
     * @param requiredAgents The set of agent types required to execute the task.
     * @param dependencies The set of task IDs that must be completed before this task can be executed.
     * @param metadata Additional metadata associated with the task.
     * @return The newly created Task instance.
     */
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

    /**
     * Schedules a task by calculating its priority, urgency, and importance scores, updating its metadata, and adding it to the scheduling queue.
     *
     * The queue is sorted by total score in descending order, and queue processing is initiated. Any exceptions encountered during scheduling are delegated to the error handler with relevant context.
     */
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

    /**
     * Processes the task queue, executing eligible tasks while respecting active task capacity.
     *
     * Continuously removes and starts tasks from the queue if their dependencies are satisfied and agent requirements are met, until the queue is empty or the maximum number of active tasks is reached.
     */
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

    /**
     * Determines if a task is ready for execution by verifying that all its dependencies are completed.
     *
     * Returns `true` if every dependency has a status of `COMPLETED`. Agent availability is not currently checked and is assumed to be satisfied.
     *
     * @param task The task to check for execution readiness.
     * @return `true` if the task can be executed; `false` otherwise.
     */
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
     * Marks the specified task as in progress, assigns its required agents, and updates active and tracked task collections.
     *
     * @param task The task to be executed.
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
     * Moves the task to the appropriate collection based on the new status, triggers error handling if the task failed, updates the task record and statistics, and continues scheduling pending tasks.
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
     * Calculates the weighted priority score for a task based on its priority value and the configured priority weight.
     *
     * @param task The task whose priority score is being calculated.
     * @return The weighted priority score as a Float.
     */
    private fun calculatePriorityScore(task: Task): Float {
        return task.priority.value * config.priorityWeight
    }

    /**
     * Computes the weighted urgency score for the given task.
     *
     * Multiplies the task's urgency value by the configured urgency weight to determine its scheduling score.
     *
     * @param task The task whose urgency score is being calculated.
     * @return The weighted urgency score for scheduling.
     */
    private fun calculateUrgencyScore(task: Task): Float {
        return task.urgency.value * config.urgencyWeight
    }

    /**
     * Calculates the weighted importance score for a task based on its importance value and the configured importance weight.
     *
     * @param task The task whose importance score is being calculated.
     * @return The weighted importance score as a Float.
     */
    private fun calculateImportanceScore(task: Task): Float {
        return task.importance.value * config.importanceWeight
    }

    /**
     * Updates the aggregated task statistics to reflect the current state after a task is added or its status changes.
     *
     * Increments the total task count, updates counts for active, completed, and pending tasks, refreshes the last updated timestamp, and adjusts the count for the task's current status.
     *
     * @param task The task whose addition or status change triggers the statistics update.
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

@kotlinx.serialization.Serializable // Added annotation
data class TaskStats(
    val totalTasks: Int = 0,
    val activeTasks: Int = 0,
    val completedTasks: Int = 0,
    val pendingTasks: Int = 0,
    val taskCounts: Map<TaskStatus, Int> = emptyMap(),
    @kotlinx.serialization.Serializable(with = dev.aurakai.auraframefx.serialization.InstantSerializer::class) val lastUpdated: Instant = Clock.System.now(),
)
