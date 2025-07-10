package dev.aurakai.auraframefx.ai.task

import dev.aurakai.auraframefx.ai.error.ErrorHandler
import dev.aurakai.auraframefx.ai.pipeline.AIPipelineConfig
import dev.aurakai.auraframefx.model.AgentType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import dev.aurakai.auraframefx.serialization.InstantSerializer // Added import
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

    /**
     * Calculates weighted scores for a task, updates its metadata with these scores, adds it to the task queue,
     * sorts the queue by total score in descending order, and triggers processing of the queue.
     *
     * If an error occurs during scheduling, it is handled via the error handler with relevant context and task ID.
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
     * Executes eligible tasks from the queue until reaching the maximum allowed active tasks or encountering a task that cannot be executed.
     *
     * Removes each executed task from the queue. Stops processing when a task's dependencies or agent requirements are not satisfied.
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

<<<<<<< HEAD
    /**
     * Determines whether a task is ready to be executed based on its dependencies and required agents.
     *
     * Returns `true` if all dependencies are completed and agent requirements (currently always allowed) are met; otherwise, returns `false`.
     */
=======
>>>>>>> pr458merge
    /**
     * Determines whether a task is eligible for execution based on its dependencies and required agents.
     *
     * Returns `true` if all dependencies are completed and agent requirements are considered satisfied; otherwise, returns `false`.
     */
    /**
     * Determines whether a task is eligible for execution based on its dependencies and required agents.
     *
     * Returns `true` if all dependencies have been completed and agent requirements are considered satisfied.
     * Currently, agent availability is always assumed to be met if required agents are specified.
     *
     * @return `true` if the task can be executed; otherwise, `false`.
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

<<<<<<< HEAD
    /**
     * Marks the given task as in progress, assigns required agents, and updates active and tracked tasks.
     *
     * @param task The task to be executed.
     */
=======
>>>>>>> pr458merge
    /**
     * Marks the given task as in progress, assigns required agents, and updates internal tracking of active and all tasks.
     *
     * Updates the task's status to `IN_PROGRESS`, assigns the required agents, adds it to the active tasks map, and updates the overall tasks state.
     */
    /**
     * Marks the given task as in progress, assigns required agents, and updates the active and overall task state.
     *
     * The task's status is set to `IN_PROGRESS`, its assigned agents are updated to match the required agents,
     * and it is added to the active tasks map and the overall tasks state.
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

<<<<<<< HEAD
    /**
     * Updates the status of a task and manages its transition between active, completed, and failed states.
     *
     * If the status is `COMPLETED`, moves the task from active to completed tasks. If the status is `FAILED`, removes the task from active tasks and triggers error handling. Updates the task record, refreshes task statistics, and processes the scheduling queue.
     *
     * @param taskId The unique identifier of the task to update.
     * @param status The new status to assign to the task.
     */
=======
>>>>>>> pr458merge
    /**
     * Updates the status of a task and manages its state transitions.
     *
     * If the status is `COMPLETED`, moves the task from active to completed tasks.
     * If the status is `FAILED`, removes the task from active tasks and triggers error handling.
     * Updates the task in the internal task map, refreshes task statistics, and processes the task queue.
     *
     * @param taskId The unique identifier of the task to update.
     * @param status The new status to assign to the task.
     */
    /**
     * Updates the status of a task and manages its state transitions.
     *
     * Moves the task between active, completed, or failed collections based on the new status.
     * On failure, triggers error handling with relevant context and metadata.
     * Refreshes the task in the main task map, updates aggregated statistics, and reprocesses the task queue.
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

<<<<<<< HEAD
    /**
     * Calculates the weighted priority score for a task based on its priority value and the configured priority weight.
     *
     * @return The computed priority score as a floating-point value.
     */
=======
>>>>>>> pr458merge
    private fun calculatePriorityScore(task: Task): Float {
        return task.priority.value * config.priorityWeight
    }

<<<<<<< HEAD
    /**
     * Calculates the urgency score for a task based on its urgency value and the configured urgency weight.
     *
     * @return The weighted urgency score as a Float.
     */
=======
>>>>>>> pr458merge
    private fun calculateUrgencyScore(task: Task): Float {
        return task.urgency.value * config.urgencyWeight
    }

<<<<<<< HEAD
    /**
     * Calculates the importance score for a task based on its importance value and the configured importance weight.
     *
     * @return The weighted importance score as a Float.
     */
=======
>>>>>>> pr458merge
    private fun calculateImportanceScore(task: Task): Float {
        return task.importance.value * config.importanceWeight
    }

<<<<<<< HEAD
    /**
     * Updates the aggregated task statistics to reflect the addition or status change of a task.
     *
     * Increments the total task count, updates counts for active, completed, and pending tasks,
     * refreshes the last updated timestamp, and adjusts the count for the task's current status.
     *
     * @param task The task whose status or addition triggers the statistics update.
     */
=======
>>>>>>> pr458merge
    /**
     * Updates the aggregated task statistics to reflect the addition or status change of the specified task.
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
