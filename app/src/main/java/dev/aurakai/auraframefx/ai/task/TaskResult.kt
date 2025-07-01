package dev.aurakai.auraframefx.ai.task

import kotlinx.serialization.Serializable

@Serializable
data class TaskResult(
    val taskId: String,
    val status: TaskStatus,
    val message: String? = null,
    val timestamp: Long = System.currentTimeMillis(),
    val durationMs: Long? = null, // How long the task took
)

// Removed local TaskStatus enum.
// The 'status: TaskStatus' field in TaskResult data class
// should now refer to TaskStatus from TaskModel.kt in the same package.
