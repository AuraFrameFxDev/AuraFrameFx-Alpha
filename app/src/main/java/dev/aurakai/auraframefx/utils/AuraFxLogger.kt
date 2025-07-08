package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting purposes.
 *
 * @param tag Identifies the source or component generating the log entry.
 * @param message The debug message to record.
 * @param throwable An optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message indicating normal application operation.
 *
 * @param tag Identifier for the log source or component.
 * @param message The informational message to log.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message indicating a potential issue or abnormal condition.
 *
 * @param tag Identifies the category or source of the warning.
 * @param message The warning message to log.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message indicating a failure or exception.
 *
 * @param tag Identifies the category or source of the error.
 * @param message Describes the error or failure.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event that requires immediate attention.
 *
 * @param tag Identifies the source or category of the security event.
 * @param message Describes the security incident or concern.
 * @param throwable Optional exception or error related to the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, including its duration and optional contextual metadata.
     *
     * @param tag Identifier for the source or component generating the log entry.
     * @param operation Name or description of the operation being measured.
     * @param durationMs Duration of the operation in milliseconds.
     * @param metadata Optional map with additional context for the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Logs a user interaction event for analytics and user experience monitoring.
 *
 * @param tag The category or component associated with the interaction.
 * @param action The specific user action performed.
 * @param metadata Optional contextual details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event with the specified operation name, confidence score, and optional metadata.
     *
     * Use this method to record AI-related activities for monitoring, analytics, or auditing purposes.
     *
     * @param tag The category or component associated with the AI operation.
     * @param operation The name or description of the AI operation performed.
     * @param confidence The confidence score of the AI result.
     * @param metadata Optional additional details about the operation.
     */
    fun aiOperation(
        tag: String,
        operation: String,
        confidence: Float,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Enables or disables the logging system at runtime.
 *
 * @param enabled If true, logging is enabled; if false, all logging is suppressed.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for logging.
 *
 * Only log entries at or above the specified level will be recorded; lower-priority entries are ignored.
 *
 * @param level The minimum log level to process.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Suspends execution to write all buffered log entries to persistent storage.
 *
 * Ensures that any pending log data is flushed, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Releases all resources and shuts down the logging system.
 *
 * Call this method to perform cleanup when the logger is no longer needed.
 */
    fun cleanup()
}

/**
 * Log levels for filtering and prioritization.
 */
enum class LogLevel(val priority: Int) {
    DEBUG(0),
    INFO(1),
    WARN(2),
    ERROR(3),
    SECURITY(4)
}
