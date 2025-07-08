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
 * @param message The debug message to log.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message representing standard application activity.
 *
 * @param tag Identifier for the source or component generating the log entry.
 * @param message The informational message to record.
 * @param throwable Optional exception or error to associate with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message indicating a potential issue or abnormal condition.
 *
 * @param tag The category or source of the warning.
 * @param message The warning message content.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message indicating a failure or exception.
 *
 * @param tag The category or source of the error event.
 * @param message The error message describing the failure or issue.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a security-related event that requires immediate attention.
 *
 * @param tag Identifies the category or source of the security event.
 * @param message Describes the security event.
 * @param throwable An optional exception or error related to the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for an operation, including its duration and optional metadata.
     *
     * @param tag Identifier for the source or component generating the log entry.
     * @param operation Name or description of the operation being measured.
     * @param durationMs Duration of the operation in milliseconds.
     * @param metadata Additional context for the performance event.
     */
    fun performance(
        tag: String,
        operation: String,
        durationMs: Long,
        metadata: Map<String, Any> = emptyMap()
    )

    /**
 * Logs a user interaction event for analytics and user experience tracking.
 *
 * @param tag The category or component associated with the event.
 * @param action The specific user action to log.
 * @param metadata Optional additional details providing context about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event with its name, confidence score, and optional metadata for analytics or tracking.
     *
     * @param tag The category or component associated with the AI operation.
     * @param operation The specific AI operation performed.
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
 * Toggles logging on or off during runtime.
 *
 * @param enabled If true, logging is enabled; if false, logging is disabled.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level for logging.
 *
 * Only log entries at or above the specified level are recorded; lower-priority entries are ignored.
 *
 * @param level The minimum log level to record.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Suspends execution to write all buffered log entries to persistent storage.
 *
 * Ensures that any pending log data is flushed, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Releases resources and terminates the logging system.
 *
 * Call this method to properly shut down the logger and free any associated resources.
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
