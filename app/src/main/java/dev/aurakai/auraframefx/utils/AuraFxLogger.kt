package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {

    /**
 * Logs a debug-level message for development and troubleshooting purposes.
 *
 * @param tag Identifier for the source or component of the log entry.
 * @param message The message to be logged.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an informational message that represents normal application operation.
 *
 * @param tag Identifies the source or component of the log entry.
 * @param message The informational message to record.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a warning message indicating a potential issue or abnormal condition.
 *
 * @param tag Identifies the source or component of the log entry.
 * @param message The warning message content.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message indicating a failure or exception.
 *
 * @param tag Identifies the source or component of the error.
 * @param message Describes the error or failure.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a critical security event requiring immediate attention.
 *
 * @param tag Identifies the source or component of the security event.
 * @param message The security-related message to log.
 * @param throwable Optional exception or error associated with the event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)

    /**
     * Logs a performance event for a specific operation, including its duration and optional contextual metadata.
     *
     * @param tag Identifies the source or component of the log entry.
     * @param operation The name or description of the operation being measured.
     * @param durationMs The duration of the operation in milliseconds.
     * @param metadata Optional map providing additional context for the performance event.
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
 * @param tag Identifies the source or component of the interaction.
 * @param action The specific user action to record.
 * @param metadata Optional contextual details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event, including the operation name, confidence score, and optional metadata for monitoring or analytics.
     *
     * @param tag Identifier for the category or component related to the AI operation.
     * @param operation Name or description of the AI operation performed.
     * @param confidence Confidence score associated with the AI result.
     * @param metadata Optional additional context or details about the operation.
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
 * @param enabled True to enable logging; false to disable all log output.
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
 * Suspends execution to flush all buffered log entries to persistent storage.
 *
 * Ensures that any pending log data is written, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Releases all resources and terminates the logging system.
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
