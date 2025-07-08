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
 * @param tag Identifies the source or category of the warning.
 * @param message The warning message to log.
 * @param throwable An optional exception or error related to the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs an error message indicating a failure or exception.
 *
 * @param tag The category or source of the error.
 * @param message The error message describing the failure.
 * @param throwable An optional exception associated with the error.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)

    /**
 * Logs a security event that requires immediate attention.
 *
 * @param tag Identifies the source or category of the security event.
 * @param message Describes the security-related incident.
 * @param throwable An optional exception or error related to the event.
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
 * @param action The specific user action to record.
 * @param metadata Optional contextual details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())

    /**
     * Logs an AI operation event, including the operation name, confidence score, and optional metadata for monitoring or analytics purposes.
     *
     * @param tag The category or component associated with the AI operation.
     * @param operation The name or description of the AI operation performed.
     * @param confidence The confidence score of the AI result.
     * @param metadata Optional additional details or context about the operation.
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
 * @param enabled If true, logging is enabled; if false, logging is disabled.
 */
    fun setLoggingEnabled(enabled: Boolean)

    /**
 * Sets the minimum log level threshold for logging.
 *
 * Only log entries at or above the specified level will be recorded; entries below this level are ignored.
 *
 * @param level The minimum log level to enable.
 */
    fun setLogLevel(level: LogLevel)

    /**
 * Flushes all buffered log entries to persistent storage.
 *
 * This suspend function ensures that any pending log data is written, which may involve I/O operations.
 */
    suspend fun flush()

    /**
 * Releases all resources and shuts down the logging system.
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
