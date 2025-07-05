package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {
    
    /**
 * Logs a debug-level message for development and troubleshooting.
 *
 * @param tag Identifier for the log source or component.
 * @param message The debug message to log.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an informational message indicating normal application operation.
 *
 * @param tag Identifier for the source or category of the log entry.
 * @param message The informational message to record.
 * @param throwable Optional exception or error to include with the log entry.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a warning message to indicate a potential issue or abnormal situation.
 *
 * @param tag Identifies the category or source of the warning.
 * @param message The warning message describing the potential issue.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an error message to indicate a failure or exception in the application.
 *
 * @param tag Identifies the category or source of the error.
 * @param message Describes the error condition.
 * @param throwable An optional exception or error object associated with the failure.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a security event indicating a critical issue or potential threat.
 *
 * @param tag Identifies the source or category of the security event.
 * @param message Describes the security incident or concern.
 * @param throwable An optional exception or error related to the security event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs performance metrics for a given operation, including its duration and optional metadata.
 *
 * @param tag Identifies the source or category of the log entry.
 * @param operation Describes the operation being measured.
 * @param durationMs The duration of the operation in milliseconds.
 * @param metadata Additional contextual data about the operation, if any.
 */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Records a user interaction event for analytics and user experience improvement.
 *
 * @param tag Identifies the category or source of the interaction.
 * @param action Describes the specific user action being logged.
 * @param metadata Optional map containing additional contextual data about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs an AI operation with its name, confidence score, and optional metadata for monitoring or auditing AI-driven actions.
 *
 * @param tag Identifier for the source or component generating the log.
 * @param operation Name or description of the AI operation performed.
 * @param confidence Confidence score representing the certainty of the AI operation.
 * @param metadata Optional map containing additional contextual information about the AI operation.
 */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Enables or disables logging at runtime.
 *
 * @param enabled If true, logging is enabled; if false, logging is disabled.
 */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
 * Sets the minimum log level for which log entries will be recorded.
 *
 * Only messages at the specified level or higher will be logged; lower-priority messages are ignored.
 *
 * @param level The minimum log level to enable.
 */
    fun setLogLevel(level: LogLevel)
    
    /**
 * Immediately writes all buffered or pending log entries to persistent storage.
 *
 * This suspend function ensures that any in-memory or queued log data is flushed, guaranteeing durability of all logs up to the point of invocation.
 */
    suspend fun flush()
    
    /**
 * Releases any resources held by the logger and terminates logging operations.
 *
 * Call this method to properly shut down the logger and free associated resources when logging is no longer needed.
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
