package dev.aurakai.auraframefx.utils

/**
 * Interface for AuraFrameFX logging system.
 * Provides structured logging with security awareness and performance monitoring.
 */
interface AuraFxLogger {
    
    /**
 * Logs a debug-level message for development and troubleshooting.
 *
 * @param tag The source or component generating the log entry.
 * @param message The debug message to log.
 * @param throwable Optional exception or error associated with the log entry.
 */
    fun debug(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an informational message indicating normal application activity.
 *
 * @param tag Identifies the source or category of the log entry.
 * @param message The informational message to log.
 * @param throwable Optional exception or error associated with the event.
 */
    fun info(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a warning message indicating a potential issue or abnormal situation.
 *
 * @param tag Identifies the source or category of the warning.
 * @param message Describes the warning condition.
 * @param throwable An optional exception or error associated with the warning.
 */
    fun warn(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs an error-level message to record failures or exceptions within the application.
 *
 * @param tag Identifies the source or category of the error event.
 * @param message Describes the error condition.
 * @param throwable An optional exception or error associated with the failure.
 */
    fun error(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a security-related event, such as a critical issue or potential threat.
 *
 * @param tag Identifies the source or category of the security event.
 * @param message Describes the security incident or concern.
 * @param throwable An optional exception related to the security event.
 */
    fun security(tag: String, message: String, throwable: Throwable? = null)
    
    /**
 * Logs a performance event for an operation, capturing its duration and optional metadata.
 *
 * @param tag Identifies the source or category of the performance event.
 * @param operation The name or description of the measured operation.
 * @param durationMs The duration of the operation in milliseconds.
 * @param metadata Additional contextual information about the operation.
 */
    fun performance(tag: String, operation: String, durationMs: Long, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Records a user interaction event for analytics and tracking purposes.
 *
 * @param tag Identifies the category or source of the interaction.
 * @param action Describes the specific user action performed.
 * @param metadata Optional contextual data providing additional details about the interaction.
 */
    fun userInteraction(tag: String, action: String, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Logs an AI operation event with its name, confidence score, and optional metadata for monitoring or auditing AI-driven actions.
 *
 * @param tag Identifier for the source or component generating the log entry.
 * @param operation The name or description of the AI operation performed.
 * @param confidence The confidence score representing the certainty of the AI operation.
 * @param metadata Optional contextual information about the AI operation.
 */
    fun aiOperation(tag: String, operation: String, confidence: Float, metadata: Map<String, Any> = emptyMap())
    
    /**
 * Enables or disables logging at runtime.
 *
 * When logging is disabled, all log messages are ignored until re-enabled.
 *
 * @param enabled If true, logging is enabled; if false, logging is disabled.
 */
    fun setLoggingEnabled(enabled: Boolean)
    
    /**
 * Sets the minimum severity level for log messages to be recorded.
 *
 * Log entries below the specified level will be ignored.
 *
 * @param level The minimum log level to enable.
 */
    fun setLogLevel(level: LogLevel)
    
    /**
 * Suspends execution to persist all buffered or pending log entries.
 *
 * Ensures that all log data up to the point of invocation is written to durable storage, preventing loss in case of failure.
 */
    suspend fun flush()
    
    /**
 * Releases resources and shuts down all logging operations.
 *
 * Call this method to ensure proper cleanup when the logger is no longer needed.
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
