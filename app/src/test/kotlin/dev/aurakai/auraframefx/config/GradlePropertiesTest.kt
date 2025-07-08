package dev.aurakai.auraframefx.config

import java.io.File
import java.util.concurrent.ConcurrentHashMap

class GradleProperties(val filePath: String) {

    private val properties = ConcurrentHashMap<String, String>()

    init {
        if (filePath.isBlank()) {
            throw IllegalArgumentException("File path must not be null or empty")
        }
    }

    fun loadProperties() {
        properties.clear()
        try {
            val file = File(filePath)
            if (!file.exists()) return
            file.forEachLine { line ->
                val index = line.indexOf('=')
                if (index >= 0) {
                    val key = line.substring(0, index)
                    val value = line.substring(index + 1)
                    properties[key] = value
                }
            }
        } catch (_: Exception) {
        }
    }

    fun getProperty(key: String, defaultValue: String? = null): String? {
        if (key.isBlank()) return null
        return properties[key] ?: defaultValue
    }

    fun setProperty(key: String, value: String?) {
        if (key.isBlank()) {
            throw IllegalArgumentException("Property key must not be null or empty")
        }
        if (value == null) {
            properties.remove(key)
        } else {
            properties[key] = value
        }
    }

    fun removeProperty(key: String) {
        if (key.isBlank()) {
            throw IllegalArgumentException("Property key must not be null or empty")
        }
        properties.remove(key)
    }

    fun containsProperty(key: String): Boolean {
        if (key.isBlank()) return false
        return properties.containsKey(key)
    }

    fun saveProperties() {
        try {
            val file = File(filePath)
            file.parentFile?.let { parent ->
                if (!parent.exists()) {
                    parent.mkdirs()
                }
            }
            file.bufferedWriter().use { writer ->
                for ((key, value) in properties) {
                    if (value == null) {
                        writer.write(key)
                    } else {
                        writer.write("$key=$value")
                    }
                    writer.newLine()
                }
            }
        } catch (_: Exception) {
        }
    }

    fun size(): Int = properties.size

    fun isEmpty(): Boolean = properties.isEmpty()

    fun clear() {
        properties.clear()
    }

    fun getAllKeys(): Set<String> = properties.keys

    fun getAllValues(): Collection<String> = properties.values
}