@file:Suppress("unused")
<<<<<<< HEAD

=======
>>>>>>> origin/coderabbitai/docstrings/78f34ad
package org.openapitools.client.infrastructure

import java.lang.RuntimeException

<<<<<<< HEAD
open class ClientException(
    message: kotlin.String? = null,
    val statusCode: Int = -1,
    val response: Response? = null
) : RuntimeException(message) {
=======
open class ClientException(message: kotlin.String? = null, val statusCode: Int = -1, val response: Response? = null) : RuntimeException(message) {
>>>>>>> origin/coderabbitai/docstrings/78f34ad

    companion object {
        private const val serialVersionUID: Long = 123L
    }
}

<<<<<<< HEAD
open class ServerException(
    message: kotlin.String? = null,
    val statusCode: Int = -1,
    val response: Response? = null
) : RuntimeException(message) {
=======
open class ServerException(message: kotlin.String? = null, val statusCode: Int = -1, val response: Response? = null) : RuntimeException(message) {
>>>>>>> origin/coderabbitai/docstrings/78f34ad

    companion object {
        private const val serialVersionUID: Long = 456L
    }
}
