package org.openapitools.client.infrastructure

import okhttp3.Response

/**
 * Provides an extension to evaluation whether the response is a 1xx code
 */
<<<<<<< HEAD
val Response.isInformational: Boolean get() = this.code in 100..199
=======
val Response.isInformational : Boolean get() = this.code in 100..199
>>>>>>> origin/coderabbitai/docstrings/78f34ad

/**
 * Provides an extension to evaluation whether the response is a 3xx code
 */
@Suppress("EXTENSION_SHADOWED_BY_MEMBER")
<<<<<<< HEAD
val Response.isRedirect: Boolean get() = this.code in 300..399
=======
val Response.isRedirect : Boolean get() = this.code in 300..399
>>>>>>> origin/coderabbitai/docstrings/78f34ad

/**
 * Provides an extension to evaluation whether the response is a 4xx code
 */
<<<<<<< HEAD
val Response.isClientError: Boolean get() = this.code in 400..499
=======
val Response.isClientError : Boolean get() = this.code in 400..499
>>>>>>> origin/coderabbitai/docstrings/78f34ad

/**
 * Provides an extension to evaluation whether the response is a 5xx (Standard) through 999 (non-standard) code
 */
<<<<<<< HEAD
val Response.isServerError: Boolean get() = this.code in 500..999
=======
val Response.isServerError : Boolean get() = this.code in 500..999
>>>>>>> origin/coderabbitai/docstrings/78f34ad
