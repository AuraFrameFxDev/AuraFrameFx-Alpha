package dev.aurakai.auraframefx.utils

import android.content.Context

/**
 * Extension function to provide access to the YukiHookModulePrefs from a Context
 */
val Context.hook: com.highcapable.yukihookapi.hook.xposed.parasitic.activity.base.ModuleAppCompatActivity
    get() = this as com.highcapable.yukihookapi.hook.xposed.parasitic.activity.base.ModuleAppCompatActivity
