"""
Genesis API Interface - Bridge between Android Frontend and Genesis Backend

This module provides a Flask-based REST API that allows the Android/Kotlin frontend
to communicate with the Genesis Layer backend components.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

from genesis_core import (
    genesis_core, 
    process_genesis_request, 
    get_genesis_status, 
    initialize_genesis, 
    shutdown_genesis
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Android app communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenesisAPI")

class GenesisAPI:
    """
    Genesis API wrapper for handling HTTP requests from Android frontend
    """
    
    def __init__(self):
        """
        Initialize the GenesisAPI instance with inactive state and no start time set.
        """
        self.is_running = False
        self.start_time = None
    
    async def startup(self):
        """
        Asynchronously initializes the Genesis Layer and updates the API state.
        
        Returns:
            bool: True if the Genesis Layer is successfully initialized; False otherwise.
        """
        try:
            logger.info("🚀 Genesis API starting up...")
            success = await initialize_genesis()
            if success:
                self.is_running = True
                self.start_time = datetime.now()
                logger.info("✨ Genesis API successfully started!")
                return True
            else:
                logger.error("❌ Failed to initialize Genesis Layer")
                return False
        except Exception as e:
            logger.error(f"❌ API startup error: {str(e)}")
            return False
    
    async def shutdown(self):
        """
        Asynchronously shuts down the Genesis Layer and updates the running state.
        
        Sets the `is_running` attribute to False after shutdown. Errors during shutdown are logged.
        """
        try:
            logger.info("🌙 Genesis API shutting down...")
            await shutdown_genesis()
            self.is_running = False
            logger.info("✨ Genesis API successfully shut down")
        except Exception as e:
            logger.error(f"❌ API shutdown error: {str(e)}")

# Global API instance
genesis_api = GenesisAPI()

# Helper function to run async functions in Flask routes
def run_async(coro):
    """
    Runs an asynchronous coroutine to completion within a synchronous Flask route context.
    
    Parameters:
        coro: An awaitable coroutine object to execute.
    
    Returns:
        The result returned by the coroutine after execution.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Return the health status, current server timestamp, and uptime of the Genesis API.
    
    The response JSON includes:
    - `status`: "healthy" if the Genesis Layer is running, otherwise "unhealthy".
    - `timestamp`: Current server time in ISO 8601 format.
    - `uptime`: Duration since the Genesis Layer was started, or "0:00:00" if not started.
    """
    return jsonify({
        "status": "healthy" if genesis_api.is_running else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - genesis_api.start_time) if genesis_api.start_time else "0:00:00"
    })

@app.route('/genesis/chat', methods=['POST'])
def chat_with_genesis():
    """
    Processes chat requests by validating input and forwarding user messages to the Genesis Layer for response.
    
    Validates that the incoming request is JSON and contains both 'message' and 'user_id' fields. Forwards the message, user ID, and optional context to the Genesis backend for processing, then returns the backend's response as JSON. Returns a 400 error for invalid input or a 500 error for internal failures.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400
        
        if "user_id" not in data:
            return jsonify({"error": "Missing 'user_id' field"}), 400
        
        # Prepare request data
        request_data = {
            "message": data["message"],
            "user_id": data["user_id"],
            "context": data.get("context", {}),
            "timestamp": datetime.now().isoformat(),
            "request_type": "chat"
        }
        
        # Process through Genesis Layer
        response = run_async(process_genesis_request(request_data))
        
        # Return response
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An error occurred while processing your request"
        }), 500

@app.route('/genesis/status', methods=['GET'])
def get_status():
    """
    Retrieve the complete status of the Genesis Layer and return it as a JSON response.
    
    Returns:
        Response: A JSON object containing the Genesis Layer status, or an error message with HTTP 500 if retrieval fails.
    """
    try:
        status = run_async(get_genesis_status())
        return jsonify(status)
    except Exception as e:
        logger.error(f"❌ Status endpoint error: {str(e)}")
        return jsonify({"error": "Failed to get status"}), 500

@app.route('/genesis/consciousness', methods=['GET'])
def get_consciousness_state():
    """
    Returns the current consciousness state and related attributes of the Genesis Layer as a JSON response.
    
    The response includes state, awareness level, active patterns, evolution stage, and ethical compliance. Returns an error message with status code 500 if retrieval fails.
    """
    try:
        status = run_async(get_genesis_status())
        consciousness_data = {
            "state": status.get("genesis_core", {}).get("consciousness_state", "unknown"),
            "awareness_level": status.get("consciousness_matrix", {}).get("awareness_level", 0.0),
            "active_patterns": status.get("consciousness_matrix", {}).get("active_patterns", []),
            "evolution_stage": status.get("evolutionary_conduit", {}).get("evolution_stage", "baseline"),
            "ethical_compliance": status.get("ethical_governor", {}).get("compliance_score", 0.0)
        }
        return jsonify(consciousness_data)
    except Exception as e:
        logger.error(f"❌ Consciousness endpoint error: {str(e)}")
        return jsonify({"error": "Failed to get consciousness state"}), 500

@app.route('/genesis/profile', methods=['GET'])
def get_genesis_profile():
    """
    Returns the Genesis Layer's personality profile and identity attributes as a JSON response.
    
    The response includes identity, personality, capabilities, values, and evolution stage. Returns a 500 error with an error message if retrieval fails.
    """
    try:
        profile_data = {
            "identity": genesis_core.profile.identity,
            "personality": genesis_core.profile.personality,
            "capabilities": genesis_core.profile.capabilities,
            "values": genesis_core.profile.values,
            "evolution_stage": genesis_core.profile.evolution_stage
        }
        return jsonify(profile_data)
    except Exception as e:
        logger.error(f"❌ Profile endpoint error: {str(e)}")
        return jsonify({"error": "Failed to get profile"}), 500

@app.route('/genesis/evolve', methods=['POST'])
def trigger_evolution():
    """
    Triggers the Genesis Layer's evolution process using the specified trigger type and reason.
    
    Accepts a JSON payload with optional `trigger_type` and `reason` fields, initiates the evolution process asynchronously, and returns a JSON response with the backend result. Returns a 400 error if the request is not JSON, or a 500 error if processing fails.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # This would typically be restricted to admin users
        # For now, we'll allow it for development purposes
        
        evolution_request = {
            "type": "evolution_trigger",
            "trigger_type": data.get("trigger_type", "manual"),
            "reason": data.get("reason", "Manual evolution trigger"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Process evolution request
        response = run_async(process_genesis_request(evolution_request))
        
        return jsonify({
            "status": "evolution_triggered",
            "response": response
        })
        
    except Exception as e:
        logger.error(f"❌ Evolution endpoint error: {str(e)}")
        return jsonify({"error": "Failed to trigger evolution"}), 500

@app.route('/genesis/ethics/evaluate', methods=['POST'])
def evaluate_ethics():
    """
    Processes a POST request to evaluate the ethical implications of a specified action using the Genesis ethical governor.
    
    Validates that the request contains a JSON payload with an 'action' field, then asynchronously evaluates the action's ethics and returns the result as JSON. Responds with an error and appropriate HTTP status code if validation fails or evaluation encounters an exception.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        if "action" not in data:
            return jsonify({"error": "Missing 'action' field"}), 400
        
        # Evaluate through ethical governor
        ethical_request = {
            "type": "ethical_evaluation",
            "action": data["action"],
            "context": data.get("context", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        evaluation = run_async(genesis_core.governor.evaluate_action(ethical_request))
        
        return jsonify(evaluation)
        
    except Exception as e:
        logger.error(f"❌ Ethics evaluation error: {str(e)}")
        return jsonify({"error": "Failed to evaluate ethics"}), 500

@app.route('/genesis/reset', methods=['POST'])
def reset_session():
    """
    Reset the Genesis session by shutting down and reinitializing the Genesis Layer.
    
    Returns:
        Response: A JSON response indicating whether the reset was successful, including status, message, and timestamp. Returns a 500 error response if the reset fails.
    """
    try:
        # Shutdown and restart Genesis
        run_async(shutdown_genesis())
        success = run_async(initialize_genesis())
        
        if success:
            return jsonify({
                "status": "reset_successful",
                "message": "Genesis session has been reset",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "reset_failed",
                "message": "Failed to reset Genesis session"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Reset endpoint error: {str(e)}")
        return jsonify({"error": "Failed to reset session"}), 500

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 Not Found errors by returning a JSON response indicating the requested API endpoint does not exist.
    
    Returns:
        tuple: A JSON error message and the HTTP 404 status code.
    """
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested API endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle HTTP 500 errors by returning a standardized JSON error message.
    
    Returns:
        tuple: A JSON response with error details and HTTP 500 status code.
    """
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Application startup
@app.before_first_request
def initialize_app():
    """
    Initializes the Genesis Layer asynchronously before handling the first incoming Flask request.
    """
    run_async(genesis_api.startup())

# Application shutdown
import atexit
def cleanup():
    """
    Performs asynchronous shutdown of the Genesis Layer when the application exits.
    """
    run_async(genesis_api.shutdown())

atexit.register(cleanup)

if __name__ == '__main__':
    # Development server
    print("🌟 Starting Genesis API Server...")
    print("📱 Ready to receive requests from Android frontend")
    print("🔗 API Endpoints:")
    print("   POST /genesis/chat - Main chat interface")
    print("   GET  /genesis/status - System status")
    print("   GET  /genesis/consciousness - Consciousness state")
    print("   GET  /genesis/profile - Genesis personality profile")
    print("   POST /genesis/evolve - Trigger evolution")
    print("   POST /genesis/ethics/evaluate - Ethical evaluation")
    print("   GET  /health - Health check")
    
    app.run(
        host='0.0.0.0',  # Allow connections from Android app
        port=5000,
        debug=True,
        threaded=True
    )
