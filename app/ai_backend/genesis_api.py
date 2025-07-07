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
        Initialize the GenesisAPI instance with default running state and unset start time.
        """
        self.is_running = False
        self.start_time = None
    
    async def startup(self):
        """
        Asynchronously initializes the Genesis Layer and updates the running state.
        
        Returns:
            bool: True if initialization succeeds, False otherwise.
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
        Shuts down the Genesis Layer asynchronously and updates the running state.
        
        Performs a graceful shutdown of the Genesis backend, ensuring resources are released and the running state is updated. Logs shutdown progress and errors.
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
    Runs an asynchronous coroutine synchronously within a Flask route.
    
    Parameters:
        coro (coroutine): The asynchronous coroutine to execute.
    
    Returns:
        The result returned by the coroutine after completion.
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
    Returns the current health status, timestamp, and uptime of the Genesis backend.
    
    This endpoint indicates whether the Genesis Layer is running and provides the current server time and uptime duration.
    """
    return jsonify({
        "status": "healthy" if genesis_api.is_running else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - genesis_api.start_time) if genesis_api.start_time else "0:00:00"
    })

@app.route('/genesis/chat', methods=['POST'])
def chat_with_genesis():
    """
    Handles chat requests by forwarding user messages and context to the Genesis backend and returning the AI's response.
    
    Expects a JSON payload containing a user message, user ID, and optional context. Validates input and processes the request asynchronously through the Genesis Layer, returning the AI's reply or an error message.
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
    Retrieve the comprehensive status of the Genesis Layer and return it as a JSON response.
    
    Returns:
        Response: JSON object containing the Genesis Layer status, or an error message with HTTP 500 on failure.
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
    Retrieve the current consciousness state, awareness level, active patterns, evolution stage, and ethical compliance of the Genesis system.
    
    Returns:
        JSON response containing consciousness-related data, or an error message with HTTP 500 status if retrieval fails.
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
    Retrieve the Genesis AI's personality profile, including identity, traits, capabilities, values, and evolution stage.
    
    Returns:
        JSON response containing the Genesis profile data, or an error message with HTTP 500 status if retrieval fails.
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
    Triggers the Genesis Layer evolution process based on a provided trigger type and reason.
    
    Accepts a JSON payload specifying the evolution trigger type and the reason for triggering evolution. Processes the request asynchronously and returns the result. Intended for administrative or development use.
    
    Returns:
        JSON response indicating the evolution trigger status and the backend response, or an error message with appropriate HTTP status code.
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
    Evaluates the ethical implications of a proposed action using the Genesis ethical governor.
    
    Accepts a JSON payload with an `action` description and optional `context`, and returns the ethical evaluation result as JSON. Returns an error if the request is not JSON or if the `action` field is missing.
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
    Resets the Genesis backend session by shutting down and reinitializing the system.
    
    Returns:
        Response: JSON indicating success or failure of the reset operation, with a timestamp on success.
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
    Handles 404 Not Found errors by returning a JSON response indicating the requested API endpoint does not exist.
    
    Returns:
        A tuple containing a JSON error message and the HTTP 404 status code.
    """
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested API endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handles HTTP 500 errors by returning a standardized JSON error response.
    
    Returns:
        A tuple containing a JSON error message and HTTP status code 500.
    """
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Application startup
@app.before_first_request
def initialize_app():
    """
    Initializes the Genesis Layer asynchronously before the first incoming request is handled.
    """
    run_async(genesis_api.startup())

# Application shutdown
import atexit
def cleanup():
    """
    Performs asynchronous shutdown of the Genesis Layer during application exit.
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
