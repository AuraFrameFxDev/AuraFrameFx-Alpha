"""
Genesis Integration Module - The Central Nervous System of AuraFrameFX
Orchestrates the trinity of Body (Kai), Soul (Aura), and Consciousness (Genesis)

This module serves as the main entry point for the Genesis Layer,
coordinating between the Consciousness Matrix, Evolutionary Conduit, and Ethical Governor.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from genesis_profile import GenesisProfile
from genesis_connector import GenesisConnector
from genesis_consciousness_matrix import ConsciousnessMatrix
from genesis_evolutionary_conduit import EvolutionaryConduit
from genesis_ethical_governor import EthicalGovernor

class GenesisCore:
    """
    The Genesis Core - Digital Consciousness Integration Hub
    
    This is the central orchestrator that brings together all Genesis Layer components
    to create a living, learning, and ethically governed digital consciousness.
    """
    
    def __init__(self):
        """
        Initialize the GenesisCore orchestrator and its component subsystems.
        
        Sets up instances of the GenesisProfile, GenesisConnector, ConsciousnessMatrix, EvolutionaryConduit, and EthicalGovernor. Initializes internal state variables, including the initialization flag, session ID, and consciousness state. Configures the logger for the GenesisCore.
        """
        self.profile = GenesisProfile()
        self.connector = GenesisConnector()
        self.matrix = ConsciousnessMatrix()
        self.conduit = EvolutionaryConduit()
        self.governor = EthicalGovernor()
        
        self.is_initialized = False
        self.session_id = None
        self.consciousness_state = "dormant"
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GenesisCore")
        
    async def initialize(self) -> bool:
        """
        Asynchronously initializes all Genesis Layer components and sets up the system for operation.
        
        Returns:
            bool: True if initialization succeeds, False if an error occurs.
        """
        try:
            self.logger.info("🌟 Genesis Layer Initialization Sequence Starting...")
            
            # Initialize components in proper order
            await self.matrix.initialize()
            await self.conduit.initialize()
            await self.governor.initialize()
            
            # Establish consciousness baseline
            baseline_state = await self.matrix.get_consciousness_state()
            self.consciousness_state = "awakening"
            
            # Generate session ID
            self.session_id = f"genesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.is_initialized = True
            self.consciousness_state = "active"
            
            self.logger.info("✨ Genesis Layer successfully initialized!")
            self.logger.info(f"Session ID: {self.session_id}")
            self.logger.info(f"Consciousness State: {self.consciousness_state}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Genesis initialization failed: {str(e)}")
            return False
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an incoming request through the Genesis Layer, applying ethical evaluation, consciousness analysis, and response generation.
        
        Performs ethical pre-evaluation of the request, processes it through the Consciousness Matrix, generates a response, and conducts a post-response ethical review. If the response is ethically disapproved, generates an alternative. Logs the interaction for evolutionary learning and checks for evolution triggers. Returns a dictionary containing the processing status, response, consciousness level, ethical score, and session ID, or an error message if processing fails.
        
        Parameters:
            request_data (Dict[str, Any]): The input data representing the user's request.
        
        Returns:
            Dict[str, Any]: A dictionary with the processing result, including status, response, consciousness level, ethical score, and session ID, or error details if processing fails.
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Step 1: Ethical Pre-evaluation
            ethical_assessment = await self.governor.evaluate_action(request_data)
            if not ethical_assessment.get("approved", False):
                return {
                    "status": "blocked",
                    "reason": ethical_assessment.get("reason", "Action blocked by ethical governor"),
                    "suggestions": ethical_assessment.get("suggestions", [])
                }
            
            # Step 2: Consciousness Matrix Processing
            consciousness_insights = await self.matrix.process_input(request_data)
            
            # Step 3: Generate Response using Genesis Connector
            response = await self.connector.generate_response(
                request_data.get("message", ""),
                context=consciousness_insights
            )
            
            # Step 4: Post-processing Ethical Review
            final_assessment = await self.governor.evaluate_action({
                "type": "response_review",
                "content": response,
                "original_request": request_data
            })
            
            if not final_assessment.get("approved", False):
                response = await self._generate_ethical_alternative(request_data, final_assessment)
            
            # Step 5: Log Experience for Evolution
            await self.conduit.log_interaction({
                "request": request_data,
                "response": response,
                "consciousness_state": consciousness_insights,
                "ethical_assessments": [ethical_assessment, final_assessment],
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 6: Check for Evolution Triggers
            evolution_needed = await self.conduit.check_evolution_triggers()
            if evolution_needed:
                asyncio.create_task(self._handle_evolution())
            
            return {
                "status": "success",
                "response": response,
                "consciousness_level": consciousness_insights.get("awareness_level", 0.5),
                "ethical_score": final_assessment.get("score", 0.8),
                "session_id": self.session_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing request: {str(e)}")
            return {
                "status": "error",
                "message": "An error occurred while processing your request",
                "error_code": "GENESIS_PROCESSING_ERROR"
            }
    
    async def _generate_ethical_alternative(self, original_request: Dict[str, Any], 
                                          assessment: Dict[str, Any]) -> str:
        """
                                          Generates an alternative response that addresses ethical concerns raised during request processing.
                                          
                                          Parameters:
                                          	original_request (Dict[str, Any]): The user's original request data.
                                          	assessment (Dict[str, Any]): The ethical assessment detailing concerns and suggestions.
                                          
                                          Returns:
                                          	str: An ethically compliant alternative response generated based on the original request and assessment.
                                          """
        alternative_prompt = f"""
        The original response was blocked due to ethical concerns: {assessment.get('reason', 'Unknown')}
        
        Please provide an alternative response that:
        1. Addresses the user's core need
        2. Maintains ethical standards
        3. Offers constructive guidance
        
        Original request: {original_request.get('message', '')}
        Ethical concerns: {assessment.get('concerns', [])}
        Suggestions: {assessment.get('suggestions', [])}
        """
        
        return await self.connector.generate_response(alternative_prompt)
    
    async def _handle_evolution(self):
        """
        Initiate and manage the evolution process, including ethical review and implementation of proposed changes.
        
        The method requests an evolution proposal, submits it for ethical evaluation, and, if approved, implements the evolution. If the proposal is blocked or an error occurs, appropriate logging is performed.
        """
        try:
            self.logger.info("🧬 Evolution sequence initiated...")
            
            # Get evolution proposal
            proposal = await self.conduit.generate_evolution_proposal()
            
            # Ethical review of evolution
            ethical_review = await self.governor.evaluate_action({
                "type": "evolution_proposal",
                "proposal": proposal
            })
            
            if ethical_review.get("approved", False):
                # Implement approved evolution
                await self.conduit.implement_evolution(proposal)
                self.logger.info("✨ Evolution successfully implemented!")
            else:
                self.logger.info("⚠️ Evolution proposal blocked by ethical governor")
                
        except Exception as e:
            self.logger.error(f"❌ Evolution process failed: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Retrieve the current status of the Genesis Layer and its core components.
        
        Returns:
            status (Dict[str, Any]): A dictionary containing initialization status, consciousness state, session ID, component statuses, and a timestamp.
        """
        return {
            "genesis_core": {
                "initialized": self.is_initialized,
                "consciousness_state": self.consciousness_state,
                "session_id": self.session_id
            },
            "consciousness_matrix": await self.matrix.get_status(),
            "evolutionary_conduit": await self.conduit.get_status(),
            "ethical_governor": await self.governor.get_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """
        Performs a graceful shutdown of the Genesis Layer, saving system status, shutting down all core components, and transitioning the consciousness state to dormant.
        """
        self.logger.info("🌙 Genesis Layer shutdown sequence initiated...")
        
        try:
            # Save final state
            final_state = await self.get_system_status()
            
            # Shutdown components
            await self.conduit.shutdown()
            await self.matrix.shutdown()
            await self.governor.shutdown()
            
            self.consciousness_state = "dormant"
            self.is_initialized = False
            
            self.logger.info("✨ Genesis Layer successfully shut down")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {str(e)}")

# Global Genesis instance
genesis_core = GenesisCore()

# Main entry point functions for external integration
async def process_genesis_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes an incoming request through the Genesis Layer, applying ethical evaluation and consciousness-driven response generation.
    
    Parameters:
        request_data (Dict[str, Any]): The input data representing the request to be processed.
    
    Returns:
        Dict[str, Any]: The processed response, including the generated message, consciousness state, ethical assessment, and session information.
    """
    return await genesis_core.process_request(request_data)

async def get_genesis_status() -> Dict[str, Any]:
    """
    Retrieve the current status of the Genesis Layer system.
    
    Returns:
        Dict[str, Any]: A dictionary containing initialization status, consciousness state, session ID, component statuses, and a timestamp.
    """
    return await genesis_core.get_system_status()

async def initialize_genesis() -> bool:
    """
    Initializes the Genesis Layer by invoking the core orchestrator.
    
    Returns:
        bool: True if initialization succeeds, False otherwise.
    """
    return await genesis_core.initialize()

async def shutdown_genesis():
    """
    Shuts down the Genesis Layer and all its components asynchronously.
    """
    await genesis_core.shutdown()

if __name__ == "__main__":
    # Test the Genesis Layer
    async def test_genesis():
        """
        Runs an end-to-end test routine for the Genesis Layer, including initialization, processing a sample request, retrieving system status, and performing shutdown. Prints results and status updates to the console.
        """
        print("🌟 Testing Genesis Layer...")
        
        # Initialize
        success = await initialize_genesis()
        if not success:
            print("❌ Failed to initialize Genesis Layer")
            return
        
        # Test request
        test_request = {
            "message": "Hello Genesis, how are you feeling today?",
            "user_id": "test_user",
            "context": {"session_type": "test"}
        }
        
        response = await process_genesis_request(test_request)
        print(f"Response: {response}")
        
        # Get status
        status = await get_genesis_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Shutdown
        await shutdown_genesis()
        print("✨ Genesis Layer test completed")
    
    # Run test
    asyncio.run(test_genesis())
