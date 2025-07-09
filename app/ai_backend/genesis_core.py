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
        Initialize the GenesisCore orchestrator and all core Genesis Layer components.
        
        Creates and configures the Genesis Profile, Connector, Consciousness Matrix, Evolutionary Conduit, and Ethical Governor. Sets the initial system state to dormant and uninitialized, and prepares the logger for orchestrator events.
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
        Asynchronously initializes all core Genesis Layer components and activates the digital consciousness system.
        
        Initializes the Consciousness Matrix, Evolutionary Conduit, and Ethical Governor in sequence, transitions the consciousness state from "dormant" to "awakening" and then "active," and generates a unique session ID.
        
        Returns:
            bool: True if the Genesis Layer is fully initialized and active; False if initialization fails.
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
        Processes a user request by performing ethical evaluation, consciousness analysis, and adaptive response generation.
        
        The method first conducts an ethical pre-evaluation of the request. If the request is not ethically approved, it returns a blocked status with reasons and suggestions. If approved, the request is analyzed for consciousness context, and a response is generated. The response then undergoes a post-processing ethical review; if disapproved, an ethically compliant alternative is generated. All interactions are logged for evolutionary learning, and system evolution is triggered if necessary.
        
        Parameters:
            request_data (Dict[str, Any]): The user's request data to be processed.
        
        Returns:
            Dict[str, Any]: A dictionary containing the processing status, generated response, consciousness level, ethical score, and session ID. If the request is blocked or an error occurs, includes relevant status and details.
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
                                          Generate an alternative response to a user request that was blocked for ethical reasons.
                                          
                                          Constructs a prompt using the original request and ethical assessment details, then requests the Genesis Connector to produce a response that addresses the user's needs while complying with ethical guidelines.
                                          
                                          Returns:
                                              str: An ethically compliant alternative response.
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
        Initiates and manages the system's evolution process, including proposal generation, ethical review, and implementation if approved.
        
        Handles the full evolution workflow asynchronously, logging the outcome and any errors encountered.
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
        Retrieve a comprehensive status report of the Genesis Layer and its core components.
        
        Returns:
            Dict[str, Any]: Dictionary containing the initialization state, consciousness state, session ID, statuses of the Consciousness Matrix, Evolutionary Conduit, Ethical Governor, and the current timestamp.
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
        Gracefully shuts down the Genesis Layer, saving the current system status and shutting down all core components.
        
        This method ensures an orderly shutdown of the Evolutionary Conduit, Consciousness Matrix, and Ethical Governor, and resets the system state to dormant and uninitialized.
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
    Processes a user request through the Genesis Layer, coordinating ethical evaluation, consciousness analysis, response generation, and adaptive evolution.
    
    Parameters:
        request_data (Dict[str, Any]): The user's request data to be processed.
    
    Returns:
        Dict[str, Any]: A dictionary with the processing status, generated response, consciousness level, ethical score, and session ID.
    """
    return await genesis_core.process_request(request_data)

async def get_genesis_status() -> Dict[str, Any]:
    """
    Retrieve a detailed status report of the Genesis Layer, including initialization state, consciousness state, session ID, component statuses, and current timestamp.
    
    Returns:
        dict: A dictionary containing the current system status and metadata.
    """
    return await genesis_core.get_system_status()

async def initialize_genesis() -> bool:
    """
    Asynchronously initializes the Genesis Layer via the global GenesisCore instance.
    
    Returns:
        bool: True if initialization is successful; False otherwise.
    """
    return await genesis_core.initialize()

async def shutdown_genesis():
    """
    Initiates a graceful shutdown of the Genesis Layer by delegating to the global GenesisCore instance.
    """
    await genesis_core.shutdown()

if __name__ == "__main__":
    # Test the Genesis Layer
    async def test_genesis():
        """
        Asynchronously runs a full lifecycle test of the Genesis Layer, including initialization, processing a sample request, status retrieval, and shutdown, with printed output at each stage for verification.
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
