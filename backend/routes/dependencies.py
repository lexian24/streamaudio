"""
Shared dependencies for routes.
"""

from services.auto_recorder import AutoRecorder

# Global instances (will be set by main.py during startup)
auto_recorder: AutoRecorder = None

def get_auto_recorder() -> AutoRecorder:
    """Get the global auto recorder instance"""
    return auto_recorder

def set_auto_recorder(recorder: AutoRecorder):
    """Set the global auto recorder instance (called from main.py)"""
    global auto_recorder
    auto_recorder = recorder